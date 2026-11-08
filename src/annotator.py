# -*- coding: utf-8 -*-
"""单个视频的标注编译：切分结果 → 抽取证据帧 → 实体融合 → VQA 生成 → 校验

设计要点：
  - 片段级抽帧：只抽取有效操作片段的帧，不做全视频固定频率抽帧，成本可降一个数量级
  - 逐段编译：每个操作片段独立走完 VLM 帧描述 → 实体融合 → VQA 生成 → 校验，
    避免一次性把数百帧喂给模型导致输出退化
  - 幂等：帧图落盘后按路径缓存，中断重跑不会重复抽帧
"""

import json
from pathlib import Path

import cv2

from .config import Config
from .cost_tracker import CostTracker
from .llm_client import LLMClient
from .sampler import get_video_info
from .utils import write_json
from .schemas import FrameInfo
from .validator import Validator
from .vocab import VocabManager
from .vlm_client import VLMClient

# 帧缓存落盘规格：与 VLM 端降采样目标（960px）保持一致，避免存原分辨率 PNG
FRAME_EXT = ".jpg"
FRAME_MAX_W = 960
FRAME_JPEG_Q = 92


def _downscale(frame, max_w):
    """按宽度上限等比缩小（不做放大），与 VLM 端编码尺寸对齐"""
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
    return frame


def extract_segment_frames(video_path, segments, frames_per_seg, clarity_threshold, frames_dir):
    """片段级抽帧：每个片段抽 N 帧（起始 / 峰时 / 中点 / 结束）

    Returns:
        (FrameInfo 列表, 被清晰度阈值过滤掉的帧数)
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    stem = Path(video_path).stem
    frames, filtered = [], 0
    for seg in segments:
        a, b = seg["start_sec"], seg["end_sec"]
        pk = seg.get("peak_time_sec", (a + b) / 2)
        # 候选时间点：起始+0.3s、峰时、中点、结束-0.3s（按数量截取）
        cands = [a + 0.3, min(max(pk, a + 0.3), b - 0.3), (a + b) / 2, b - 0.3]
        seen, picked = set(), []
        for t in cands:
            t = round(min(max(t, a), b), 2)
            if t not in seen:
                seen.add(t)
                picked.append(t)
        # 按时间升序排列后再编号，避免峰时晚于中点时出现 f3 时间戳早于 f2 的倒序，
        # 那会让模型看到的帧序与时间标注矛盾，污染 temporal / trajectory 类问答
        picked = sorted(picked)[:frames_per_seg]

        for k, t in enumerate(picked):
            frame_idx = int(t * video_fps)
            fid = f"s{seg['index']:03d}_f{k + 1}"
            img_path = frames_dir / f"{stem}_{fid}{FRAME_EXT}"
            if img_path.exists():  # 命中缓存，便于中断重跑
                frames.append(FrameInfo(
                    frame_id=fid, timestamp=round(t, 2), image_path=str(img_path),
                    clarity_score=100.0, scene_id=seg["index"]))
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clarity = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if clarity < clarity_threshold:
                filtered += 1
                continue
            # 存 960px JPEG 而非原分辨率 PNG：模型端本就会降采样，
            # 原分辨率对效果零增益却让帧缓存膨胀一个数量级
            frame = _downscale(frame, FRAME_MAX_W)
            cv2.imencode(FRAME_EXT, frame,
                         [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_Q])[1].tofile(str(img_path))
            frames.append(FrameInfo(
                frame_id=fid, timestamp=round(t, 2), image_path=str(img_path),
                clarity_score=round(clarity, 1), scene_id=seg["index"],
            ))
    cap.release()
    return frames, filtered


def annotate_one(video_path, seg_path, out_dir, frames_dir,
                 frames_per_seg=4, clarity=60.0, workers=4, verbose=True):
    """对单个视频执行完整编译，写出 <视频名>.compiled.json

    Returns:
        (结果 dict, 产物路径, 本次成本)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    seg_data = json.load(open(seg_path, encoding="utf-8"))
    segments = seg_data.get("segments", [])
    if not segments:
        raise RuntimeError("切分结果无有效片段")

    video_info = get_video_info(str(video_path))
    cost_tracker = CostTracker()
    vocab = VocabManager()
    vlm = VLMClient(cost_tracker)
    llm = LLMClient(cost_tracker)
    validator = Validator(vocab)

    # 1. 片段级抽帧（全部片段一次抽完，带缓存跳过）
    frames, filtered = extract_segment_frames(
        video_path, segments, frames_per_seg, clarity, frames_dir)
    if not frames:
        raise RuntimeError("无帧通过清晰度筛选，可尝试调低 --clarity 阈值")

    # 2. 按片段分组
    groups = {}
    for f in frames:
        groups.setdefault(f.scene_id, []).append(f)
    seg_map = {s["index"]: s for s in segments}
    todo = [(idx, fs) for idx, fs in sorted(groups.items()) if idx in seg_map]
    if verbose:
        print(f"   逐段编译: {len(todo)} 个片段, {workers} 线程并行", flush=True)

    def _compile_segment(item):
        idx, seg_frames = item
        seg = seg_map[idx]
        try:
            descs = vlm.describe_frames(seg_frames)
            entities = llm.fuse_entities(descs)
            vqa_items = llm.generate_vqa(entities, descs)
            validation = validator.validate(vqa_items, entities, descs)
            # 校验失败递归修复一次
            if validation.status == "fail" and validation.errors:
                vqa_json = json.dumps(
                    {"vqa_items": [v.model_dump() for v in vqa_items]}, ensure_ascii=False)
                schema_hint = ("VQA items: mode(7种), question, answer, "
                               "evidence_frame_ids, reasoning, difficulty(easy/medium/hard)")
                repaired = llm.repair_json(vqa_json, validation.errors[:10], schema_hint)
                if repaired and "vqa_items" in repaired:
                    from .schemas import VQAItem, VQA_MODES
                    vqa_items = [VQAItem(
                        mode=it.get("mode", "summary") if it.get("mode") in VQA_MODES else "summary",
                        question=it.get("question", ""), answer=it.get("answer", ""),
                        evidence_frame_ids=it.get("evidence_frame_ids", []),
                        reasoning=it.get("reasoning", ""),
                        difficulty=it.get("difficulty", "easy"),
                    ) for it in repaired.get("vqa_items", [])]
                    validation.repaired_count = 1
            return idx, {
                "segment": seg,
                "frames": [{"frame_id": f.frame_id, "timestamp": f.timestamp,
                            "image_path": f.image_path, "clarity": f.clarity_score}
                           for f in seg_frames],
                "entities": [e.model_dump() for e in entities],
                "vqa_items": [v.model_dump() for v in vqa_items],
                "validation": validation.model_dump(),
            }, None
        except Exception as e:
            return idx, None, f"{type(e).__name__}: {str(e)[:150]}"

    units, failed = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_compile_segment, t): t[0] for t in todo}
        for fut in as_completed(futs):
            idx, unit, err = fut.result()
            done += 1
            if err:
                failed.append((idx, err))
                if verbose:
                    print(f"   片段{idx} 失败 {err} ({done}/{len(todo)})", flush=True)
            else:
                units.append(unit)
                if verbose:
                    print(f"   片段{idx} 完成 实体{len(unit['entities'])} "
                          f"VQA{len(unit['vqa_items'])}条 ({done}/{len(todo)})", flush=True)
    units.sort(key=lambda u: u["segment"]["index"])

    total_entities = sum(len(u["entities"]) for u in units)
    total_vqa = sum(len(u["vqa_items"]) for u in units)
    mode_dist = {}
    for u in units:
        for v in u["vqa_items"]:
            mode_dist[v["mode"]] = mode_dist.get(v["mode"], 0) + 1
    empty = sum(1 for u in units if not u["entities"] and not u["vqa_items"])

    result = {
        "video_name": Path(video_path).name,
        "video_info": video_info,
        "segmentation": {
            "segment_count": seg_data.get("segment_count"),
            "period_sec": seg_data.get("period_sec"),
            "period_corr": seg_data.get("period_corr"),
            "effective_ratio": seg_data.get("effective_ratio"),
            "segments": segments,
            "warnings": seg_data.get("warnings", []),
        },
        "segment_units": units,
        "stats": {
            "segments_total": len(segments),
            "segments_compiled": len(units),
            "segments_failed": len(failed),
            "total_entities": total_entities,
            "total_vqa": total_vqa,
            "vqa_mode_distribution": mode_dist,
            "empty_segments": empty,
            "empty_rate": round(empty / len(units), 4) if units else 0.0,
        },
        "frame_filtering": {"extracted": len(frames), "filtered_by_clarity": filtered},
        "total_cost_cny": cost_tracker.total_cost(),
        "cost_records": [
            r.model_dump() if hasattr(r, "model_dump") else r
            for r in cost_tracker.records
        ],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(video_path).stem}.compiled.json"
    write_json(out_path, result)
    return result, out_path, cost_tracker.total_cost()
