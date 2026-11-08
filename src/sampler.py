"""帧采样模块：抽帧 + 场景分段 + 清晰度预筛 + 时间戳水印"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import logging

from .config import Config
from .schemas import FrameInfo

logger = logging.getLogger(__name__)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian方差，用于评估图像清晰度"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _add_timestamp(image_path: str, timestamp: float, frame_id: str):
    """在帧图像顶部加时间戳水印"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    text = f"[{timestamp:.1f}s] {frame_id}"
    # 白底黑字条
    bar_h = 28
    draw.rectangle([(0, 0), (img.width, bar_h)], fill="white", outline="black")
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill="black", font=font)
    img.save(image_path, quality=95)


def get_video_info(video_path: str) -> dict:
    """获取视频基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": 0.0,
    }
    if info["fps"] > 0:
        info["duration_sec"] = info["total_frames"] / info["fps"]
    cap.release()
    return info


def detect_scenes(video_path: str, threshold: float = 27.0) -> list[dict]:
    """
    用 PySceneDetect 检测场景边界。
    返回 [{scene_id, start_sec, end_sec}] 列表。
    """
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(video_path, ContentDetector(threshold=threshold))
        result = []
        for i, (start, end) in enumerate(scenes):
            result.append({
                "scene_id": i,
                "start_sec": start.get_seconds(),
                "end_sec": end.get_seconds(),
            })
        return result if result else [{"scene_id": 0, "start_sec": 0, "end_sec": 0}]
    except Exception as e:
        logger.warning(f"场景检测失败，降级为单场景: {e}")
        return [{"scene_id": 0, "start_sec": 0, "end_sec": 0}]


def extract_frames(
    video_path: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    clarity_threshold: Optional[float] = None,
    progress_callback=None,
) -> list[FrameInfo]:
    """
    从视频中抽帧。
    
    Args:
        video_path: 视频文件路径
        fps: 抽帧频率（默认从配置读取）
        max_frames: 最大帧数（默认从配置读取）
        clarity_threshold: 清晰度阈值（默认从配置读取）
        progress_callback: 回调函数(completed, total)
    
    Returns:
        FrameInfo列表
    """
    s_cfg = Config.sampling()
    fps = fps or s_cfg.get("fps", 2)
    max_frames = max_frames or s_cfg.get("max_frames", 500)
    clarity_threshold = clarity_threshold or s_cfg.get("min_clarity_threshold", 80)
    batch_size = s_cfg.get("batch_size", 8)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_video_frames / video_fps

    # 计算要抽的帧位置
    sample_interval = max(1, int(video_fps / fps))
    target_frame_indices = list(range(0, total_video_frames, sample_interval))
    if max_frames and len(target_frame_indices) > max_frames:
        # 均匀采样
        step = len(target_frame_indices) / max_frames
        target_frame_indices = [target_frame_indices[int(i * step)] for i in range(max_frames)]

    logger.info(f"视频: {video_fps:.1f}fps, {total_video_frames}帧, {duration:.1f}s → 抽帧{len(target_frame_indices)}张 @{fps}fps")

    # 场景检测
    scenes = detect_scenes(video_path, threshold=Config.get("scene_detection", "threshold", default=27.0))

    # 确保帧目录存在
    frames_dir = Config.FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames: list[FrameInfo] = []
    completed = 0
    total = len(target_frame_indices)

    for frame_idx in target_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"读取帧 {frame_idx} 失败，跳过")
            completed += 1
            continue

        timestamp = frame_idx / video_fps
        frame_id = f"f{len(frames) + 1:03d}"

        # 清晰度评估
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clarity = _laplacian_variance(gray)

        if clarity < clarity_threshold:
            logger.debug(f"帧 {frame_id} 清晰度 {clarity:.1f} < {clarity_threshold}，跳过")
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            continue

        # 确定场景ID
        scene_id = 0
        for s in scenes:
            if s["start_sec"] <= timestamp < s["end_sec"]:
                scene_id = s["scene_id"]
                break

        # 保存帧图片（用imencode解决中文路径问题）
        img_path = str(frames_dir / f"{frame_id}.jpg")
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            buffer.tofile(img_path)
        else:
            logger.warning(f"帧 {frame_id} 编码失败，跳过")
            continue

        # 加时间戳水印
        _add_timestamp(img_path, timestamp, frame_id)

        frames.append(FrameInfo(
            frame_id=frame_id,
            timestamp=round(timestamp, 2),
            image_path=img_path,
            clarity_score=round(clarity, 1),
            scene_id=scene_id,
        ))

        completed += 1
        if progress_callback:
            progress_callback(completed, total)

    cap.release()
    logger.info(f"抽帧完成: {len(frames)}/{total} 帧通过清晰度筛选")

    # 保存帧列表
    import json
    frames_meta = [f.model_dump() for f in frames]
    with open(Config.CACHE_DIR / "frames.json", "w", encoding="utf-8") as f:
        json.dump(frames_meta, f, ensure_ascii=False, indent=2)

    return frames
