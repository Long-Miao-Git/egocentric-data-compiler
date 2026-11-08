#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EGO 视频编译工具 —— 命令行入口

把一段第一人称（EGO）长视频，编译成结构化的训练数据单元：
    操作片段切分 → 证据帧抽取 → 实体标注 → 七模式 VQA → JSON 产出

用法示例
--------
# 1. 编译单个本地视频
python compile.py --input ./videos/demo.mp4

# 2. 编译整个目录
python compile.py --input ./videos/

# 3. 从网络地址获取视频后编译
python compile.py --url https://example.com/ego_clip.mp4

# 4. 指定输出目录与并行度
python compile.py --input ./videos/ --output ./my_dataset --workers 8

# 5. 只做切分（不调用模型，零成本，用于先看切分效果）
python compile.py --input ./videos/demo.mp4 --stage segment

# 6. 复用自己的配置文件
python compile.py --input ./videos/ --config ./my_config.yaml

# 7. 查看当前生效的全部参数
python compile.py --show-config
"""

import argparse
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.utils import write_json  # noqa: E402
from src.segmenter import segment_video_auto  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".MP4", ".AVI", ".MOV"}


# ---------------------------------------------------------------- 输入收集

def collect_local(path: Path, recursive: bool) -> list[Path]:
    """从文件或目录收集视频"""
    if path.is_file():
        if path.suffix not in VIDEO_EXTS:
            print(f"  [跳过] 不是支持的视频格式: {path.name}")
            return []
        return [path]
    if not path.is_dir():
        print(f"  [跳过] 路径不存在: {path}")
        return []
    it = path.rglob("*") if recursive else path.glob("*")
    vids = sorted(p for p in it if p.is_file() and p.suffix in VIDEO_EXTS)
    return vids


def download_url(url: str, dest_dir: Path) -> Path:
    """下载网络视频到本地（支持 http/https，自动跟随重定向）"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urllib.parse.urlparse(url).path).name or "download.mp4"
    if not Path(name).suffix:
        name += ".mp4"
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [缓存] 已存在: {dest.name} ({dest.stat().st_size / 2**20:.1f} MB)")
        return dest

    print(f"  下载中: {url}")
    # 直接写入目标文件。不使用「先写 .part 再 rename」的方案，
    # 因为部分环境会对 rename/unlink 做拦截，导致 WinError 5。
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "ego-compiler/1.0")]
    with opener.open(url, timeout=60) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        got, last = 0, time.time()
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 2:
                pct = f"{got / total * 100:.0f}%" if total else f"{got / 2**20:.0f}MB"
                print(f"    {pct}", flush=True)
                last = time.time()
    print(f"  完成: {dest.name} ({dest.stat().st_size / 2**20:.1f} MB)")
    return dest


# ---------------------------------------------------------------- 单视频流程

def do_segment(video: Path, out_dir: Path, analysis_fps: float,
               min_peak_ratio: float, force: bool) -> Path | None:
    """切分阶段：产出 <stem>.segments.json"""
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / f"{video.stem}.segments.json"
    if seg_path.exists() and not force:
        print(f"  切分已存在，跳过（--force 可重跑）")
        return seg_path

    t0 = time.time()
    print(f"  切分中（采样 {analysis_fps}fps）...", flush=True)
    r = segment_video_auto(str(video), analysis_fps=analysis_fps,
                           min_peak_ratio=min_peak_ratio)
    r["video"] = video.name
    r["video_path"] = str(video)
    write_json(seg_path, r)

    n = r.get("segment_count", 0)
    if n == 0:
        warns = "；".join(r.get("warnings") or []) or "未检出有效操作段"
        print(f"  切分失败: {warns}")
        return None
    print(f"  切分完成: {n} 个片段，有效率 {r.get('effective_ratio', 0) * 100:.0f}%，"
          f"主周期 {r.get('period_sec')}s，耗时 {time.time() - t0:.0f}s")
    return seg_path


def do_annotate(video: Path, seg_path: Path, out_dir: Path, frames_dir: Path,
                frames_per_seg: int, clarity: float, workers: int,
                force: bool) -> tuple[bool, float]:
    """标注阶段：产出 <stem>.compiled.json"""
    from src.annotator import annotate_one

    compiled_path = out_dir / f"{video.stem}.compiled.json"
    if compiled_path.exists() and not force:
        print(f"  已编译，跳过（--force 可重跑）")
        return True, 0.0

    t0 = time.time()
    result, out_path, cost = annotate_one(
        str(video), str(seg_path), out_dir, frames_dir,
        frames_per_seg=frames_per_seg, clarity=clarity, workers=workers)
    st = result["stats"]
    print(f"  编译完成: 片段 {st['segments_compiled']}/{st['segments_total']}，"
          f"实体 {st['total_entities']}，VQA {st['total_vqa']} 条，"
          f"空片段 {st['empty_segments']}，成本 ¥{cost:.2f}，"
          f"耗时 {time.time() - t0:.0f}s")
    print(f"  产物: {out_path}")
    return True, cost


# ---------------------------------------------------------------- 主流程

def show_config() -> None:
    """打印当前生效配置，便于排查"""
    keys = [
        ("API base_url", Config.DASHSCOPE_BASE_URL),
        ("API key", "已配置" if Config.DASHSCOPE_API_KEY else "未配置"),
        ("配置文件", str(Config._cfg_path) if Config._cfg_path else "未找到（使用默认值）"),
        ("输出根目录", str(Config.OUTPUT_ROOT)),
    ]
    for k, v in keys:
        print(f"  {k}: {v}")
    print(f"  VLM 模型: {Config.vlm().get('model')}")
    print(f"  LLM 模型: {Config.llm().get('model')}")
    print("\n  sampling:")
    for k, v in Config.sampling().items():
        print(f"    {k}: {v}")
    print("  vqa:")
    for k, v in Config.vqa_config().items():
        print(f"    {k}: {v}")
    print("  cost:")
    for k, v in Config.cost_rates().items():
        print(f"    {k}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="EGO 视频编译工具：长视频 → 结构化训练数据单元",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    src = ap.add_argument_group("输入")
    src.add_argument("--input", "-i", action="append", default=[],
                     help="本地视频文件或目录，可重复指定")
    src.add_argument("--url", "-u", action="append", default=[],
                     help="网络视频地址（http/https），可重复指定")
    src.add_argument("--recursive", "-r", action="store_true",
                     help="输入为目录时递归查找子目录")

    out = ap.add_argument_group("输出")
    out.add_argument("--output", "-o", default=None,
                     help="输出根目录（默认 ./output）")
    out.add_argument("--force", action="store_true",
                     help="重跑已存在的切分/编译结果")

    algo = ap.add_argument_group("算法参数（不传则用 config.yaml）")
    algo.add_argument("--stage", default="all", choices=["all", "segment", "annotate"],
                      help="只跑某阶段：segment=仅切分(不花钱)，annotate=仅标注")
    algo.add_argument("--analysis-fps", type=float, default=None,
                      help="切分时的运动曲线采样帧率，默认 5.0")
    algo.add_argument("--min-peak-ratio", type=float, default=None,
                      help="切分峰值阈值比例，默认 0.65，越小切得越细")
    algo.add_argument("--frames-per-seg", type=int, default=None,
                      help="每个片段抽取的证据帧数，默认 4")
    algo.add_argument("--clarity", type=float, default=None,
                      help="清晰度阈值（Laplacian 方差），低于此值判为模糊帧，默认 60")
    algo.add_argument("--workers", "-j", type=int, default=None,
                      help="片段级并行线程数，默认 4")

    misc = ap.add_argument_group("其他")
    misc.add_argument("--config", "-c", default=None, help="配置文件路径（默认 ./config.yaml）")
    misc.add_argument("--show-config", action="store_true", help="打印当前生效配置后退出")
    args = ap.parse_args()

    Config.load(args.config)
    if args.output:
        Config.set_output_root(args.output)

    if args.show_config:
        show_config()
        return 0

    # 参数：命令行 > config.yaml > 内置默认
    sampling = Config.sampling()
    analysis_fps = args.analysis_fps if args.analysis_fps is not None \
        else float(sampling.get("analysis_fps", 5.0))
    min_peak_ratio = args.min_peak_ratio if args.min_peak_ratio is not None \
        else float(sampling.get("min_peak_ratio", 0.65))
    frames_per_seg = args.frames_per_seg if args.frames_per_seg is not None \
        else int(sampling.get("frames_per_seg", 4))
    clarity = args.clarity if args.clarity is not None \
        else float(sampling.get("min_clarity_threshold", 60))
    workers = args.workers if args.workers is not None \
        else int(sampling.get("workers", 4))

    out_root = Config.OUTPUT_ROOT
    compiled_dir = out_root / "compiled"          # 一切产物在这里
    frames_dir = out_root / "frames_cache"        # 抽帧缓存（可删）
    download_dir = out_root / "downloads"         # 网络视频落盘位置
    for d in (compiled_dir, frames_dir, download_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 收集输入
    videos: list[Path] = []
    print("输入收集：")
    for p in args.input:
        videos += collect_local(Path(p).expanduser(), args.recursive)
    for u in args.url:
        try:
            videos.append(download_url(u, download_dir))
        except Exception as e:
            print(f"  [失败] 下载 {u}: {type(e).__name__}: {e}")

    if not videos:
        print("\n未找到任何可编译的视频。")
        print("  本地视频：python compile.py --input ./your_video.mp4")
        print("  目录批量：python compile.py --input ./videos/ -r")
        print("  网络视频：python compile.py --url https://example.com/a.mp4")
        return 1

    # 去重并保序
    seen, uniq = set(), []
    for v in videos:
        rv = v.resolve()
        if rv in seen:
            continue
        seen.add(rv)
        uniq.append(v)
    videos = uniq

    need_model = args.stage in ("all", "annotate")
    if need_model:
        try:
            Config.require_api_key()
        except RuntimeError as e:
            print(f"\n{e}")
            return 1

    print(f"\n共 {len(videos)} 个视频，阶段: {args.stage}")
    print(f"输出目录: {out_root}")
    print(f"VLM: {Config.vlm().get('model')} / LLM: {Config.llm().get('model')}")
    print("=" * 66)

    ok, failed, skipped, total_cost = 0, [], 0, 0.0
    t_all = time.time()

    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {vp.name}")
        try:
            seg_path = None
            if args.stage in ("all", "segment"):
                seg_path = do_segment(vp, compiled_dir, analysis_fps,
                                      min_peak_ratio, args.force)
                if seg_path is None:
                    failed.append((vp.name, "切分无有效片段"))
                    continue
            if args.stage in ("all", "annotate"):
                if seg_path is None:
                    seg_path = compiled_dir / f"{vp.stem}.segments.json"
                if not seg_path.exists():
                    failed.append((vp.name, f"缺少切分结果 {seg_path.name}，请先跑 --stage segment"))
                    continue
                success, cost = do_annotate(
                    vp, seg_path, compiled_dir, frames_dir,
                    frames_per_seg, clarity, workers, args.force)
                total_cost += cost
                if not success:
                    failed.append((vp.name, "编译失败"))
                    continue
            ok += 1
        except KeyboardInterrupt:
            print("\n用户中断，已完成的产物保留，可重跑续做。")
            sys.exit(130)
        except Exception as e:
            import traceback
            print(f"  失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append((vp.name, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 66)
    print(f"完成: {ok} 成功 / {len(failed)} 失败 / 共 {len(videos)} 个")
    print(f"累计模型成本: ¥{total_cost:.2f}   总耗时: {(time.time() - t_all) / 60:.1f} 分钟")
    print(f"产物目录: {compiled_dir}")
    if failed:
        print("\n失败明细：")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
