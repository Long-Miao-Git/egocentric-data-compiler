# -*- coding: utf-8 -*-
"""分析长视频运动能量曲线，输出分段时间标签（调试版）
用法: python segment_video.py <video_path> [--fps 5] [--view]
"""
import sys, os, json
from pathlib import Path
import cv2
import numpy as np

def compute_motion_curve(video_path, analysis_fps=5):
    """按 analysis_fps 采样，计算每帧的光流能量与方向一致性

    快速路径: 用 ffmpeg 直接抽取低分辨率帧序列(H.265解码在C层多线程, 比cv2顺序解码快5-10倍),
    失败时回退 cv2 顺序解码。
    """
    import subprocess
    import tempfile
    import shutil

    video_path = str(video_path)
    cap0 = cv2.VideoCapture(video_path)
    if not cap0.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')
    video_fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    cap0.release()

    # 找 ffmpeg: bin/ 下的随包版本优先, 其次系统 PATH, 最后 imageio-ffmpeg 自带版本
    ffmpeg_exe = Path(__file__).resolve().parent.parent / "bin" / "ffmpeg.exe"
    if not ffmpeg_exe.exists():
        from shutil import which
        sys_ff = which("ffmpeg")
        if sys_ff:
            ffmpeg_exe = Path(sys_ff)
    if not ffmpeg_exe.exists():
        try:
            import imageio_ffmpeg
            ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            ffmpeg_exe = None

    times, energies, consistencies = [], [], []
    prev_gray = None

    def _process(img, t):
        nonlocal prev_gray
        small = cv2.resize(img, (400, 225))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask = mag > 1.0
            energy = float(np.percentile(mag, 90))
            if mask.sum() > 50:
                sin_s = np.sum(np.sin(ang[mask])); cos_s = np.sum(np.cos(ang[mask]))
                consist = float(np.hypot(sin_s, cos_s) / mask.sum())
            else:
                consist = 1.0
            times.append(t); energies.append(energy); consistencies.append(consist)
        prev_gray = gray

    if ffmpeg_exe and ffmpeg_exe.exists():
        tmp = Path(tempfile.mkdtemp(prefix="motion_"))
        try:
            cmd = [str(ffmpeg_exe), "-hide_banner", "-loglevel", "error",
                   "-i", video_path, "-vf", f"fps={analysis_fps},scale=400:225",
                   "-q:v", "5", str(tmp / "f_%06d.jpg")]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            imgs = sorted(tmp.glob("f_*.jpg"))
            if r.returncode == 0 and imgs:
                interval = 1.0 / analysis_fps
                for i, p in enumerate(imgs):
                    img = cv2.imread(str(p))
                    if img is not None:
                        _process(img, i * interval)
                if times:
                    return np.array(times), np.array(energies), np.array(consistencies), duration
            print(f"[compute_motion_curve] ffmpeg快速路径失败({r.returncode}), 回退cv2顺序解码")
        except Exception as e:
            print(f"[compute_motion_curve] ffmpeg路径异常({e}), 回退cv2顺序解码")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # 回退: cv2 顺序解码
    cap = cv2.VideoCapture(video_path)
    step = max(1, int(round(video_fps / analysis_fps)))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            _process(frame, idx / video_fps)
        idx += 1
    cap.release()
    return np.array(times), np.array(energies), np.array(consistencies), duration


def smooth(x, k=5):
    if len(x) < k:
        return x.copy()
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode='same')


def find_segments(times, energy, consist, duration, cfg):
    """L1: 能量阈值找活动段; L2: 自相关找主周期精修边界"""
    e = smooth(energy, k=5)
    # 自适应阈值: 用中位数+若干倍MAD，避免全活动/全静止失效
    med = np.median(e)
    mad = np.median(np.abs(e - med))
    thr = max(cfg.get('min_thr', 1.5), med + 1.5 * max(mad, 0.3))
    active = e > thr

    # 形态学: 膨胀合并小间隙、腐蚀去除毛刺
    def morph(mask, op, k):
        out = mask.copy()
        for i in range(len(mask)):
            lo, hi = max(0, i - k), min(len(mask), i + k + 1)
            seg = mask[lo:hi]
            out[i] = (np.any(seg) if op == 'dilate' else np.all(seg))
        return out
    gap = cfg.get('merge_gap_s', 1.5)
    k_merge = int(gap * 2)  # 采样间隔约0.2s
    active = morph(active, 'dilate', k_merge)
    k_min = int(cfg.get('min_seg_s', 1.0) * 2)
    active = morph(active, 'erode', k_min)

    # 提取连续活动段
    segs = []
    i = 0
    n = len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            segs.append((times[i], times[j - 1]))
            i = j
        else:
            i += 1

    # 周期分析: 活动段时长中位数作为单元时长先验
    durs = [b - a for a, b in segs if b - a > 0.3]
    period = float(np.median(durs)) if durs else None
    return segs, thr, period, e


def refine_by_valleys(times, energy, segs, cfg):
    """在段内找能量谷底精修起止点（段内前20%找起始谷、后20%找结束谷）"""
    e = smooth(energy, k=3)
    refined = []
    for a, b in segs:
        ia = np.searchsorted(times, a); ib = np.searchsorted(times, b)
        ia = max(0, ia); ib = min(len(times) - 1, ib)
        if ib - ia < 5:
            refined.append((float(times[ia]), float(times[ib])))
            continue
        # 起始点: 在[ia, ia+20%]内找谷底
        w = max(2, int((ib - ia) * 0.2))
        sa = ia + int(np.argmin(e[ia:ia + w]))
        sb = ib - w + int(np.argmin(e[ib - w:ib + 1]))
        refined.append((float(times[sa]), float(times[sb])))
    return refined


def detect_period(times, energy, min_lag_s=2.0, max_lag_s=60.0):
    """自相关检测主周期(秒)。返回 (period_sec, corr_coef)，无周期返回 (None, 0)"""
    if len(times) < 30:
        return None, 0.0
    dt = float(times[1] - times[0])
    e = energy - energy.mean()
    e = np.convolve(e, np.ones(max(1, int(1.0 / dt))) / max(1, int(1.0 / dt)), mode='same')
    lo, hi = int(min_lag_s / dt), min(int(max_lag_s / dt), len(e) - 2)
    if hi <= lo:
        return None, 0.0
    lags = np.arange(lo, hi)
    acs = np.array([np.corrcoef(e[:-l], e[l:])[0, 1] for l in lags])
    best = int(np.argmax(acs))
    return float(lags[best] * dt), float(acs[best])


def segment_video_auto(video_path, analysis_fps=5, min_peak_ratio=0.65):
    """峰值锚定+谷底定界切分（7分钟实测验证版，周期自适应）

    Returns:
        dict: {
            video, duration_sec, analysis_fps,
            period_sec, period_corr, threshold,
            segment_count, segments: [{index,start_sec,end_sec,duration_sec,peak_time_sec,peak_energy,flag}],
            effective_ratio, effective_sec, warnings: [...]
        }
    """
    times, energy, consist, duration = compute_motion_curve(video_path, analysis_fps)
    if len(times) < 30:
        return {'video': video_path, 'duration_sec': round(duration, 2), 'segment_count': 0,
                'segments': [], 'warnings': ['采样帧不足，无法切分']}

    es = np.convolve(energy, np.ones(int(2.0 / (times[1] - times[0]))) / int(2.0 / (times[1] - times[0])), mode='same')

    # 1. 周期检测
    period, corr = detect_period(times, energy)

    # 2. 峰值检测(最小间距按周期自适应: 0.6*T, 兜底6s)
    p_thr = np.percentile(es, min_peak_ratio * 100)
    min_dist = max(2.0, 0.6 * period) if period else 6.0
    dt = float(times[1] - times[0])
    k_min = max(1, int(min_dist / dt))
    peaks = []
    i = 1
    while i < len(es) - 1:
        if es[i] >= es[i - 1] and es[i] >= es[i + 1] and es[i] > p_thr:
            if not peaks or (times[i] - peaks[-1]) > min_dist:
                peaks.append(float(times[i]))
            elif es[i] > es[np.searchsorted(times, peaks[-1])]:
                peaks[-1] = float(times[i])
            i += 2
        else:
            i += 1

    # 3. 谷底定界
    bounds = [float(times[0])]
    for a, b in zip(peaks[:-1], peaks[1:]):
        m = (times >= a) & (times <= b)
        seg = np.where(m)[0]
        if len(seg):
            bounds.append(float(times[seg[np.argmin(es[seg])]]))
    bounds.append(float(times[-1]))
    raw = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    # 4. 组装结果+异常标记
    segments, warnings = [], []
    for i, (a, b) in enumerate(raw, 1):
        m = (times >= a) & (times <= b)
        if not m.any():
            continue
        pk_t = float(times[m][np.argmax(es[m])])
        pk_e = float(es[m].max())
        dur = b - a
        flag = ''
        if period and dur > 1.8 * period:
            flag = 'long'      # 可能合并了两次操作或中途被打岔
        elif period and dur < 0.5 * period:
            flag = 'short'     # 可能误检
        segments.append({'index': i, 'start_sec': round(a, 2), 'end_sec': round(b, 2),
                         'duration_sec': round(dur, 2), 'peak_time_sec': round(pk_t, 2),
                         'peak_energy': round(pk_e, 2), 'flag': flag})

    total = sum(s['duration_sec'] for s in segments)
    ratio = total / duration if duration else 0
    if not segments:
        warnings.append('未检测到有效片段')
    if period and corr < 0.2:
        warnings.append(f'周期信号弱(r={corr:.2f})，视频可能非单一重复操作，切分结果需人工复核')
    if ratio > 0.98:
        warnings.append('有效率接近100%，可能未检出间歇，需人工复核')
    if ratio < 0.3:
        warnings.append('有效率偏低(<30%)，可能大量操作未被检出')

    return {
        'video': video_path, 'duration_sec': round(duration, 2), 'analysis_fps': analysis_fps,
        'period_sec': round(period, 2) if period else None, 'period_corr': round(corr, 3),
        'threshold': round(float(p_thr), 3),
        'segment_count': len(segments), 'segments': segments,
        'effective_sec': round(total, 2), 'effective_ratio': round(ratio, 4),
        'warnings': warnings,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="EGO长视频操作切分: 峰值锚定+谷底定界, 周期自适应",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("video", nargs="?", default="test-7min.mp4", help="输入视频路径")
    ap.add_argument("-o", "--output", default="output/segments_auto.json", help="输出JSON路径")
    ap.add_argument("--analysis-fps", type=float, default=5.0, help="分析采样频率")
    args = ap.parse_args()

    video_path = args.video
    analysis_fps = args.analysis_fps
    print(f'分析视频: {video_path} @ {analysis_fps}fps ...')

    # 直接调用自动切分(内部完成曲线计算, 避免重复计算)
    result = segment_video_auto(video_path, analysis_fps)
    if result.get('warnings') and result['segment_count'] == 0:
        print(f"❌ {result['warnings']}")
        return

    print(f"采样完成, 视频时长 {result['duration_sec']}s")
    print(f"主周期: {result['period_sec']}s (r={result['period_corr']}), 阈值: {result['threshold']}")
    print(f"=== 切分结果 ({result['segment_count']} 个有效单元, 有效率 {result['effective_ratio']*100:.1f}%) ===")
    for s in result['segments']:
        flag = f"  [{s['flag']}]" if s['flag'] else ''
        print(f"单元{s['index']:3d}: {s['start_sec']:7.1f}s - {s['end_sec']:7.1f}s  (时长{s['duration_sec']:5.1f}s){flag}")
    if result['warnings']:
        print('\n⚠️ 警告:')
        for w in result['warnings']:
            print('  -', w)

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'已保存: {out_path}')

if __name__ == '__main__':
    main()
