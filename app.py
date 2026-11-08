#!/usr/bin/env python
"""EGO片段编译工具 - Gradio 界面"""

import sys
import os
import json
import logging
import traceback
from pathlib import Path

# 确保项目根目录在path中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 将 imageio-ffmpeg 的 ffmpeg 复制为标准文件名 ffmpeg.exe 并加入 PATH，
# 确保 Gradio 视频组件能找到（imageio 自带文件名是 ffmpeg-win-x86_64-v7.1.exe，
# Windows 按 PATH 查找 "ffmpeg" 时无法匹配非标准文件名）
try:
    import shutil
    import imageio_ffmpeg
    _ff_src = imageio_ffmpeg.get_ffmpeg_exe()
    _ff_dir = PROJECT_ROOT / "bin"
    _ff_dir.mkdir(exist_ok=True)
    _ff_dst = _ff_dir / "ffmpeg.exe"
    if not _ff_dst.exists():
        shutil.copy2(_ff_src, _ff_dst)
    os.environ["PATH"] = str(_ff_dir) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import gradio as gr
import pandas as pd

from src.config import Config
from src.sampler import extract_frames, get_video_info
from src.vlm_client import VLMClient
from src.llm_client import LLMClient
from src.validator import Validator
from src.compiler import Compiler
from src.cost_tracker import CostTracker
from src.vocab import VocabManager, L0_OBJECT_CATEGORIES, L0_ACTION_CATEGORIES
from src.schemas import VQA_MODE_CN, VQA_MODES

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# 全局状态
_last_output = None


def run_pipeline(video_path, fps, batch_size, use_cache):
    """
    主流程：视频 → 抽帧 → VLM描述 → 实体融合 → VQA生成 → 校验 → 编译
    使用 generator 返回中间进度，最终返回完整结果。
    """
    global _last_output

    if not video_path:
        yield "❌ 请先上传视频文件", "", None, None, None, None
        return

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        logger.info(msg)
        return "\n".join(log_lines)

    try:
        # 0. 初始化
        Config._ensure_loaded()
        log_text = log("✅ 配置加载完成")
        yield log_text, "", None, None, None, None

        cost_tracker = CostTracker()
        vocab = VocabManager()
        vlm = VLMClient(cost_tracker)
        llm = LLMClient(cost_tracker)
        validator = Validator(vocab)
        compiler = Compiler()

        # 1. 获取视频信息
        log_text = log(f"📹 正在读取视频: {Path(video_path).name}")
        yield log_text, "", None, None, None, None

        video_info = get_video_info(video_path)
        log_text = log(
            f"   分辨率: {video_info['width']}x{video_info['height']}, "
            f"帧率: {video_info['fps']:.1f}fps, "
            f"时长: {video_info['duration_sec']:.1f}s"
        )
        yield log_text, "", None, None, None, None

        # 2. 抽帧
        log_text = log(f"🎬 开始抽帧 (fps={fps}, batch_size={batch_size})...")
        yield log_text, "", None, None, None, None

        def frame_progress(completed, total):
            log_lines.append(f"   抽帧进度: {completed}/{total}")

        frames = extract_frames(video_path, fps=fps)
        log_text = log(f"   ✅ 抽帧完成: {len(frames)} 帧通过清晰度筛选")
        yield log_text, "", None, None, None, None

        if not frames:
            log_text = log("❌ 没有帧通过清晰度筛选，请检查视频质量")
            yield log_text, "", None, None, None, None
            return

        # 3. VLM 帧描述
        log_text = log(f"🔍 调用 VLM (qwen-vl-max) 描述 {len(frames)} 帧...")
        yield log_text, "", None, None, None, None

        frame_descs = vlm.describe_frames(frames)
        total_objects = sum(len(fd.objects) for fd in frame_descs)
        total_actions = sum(len(fd.actions) for fd in frame_descs)
        log_text = log(
            f"   ✅ VLM描述完成: {len(frame_descs)} 帧, "
            f"共识别 {total_objects} 个物体, {total_actions} 个动作"
        )
        yield log_text, "", None, None, None, None

        # 4. 实体融合
        log_text = log("🔗 调用 LLM (qwen-max) 进行跨帧实体融合...")
        yield log_text, "", None, None, None, None

        entities = llm.fuse_entities(frame_descs)
        log_text = log(f"   ✅ 实体融合完成: {len(entities)} 个统一实体")
        yield log_text, "", None, None, None, None

        # 5. VQA 生成
        log_text = log("❓ 调用 LLM 生成七模式 VQA 问答对...")
        yield log_text, "", None, None, None, None

        vqa_items = llm.generate_vqa(entities, frame_descs)
        log_text = log(f"   ✅ VQA生成完成: {len(vqa_items)} 条问答对")
        yield log_text, "", None, None, None, None

        # 6. 校验 + 归一
        log_text = log("✔️ 执行硬校验 + 同义词归一...")
        yield log_text, "", None, None, None, None

        validation = validator.validate(vqa_items, entities, frame_descs)
        summary = validator.get_validation_summary(validation)
        log_text = log(f"   校验状态: {validation.status.upper()}\n   错误: {len(validation.errors)} | 警告: {len(validation.warnings)}")
        yield log_text, "", None, None, None, None

        # 6.1 如果有错误，尝试递归修复
        if validation.status == "fail" and validation.errors:
            log_text = log("🔧 检测到校验失败，尝试递归修复...")
            yield log_text, "", None, None, None, None

            # 把VQA转回JSON让LLM修复
            vqa_json = json.dumps(
                {"vqa_items": [v.model_dump() for v in vqa_items]},
                ensure_ascii=False,
            )
            schema_hint = "VQA items: mode(7种), question, answer, evidence_frame_ids, reasoning, difficulty(easy/medium/hard)"
            repaired = llm.repair_json(vqa_json, validation.errors[:10], schema_hint)

            if repaired and "vqa_items" in repaired:
                from src.schemas import VQAItem
                new_items = []
                for item in repaired.get("vqa_items", []):
                    mode = item.get("mode", "summary")
                    if mode not in VQA_MODES:
                        mode = "summary"
                    new_items.append(VQAItem(
                        mode=mode,
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        evidence_frame_ids=item.get("evidence_frame_ids", []),
                        reasoning=item.get("reasoning", ""),
                        difficulty=item.get("difficulty", "easy"),
                    ))
                vqa_items[:] = new_items
                validation.repaired_count = 1
                log_text = log("   ✅ 递归修复完成")
                yield log_text, "", None, None, None, None
            else:
                log_text = log("   ⚠️ 递归修复未成功，使用过滤后的结果")
                yield log_text, "", None, None, None, None

        # 7. 编译输出
        log_text = log("📦 正在编译最终输出JSON...")
        yield log_text, "", None, None, None, None

        output = compiler.compile(
            video_path=video_path,
            video_info_raw=video_info,
            frames_sampled=len(frames),
            frame_descriptions=frame_descs,
            entities=entities,
            vqa_items=vqa_items,
            validation=validation,
            cost_tracker=cost_tracker,
        )

        # 8. 生成展示内容
        output_summary = compiler.format_output_summary(output)
        log_text = log(f"\n✅ 编译完成！\n   总成本: ¥{output.total_cost_cny}")
        yield log_text, output_summary, str(Config.OUTPUT_DIR / "latest_compiled.json"), None, None, None

        # 9. VQA表格
        vqa_data = []
        for i, item in enumerate(vqa_items):
            vqa_data.append({
                "序号": i + 1,
                "模式": VQA_MODE_CN.get(item.mode, item.mode),
                "难度": item.difficulty,
                "问题": item.question[:80],
                "答案": item.answer[:80],
                "证据帧": ", ".join(item.evidence_frame_ids[:3]),
            })
        vqa_df = pd.DataFrame(vqa_data) if vqa_data else pd.DataFrame(columns=["序号", "模式", "难度", "问题", "答案", "证据帧"])

        # 10. 成本报告
        cost_report = cost_tracker.get_report()

        # 11. 词汇表
        vocab_data = vocab.get_all_words()
        word_rows = []
        for label, canonical in vocab_data["word_table"].items():
            word_rows.append({"自由文本": label, "规范名": canonical})
        word_df = pd.DataFrame(word_rows) if word_rows else pd.DataFrame(columns=["自由文本", "规范名"])

        candidate_rows = []
        for c in vocab_data["candidates"]:
            candidate_rows.append({"候选词": c["term"], "上下文": c.get("context", ""), "帧": c.get("frame_id", "")})
        cand_df = pd.DataFrame(candidate_rows) if candidate_rows else pd.DataFrame(columns=["候选词", "上下文", "帧"])

        _last_output = output
        yield log_text, output_summary, str(Config.OUTPUT_DIR / "latest_compiled.json"), vqa_df, cost_report, word_df

    except Exception as e:
        error_msg = f"❌ 流程出错: {e}\n\n{traceback.format_exc()}"
        log_text = log(error_msg)
        logger.error(error_msg)
        yield log_text, "", None, None, None, None


def load_vocab():
    """加载词汇表用于展示"""
    vocab = VocabManager()
    data = vocab.get_all_words()
    word_rows = []
    for label, canonical in data["word_table"].items():
        word_rows.append({"自由文本": label, "规范名": canonical})
    word_df = pd.DataFrame(word_rows) if word_rows else pd.DataFrame(columns=["自由文本", "规范名"])

    cand_rows = []
    for c in data["candidates"]:
        cand_rows.append({"候选词": c["term"], "上下文": c.get("context", ""), "帧": c.get("frame_id", "")})
    cand_df = pd.DataFrame(cand_rows) if cand_rows else pd.DataFrame(columns=["候选词", "上下文", "帧"])

    return word_df, cand_df


def add_synonym(canonical, synonym):
    """添加同义词"""
    if not canonical or not synonym:
        return *load_vocab(), "❌ 规范名和同义词不能为空"
    vocab = VocabManager()
    vocab.add_synonym(canonical.strip(), synonym.strip())
    word_df, cand_df = load_vocab()
    return word_df, cand_df, f"✅ 已添加: '{synonym}' → '{canonical}'"


def merge_entities_ui(source, target):
    """合并实体"""
    if not source or not target:
        return *load_vocab(), "❌ 源名和目标名不能为空"
    vocab = VocabManager()
    vocab.merge_entities(source.strip(), target.strip())
    word_df, cand_df = load_vocab()
    return word_df, cand_df, f"✅ 已合并: '{source}' → '{target}'"


def clear_candidates_ui():
    vocab = VocabManager()
    vocab.clear_candidates()
    word_df, _ = load_vocab()
    return word_df, pd.DataFrame(columns=["候选词", "上下文", "帧"]), "✅ 候选词已清空"


# ============================================================
#  视频理解预览（7维度）
# ============================================================

ASPECTS = [
    ("temporal", "⏱️ 时序推理", "事件先后顺序、持续时间、过程步骤"),
    ("spatial", "📍 空间布局", "物体间空间关系、位置、相对方位"),
    ("attribute", "🏷️ 物体属性", "颜色、大小、材质、状态等物理属性"),
    ("mechanics", "🔧 力学/动力学", "力的方向、大小、作用方式、物理效果"),
    ("reasoning", "🧠 因果推理", "为什么这样做、因果关系、目的推断"),
    ("summary", "📝 任务总结", "整段视频的任务目标、整体过程概括"),
    ("trajectory", "🛤️ 运动轨迹", "物体或手部的运动路径、轨迹描述"),
]

ASPECT_COLORS = {
    "temporal": "#E3F2FD", "spatial": "#E8F5E9", "attribute": "#FFF3E0",
    "mechanics": "#FCE4EC", "reasoning": "#F3E5F5", "summary": "#E0F7FA",
    "trajectory": "#FFFDE7",
}
ASPECT_BORDERS = {
    "temporal": "#1976D2", "spatial": "#388E3C", "attribute": "#F57C00",
    "mechanics": "#C2185B", "reasoning": "#7B1FA2", "summary": "#0097A7",
    "trajectory": "#FBC02D",
}


def format_understanding_html(output_file_path):
    """读取编译JSON，生成7维度理解预览HTML"""
    if not output_file_path:
        return "<div style='text-align:center;padding:60px;color:#999;font-size:16px;'>尚未编译，请先在「编译流程」页上传视频并点击「开始编译」</div>"

    p = Path(output_file_path)
    if not p.exists():
        return "<div style='text-align:center;padding:60px;color:#999;font-size:16px;'>输出文件不存在，请先编译</div>"

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "<div style='text-align:center;padding:60px;color:#f44336;'>读取输出文件失败</div>"

    vqa_items = data.get("vqa_items", [])
    entities = data.get("entities", [])
    video_info = data.get("video_info", {})

    # 按7维度分组
    by_mode = {}
    for v in vqa_items:
        m = v.get("mode", "summary")
        by_mode.setdefault(m, []).append(v)

    # 构建HTML
    html_parts = []

    # 顶部摘要条
    html_parts.append(f"""
    <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:16px 24px;border-radius:10px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;">
        <div>
            <span style="font-size:20px;font-weight:bold;">📹 视频理解总览</span>
            <span style="font-size:13px;opacity:0.85;margin-left:12px;">
                {video_info.get('width','?')}x{video_info.get('height','?')} ·
                {video_info.get('duration_sec',0):.1f}s ·
                {video_info.get('total_frames',0)}帧
            </span>
        </div>
        <div style="font-size:14px;">
            <span style="background:rgba(255,255,255,0.2);padding:4px 12px;border-radius:6px;margin-right:8px;">🎯 {len(entities)} 个实体</span>
            <span style="background:rgba(255,255,255,0.2);padding:4px 12px;border-radius:6px;">❓ {len(vqa_items)} 条VQA</span>
        </div>
    </div>""")

    # 实体列表（如果有的话）
    if entities:
        entity_chips = []
        for e in entities[:20]:
            name = e.get("canonical_name", e.get("label", "?"))
            cat = e.get("category", "")
            chip_color = ASPECT_COLORS.get("attribute", "#FFF3E0")
            entity_chips.append(
                f'<span style="display:inline-block;background:{chip_color};color:#333;border:1px solid #ddd;border-radius:12px;padding:3px 10px;margin:3px;font-size:13px;">{name}<span style="color:#888;font-size:11px;margin-left:4px;">({cat})</span></span>'
            )
        html_parts.append(
            f'<div style="margin-bottom:18px;padding:12px;background:#fafafa;border-radius:8px;">'
            f'<div style="font-weight:bold;margin-bottom:6px;font-size:14px;">📋 检测到的实体</div>'
            f'{"".join(entity_chips)}</div>'
        )

    # 7维度卡片（2列网格）
    html_parts.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">')

    for mode_key, title, desc in ASPECTS:
        items = by_mode.get(mode_key, [])
        bg = ASPECT_COLORS.get(mode_key, "#f5f5f5")
        border = ASPECT_BORDERS.get(mode_key, "#ccc")
        count_badge = f'<span style="background:{border};color:white;border-radius:10px;padding:1px 8px;font-size:12px;margin-left:6px;">{len(items)}</span>'

        card = f"""
        <div style="background:{bg};border-left:4px solid {border};border-radius:8px;padding:14px;min-height:120px;">
            <div style="font-weight:bold;font-size:15px;margin-bottom:4px;">{title}{count_badge}</div>
            <div style="color:#666;font-size:11px;margin-bottom:10px;">{desc}</div>"""

        if items:
            for item in items[:3]:
                q = item.get("question", "")[:100]
                a = item.get("answer", "")[:150]
                difficulty = item.get("difficulty", "")
                diff_color = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#f44336"}.get(difficulty, "#999")
                card += f"""
                <div style="background:rgba(255,255,255,0.7);border-radius:6px;padding:8px 10px;margin-bottom:6px;">
                    <div style="font-size:13px;color:#333;"><b>Q:</b> {q}</div>
                    <div style="font-size:13px;color:#555;margin-top:3px;"><b>A:</b> {a}</div>
                    <span style="display:inline-block;background:{diff_color};color:white;border-radius:3px;padding:0px 6px;font-size:10px;margin-top:3px;">{difficulty}</span>
                </div>"""
            if len(items) > 3:
                card += f'<div style="color:#888;font-size:12px;text-align:center;padding:4px;">还有 {len(items)-3} 条...</div>'
        else:
            card += '<div style="color:#aaa;font-size:13px;text-align:center;padding:20px;">暂无数据</div>'

        card += "</div>"
        html_parts.append(card)

    html_parts.append("</div>")
    return "\n".join(html_parts)


# ============================================================
#  Gradio 界面
# ============================================================

with gr.Blocks(
    title="EGO片段编译工具",
) as app:

    gr.Markdown("# EGO 片段编译工具")
    gr.Markdown("将第一人称视角视频自动编译为物理常识 VQA 训练数据")

    # tooltip JS注入
    gr.HTML("""
    <script>
    (function() {
        const tooltips = {
            'btn-run': '上传视频后点击此按钮，自动执行抽帧→VLM描述→实体融合→VQA生成→校验→编译全流程',
            'btn-syn': '将一个同义词映射到规范名，如 银色扳手→扳手，之后所有该同义词自动归一',
            'btn-merge': '将两个实体合并为一个，源名被并入目标名，词表自动更新',
            'btn-clear-cand': '清空所有待人工审核的候选词列表',
            'slider-fps': '每秒从视频中采样几帧，2fps=每秒2帧，值越大帧越多但API成本越高',
            'slider-batch': '每次VLM API调用同时送几帧图片，8帧为推荐值，太多可能超token限制',
            'video-input': '点击或拖拽上传第一人称视角视频（MP4格式，建议30-60秒）',
        };
        function applyTooltips() {
            document.querySelectorAll('[id]').forEach(el => {
                const id = el.id || '';
                for (const [key, tip] of Object.entries(tooltips)) {
                    if (id.includes(key)) { el.title = tip; break; }
                }
            });
        }
        // Gradio动态渲染后重新注入
        const observer = new MutationObserver(() => applyTooltips());
        observer.observe(document.body, {childList: true, subtree: true});
        setTimeout(applyTooltips, 500);
        setTimeout(applyTooltips, 2000);
    })();
    </script>
    """)

    with gr.Tab("📋 编译流程"):
        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                video_input = gr.Video(
                    label="上传 EGO 视频（30-60秒）",
                    format="mp4",
                    sources=["upload"],
                    include_audio=False,
                    elem_id="video-input",
                    height=320,
                )
                with gr.Row():
                    fps_slider = gr.Slider(1, 5, value=2, step=1, label="抽帧频率 (fps)",
                                           info="每秒采样几帧", elem_id="slider-fps")
                    batch_slider = gr.Slider(4, 12, value=8, step=2, label="VLM 批次大小",
                                            info="每次API送几帧图", elem_id="slider-batch")
                run_btn = gr.Button("🚀 开始编译", variant="primary", size="lg", elem_id="btn-run")

            with gr.Column(scale=1, min_width=400):
                log_output = gr.Textbox(label="执行日志", lines=18, max_lines=30, interactive=False)
                summary_output = gr.Textbox(label="编译结果摘要", lines=12, interactive=False)
                output_file = gr.File(label="下载编译JSON")

    with gr.Tab("🎬 视频理解"):
        gr.Markdown("### 七维度理解预览")
        gr.Markdown("编译完成后，按7个物理常识维度展示工具对视频的理解内容")
        understanding_html = gr.HTML(
            value="<div style='text-align:center;padding:60px;color:#999;font-size:16px;'>尚未编译，请先在「编译流程」页上传视频并点击「开始编译」</div>"
        )

    with gr.Tab("❓ VQA 预览"):
        gr.Markdown("### 生成的七模式VQA问答对")
        vqa_table = gr.DataFrame(
            headers=["序号", "模式", "难度", "问题", "答案", "证据帧"],
            label="VQA条目",
            interactive=False,
            wrap=True,
        )
        gr.Markdown("""
**七种VQA模式说明：**
| 模式 | 说明 |
|---|---|
| 时序推理 | 事件先后顺序、持续时间、过程步骤 |
| 空间布局 | 物体间空间关系、位置、相对方位 |
| 物体属性 | 颜色、大小、材质、状态等物理属性 |
| 力学/动力学 | 力的方向、大小、作用方式、物理效果 |
| 因果推理 | 为什么这样做、因果关系、目的推断 |
| 任务总结 | 整段视频的任务目标、整体过程概括 |
| 运动轨迹 | 物体或手部的运动路径、轨迹描述 |
""")

    with gr.Tab("💰 成本报告"):
        cost_output = gr.Textbox(label="API调用成本", lines=15, interactive=False)

    with gr.Tab("📚 词汇管理"):
        gr.Markdown("### L1 动态词表")
        word_table = gr.DataFrame(
            headers=["自由文本", "规范名"],
            label="词表",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("### 添加同义词")
        with gr.Row():
            syn_canonical = gr.Textbox(label="规范名", placeholder="如: 扳手")
            syn_synonym = gr.Textbox(label="同义词", placeholder="如: 银色扳手")
            syn_btn = gr.Button("添加", elem_id="btn-syn")
        syn_status = gr.Textbox(label="状态", interactive=False)

        gr.Markdown("### 合并实体")
        with gr.Row():
            merge_source = gr.Textbox(label="源名（被合并）", placeholder="如: 金属扳手")
            merge_target = gr.Textbox(label="目标名（保留）", placeholder="如: 扳手")
            merge_btn = gr.Button("合并", elem_id="btn-merge")
        merge_status = gr.Textbox(label="状态", interactive=False)

        gr.Markdown("### 候选词（待人工归一）")
        cand_table = gr.DataFrame(
            headers=["候选词", "上下文", "帧"],
            label="候选词列表",
            interactive=False,
        )
        clear_btn = gr.Button("清空候选词", elem_id="btn-clear-cand")

        gr.Markdown("### L0 硬枚举大类")
        gr.Markdown(f"""
**物体大类：**
{chr(10).join(f'- **{k}**: {v}' for k, v in L0_OBJECT_CATEGORIES.items())}

**动作大类：**
{chr(10).join(f'- **{k}**: {v}' for k, v in L0_ACTION_CATEGORIES.items())}
""")

    # ============================================================
    #  事件绑定
    # ============================================================

    run_btn.click(
        fn=run_pipeline,
        inputs=[video_input, fps_slider, batch_slider, gr.State(False)],
        outputs=[log_output, summary_output, output_file, vqa_table, cost_output, word_table],
    ).then(
        fn=format_understanding_html,
        inputs=[output_file],
        outputs=[understanding_html],
    )

    # 词汇管理
    syn_btn.click(
        fn=add_synonym,
        inputs=[syn_canonical, syn_synonym],
        outputs=[word_table, cand_table, syn_status],
    )

    merge_btn.click(
        fn=merge_entities_ui,
        inputs=[merge_source, merge_target],
        outputs=[word_table, cand_table, merge_status],
    )

    clear_btn.click(
        fn=clear_candidates_ui,
        outputs=[word_table, cand_table, syn_status],
    )

    # 页面加载时显示词表
    app.load(
        fn=load_vocab,
        outputs=[word_table, cand_table],
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="EGO 视频编译工具 - 可视化界面")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，局域网访问可用 0.0.0.0")
    ap.add_argument("--port", type=int, default=7860, help="端口")
    ap.add_argument("--share", action="store_true", help="生成公网临时分享链接")
    ap.add_argument("--config", "-c", default=None, help="配置文件路径，默认 ./config.yaml")
    ap.add_argument("--output", "-o", default=None, help="输出目录，默认 ./output")
    args = ap.parse_args()

    Config.load(args.config)
    if args.output:
        Config.set_output_root(args.output)

    logger.info(f"项目根目录: {Config.PROJECT_ROOT}")
    logger.info(f"输出目录: {Config.OUTPUT_ROOT}")
    logger.info(f"接口地址: {Config.DASHSCOPE_BASE_URL}")
    logger.info(f"VLM 模型: {Config.vlm().get('model')} / LLM 模型: {Config.llm().get('model')}")
    if Config.DASHSCOPE_API_KEY:
        logger.info(f"API Key: {Config.DASHSCOPE_API_KEY[:8]}...")
    else:
        logger.warning("未检测到 API Key，界面可打开但编译会失败。"
                       "请配置 .env 中的 DASHSCOPE_API_KEY")

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {max-width: 100% !important; padding: 12px 20px !important;}
        .gradio-container > div {max-width: 100% !important;}
        footer {display: none !important;}
        """,
    )
