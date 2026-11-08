"""编译器模块：组装最终输出"""

import json
import logging
from datetime import datetime
from pathlib import Path

from .config import Config
from .schemas import (
    CompilerOutput, VideoInfo, Entity, VQAItem,
    ValidationResult, CostRecord, FrameDescription,
)
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class Compiler:
    """编译最终输出JSON"""

    def compile(
        self,
        video_path: str,
        video_info_raw: dict,
        frames_sampled: int,
        frame_descriptions: list[FrameDescription],
        entities: list[Entity],
        vqa_items: list[VQAItem],
        validation: ValidationResult,
        cost_tracker: CostTracker,
    ) -> CompilerOutput:
        """组装最终输出"""

        video_info = VideoInfo(
            path=video_path,
            duration_sec=video_info_raw.get("duration_sec", 0),
            fps=video_info_raw.get("fps", 0),
            total_frames=video_info_raw.get("total_frames", 0),
            resolution=f"{video_info_raw.get('width', 0)}x{video_info_raw.get('height', 0)}",
        )

        total_cost = cost_tracker.total_cost()

        output = CompilerOutput(
            video_info=video_info,
            frames_sampled=frames_sampled,
            objects=entities,
            vqa_items=vqa_items,
            validation=validation,
            cost_records=cost_tracker.records,
            total_cost_cny=total_cost,
        )

        # 保存到文件
        output_path = Config.OUTPUT_DIR / f"compiled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, ensure_ascii=False, indent=2)

        # 同时保存一份最新版
        latest_path = Config.OUTPUT_DIR / "latest_compiled.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info(f"编译完成: {output_path}")
        logger.info(f"  帧数: {frames_sampled}, 实体: {len(entities)}, VQA: {len(vqa_items)}")
        logger.info(f"  成本: ¥{total_cost}")

        return output

    def format_output_summary(self, output: CompilerOutput) -> str:
        """生成可读的输出摘要"""
        lines = ["=" * 50]
        lines.append("EGO片段编译结果摘要")
        lines.append("=" * 50)
        lines.append(f"视频: {output.video_info.path}")
        lines.append(f"时长: {output.video_info.duration_sec:.1f}s | 分辨率: {output.video_info.resolution}")
        lines.append(f"采样帧数: {output.frames_sampled}")
        lines.append(f"融合实体: {len(output.objects)} 个")
        lines.append(f"VQA条数: {len(output.vqa_items)} 条")
        lines.append(f"校验状态: {output.validation.status}")
        lines.append(f"总成本: ¥{output.total_cost_cny}")

        # VQA模式分布
        mode_dist = {}
        for item in output.vqa_items:
            mode_dist[item.mode] = mode_dist.get(item.mode, 0) + 1
        lines.append("\nVQA模式分布:")
        for mode, count in sorted(mode_dist.items()):
            lines.append(f"  {mode}: {count}条")

        # 实体列表
        lines.append(f"\n实体列表 ({len(output.objects)}):")
        for e in output.objects[:10]:
            lines.append(f"  - {e.canonical_name} ({e.category}) [证据帧: {len(e.evidence_frames)}]")
        if len(output.objects) > 10:
            lines.append(f"  ... 共{len(output.objects)}个")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)
