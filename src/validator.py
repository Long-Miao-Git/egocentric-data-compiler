"""校验模块：Pydantic验证 + 规则引擎 + 同义词归一"""

import json
import logging
from .schemas import (
    FrameDescription, Entity, VQAItem, ValidationResult,
    VQA_MODES, VQA_MODE_CN,
)
from .vocab import VocabManager, L0_OBJECT_CATEGORIES, L0_ACTION_CATEGORIES

logger = logging.getLogger(__name__)


class Validator:
    """硬校验 + 同义词归一"""

    def __init__(self, vocab: VocabManager):
        self.vocab = vocab
        self.all_frame_ids: set[str] = set()
        self.all_entity_names: set[str] = set()

    def validate(
        self,
        vqa_items: list[VQAItem],
        entities: list[Entity],
        frame_descriptions: list[FrameDescription],
    ) -> ValidationResult:
        """
        执行完整校验。
        返回 ValidationResult，包含 errors/warnings/status。
        """
        errors = []
        warnings = []

        # 收集所有帧ID和实体名
        self.all_frame_ids = {fd.frame_id for fd in frame_descriptions}
        self.all_entity_names = {e.canonical_name for e in entities}

        # 1. 校验实体
        for i, e in enumerate(entities):
            if not e.canonical_name or e.canonical_name == "unknown":
                errors.append(f"实体#{i}: canonical_name为空")

            if e.category not in L0_OBJECT_CATEGORIES:
                errors.append(f"实体'{e.canonical_name}': category '{e.category}' 不在L0大类中")

            for ef in e.evidence_frames:
                if ef and ef not in self.all_frame_ids:
                    warnings.append(f"实体'{e.canonical_name}': 证据帧 '{ef}' 不在帧列表中")

        # 2. 同义词归一（对实体）
        for e in entities:
            normalized, was_normalized = self.vocab.normalize(e.canonical_name, e.category)
            if was_normalized and normalized != e.canonical_name:
                # 如果归一到了一个不同的规范名
                old_name = e.canonical_name
                e.canonical_name = normalized
                if old_name not in e.synonyms:
                    e.synonyms.append(old_name)
                logger.info(f"实体归一: '{old_name}' → '{normalized}'")

        # 3. 校验VQA条目
        valid_vqa = []
        for i, item in enumerate(vqa_items):
            item_errors = []

            if item.mode not in VQA_MODES:
                item_errors.append(f"mode '{item.mode}' 不在七模式中")

            if not item.question.strip():
                item_errors.append("question为空")
            if not item.answer.strip():
                item_errors.append("answer为空")

            # 证据帧校验
            for ef in item.evidence_frame_ids:
                if ef and ef not in self.all_frame_ids:
                    warnings.append(f"VQA#{i} ({item.mode}): 证据帧 '{ef}' 不存在")

            # reasoning模式特殊校验
            if item.mode == "reasoning":
                if not item.reasoning.strip():
                    warnings.append(f"VQA#{i} (reasoning): reasoning字段为空")
                if not item.answer.startswith("基于观察"):
                    warnings.append(f"VQA#{i} (reasoning): answer未以'基于观察'开头")

            # difficulty校验
            if item.difficulty not in ("easy", "medium", "hard"):
                item_errors.append(f"difficulty '{item.difficulty}' 不合法")

            # answer长度警告
            if len(item.answer) > 100:
                warnings.append(f"VQA#{i} ({item.mode}): answer超100字({len(item.answer)}字)")

            if item_errors:
                errors.extend([f"VQA#{i} ({item.mode}): {e}" for e in item_errors])
            else:
                valid_vqa.append(item)

        # 4. 帧描述校验
        prev_ts = -1.0
        for fd in frame_descriptions:
            # 时间戳单调性（容差0.1s）
            if fd.timestamp < prev_ts - 0.1:
                warnings.append(f"帧 {fd.frame_id}: 时间戳{fd.timestamp} < 前一帧{prev_ts}（非单调）")
            prev_ts = fd.timestamp

            # 物体category校验
            for obj in fd.objects:
                if obj.category not in L0_OBJECT_CATEGORIES:
                    errors.append(f"帧{fd.frame_id} 物体'{obj.label}': category '{obj.category}' 不合法")

            # 动作category校验
            for act in fd.actions:
                if act.category not in L0_ACTION_CATEGORIES:
                    errors.append(f"帧{fd.frame_id} 动作'{act.verb}': category '{act.category}' 不合法")

        # 5. 确定状态
        status = "pass" if not errors else "fail"
        if errors:
            status = "fail"
        elif warnings:
            status = "warning"

        # 过滤掉无效VQA（原地替换）
        vqa_items[:] = valid_vqa

        return ValidationResult(
            status=status,
            errors=errors,
            warnings=warnings,
            repaired_count=0,
        )

    def get_validation_summary(self, result: ValidationResult) -> str:
        """生成校验报告文本"""
        lines = [f"校验状态: {result.status.upper()}"]
        lines.append(f"错误: {len(result.errors)} 条")
        lines.append(f"警告: {len(result.warnings)} 条")
        lines.append(f"修复次数: {result.repaired_count}")
        if result.errors:
            lines.append("\n--- 错误详情 ---")
            for e in result.errors[:20]:
                lines.append(f"  × {e}")
            if len(result.errors) > 20:
                lines.append(f"  ... 共{len(result.errors)}条")
        if result.warnings:
            lines.append("\n--- 警告详情 ---")
            for w in result.warnings[:10]:
                lines.append(f"  ⚠ {w}")
            if len(result.warnings) > 10:
                lines.append(f"  ... 共{len(result.warnings)}条")
        return "\n".join(lines)
