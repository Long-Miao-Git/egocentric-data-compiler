"""LLM客户端：调用 qwen-max 进行实体融合、VQA生成和递归修复"""

import json
import logging
import os
import threading
import time
from openai import OpenAI

from .config import Config
from .schemas import (
    FrameDescription, Entity, VQAItem,
    ENTITY_FUSION_SCHEMA, VQA_OUTPUT_SCHEMA,
    VQA_MODES,
)
from .cost_tracker import CostTracker
from .vocab import L0_OBJECT_CATEGORIES
from .prompts.fusion_prompt import FUSION_SYSTEM_PROMPT, build_fusion_user_prompt
from .prompts.vqa_prompt import VQA_SYSTEM_PROMPT, build_vqa_user_prompt
from .prompts.repair_prompt import REPAIR_SYSTEM_PROMPT, build_repair_user_prompt

logger = logging.getLogger(__name__)


class LLMClient:
    """文本大模型客户端（当前：qwen3.8-flash）

    重要：思考型模型（qwen3.8-flash 等）在**开启思考**时不接受 tool_choice，
    会返回 400 InternalError.Algo.InvalidParameter 并降级到 response_format，
    导致输出 token 暴涨约 10 倍、耗时约 9 倍且成本反超 qwen-max。
    因此必须显式 enable_thinking=False（由 config.llm.enable_thinking 控制）。
    实测（同一段 4 帧数据）：
        qwen-max                  7 VQA / 805 out / 20.8s / ¥0.0268
        qwen3.8-flash 思考开      18 VQA / 9903 out / 180.3s / ¥0.0299（降级路径）
        qwen3.8-flash 思考关      10 VQA / 1015 out / 16.3s / ¥0.0065（推荐）
    """

    def __init__(self, cost_tracker: CostTracker):
        self.client = OpenAI(
            api_key=Config.DASHSCOPE_API_KEY,
            base_url=Config.DASHSCOPE_BASE_URL,
        )
        self.cost_tracker = cost_tracker
        llm_cfg = Config.llm()
        self.model = llm_cfg.get("model", "qwen-max")
        self.max_tokens = llm_cfg.get("max_tokens", 8192)
        self.temperature = llm_cfg.get("temperature", 0.3)
        # None = 不传参（用模型默认）；True/False = 显式开关
        self.enable_thinking = llm_cfg.get("enable_thinking", None)

    def _extra_body(self):
        """思考开关（DashScope OpenAI 兼容模式的参数）"""
        if self.enable_thinking is None:
            return {}
        return {"extra_body": {"enable_thinking": bool(self.enable_thinking)}}

    def _call_with_function_calling(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        function_name: str,
        function_desc: str,
        module_name: str = "llm",
    ) -> dict:
        """
        使用 Function Calling + response_format 双保险调用LLM。
        优先解析 tool_calls，其次解析 content。
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": function_desc,
                    "parameters": schema,
                },
            }
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": function_name}},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self._extra_body(),
            )
        except Exception as e:
            logger.warning(f"Function Calling失败，降级为response_format: {e}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self._extra_body(),
            )

        # 记录成本
        usage = response.usage
        self.cost_tracker.record_llm(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            module=module_name,
        )

        msg = response.choices[0].message

        # 优先从 tool_calls 解析
        if msg.tool_calls:
            args = msg.tool_calls[0].function.arguments
            return json.loads(args)

        # 从 content 解析
        raw = msg.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)

    # ============================================================
    #  实体融合
    # ============================================================

    def fuse_entities(self, frame_descriptions: list[FrameDescription]) -> list[Entity]:
        """调用LLM进行跨帧实体融合"""
        if not frame_descriptions:
            return []

        fd_dicts = [fd.model_dump() for fd in frame_descriptions]
        user_prompt = build_fusion_user_prompt(fd_dicts)

        logger.info(f"实体融合: 输入{len(frame_descriptions)}帧描述")

        try:
            data = self._call_with_function_calling(
                system_prompt=FUSION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=ENTITY_FUSION_SCHEMA,
                function_name="output_entities",
                function_desc="Output the fused entity list after cross-frame entity fusion",
                module_name="llm_fusion",
            )
        except Exception as e:
            logger.error(f"实体融合调用失败: {e}")
            # 降级：直接用帧描述中的物体作为实体
            return self._fallback_fusion(frame_descriptions)

        entities = []
        for e in data.get("entities", []):
            category = e.get("category", "other")
            if category not in L0_OBJECT_CATEGORIES:
                category = "other"
            entities.append(Entity(
                canonical_name=e.get("canonical_name", "unknown"),
                category=category,
                synonyms=e.get("synonyms", []),
                first_seen_frame=e.get("first_seen_frame", ""),
                evidence_frames=e.get("evidence_frames", []),
                attributes=e.get("attributes", {}),
            ))

        # 保存中间结果（仅用于调试排查）。片段级并行时会并发写，文件名需唯一，
        # 且该缓存不是关键路径，写入失败不应中断编译。
        try:
            save_path = Config.CACHE_DIR / (
                f"entities_{os.getpid()}_{threading.get_ident()}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in entities],
                          f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"实体缓存写入失败（不影响编译）: {e}")
        logger.info(f"实体融合完成: {len(entities)} 个实体")

        return entities

    def _fallback_fusion(self, frame_descriptions: list[FrameDescription]) -> list[Entity]:
        """降级方案：直接从帧描述中提取物体作为实体"""
        seen = {}
        for fd in frame_descriptions:
            for obj in fd.objects:
                key = obj.label.lower()
                if key not in seen:
                    seen[key] = Entity(
                        canonical_name=obj.label,
                        category=obj.category,
                        synonyms=[obj.label],
                        first_seen_frame=fd.frame_id,
                        evidence_frames=[fd.frame_id],
                        attributes=obj.attributes,
                    )
                else:
                    if fd.frame_id not in seen[key].evidence_frames:
                        seen[key].evidence_frames.append(fd.frame_id)
        return list(seen.values())

    # ============================================================
    #  VQA 生成
    # ============================================================

    def generate_vqa(
        self,
        entities: list[Entity],
        frame_descriptions: list[FrameDescription],
    ) -> list[VQAItem]:
        """调用LLM生成七模式VQA"""
        if not entities or not frame_descriptions:
            return []

        e_dicts = [e.model_dump() for e in entities]
        fd_dicts = [fd.model_dump() for fd in frame_descriptions]
        user_prompt = build_vqa_user_prompt(e_dicts, fd_dicts)

        logger.info(f"VQA生成: {len(entities)}实体, {len(frame_descriptions)}帧")

        try:
            data = self._call_with_function_calling(
                system_prompt=VQA_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=VQA_OUTPUT_SCHEMA,
                function_name="output_vqa",
                function_desc="Output the generated VQA items in 7 modes",
                module_name="llm_vqa",
            )
        except Exception as e:
            logger.error(f"VQA生成调用失败: {e}")
            return []

        vqa_items = []
        for item in data.get("vqa_items", []):
            mode = item.get("mode", "summary")
            if mode not in VQA_MODES:
                mode = "summary"
            vqa_items.append(VQAItem(
                mode=mode,
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                evidence_frame_ids=item.get("evidence_frame_ids", []),
                reasoning=item.get("reasoning", ""),
                difficulty=item.get("difficulty", "easy"),
            ))

        logger.info(f"VQA生成完成: {len(vqa_items)} 条")
        return vqa_items

    # ============================================================
    #  递归修复
    # ============================================================

    def repair_json(self, failed_json: str, errors: list[str], schema_hint: str = "") -> dict:
        """
        调用LLM修复格式错误的JSON。
        最多调用1次。
        """
        user_prompt = build_repair_user_prompt(failed_json, errors, schema_hint)

        logger.info(f"递归修复: {len(errors)} 个错误")

        messages = [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=self.max_tokens,
                **self._extra_body(),
            )
        except Exception as e:
            logger.error(f"递归修复调用失败: {e}")
            return {}

        usage = response.usage
        self.cost_tracker.record_llm(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            module="llm_repair",
        )

        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("修复后JSON仍无法解析")
            return {}
