"""VLM客户端：调用 qwen-vl-max 进行帧描述"""

import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from openai import OpenAI

from .config import Config
from .schemas import FrameInfo, FrameDescription, ObjectDesc, ActionDesc
from .cost_tracker import CostTracker
from .prompts.vlm_prompt import VLM_SYSTEM_PROMPT, build_vlm_user_prompt
from .vocab import L0_OBJECT_CATEGORIES, L0_ACTION_CATEGORIES

# VLM 单批次最大尝试次数(含首次)。模型偶发返回非预期结构或空内容,
# 多试几次的成本远低于产出"实体0/VQA0"废片段。
MAX_VLM_RETRY = 4

logger = logging.getLogger(__name__)


class VLMClient:
    """qwen-vl-max 视觉语言模型客户端"""

    def __init__(self, cost_tracker: CostTracker):
        self.client = OpenAI(
            api_key=Config.DASHSCOPE_API_KEY,
            base_url=Config.DASHSCOPE_BASE_URL,
        )
        self.cost_tracker = cost_tracker
        vlm_cfg = Config.vlm()
        self.model = vlm_cfg.get("model", "qwen-vl-max")
        self.max_tokens = vlm_cfg.get("max_tokens", 4096)
        self.temperature = vlm_cfg.get("temperature", 0.1)
        self.batch_size = Config.sampling().get("batch_size", 8)
        self.max_workers = Config.sampling().get("vlm_workers", 4)
        self._batch_counter = 0

    def _encode_image(self, image_path: str) -> str:
        """读取图片(兼容中文路径), 降采样到宽960px, JPEG压缩后base64(载荷减小10-20倍, 大幅提速)"""
        try:
            import cv2
            import numpy as np
            data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                if w > 960:
                    img = cv2.resize(img, (960, int(h * 960 / w)))
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    return base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception:
            pass
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def describe_frames(
        self,
        frames: list[FrameInfo],
        progress_callback=None,
    ) -> list[FrameDescription]:
        """
        分批调用VLM描述所有帧。
        
        Args:
            frames: FrameInfo列表
            progress_callback: 回调函数(completed, total, batch_id)
        
        Returns:
            FrameDescription列表
        """
        if not frames:
            return []

        # 分批
        batches = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            self._batch_counter += 1
            batches.append((batch, f"batch_{self._batch_counter:03d}"))

        total = len(frames)
        print(f"   VLM共{len(batches)}批(每批≤{self.batch_size}帧), {self.max_workers}线程并行", flush=True)

        def _run_one(item):
            batch, batch_id = item
            # 批次级缓存: 重跑跳过已成功批次(长任务中断后不重复花钱)
            cache_dir = Config.CACHE_DIR / "vlm_batches"
            cache_dir.mkdir(parents=True, exist_ok=True)
            import hashlib
            key = hashlib.md5(("|".join(f.image_path for f in batch) + self.model).encode()).hexdigest()[:16]
            cache_file = cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    data = json.load(open(cache_file, encoding="utf-8"))
                    descs = [FrameDescription(**d) for d in data]
                    print(f"   VLM {batch_id} ⏩ 缓存命中 {len(descs)}帧", flush=True)
                    return batch_id, descs, None
                except Exception:
                    pass
            # 多次重试: 模型偶发返回非预期结构/空内容, 多给几次机会远比产出废片段划算
            # 第3次起把温度降到0, 减少输出结构抖动
            attempts = [(None, 0)] * (MAX_VLM_RETRY - 2) + [(0.0, 2), (0.0, 4)]
            last_err = ""
            for i, (temp, backoff) in enumerate(attempts, start=1):
                try:
                    desc = self._call_vlm_batch(batch, batch_id, temperature=temp)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump([d.model_dump() for d in desc], f, ensure_ascii=False)
                    if i > 1:
                        print(f"   VLM {batch_id} ✅ 第{i}次重试成功 {len(desc)}帧", flush=True)
                    return batch_id, desc, None
                except Exception as e:
                    last_err = str(e)
                    if i < len(attempts):
                        if backoff:
                            time.sleep(backoff)
                        else:
                            time.sleep(2)
            return batch_id, None, last_err

        done = 0
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(_run_one, b): b[1] for b in batches}
            for fut in as_completed(futs):
                batch_id, desc, err = fut.result()
                done += 1
                if err:
                    logger.error(f"VLM批次 {batch_id} 失败: {err[:200]}")
                    print(f"   VLM {batch_id} ❌ 失败({done}/{len(batches)})", flush=True)
                    results[batch_id] = None
                else:
                    results[batch_id] = desc
                    print(f"   VLM {batch_id} ✅ {len(desc)}帧 ({done}/{len(batches)})", flush=True)

        # 按批次顺序组装, 失败的批次填占位
        all_descriptions: list[FrameDescription] = []
        failed_batches = []
        for batch, batch_id in batches:
            desc = results.get(batch_id)
            if desc is None:
                failed_batches.append(batch_id)
                for f in batch:
                    all_descriptions.append(FrameDescription(
                        frame_id=f.frame_id, timestamp=f.timestamp,
                        scene_summary="(描述失败)"))
            else:
                all_descriptions.extend(desc)
        if failed_batches:
            logger.error(f"VLM失败批次: {failed_batches}")

        # 保存中间结果（仅用于调试排查）。片段级并行时多个线程会同时走到这里，
        # 因此文件名必须唯一；且该缓存不是关键路径，写入失败不应中断编译。
        try:
            tag = all_descriptions[0].frame_id if all_descriptions else "empty"
            save_path = Config.CACHE_DIR / (
                f"frames_desc_{tag}_{os.getpid()}_{threading.get_ident()}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([d.model_dump() for d in all_descriptions],
                          f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"帧描述缓存写入失败（不影响编译）: {e}")
        logger.info(f"VLM帧描述完成: {len(all_descriptions)} 帧已描述")

        return all_descriptions

    def _call_vlm_batch(
        self,
        batch: list[FrameInfo],
        batch_id: str,
        temperature: float | None = None,
    ) -> list[FrameDescription]:
        """调用VLM处理单个批次

        temperature 传 None 时用实例默认温度; 重试阶段可显式降为 0 提升结构稳定性。
        """
        # 构建消息内容
        content = []
        for frame in batch:
            b64 = self._encode_image(frame.image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        # 用户提示
        frames_info = [{"frame_id": f.frame_id, "timestamp": f.timestamp} for f in batch]
        user_prompt = build_vlm_user_prompt(batch_id, frames_info)
        content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        )

        # 记录成本
        usage = response.usage
        self.cost_tracker.record_vlm(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            module=f"vlm_{batch_id}",
        )

        # 解析响应
        raw = response.choices[0].message.content
        result = self._parse_vlm_response(raw, batch, batch_id)

        # 内容级校验: 偶发情况下模型返回结构合法但objects/actions全空,
        # 若不拦截会静默产出"实体0/VQA0"的废片段。抛错触发重试更划算。
        n_obj = sum(len(d.objects) for d in result)
        n_act = sum(len(d.actions) for d in result)
        if n_obj == 0 and n_act == 0:
            raise ValueError(
                f"VLM返回内容为空(0物体/0动作, {len(result)}帧), 疑为偶发生成失败, 触发重试"
            )
        return result

    @staticmethod
    def _extract_frames(data) -> list:
        """从VLM响应中稳健提取帧数组。

        不同模型/偶发情况下顶层结构可能是:
          {"frames": [...]} / [...] / {"data": [...]} / {"result": {...}}
        """
        if isinstance(data, list):
            # 元素需是含 frame_id/objects 的字典
            return [x for x in data if isinstance(x, dict)]
        if not isinstance(data, dict):
            return []
        for key in ("frames", "data", "results", "result", "items"):
            v = data.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
            if isinstance(v, dict):
                # 再剥一层
                for k2 in ("frames", "data", "results"):
                    v2 = v.get(k2)
                    if isinstance(v2, list):
                        return [x for x in v2 if isinstance(x, dict)]
        # 兜底: 取 dict 中唯一一个"由含objects的字典组成的列表"
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "objects" in v[0]:
                return [x for x in v if isinstance(x, dict)]
        return []

    def _parse_vlm_response(
        self,
        raw_json: str,
        batch: list[FrameInfo],
        batch_id: str,
    ) -> list[FrameDescription]:
        """解析VLM返回的JSON为FrameDescription列表"""
        # 清理可能的markdown标记
        raw_json = raw_json.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(raw_json)
        frames_data = self._extract_frames(data)

        if not frames_data:
            # 解析出0帧多为模型偶发返回非预期结构，抛错触发重试，避免静默产出空片段
            raise ValueError(f"VLM响应未解析出任何帧数据 (顶层类型: {type(data).__name__})")

        # 按frame_id建立索引
        frame_map = {f.frame_id: f for f in batch}
        result = []

        for fd in frames_data:
            frame_id = fd.get("frame_id", "")
            frame_info = frame_map.get(frame_id)

            # 构建物体列表
            objects = []
            for obj in fd.get("objects", []):
                category = obj.get("category", "other")
                if category not in L0_OBJECT_CATEGORIES:
                    category = "other"
                objects.append(ObjectDesc(
                    label=obj.get("label", "unknown"),
                    category=category,
                    raw_description=obj.get("raw_description", ""),
                    batch_id=batch_id,
                    local_id=obj.get("local_id", "obj_00"),
                    attributes=obj.get("attributes", {}),
                    confidence_grade=obj.get("confidence_grade", "B"),
                ))

            # 构建动作列表
            actions = []
            for act in fd.get("actions", []):
                category = act.get("category", "navigate")
                if category not in L0_ACTION_CATEGORIES:
                    category = "navigate"
                actions.append(ActionDesc(
                    verb=act.get("verb", "unknown"),
                    category=category,
                    raw_description=act.get("raw_description", ""),
                    object_ref=act.get("object_ref", ""),
                    direction=act.get("direction"),
                    confidence_grade=act.get("confidence_grade", "B"),
                ))

            result.append(FrameDescription(
                frame_id=frame_id,
                timestamp=frame_info.timestamp if frame_info else fd.get("timestamp", 0),
                objects=objects,
                actions=actions,
                scene_summary=fd.get("scene_summary", ""),
            ))

        # 如果返回的帧数不匹配，补全缺失帧
        result_ids = {r.frame_id for r in result}
        for f in batch:
            if f.frame_id not in result_ids:
                result.append(FrameDescription(
                    frame_id=f.frame_id,
                    timestamp=f.timestamp,
                    scene_summary="(未描述)",
                ))

        return result
