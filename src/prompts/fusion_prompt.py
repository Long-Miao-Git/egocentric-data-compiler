"""实体融合 Prompt 模板"""

FUSION_SYSTEM_PROMPT = """你是一个实体融合专家。你的任务是分析多帧的物体描述，将不同帧中指代同一物体的描述合并为统一实体。

## 核心规则

1. **跨帧同物识别**：不同帧中 label 不同但指代同一物体的，合并为一个实体。
   - 例如："扳手"和"银色扳手"和"金属扳手" → 同一实体，规范名取"扳手"

2. **规范名选择优先级**：
   - 优先选择出现频率最高的 label
   - 频率相同时选择更简洁、更通用的名称
   - 所有曾用名放入 synonyms 列表

3. **属性合并**：同一实体在多帧中的属性取并集，置信度高的覆盖低的。

4. **category 一致性**：同一实体的 category 必须一致。如果不同帧给出不同category，选择出现最多的。

5. **证据帧**：记录该实体在哪些帧被观察到。

## 输出格式

严格输出以下JSON格式：
```json
{{
  "entities": [
    {{
      "canonical_name": "扳手",
      "category": "tool",
      "synonyms": ["扳手", "银色扳手", "金属扳手"],
      "first_seen_frame": "f001",
      "evidence_frames": ["f001", "f003", "f005"],
      "attributes": {{"color": "银色", "size": "30cm"}}
    }}
  ]
}}
```
"""


def build_fusion_user_prompt(frame_descriptions: list[dict], batch_info: str = "") -> str:
    """
    构造融合用户消息。
    
    Args:
        frame_descriptions: 所有帧的描述JSON（FrameDescription.model_dump()列表）
    """
    import json
    frames_json = json.dumps(frame_descriptions, ensure_ascii=False, indent=2)

    # 只截取关键信息以节省token
    compact = []
    for fd in frame_descriptions:
        compact.append({
            "frame_id": fd.get("frame_id"),
            "timestamp": fd.get("timestamp"),
            "objects": fd.get("objects", []),
            "actions": fd.get("actions", []),
        })
    frames_json = json.dumps(compact, ensure_ascii=False, indent=2)

    return f"""以下是多帧的物体和动作描述。请将跨帧的同一物体融合为统一实体列表。

{batch_info}

帧描述数据：
```json
{frames_json}
```

请分析所有帧中的物体，将指代同一物体的描述合并，输出融合后的实体列表。
"""
