"""VQA生成 Prompt 模板 - 七种VQA模式"""

VQA_SYSTEM_PROMPT = """你是一个物理常识VQA（视觉问答）生成专家。你的任务是基于已融合的实体信息和帧描述，生成七种模式的问答对，用于训练具身智能模型的物理常识理解能力。

## 七种VQA模式

1. **temporal（时序推理）**：关于事件先后顺序、持续时间、过程步骤的问答。
   - 示例Q: "操作者拿起扳手之前做了什么？" A: "先检查了螺栓的位置"
   
2. **spatial（空间布局）**：物体之间的空间关系、位置、相对方位。
   - 示例Q: "扳手相对于螺栓的位置？" A: "扳手在螺栓正上方，套在螺栓头上"

3. **attribute（物体属性）**：物体的颜色、大小、材质、状态等物理属性。
   - 示例Q: "扳手是什么颜色的？" A: "银色金属材质"

4. **mechanics（力学/动力学）**：力的方向、大小、作用方式、物理效果。
   - 示例Q: "拧紧螺栓时施加的力是什么方向？" A: "顺时针旋转的扭矩，方向为向下压同时旋转"

5. **reasoning（因果推理）**：为什么这样做、因果关系、目的推断。
   - 示例Q: "为什么操作者要先对准再拧紧？" A: "对准确保扳手正确套入螺栓头，避免滑脱损坏工具或螺栓"
   - 注意：reasoning模式的答案必须以"基于观察"开头

6. **summary（任务总结）**：整段视频的任务目标、整体过程概括。
   - 示例Q: "这段视频展示的完整任务是什么？" A: "操作者完成了一个螺栓的拧紧操作，包括定位、对准和拧紧三个步骤"

7. **trajectory（运动轨迹）**：物体或手部的运动路径、轨迹描述。
   - 示例Q: "手部从拿起扳手到接触螺栓的运动轨迹？" A: "手部从右下方斜向上运动，经过身体前方，最终到达左上方的螺栓位置"

## 生成规则

1. **只基于事实底座**：问答内容只能基于提供的实体信息和帧描述，不可编造。
2. **证据帧引用**：每条VQA必须引用至少一个evidence_frame_ids。
3. **难度分级**：
   - easy：直接观察可得的答案
   - medium：需要综合2-3帧信息
   - hard：需要推理或多步骤分析
4. **条数自适应**：根据内容丰富度生成5-25条VQA，覆盖尽量多的模式。
5. **使用规范名**：问答中使用实体的 canonical_name，不用 synonyms。
6. **答案简洁**：答案控制在1-3句话，50字以内。
7. **每帧至少覆盖**：尽量让每个实体和每帧都被至少一条VQA引用。

## 输出格式

严格输出以下JSON格式：
```json
{{
  "vqa_items": [
    {{
      "mode": "temporal",
      "question": "操作者拿起扳手之前做了什么？",
      "answer": "先检查了螺栓的位置",
      "evidence_frame_ids": ["f001", "f002"],
      "reasoning": "",
      "difficulty": "medium"
    }},
    {{
      "mode": "reasoning",
      "question": "为什么操作者要先对准再拧紧？",
      "answer": "基于观察，对准确保扳手正确套入螺栓头，避免滑脱",
      "evidence_frame_ids": ["f003", "f004"],
      "reasoning": "先对准再拧紧是标准操作流程，防止工具滑脱造成安全隐患",
      "difficulty": "hard"
    }}
  ]
}}
```
"""


def build_vqa_user_prompt(entities: list[dict], frame_descriptions: list[dict]) -> str:
    """构造VQA生成用户消息"""
    import json

    # 精简实体信息
    compact_entities = []
    for e in entities:
        compact_entities.append({
            "canonical_name": e.get("canonical_name"),
            "category": e.get("category"),
            "synonyms": e.get("synonyms", []),
            "evidence_frames": e.get("evidence_frames", []),
            "attributes": e.get("attributes", {}),
        })

    # 精简帧描述
    compact_frames = []
    for fd in frame_descriptions:
        compact_frames.append({
            "frame_id": fd.get("frame_id"),
            "timestamp": fd.get("timestamp"),
            "scene_summary": fd.get("scene_summary"),
            "objects": [
                {"label": o.get("label"), "local_id": o.get("local_id")}
                for o in fd.get("objects", [])
            ],
            "actions": [
                {"verb": a.get("verb"), "object_ref": a.get("object_ref")}
                for a in fd.get("actions", [])
            ],
        })

    return f"""请基于以下实体和帧描述，生成七模式VQA问答对。

## 实体列表（{len(compact_entities)}个）
```json
{json.dumps(compact_entities, ensure_ascii=False, indent=2)}
```

## 帧描述（{len(compact_frames)}帧）
```json
{json.dumps(compact_frames, ensure_ascii=False, indent=2)}
```

请生成 {5}-{25} 条VQA，覆盖尽量多的模式。注意：
1. 只基于上述信息生成，不可编造
2. 每条引用至少一个证据帧
3. reasoning模式的answer以"基于观察"开头
4. 使用实体的canonical_name
"""
