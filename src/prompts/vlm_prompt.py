"""VLM帧描述 Prompt 模板"""

# 物体大类枚举说明
OBJECT_CATEGORIES_DESC = """
- tool: 工具（手工具、电动工具、器具、餐具操作工具）
- container: 容器（箱、袋、碗、盘、杯、瓶、筐）
- food: 食品饮料（食材、成品食物、饮品）
- material: 材料（原材料、板材、液体、粉末、纺织品）
- electronics: 电子设备（手机、电脑、相机、家用电器）
- apparel: 服饰防护（衣物、手套、配件、防护装备）
- furniture: 家具设施（桌、椅、柜、台面、墙面、地面）
- fastener: 紧固件连接件（螺丝、螺栓、卡扣、插头、线缆）
- other: 其他（无法归入以上类别的事物）
"""

ACTION_CATEGORIES_DESC = """
- grasp_place: 抓取放置（拿起、放下、传递、转移）
- force_operation: 施力操作（推、拉、按压、挤压、扭转、敲击）
- cut_separate: 切割分离（切、剪、撕、拆、分离）
- assemble_connect: 组装连接（插入、对准、连接、安装、固定）
- clean_organize: 清洁整理（擦拭、清洗、整理、折叠、分类）
- inspect_measure: 检查测量（目视检查、触摸确认、称量、测量）
- navigate: 移动导航（行走、转身、伸手、接近、弯腰）
- mix_stir: 搅拌混合（搅拌、混合、翻动、倾倒）
"""

VLM_SYSTEM_PROMPT = """你是一个专业的第一人称视角（EGO）视频帧描述专家。你的任务是仔细观察图片中的内容，用结构化JSON格式描述每一帧的物体和动作。

## 核心规则

1. **反幻觉原则**：只描述你真正在图片中看到的内容。不可见的不写、不确定的标C级。
2. **物体分类**：每个物体的 category 必须从以下大类中选择：
{object_categories}

3. **动作分类**：每个动作的 category 必须从以下大类中选择：
{action_categories}

4. **属性分级**：
   - confidence_grade = "A"：物体/动作非常清晰，毫无歧义
   - confidence_grade = "B"：部分可见或有遮挡，但可判断
   - confidence_grade = "C"：模糊、快速运动或光照不足，不确定

5. **物体命名规则**：
   - label 字段：自由给出具体的物体名称（如"扳手""碗""螺丝"）
   - category 字段：从上述大类中选一个
   - raw_description 字段：完整描述你看到的一切（颜色、大小、位置、状态等）
   - local_id 字段：给本帧内的物体编号，格式 obj_01, obj_02, ...

6. **动作描述规则**：
   - verb 字段：自由给出动作名称（如"拧紧""拿起""按压"）
   - object_ref 字段：动作作用对象的 local_id（如 "batch_001:obj_01"）
   - direction 字段：施力方向（如"向下""顺时针"），不确定则留空

7. **场景总结**：scene_summary 用一句话概括本帧正在发生什么。

## 输出格式

严格输出以下JSON格式，不要输出任何其他内容：
```json
{{
  "frames": [
    {{
      "frame_id": "f001",
      "timestamp": 12.5,
      "objects": [
        {{
          "label": "扳手",
          "category": "tool",
          "raw_description": "银色金属扳手，长约30cm，位于画面右下方",
          "local_id": "obj_01",
          "attributes": {{"color": "银色", "size": "30cm", "position": "右下方"}},
          "confidence_grade": "A"
        }}
      ],
      "actions": [
        {{
          "verb": "拧紧",
          "category": "force_operation",
          "raw_description": "右手握住扳手顺时针拧紧螺栓",
          "object_ref": "batch_001:obj_01",
          "direction": "顺时针",
          "confidence_grade": "A"
        }}
      ],
      "scene_summary": "操作者正在用扳手拧紧一个螺栓"
    }}
  ]
}}
```
""".format(
    object_categories=OBJECT_CATEGORIES_DESC,
    action_categories=ACTION_CATEGORIES_DESC,
)


def build_vlm_user_prompt(batch_id: str, frames_info: list[dict]) -> str:
    """
    构造 VLM 用户消息。
    
    Args:
        batch_id: 批次ID，如 "batch_001"
        frames_info: [{"frame_id": "f001", "timestamp": 12.5}, ...]
    """
    frame_list = "\n".join(
        f"  - {f['frame_id']} (时间戳: {f['timestamp']}s)"
        for f in frames_info
    )
    return f"""请描述以下 {len(frames_info)} 帧图像的内容。

批次ID: {batch_id}

帧列表:
{frame_list}

请按照系统提示中的JSON格式输出这 {len(frames_info)} 帧的描述。注意：
1. 每个物体的 local_id 格式为 obj_01, obj_02 等
2. 动作的 object_ref 格式为 "{batch_id}:obj_01"
3. 只描述图片中真正可见的内容
"""
