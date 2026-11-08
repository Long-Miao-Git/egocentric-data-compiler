"""Pydantic 数据模型定义"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
#  基础数据结构
# ============================================================

class FrameInfo(BaseModel):
    """抽帧后的单帧信息"""
    frame_id: str = Field(description="帧ID，如 f001")
    timestamp: float = Field(description="时间戳（秒）")
    image_path: str = Field(description="帧图片路径")
    clarity_score: float = Field(default=0.0, description="清晰度评分（Laplacian方差）")
    scene_id: int = Field(default=0, description="所属场景段ID")


class ObjectDesc(BaseModel):
    """VLM描述的单个物体"""
    label: str = Field(description="L1: 模型自由给出的物体名称")
    category: str = Field(description="L0: 物体大类，必须从L0_OBJECT_CATEGORIES中选")
    raw_description: str = Field(description="L2: 模型原始完整描述")
    batch_id: str = Field(description="所属VLM批次ID")
    local_id: str = Field(description="批内物体编号，如 obj_01")
    attributes: dict = Field(default_factory=dict, description="属性字典")
    confidence_grade: str = Field(default="B", description="可信度 A=清晰/B=部分/C=不确定")


class ActionDesc(BaseModel):
    """VLM描述的单个动作"""
    verb: str = Field(description="L1: 模型自由给出的动作名称")
    category: str = Field(description="L0: 动作大类，必须从L0_ACTION_CATEGORIES中选")
    raw_description: str = Field(description="L2: 模型原始完整描述")
    object_ref: str = Field(default="", description="动作作用对象引用 batch_id:local_id")
    direction: Optional[str] = Field(default=None, description="施力方向，如 向下/向左")
    confidence_grade: str = Field(default="B")


class FrameDescription(BaseModel):
    """VLM对单帧的完整描述"""
    frame_id: str
    timestamp: float
    objects: list[ObjectDesc] = Field(default_factory=list)
    actions: list[ActionDesc] = Field(default_factory=list)
    scene_summary: str = Field(default="")


class Entity(BaseModel):
    """融合后的实体"""
    canonical_name: str = Field(description="规范名（归一后）")
    category: str = Field(description="L0大类")
    synonyms: list[str] = Field(default_factory=list, description="所有曾用名")
    first_seen_frame: str = Field(default="")
    evidence_frames: list[str] = Field(default_factory=list)
    attributes: dict = Field(default_factory=dict)


class VQAItem(BaseModel):
    """单条VQA问答对"""
    mode: str = Field(description="七模式之一")
    question: str
    answer: str
    evidence_frame_ids: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", description="推理过程（reasoning模式必填）")
    difficulty: str = Field(default="easy", description="easy/medium/hard")


class CostRecord(BaseModel):
    """成本记录"""
    module: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_cny: float = 0.0


class ValidationResult(BaseModel):
    """校验结果"""
    status: str = Field(default="pass", description="pass/warning/fail")
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    repaired_count: int = 0


class VideoInfo(BaseModel):
    """视频元信息"""
    path: str
    duration_sec: float = 0.0
    fps: float = 0.0
    total_frames: int = 0
    resolution: str = ""


class CompilerOutput(BaseModel):
    """编译最终输出"""
    video_info: VideoInfo
    frames_sampled: int = 0
    objects: list[Entity] = Field(default_factory=list)
    vqa_items: list[VQAItem] = Field(default_factory=list)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    cost_records: list[CostRecord] = Field(default_factory=list)
    total_cost_cny: float = 0.0


# ============================================================
#  VLM 输出 Schema（用于 Function Calling / Prompt 内嵌）
# ============================================================

VLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame_id": {"type": "string", "description": "帧编号，如 f001"},
                    "timestamp": {"type": "number", "description": "时间戳秒数"},
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "物体名称（自由描述）"},
                                "category": {"type": "string", "enum": [
                                    "tool", "container", "food", "material",
                                    "electronics", "apparel", "furniture",
                                    "fastener", "other"
                                ]},
                                "raw_description": {"type": "string"},
                                "local_id": {"type": "string", "description": "批内编号 obj_01"},
                                "attributes": {
                                    "type": "object",
                                    "description": "颜色/尺寸/位置/状态等"
                                },
                                "confidence_grade": {"type": "string", "enum": ["A", "B", "C"]}
                            },
                            "required": ["label", "category", "raw_description", "local_id", "confidence_grade"]
                        }
                    },
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "verb": {"type": "string", "description": "动作名称"},
                                "category": {"type": "string", "enum": [
                                    "grasp_place", "force_operation", "cut_separate",
                                    "assemble_connect", "clean_organize",
                                    "inspect_measure", "navigate", "mix_stir"
                                ]},
                                "raw_description": {"type": "string"},
                                "object_ref": {"type": "string", "description": "batch_id:local_id"},
                                "direction": {"type": "string"},
                                "confidence_grade": {"type": "string", "enum": ["A", "B", "C"]}
                            },
                            "required": ["verb", "category", "raw_description", "confidence_grade"]
                        }
                    },
                    "scene_summary": {"type": "string"}
                },
                "required": ["frame_id", "timestamp", "objects", "actions", "scene_summary"]
            }
        }
    },
    "required": ["frames"]
}

ENTITY_FUSION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_name": {"type": "string", "description": "规范名"},
                    "category": {"type": "string", "enum": [
                        "tool", "container", "food", "material",
                        "electronics", "apparel", "furniture",
                        "fastener", "other"
                    ]},
                    "synonyms": {"type": "array", "items": {"type": "string"}},
                    "first_seen_frame": {"type": "string"},
                    "evidence_frames": {"type": "array", "items": {"type": "string"}},
                    "attributes": {"type": "object"}
                },
                "required": ["canonical_name", "category", "synonyms", "evidence_frames"]
            }
        }
    },
    "required": ["entities"]
}

VQA_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "vqa_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": [
                        "temporal", "spatial", "attribute",
                        "mechanics", "reasoning", "summary", "trajectory"
                    ]},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "evidence_frame_ids": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]}
                },
                "required": ["mode", "question", "answer", "evidence_frame_ids", "difficulty"]
            }
        }
    },
    "required": ["vqa_items"]
}

# 七种VQA模式
VQA_MODES = ["temporal", "spatial", "attribute", "mechanics", "reasoning", "summary", "trajectory"]
VQA_MODE_CN = {
    "temporal": "时序推理",
    "spatial": "空间布局",
    "attribute": "物体属性",
    "mechanics": "力学/动力学",
    "reasoning": "因果推理",
    "summary": "任务总结",
    "trajectory": "运动轨迹",
}
