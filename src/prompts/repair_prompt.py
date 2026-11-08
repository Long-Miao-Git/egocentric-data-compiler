"""递归修复 Prompt 模板"""

REPAIR_SYSTEM_PROMPT = """你是一个JSON修复专家。你的任务是修正格式错误的JSON，使其符合指定的Schema要求。

## 核心规则

1. **保持原意**：只修正格式和结构，不改变任何语义内容。
2. **补全缺失**：如果缺少必要字段，根据上下文补全合理值。
3. **修正类型**：字符串用了数字就转成字符串，枚举值不合规就选最近的合法值。
4. **不要新增**：不要添加原始数据中没有的实体或VQA条目。

## 输出要求

直接输出修正后的完整JSON，不要输出任何解释性文字。
"""


def build_repair_user_prompt(failed_json: str, errors: list[str], schema_hint: str = "") -> str:
    """
    构造修复用户消息。
    
    Args:
        failed_json: 解析失败的原始JSON字符串
        errors: 错误信息列表
        schema_hint: 目标Schema的简要描述
    """
    errors_text = "\n".join(f"  - {e}" for e in errors)
    return f"""以下JSON解析失败，请修正格式使其合法。

## 目标格式要求
{schema_hint}

## 错误列表
{errors_text}

## 待修复JSON
```json
{failed_json}
```

请输出修正后的完整JSON。注意：
1. 保持原始内容的语义不变
2. 只修正格式错误，不增删条目
3. 枚举值不在合法范围内的，选最接近的合法值
4. 直接输出JSON，不要有任何其他文字
"""
