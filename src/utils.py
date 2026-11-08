# -*- coding: utf-8 -*-
"""通用小工具"""

import json
import time
from pathlib import Path


def write_json(path, data, retries: int = 5, delay: float = 1.0) -> None:
    """写 JSON 文件，遇到 PermissionError 自动重试。

    为什么需要重试：
        刚被创建或刚被写入的文件，在部分 Windows 环境下会被杀毒软件、
        文件索引服务或编辑器预览短暂持有句柄，此时立刻覆盖写会抛
        PermissionError。等待 1~2 秒后句柄释放即可正常写入。

    为什么不直接用「临时文件 + os.replace」：
        带文件安全钩子的环境会拦截 rename/unlink，os.replace 会抛 WinError 5。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for i in range(retries):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        except PermissionError as e:
            last_err = e
            if i < retries - 1:
                time.sleep(delay)
    raise last_err
