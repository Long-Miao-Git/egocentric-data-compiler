"""配置加载器

配置来源与优先级（高 → 低）：
    1. 命令行参数（由 compile.py / app.py 显式覆盖）
    2. 环境变量（见下方 ENV_ALIASES）
    3. config.yaml
    4. 代码内置默认值

API 密钥只从环境变量读取，不写入配置文件，避免误提交到仓库。
支持任何 OpenAI 兼容接口（DashScope / OpenAI / vLLM / Ollama 等），
只需改 base_url 与 model 即可。
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # 允许未装 python-dotenv 时降级运行
    def load_dotenv(*a, **kw):
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env 优先级：当前工作目录 > 项目根目录
load_dotenv(Path.cwd() / ".env")
load_dotenv(PROJECT_ROOT / ".env")

# 环境变量别名表：同一个配置项可以有多个环境变量名，按顺序取第一个非空值
ENV_ALIASES = {
    "api_key": ["DASHSCOPE_API_KEY", "OPENAI_API_KEY", "API_KEY"],
    "base_url": ["DASHSCOPE_BASE_URL", "OPENAI_BASE_URL", "API_BASE_URL"],
    "vlm_model": ["VLM_MODEL"],
    "llm_model": ["LLM_MODEL"],
}


def _env(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


class Config:
    """全局配置单例（类属性方式访问，保持与既有模块的兼容）"""

    # --- API ---
    DASHSCOPE_API_KEY: str = _env(*ENV_ALIASES["api_key"]) or ""
    DASHSCOPE_BASE_URL: str = (
        _env(*ENV_ALIASES["base_url"])
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # --- 路径 ---
    PROJECT_ROOT: Path = PROJECT_ROOT
    OUTPUT_ROOT: Path = PROJECT_ROOT / "output"
    CACHE_DIR: Path = OUTPUT_ROOT / "cache"
    OUTPUT_DIR: Path = OUTPUT_ROOT / "output"
    FRAMES_DIR: Path = OUTPUT_ROOT / "cache" / "frames"

    _cfg: dict = {}
    _cfg_path: Optional[Path] = None

    # ------------------------------------------------------------------ 加载

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "Config":
        """加载配置文件。传 None 时按 项目根目录/config.yaml 查找。"""
        path = Path(config_path) if config_path else (PROJECT_ROOT / "config.yaml")
        cls._cfg_path = path if path.exists() else None
        if cls._cfg_path:
            with open(cls._cfg_path, "r", encoding="utf-8") as f:
                cls._cfg = yaml.safe_load(f) or {}
        else:
            cls._cfg = {}

        # config.yaml 里的 api 段可覆盖 base_url（不覆盖密钥）
        api = cls._cfg.get("api", {}) or {}
        if api.get("base_url") and not _env(*ENV_ALIASES["base_url"]):
            cls.DASHSCOPE_BASE_URL = api["base_url"]

        # 输出根目录：命令行 > 环境变量 > config.yaml > 默认
        out = (
            _env("COMPILER_OUTPUT_DIR")
            or (cls._cfg.get("paths", {}) or {}).get("output_root")
        )
        if out:
            cls.set_output_root(out)

        cls._ensure_dirs()
        return cls

    @classmethod
    def set_output_root(cls, path: str | Path) -> None:
        """设置输出根目录（cache / output / frames 都会在其下重建）"""
        root = Path(path).expanduser().resolve()
        cls.OUTPUT_ROOT = root
        cls.CACHE_DIR = root / "cache"
        cls.OUTPUT_DIR = root / "output"
        cls.FRAMES_DIR = root / "cache" / "frames"
        cls._ensure_dirs()

    @classmethod
    def _ensure_dirs(cls) -> None:
        for d in (cls.CACHE_DIR, cls.OUTPUT_DIR, cls.FRAMES_DIR):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if not cls._cfg and cls._cfg_path is None:
            cls.load()

    # ------------------------------------------------------------------ 读取

    @classmethod
    def get(cls, *keys: str, default: Any = None) -> Any:
        """按点路径取配置，如 Config.get('sampling', 'fps')"""
        cls._ensure_loaded()
        val: Any = cls._cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    @classmethod
    def sampling(cls) -> dict:
        cls._ensure_loaded()
        return cls._cfg.get("sampling", {}) or {}

    @classmethod
    def vlm(cls) -> dict:
        cls._ensure_loaded()
        cfg = dict(cls._cfg.get("vlm", {}) or {})
        if _env(*ENV_ALIASES["vlm_model"]):
            cfg["model"] = _env(*ENV_ALIASES["vlm_model"])
        return cfg

    @classmethod
    def llm(cls) -> dict:
        cls._ensure_loaded()
        cfg = dict(cls._cfg.get("llm", {}) or {})
        if _env(*ENV_ALIASES["llm_model"]):
            cfg["model"] = _env(*ENV_ALIASES["llm_model"])
        return cfg

    @classmethod
    def vqa_config(cls) -> dict:
        cls._ensure_loaded()
        return cls._cfg.get("vqa", {}) or {}

    @classmethod
    def cost_rates(cls) -> dict:
        cls._ensure_loaded()
        return cls._cfg.get("cost", {}) or {}

    @classmethod
    def require_api_key(cls) -> str:
        """需要调用模型时统一走这里，缺失则给出明确指引"""
        key = cls.DASHSCOPE_API_KEY or _env(*ENV_ALIASES["api_key"])
        if not key:
            raise RuntimeError(
                "未配置 API 密钥。请设置环境变量 DASHSCOPE_API_KEY"
                "（或其他 OpenAI 兼容服务的 API Key），\n"
                "  方式一：复制 .env.example 为 .env 并填入密钥\n"
                "  方式二：export DASHSCOPE_API_KEY=sk-xxxx"
            )
        return key
