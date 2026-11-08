"""词汇管理模块：L0硬枚举 + L1动态词表 + L2描述兜底 + 同义词归一"""

import json
import os
import threading
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .config import Config

# 词表落盘的进程内互斥锁：片段级并行时多个线程会同时写同一批文件
_IO_LOCK = threading.RLock()


def _atomic_write_json(path: Path, data) -> None:
    """写入 JSON。

    注意：这里**不用**「临时文件 + os.replace」的原子替换方案。
    部分环境（带文件安全钩子的 Windows）会拦截 rename/unlink，导致
    `os.replace` 抛 WinError 5。并发安全改由调用方的 `_IO_LOCK` 串行化保证。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
#  L0 硬枚举大类（通用，不限行业）
# ============================================================

L0_OBJECT_CATEGORIES = {
    "tool":        "工具（手工具、电动工具、器具、餐具操作工具）",
    "container":   "容器（箱、袋、碗、盘、杯、瓶、筐）",
    "food":        "食品饮料（食材、成品食物、饮品）",
    "material":    "材料（原材料、板材、液体、粉末、纺织品）",
    "electronics": "电子设备（手机、电脑、相机、家用电器）",
    "apparel":     "服饰防护（衣物、手套、配件、防护装备）",
    "furniture":   "家具设施（桌、椅、柜、台面、墙面、地面）",
    "fastener":    "紧固件连接件（螺丝、螺栓、卡扣、插头、线缆）",
    "other":       "其他（无法归入以上类别的事物）",
}

L0_ACTION_CATEGORIES = {
    "grasp_place":    "抓取放置（拿起、放下、传递、转移）",
    "force_operation":"施力操作（推、拉、按压、挤压、扭转、敲击）",
    "cut_separate":   "切割分离（切、剪、撕、拆、分离）",
    "assemble_connect":"组装连接（插入、对准、连接、安装、固定）",
    "clean_organize": "清洁整理（擦拭、清洗、整理、折叠、分类）",
    "inspect_measure":"检查测量（目视检查、触摸确认、称量、测量）",
    "navigate":       "移动导航（行走、转身、伸手、接近、弯腰）",
    "mix_stir":       "搅拌混合（搅拌、混合、翻动、倾倒）",
}


class VocabManager:
    """L1动态词表 + 同义词归一管理"""

    def __init__(self):
        self._word_table_path = Config.CACHE_DIR / "vocab" / "word_table.json"
        self._synonym_path = Config.CACHE_DIR / "vocab" / "synonyms.json"
        self._candidate_path = Config.CACHE_DIR / "vocab" / "candidates.json"
        self._word_table: dict[str, str] = {}     # label -> canonical_name
        self._synonyms: dict[str, list[str]] = {}  # canonical_name -> [synonyms]
        self._candidates: list[dict] = []          # 未归一候选词
        self._ensure_dirs()
        self._load()

    def _ensure_dirs(self):
        d = self._word_table_path.parent
        d.mkdir(parents=True, exist_ok=True)

    # --- 加载/保存 ---

    def _load(self):
        for path, attr in [
            (self._word_table_path, "_word_table"),
            (self._synonym_path, "_synonyms"),
            (self._candidate_path, "_candidates"),
        ]:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    setattr(self, attr, json.load(f))

    def _save(self):
        # 多个片段线程会各自持有 VocabManager 实例并并发落盘，必须串行化，
        # 否则同一路径被同时打开写入会触发 PermissionError，且后写会覆盖先写导致丢词。
        with _IO_LOCK:
            # 先合并磁盘上的最新内容，保证并发写入是"累加"而不是"覆盖"
            try:
                if self._word_table_path.exists():
                    disk = json.loads(self._word_table_path.read_text(encoding="utf-8"))
                    merged = dict(disk)
                    merged.update(self._word_table)
                    self._word_table = merged
            except Exception:
                pass

            for path, data in [
                (self._word_table_path, self._word_table),
                (self._synonym_path, self._synonyms),
                (self._candidate_path, self._candidates),
            ]:
                _atomic_write_json(path, data)

    # --- 同义词归一 ---

    def normalize(self, label: str, category: str = "") -> tuple[str, bool]:
        """
        将自由文本 label 归一为规范名。
        返回 (canonical_name, was_normalized)
        """
        if not label:
            return ("unknown", False)

        label_clean = label.strip().lower()

        # 1) 精确匹配已有词表
        if label_clean in self._word_table:
            return (self._word_table[label_clean], True)

        # 2) 模糊匹配：编辑距离 / SequenceMatcher
        best_match = None
        best_ratio = 0.0
        for known_label, canonical in self._word_table.items():
            ratio = SequenceMatcher(None, label_clean, known_label).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = canonical
        if best_ratio >= 0.8 and best_match:
            # 自动归一
            self._word_table[label_clean] = best_match
            self._save()
            return (best_match, True)

        # 3) 无法归一 → 作为新词入库，canonical = 原始label
        canonical = label.strip()
        self._word_table[label_clean] = canonical
        if canonical not in self._synonyms:
            self._synonyms[canonical] = []
        if label_clean not in self._synonyms[canonical]:
            self._synonyms[canonical].append(label_clean)
        self._save()
        return (canonical, False)

    # --- 候选词上报 ---

    def report_candidate(self, term: str, context: str = "", frame_id: str = ""):
        """上报一个无法自动归一的候选词，等待用户审核"""
        self._candidates.append({
            "term": term,
            "context": context,
            "frame_id": frame_id,
        })
        self._save()

    # --- 用户反馈接口 ---

    def add_synonym(self, canonical: str, synonym: str):
        """用户审核后添加同义词条"""
        synonym_clean = synonym.strip().lower()
        self._word_table[synonym_clean] = canonical
        if canonical not in self._synonyms:
            self._synonyms[canonical] = []
        if synonym_clean not in self._synonyms[canonical]:
            self._synonyms[canonical].append(synonym_clean)
        self._save()

    def merge_entities(self, source: str, target: str):
        """将 source 规范名合并到 target"""
        if source not in self._synonyms:
            return
        for syn in self._synonyms[source]:
            self._word_table[syn] = target
            if target not in self._synonyms:
                self._synonyms[target] = []
            if syn not in self._synonyms[target]:
                self._synonyms[target].append(syn)
        del self._synonyms[source]
        self._save()

    # --- 查询 ---

    def get_all_words(self) -> dict:
        return {
            "word_table": self._word_table,
            "synonyms": self._synonyms,
            "candidates": self._candidates,
        }

    def get_candidates(self) -> list[dict]:
        return self._candidates

    def clear_candidates(self):
        self._candidates = []
        self._save()
