"""上游改动指针 manifest（M6）。

每次对 sglang fork 的改动都登记一条 PatchEntry：fork 分支 / 基线 commit / 意图 /
状态 / 指标结果 / 关联模块。便于 review 与回退，也方便 M6 A/B 对照。
manifest 是机器可读的 JSON，本模块负责解析与字段校验。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REQUIRED = ("id", "fork_repo", "branch", "commit", "intent")


@dataclass
class PatchEntry:
    id: str
    fork_repo: str
    branch: str
    commit: str
    intent: str
    status: str = "pending"          # pending / in_review / merged / reverted
    metric_result: str = ""
    related_module: str = ""


def load_manifest(path: str | Path) -> list[PatchEntry]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("manifest 顶层必须是列表")
    out: list[PatchEntry] = []
    for i, item in enumerate(data):
        missing = [k for k in REQUIRED if k not in item]
        if missing:
            raise ValueError(f"第 {i} 条缺少必填字段: {missing}")
        out.append(PatchEntry(**{k: item.get(k, "") for k in PatchEntry.__dataclass_fields__}))
    return out


def validate(entries: list[PatchEntry]) -> list[str]:
    """返回问题列表；空列表表示全部合规。"""
    problems: list[str] = []
    ids = set()
    for e in entries:
        if e.id in ids:
            problems.append(f"重复 id: {e.id}")
        ids.add(e.id)
        if e.status not in ("pending", "in_review", "merged", "reverted"):
            problems.append(f"{e.id}: 非法 status {e.status}")
    return problems
