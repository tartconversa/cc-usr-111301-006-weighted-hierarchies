"""层级归集桥表构建器。

本模块把已冻结、允许多父归属的有向无环组织版本展开为：

* 闭包桥（closure bridge）：节点到每个显式声明目标的全部有向路径权重乘积之和；
* 根口径（roots caliber）：节点到所有根的闭包桥之和；
* frontier 桥（first-hit bridge）：沿每条向根路径遇到第一个目标即停止；
* 叶事实联接与逐事实冻结的整数分归集。

核心约束：

* 边由 child 指向 parent，权重为非负有理数；
* 每个有出边的节点，其出边权重和必须恰为 1；无出边即为根；
* 有环、悬空引用、非法权重或权重不守恒的版本整体拒绝；
* 桥权重用精确 Fraction 计算，路径条数独立保留为证据，但事实绝不按路径条数复制；
* closure 权重和不为 1 或目标跨越祖先层次时拒绝直接归集，不做事后归一化；
* frontier 归集天然保证每个起点总权重为 1，未命中目标落入 UNASSIGNED；
* 每笔事实按最大余数法单独舍入、冻结并计算摘要，之后才做整数分汇总；
* 负费用先对绝对值分配，再恢复符号，保持逐笔与汇总的整数分守恒。

路径证据采用“首步共享 DAG”压缩：每个 (起点, 目标) 只保存一步转移的父节点、
该步贡献权重与路径条数，递归证据由同一张表按 (父节点, 目标) 引用，因此不需要
枚举指数级路径即可同时得到完整权重、路径条数和可审计依据。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

UNASSIGNED_KEY = "UNASSIGNED"
SUPPORTED_TIE_BREAK = "target_key_ascending"
SUPPORTED_NEGATIVE_ROUNDING = "allocate_absolute_then_restore_sign"
DEFAULT_FRONTIER_UNMATCHED = UNASSIGNED_KEY
ZERO = Fraction(0)
ONE = Fraction(1)
_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_RATIONAL = re.compile(r"^[+-]?(?:[0-9]+/[0-9]+|[0-9]+(?:\.[0-9]+)?)$")


# ---------------------------------------------------------------------------
# 错误与诊断
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


class BridgeInputError(ValueError):
    """输入版本、目标集合或归集配置被拒绝，携带全部可定位的问题。"""

    def __init__(self, issues: Sequence[ValidationIssue], summary: str = "输入被拒绝"):
        self.issues = list(issues)
        self.summary = summary
        super().__init__(summary + ": " + "; ".join(f"{item.code}: {item.message}" for item in self.issues))

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.summary, "issues": [item.as_dict() for item in self.issues]}


def _fail(code: str, message: str, summary: str = "输入被拒绝", **context: Any) -> None:
    raise BridgeInputError([ValidationIssue(code, message, context)], summary=summary)


def fraction_text(value: Fraction) -> str:
    """规范有理数文本，始终带分母；零为 ``0/1``。"""

    return f"{value.numerator}/{value.denominator}"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 输入模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    child: str
    parent: str
    weight: Fraction
    raw_weight: str

    def as_dict(self) -> dict[str, Any]:
        return {"child": self.child, "parent": self.parent, "weight": fraction_text(self.weight)}


@dataclass(frozen=True)
class Version:
    """已冻结的组织版本；只暴露构造后不可重排的逻辑内容与统计。"""

    version_id: str
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    roots: tuple[str, ...]
    leaves: tuple[str, ...]
    digest: str
    source_digests: Mapping[str, str]
    out_edges: Mapping[str, tuple[tuple[str, Fraction], ...]]
    in_edges: Mapping[str, tuple[str, ...]]
    topo_order: tuple[str, ...]

    @property
    def non_roots(self) -> tuple[str, ...]:
        return tuple(node for node in self.nodes if self.out_edges[node])

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version_id,
            "digest": self.digest,
            "nodes": list(self.nodes),
            "edges": [edge.as_dict() for edge in self.edges],
            "roots": list(self.roots),
            "leaves": list(self.leaves),
            "non_roots": list(self.non_roots),
            "source_digests": dict(self.source_digests),
        }


@dataclass(frozen=True)
class HierarchyConfig:
    target_sets: tuple[tuple[str, ...], ...]
    negative_rounding: str = SUPPORTED_NEGATIVE_ROUNDING
    tie_break: str = SUPPORTED_TIE_BREAK
    frontier_unmatched: str = DEFAULT_FRONTIER_UNMATCHED

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_sets": [list(targets) for targets in self.target_sets],
            "negative_rounding": self.negative_rounding,
            "tie_break": self.tie_break,
            "frontier_unmatched": self.frontier_unmatched,
        }


def parse_weight(raw: object) -> Fraction:
    """只接受非负整数、有限小数或 p/q；空白、负号结果、科学计数法均拒绝。"""

    if not isinstance(raw, str):
        raise ValueError("权重必须是字符串")
    text = raw.strip()
    if not text:
        raise ValueError("权重为空")
    if not _RATIONAL.match(text):
        raise ValueError(f"权重不是非负有理数: {raw!r}")
    if "/" in text:
        numerator_text, denominator_text = text.split("/", 1)
        denominator = int(denominator_text)
        if denominator == 0:
            raise ValueError(f"权重分母为零: {raw!r}")
        value = Fraction(int(numerator_text), denominator)
    else:
        value = Fraction(text)
    if value < 0:
        raise ValueError(f"权重为负: {raw!r}")
    return value


def parse_cents(raw: object) -> int:
    if not isinstance(raw, str) or not _INTEGER.match(raw.strip()):
        raise ValueError(f"金额必须是整数分: {raw!r}")
    return int(raw.strip())


def _read_json(path: Path) -> tuple[Any, str]:
    payload = path.read_bytes()
    return json.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    payload = path.read_bytes()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise BridgeInputError([ValidationIssue("MISSING_HEADER", f"{path} 缺少 CSV 表头")])
        rows = list(reader)
    return rows, hashlib.sha256(payload).hexdigest()


def _find_cycle(out_edges: Mapping[str, tuple[tuple[str, Fraction], ...]]) -> list[str] | None:
    white, gray, black = 0, 1, 2
    color = {node: white for node in out_edges}
    for start in sorted(out_edges):
        if color[start] != white:
            continue
        color[start] = gray
        path = [start]
        stack: list[tuple[str, Iterator[tuple[str, Fraction]]]] = [(start, iter(out_edges[start]))]
        while stack:
            node, edges = stack[-1]
            try:
                parent, _weight = next(edges)
            except StopIteration:
                color[node] = black
                stack.pop()
                path.pop()
                continue
            state = color.get(parent, white)
            if state == gray:
                position = path.index(parent)
                return path[position:] + [parent]
            if state == white:
                color[parent] = gray
                path.append(parent)
                stack.append((parent, iter(out_edges[parent])))
    return None


def _topological_order(
    nodes: Sequence[str],
    out_edges: Mapping[str, tuple[tuple[str, Fraction], ...]],
    in_edges: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    """按“根 -> 后代”排序，使任一节点的父节点都已先行处理。"""

    pending_parents = {node: len(out_edges[node]) for node in nodes}
    ready = [node for node in nodes if pending_parents[node] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for child in in_edges[node]:
            pending_parents[child] -= 1
            if pending_parents[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(nodes):
        return None
    return tuple(order)


def load_version(
    nodes_path: str | Path,
    edges_path: str | Path,
    *,
    expected_digest: str | None = None,
) -> Version:
    """加载并校验冻结版本；任何结构性或守恒缺陷都会整体拒绝。"""

    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)
    nodes_doc, nodes_source_digest = _read_json(nodes_path)
    edge_rows, edges_source_digest = _read_csv(edges_path)

    issues: list[ValidationIssue] = []
    version_id: str | None = None
    if not isinstance(nodes_doc, dict):
        issues.append(ValidationIssue("INVALID_NODES_DOCUMENT", "nodes.json 顶层必须是对象"))
        node_ids: list[str] = []
    else:
        version_id = nodes_doc.get("version")
        if not isinstance(version_id, str) or not version_id.strip():
            issues.append(ValidationIssue("MISSING_VERSION", "版本必须是非空字符串"))
            version_id = ""
        raw_nodes = nodes_doc.get("nodes")
        if not isinstance(raw_nodes, list):
            issues.append(ValidationIssue("INVALID_NODES", "nodes 必须是数组"))
            node_ids = []
        else:
            node_ids = []
            for index, item in enumerate(raw_nodes):
                if not isinstance(item, str) or not item.strip():
                    issues.append(ValidationIssue("INVALID_NODE", f"第 {index} 个节点不是非空字符串", {"index": index}))
                elif item in node_ids:
                    issues.append(ValidationIssue("DUPLICATE_NODE", f"节点重复: {item}", {"node": item}))
                else:
                    node_ids.append(item)

    nodes = tuple(node_ids)
    node_set = set(nodes)
    edges: list[Edge] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row_number, row in enumerate(edge_rows, start=2):
        child = (row.get("child") or "").strip()
        parent = (row.get("parent") or "").strip()
        raw_weight = (row.get("weight") or "").strip()
        if not child or not parent:
            issues.append(
                ValidationIssue(
                    "INVALID_EDGE_REFERENCE",
                    f"第 {row_number} 行 child/parent 为空",
                    {"row": row_number},
                )
            )
            continue
        if child not in node_set or parent not in node_set:
            missing = [name for name in (child, parent) if name not in node_set]
            issues.append(
                ValidationIssue(
                    "DANGLING_REFERENCE",
                    f"第 {row_number} 行引用未知节点: {', '.join(missing)}",
                    {"row": row_number, "child": child, "parent": parent, "missing": missing},
                )
            )
        if (child, parent) in seen_pairs:
            issues.append(
                ValidationIssue(
                    "DUPLICATE_EDGE",
                    f"重复边 {child} -> {parent}",
                    {"child": child, "parent": parent, "row": row_number},
                )
            )
            continue
        seen_pairs.add((child, parent))
        try:
            weight = parse_weight(raw_weight)
        except ValueError as error:
            issues.append(
                ValidationIssue(
                    "INVALID_WEIGHT",
                    str(error),
                    {"row": row_number, "child": child, "parent": parent, "raw_weight": raw_weight},
                )
            )
            continue
        edges.append(Edge(child=child, parent=parent, weight=weight, raw_weight=raw_weight))

    out_edges_map: dict[str, tuple[tuple[str, Fraction], ...]] = {node: () for node in nodes}
    in_edges_map: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        if edge.child not in node_set or edge.parent not in node_set:
            # 悬空引用已记入 issues，不能污染邻接表。
            continue
        out_edges_map[edge.child] = out_edges_map[edge.child] + ((edge.parent, edge.weight),)
        in_edges_map[edge.parent].append(edge.child)
    out_edges = {node: tuple(sorted(values, key=lambda pair: pair[0])) for node, values in out_edges_map.items()}
    in_edges = {node: tuple(sorted(children)) for node, children in in_edges_map.items()}

    for node in nodes:
        outgoing = out_edges[node]
        if outgoing:
            total = sum((weight for _parent, weight in outgoing), ZERO)
            if total != ONE:
                issues.append(
                    ValidationIssue(
                        "WEIGHT_NOT_CONSERVED",
                        f"非根节点 {node} 的出边权重和为 {fraction_text(total)}，必须为 1",
                        {
                            "node": node,
                            "sum": fraction_text(total),
                            "parents": [parent for parent, _weight in outgoing],
                        },
                    )
                )
        # 根按定义是无出边节点，因此“根无出边”由构造保证。

    cycle = _find_cycle(out_edges)
    if cycle is not None:
        issues.append(
            ValidationIssue(
                "CYCLE",
                "组织边中存在有向环: " + " -> ".join(cycle),
                {"path": list(cycle)},
            )
        )

    topo_order = _topological_order(nodes, out_edges, in_edges) if cycle is None else None
    if cycle is None and topo_order is None:
        issues.append(ValidationIssue("CYCLE", "拓扑排序失败，图中存在有向环"))

    if issues:
        raise BridgeInputError(issues, summary="组织版本被拒绝")

    canonical_edges = sorted(edges, key=lambda edge: (edge.child, edge.parent))
    canonical_payload = {
        "version": version_id,
        "nodes": sorted(nodes),
        "edges": [
            {"child": edge.child, "parent": edge.parent, "weight": fraction_text(edge.weight)}
            for edge in canonical_edges
        ],
    }
    version_digest = _digest(canonical_payload)
    if expected_digest is not None:
        normalized_expected = expected_digest.strip().lower()
        if normalized_expected != version_digest:
            _fail(
                "VERSION_DIGEST_MISMATCH",
                f"版本摘要不匹配：期望 {normalized_expected}，实际 {version_digest}",
                summary="冻结版本不匹配",
                expected=normalized_expected,
                actual=version_digest,
            )

    roots = tuple(sorted(node for node in nodes if not out_edges[node]))
    leaves = tuple(sorted(node for node in nodes if not in_edges[node]))
    return Version(
        version_id=version_id,
        nodes=tuple(sorted(nodes)),
        edges=tuple(canonical_edges),
        roots=roots,
        leaves=leaves,
        digest=version_digest,
        source_digests={"nodes": nodes_source_digest, "edges": edges_source_digest},
        out_edges=out_edges,
        in_edges=in_edges,
        topo_order=topo_order or tuple(),
    )


def load_hierarchy_config(path: str | Path) -> HierarchyConfig:
    """读取并严格校验层级归集配置。"""

    document, _source_digest = _read_json(Path(path))
    if not isinstance(document, dict):
        _fail("INVALID_HIERARCHY", "hierarchy.json 顶层必须是对象", summary="层级配置被拒绝")
    target_sets = document.get("target_sets")
    issues: list[ValidationIssue] = []
    normalized_sets: list[tuple[str, ...]] = []
    if not isinstance(target_sets, list) or not target_sets:
        issues.append(ValidationIssue("INVALID_TARGET_SETS", "target_sets 必须是非空数组"))
    else:
        for index, targets in enumerate(target_sets):
            if not isinstance(targets, list) or not targets:
                issues.append(
                    ValidationIssue(
                        "INVALID_TARGET_SET",
                        f"第 {index} 个目标集合必须是非空数组",
                        {"index": index},
                    )
                )
                continue
            if any(not isinstance(target, str) or not target.strip() for target in targets):
                issues.append(
                    ValidationIssue(
                        "INVALID_TARGET_SET",
                        f"第 {index} 个目标集合包含非法节点名",
                        {"index": index},
                    )
                )
                continue
            cleaned = tuple(targets)
            if len(set(cleaned)) != len(cleaned):
                issues.append(
                    ValidationIssue(
                        "DUPLICATE_TARGET",
                        f"第 {index} 个目标集合包含重复目标",
                        {"index": index},
                    )
                )
                continue
            normalized_sets.append(cleaned)

    negative_rounding = document.get("negative_rounding", SUPPORTED_NEGATIVE_ROUNDING)
    tie_break = document.get("tie_break", SUPPORTED_TIE_BREAK)
    frontier_unmatched = document.get("frontier_unmatched", DEFAULT_FRONTIER_UNMATCHED)
    if negative_rounding != SUPPORTED_NEGATIVE_ROUNDING:
        issues.append(
            ValidationIssue(
                "UNSUPPORTED_NEGATIVE_ROUNDING",
                f"不支持负费用舍入规则: {negative_rounding!r}",
                {"value": negative_rounding},
            )
        )
    if tie_break != SUPPORTED_TIE_BREAK:
        issues.append(
            ValidationIssue(
                "UNSUPPORTED_TIE_BREAK",
                f"不支持并列规则: {tie_break!r}",
                {"value": tie_break},
            )
        )
    if frontier_unmatched != UNASSIGNED_KEY:
        issues.append(
            ValidationIssue(
                "UNSUPPORTED_FRONTIER_UNMATCHED",
                "frontier 未命中桶必须是 UNASSIGNED",
                {"value": frontier_unmatched},
            )
        )
    if issues:
        raise BridgeInputError(issues, summary="层级配置被拒绝")
    return HierarchyConfig(
        target_sets=tuple(normalized_sets),
        negative_rounding=negative_rounding,
        tie_break=tie_break,
        frontier_unmatched=frontier_unmatched,
    )


# ---------------------------------------------------------------------------
# 闭包桥：全部有向路径权重乘积之和 + 压缩路径证据
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosureStep:
    """闭包证据中的一步转移；指向同一张表中的父节点单元格。"""

    parent: str
    step_weight: Fraction
    branch_cell: tuple[str, str]
    branch_weight: Fraction
    branch_path_count: int
    contribution: Fraction

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "step_weight": fraction_text(self.step_weight),
            "branch_cell": list(self.branch_cell),
            "branch_weight": fraction_text(self.branch_weight),
            "branch_path_count": self.branch_path_count,
            "contribution": fraction_text(self.contribution),
        }


@dataclass(frozen=True)
class ClosureCell:
    source: str
    target: str
    weight: Fraction
    path_count: int
    basis: str  # "self" | "paths" | "not_ancestor"
    steps: tuple[ClosureStep, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": fraction_text(self.weight),
            "path_count": self.path_count,
            "basis": self.basis,
            "steps": [step.as_dict() for step in self.steps],
            "evidence_digest": _digest(self._digest_payload()),
        }

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": fraction_text(self.weight),
            "path_count": self.path_count,
            "basis": self.basis,
            "steps": [
                {
                    "parent": step.parent,
                    "step_weight": fraction_text(step.step_weight),
                    "branch_cell": list(step.branch_cell),
                    "branch_weight": fraction_text(step.branch_weight),
                    "branch_path_count": step.branch_path_count,
                    "contribution": fraction_text(step.contribution),
                }
                for step in self.steps
            ],
        }

    @property
    def evidence_digest(self) -> str:
        return _digest(self._digest_payload())


@dataclass(frozen=True)
class ClosureTable:
    version_digest: str
    cells: Mapping[tuple[str, str], ClosureCell]
    sources: tuple[str, ...]
    roots: tuple[str, ...]

    def cell(self, source: str, target: str) -> ClosureCell | None:
        return self.cells.get((source, target))

    def weight(self, source: str, target: str) -> Fraction:
        cell = self.cells.get((source, target))
        return cell.weight if cell is not None else ZERO

    def path_count(self, source: str, target: str) -> int:
        cell = self.cells.get((source, target))
        return cell.path_count if cell is not None else 0

    def root_caliber(self, source: str) -> Fraction:
        return sum((self.weight(source, root) for root in self.roots), ZERO)

    def target_weights(self, source: str, targets: Sequence[str]) -> tuple[tuple[str, Fraction], ...]:
        return tuple((target, self.weight(source, target)) for target in targets)


def build_closure_bridges(
    version: Version,
    sources: Sequence[str] | None = None,
) -> ClosureTable:
    """一次自根向叶的动态规划得到每个起点到全部祖先的闭包桥。

    对非根节点 ``n`` 与祖先 ``t``：

        weight(n, t) = sum_{parent p} weight(n, p) * weight(p, t)
        paths(n, t)  = sum_{parent p} paths(p, t)

    每条闭包单元格的步骤只引用父节点的同目标单元格，形成共享 DAG；
    因此权重与路径条数都是多项式时间求得，证据体积正比于单元格数 × 入度，
    无需枚举任何指数级路径。
    """

    selected = tuple(version.nodes) if sources is None else tuple(sources)
    unknown = [source for source in selected if source not in version.out_edges]
    if unknown:
        _fail(
            "UNKNOWN_SOURCE",
            "起点不在冻结版本中: " + ", ".join(sorted(unknown)),
            summary="起点集合被拒绝",
            sources=sorted(unknown),
        )

    all_cells: dict[tuple[str, str], ClosureCell] = {}
    cells_by_source: dict[str, list[ClosureCell]] = {}
    for node in version.topo_order:
        # 每个节点对自身恒有一条长度为零的路径，权重 1、路径数 1，
        # 无论它是不是根；这是“包含节点自身权重 1”的定义要求。
        self_cell = ClosureCell(
            source=node, target=node, weight=ONE, path_count=1, basis="self", steps=tuple()
        )
        all_cells[(node, node)] = self_cell
        cells_by_source[node] = [self_cell]
        outgoing = version.out_edges[node]
        if not outgoing:
            continue
        aggregates: dict[str, dict[str, Any]] = {}
        for parent, step_weight in outgoing:
            for parent_cell in cells_by_source[parent]:
                target = parent_cell.target
                if target == node:
                    continue
                record = aggregates.setdefault(target, {"weight": ZERO, "path_count": 0, "steps": []})
                contribution = step_weight * parent_cell.weight
                record["weight"] += contribution
                record["path_count"] += parent_cell.path_count
                record["steps"].append(
                    ClosureStep(
                        parent=parent,
                        step_weight=step_weight,
                        branch_cell=(parent, target),
                        branch_weight=parent_cell.weight,
                        branch_path_count=parent_cell.path_count,
                        contribution=contribution,
                    )
                )
        for target, record in aggregates.items():
            cell = ClosureCell(
                source=node,
                target=target,
                weight=record["weight"],
                path_count=record["path_count"],
                basis="paths",
                steps=tuple(record["steps"]),
            )
            all_cells[(node, target)] = cell
            cells_by_source[node].append(cell)

    # 保留全部单元格，使任意单元格的证据引用都能在同一张表中解析；
    # rows 输出只按显式起点集合过滤。
    cells = dict(all_cells)

    # 每个起点的根口径必须恰为 1：这是权重守恒与 DAG 无环的直接推论。
    # 与起点不在同一连通分量的根贡献 0，仍参与该口径求和。
    for source in selected:
        caliber = sum((cells[(source, root)].weight for root in version.roots if (source, root) in cells), ZERO)
        if caliber != ONE:
            _fail(
                "ROOT_CALIBER_NOT_ONE",
                f"起点 {source} 的 roots 口径为 {fraction_text(caliber)}，必须为 1",
                summary="闭包桥不守恒",
                source=source,
                caliber=fraction_text(caliber),
            )

    return ClosureTable(
        version_digest=version.digest,
        cells=cells,
        sources=tuple(selected),
        roots=version.roots,
    )


# ---------------------------------------------------------------------------
# frontier 桥：first-hit 语义
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontierStep:
    parent: str
    step_weight: Fraction
    branch_cell: tuple[str, str]
    branch_weight: Fraction
    branch_path_count: int
    contribution: Fraction

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "step_weight": fraction_text(self.step_weight),
            "branch_cell": list(self.branch_cell),
            "branch_weight": fraction_text(self.branch_weight),
            "branch_path_count": self.branch_path_count,
            "contribution": fraction_text(self.contribution),
        }


@dataclass(frozen=True)
class FrontierCell:
    source: str
    target: str
    weight: Fraction
    path_count: int
    basis: str  # "self" | "first_hit" | "unmatched" | "not_reached"
    steps: tuple[FrontierStep, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": fraction_text(self.weight),
            "path_count": self.path_count,
            "basis": self.basis,
            "steps": [step.as_dict() for step in self.steps],
            "evidence_digest": _digest(self._digest_payload()),
        }

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": fraction_text(self.weight),
            "path_count": self.path_count,
            "basis": self.basis,
            "steps": [
                {
                    "parent": step.parent,
                    "step_weight": fraction_text(step.step_weight),
                    "branch_cell": list(step.branch_cell),
                    "branch_weight": fraction_text(step.branch_weight),
                    "branch_path_count": step.branch_path_count,
                    "contribution": fraction_text(step.contribution),
                }
                for step in self.steps
            ],
        }

    @property
    def evidence_digest(self) -> str:
        return _digest(self._digest_payload())


@dataclass(frozen=True)
class FrontierTable:
    version_digest: str
    target_set: tuple[str, ...]
    cells: Mapping[tuple[str, str], FrontierCell]
    sources: tuple[str, ...]
    roots: tuple[str, ...]

    def cell(self, source: str, target: str) -> FrontierCell | None:
        return self.cells.get((source, target))

    def distribution(self, source: str) -> tuple[tuple[str, Fraction], ...]:
        return tuple(
            (target, self.cells[(source, target)].weight)
            for target in self.target_set + (UNASSIGNED_KEY,)
            if (source, target) in self.cells
        )

    def cells_for(self, source: str) -> tuple[FrontierCell, ...]:
        return tuple(
            self.cells[(source, target)]
            for target in self.target_set + (UNASSIGNED_KEY,)
            if (source, target) in self.cells
        )


def build_frontier_bridges(
    version: Version,
    target_set: Sequence[str],
    sources: Sequence[str] | None = None,
) -> FrontierTable:
    """first-hit 桥：沿每条向根路径，遇到第一个目标即停止。

    以 roots-first 拓扑序递推：

    * 起点本身是目标：该目标权重 1、路径数 1，其余为 0；
    * 否则对每个父节点 ``p``：``dist(n)[k] = sum_p w(n,p) * dist(p)[k]``，
      其中 ``dist(p)[k]`` 已由 p 的 first-hit 语义给出；
    * 未命中任何目标的分支累计到 ``UNASSIGNED``；
    * 根且非目标：全部权重落入 ``UNASSIGNED``。

    这里从不先算普通闭包再归一化；每个起点的全部分桶权重之和严格为 1。
    """

    declared = tuple(target_set)
    if not declared:
        _fail("EMPTY_TARGET_SET", "frontier 目标集合不能为空", summary="目标集合被拒绝")
    if len(set(declared)) != len(declared):
        _fail("DUPLICATE_TARGET", "frontier 目标集合包含重复目标", summary="目标集合被拒绝")
    unknown = [target for target in declared if target not in version.out_edges]
    if unknown:
        _fail(
            "TARGET_NOT_IN_VERSION",
            "目标不在冻结版本中: " + ", ".join(sorted(unknown)),
            summary="目标集合被拒绝",
            targets=sorted(unknown),
        )

    selected = tuple(version.nodes) if sources is None else tuple(sources)
    unknown_sources = [source for source in selected if source not in version.out_edges]
    if unknown_sources:
        _fail(
            "UNKNOWN_SOURCE",
            "起点不在冻结版本中: " + ", ".join(sorted(unknown_sources)),
            summary="起点集合被拒绝",
            sources=sorted(unknown_sources),
        )

    target_lookup = set(declared)
    # node -> target/unmatched -> (weight, path_count)
    distributions: dict[str, dict[str, tuple[Fraction, int]]] = {}
    transitions: dict[tuple[str, str], list[FrontierStep]] = {}

    for node in version.topo_order:
        if node in target_lookup:
            values = {node: (ONE, 1)}
            distributions[node] = values
            for bucket in declared + (UNASSIGNED_KEY,):
                if bucket != node:
                    transitions[(node, bucket)] = []
            transitions[(node, node)] = []
            continue

        outgoing = version.out_edges[node]
        accumulator: dict[str, list[Any]] = {}
        if not outgoing:
            accumulator[UNASSIGNED_KEY] = [ONE, 1, []]
        for parent, step_weight in outgoing:
            for bucket, (parent_weight, parent_count) in distributions[parent].items():
                record = accumulator.setdefault(bucket, [ZERO, 0, []])
                contribution = step_weight * parent_weight
                record[0] += contribution
                record[1] += parent_count
                record[2].append(
                    FrontierStep(
                        parent=parent,
                        step_weight=step_weight,
                        branch_cell=(parent, bucket),
                        branch_weight=parent_weight,
                        branch_path_count=parent_count,
                        contribution=contribution,
                    )
                )
        distributions[node] = {bucket: (record[0], record[1]) for bucket, record in accumulator.items()}
        for bucket, record in accumulator.items():
            transitions[(node, bucket)] = record[2]

    cells: dict[tuple[str, str], FrontierCell] = {}
    for source in selected:
        source_distribution = distributions[source]
        reached_total = sum((weight for weight, _count in source_distribution.values()), ZERO)
        if reached_total != ONE:
            _fail(
                "FRONTIER_NOT_CONSERVED",
                f"起点 {source} 的 first-hit 总权重为 {fraction_text(reached_total)}，必须为 1",
                summary="frontier 桥不守恒",
                source=source,
                total=fraction_text(reached_total),
            )
        for target in declared:
            weight, count = source_distribution.get(target, (ZERO, 0))
            if target == source:
                basis = "self"
            elif weight > 0:
                basis = "first_hit"
            else:
                basis = "not_reached"
            cells[(source, target)] = FrontierCell(
                source=source,
                target=target,
                weight=weight,
                path_count=count,
                basis=basis,
                steps=tuple(transitions.get((source, target), ())),
            )
        unmatched_weight, unmatched_count = source_distribution.get(UNASSIGNED_KEY, (ZERO, 0))
        cells[(source, UNASSIGNED_KEY)] = FrontierCell(
            source=source,
            target=UNASSIGNED_KEY,
            weight=unmatched_weight,
            path_count=unmatched_count,
            basis="unmatched" if unmatched_weight > 0 else "not_reached",
            steps=tuple(transitions.get((source, UNASSIGNED_KEY), ())),
        )

    return FrontierTable(
        version_digest=version.digest,
        target_set=declared,
        cells=cells,
        sources=tuple(selected),
        roots=version.roots,
    )


# ---------------------------------------------------------------------------
# 目标层次交叠检测与闭包归集资格
# ---------------------------------------------------------------------------


def ancestor_set(version: Version, node: str) -> frozenset[str]:
    """结构可达祖先（含自身），与边权重无关，因此零边也算路径存在。"""

    stack = [node]
    visited = {node}
    while stack:
        current = stack.pop()
        for parent, _weight in version.out_edges[current]:
            if parent not in visited:
                visited.add(parent)
                stack.append(parent)
    return frozenset(visited)


def overlap_pairs(version: Version, target_set: Sequence[str], source: str | None = None) -> tuple[tuple[str, str], ...]:
    """返回 ``(先命中, 后命中)`` 目标对：存在一条从起点向上的路径依次经过两者。

    祖先与后代交叠时，普通闭包会把同一条路径上的两层各计一次；
    first-hit 语义只会命中第一个。``source`` 为 None 时只做结构检测。
    """

    ordered = tuple(sorted(target_set))
    ancestors = {target: ancestor_set(version, target) for target in ordered}
    applicable = ancestor_set(version, source) if source is not None else None
    pairs: list[tuple[str, str]] = []
    for first in ordered:
        for later in ordered:
            if first == later:
                continue
            if later not in ancestors[first]:
                continue
            if applicable is not None and first not in applicable:
                continue
            pairs.append((first, later))
    return tuple(pairs)


@dataclass(frozen=True)
class ClosureQuery:
    source: str
    targets: tuple[str, ...]
    weights: Mapping[str, Fraction]
    path_counts: Mapping[str, int]
    total: Fraction
    root_caliber: Fraction
    overlaps: tuple[tuple[str, str], ...]
    direct_allowed: bool
    reasons: tuple[ValidationIssue, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "targets": list(self.targets),
            "weights": {target: fraction_text(value) for target, value in self.weights.items()},
            "path_counts": dict(self.path_counts),
            "total": fraction_text(self.total),
            "root_caliber": fraction_text(self.root_caliber),
            "overlaps": [list(pair) for pair in self.overlaps],
            "direct_allowed": self.direct_allowed,
            "reasons": [reason.as_dict() for reason in self.reasons],
        }


def check_closure_query(
    version: Version,
    closure: ClosureTable,
    source: str,
    targets: Sequence[str],
) -> ClosureQuery:
    """判断普通闭包能否直接用于分层归集。

    直接归集必须同时满足：

    * 目标集合内没有祖先/后代交叠（否则同一条路径可能命中多个层次）；
    * 全部目标的闭包权重和恰为 1（否则拒绝，绝不做事后归一化）。

    roots 口径单独返回，始终按 ``sum(weight(source, root))`` 计算。
    """

    declared = tuple(targets)
    if source not in version.out_edges:
        _fail(
            "UNKNOWN_SOURCE",
            f"起点 {source} 不在冻结版本中",
            summary="起点被拒绝",
            source=source,
        )
    if not declared:
        _fail(
            "EMPTY_TARGET_SET",
            "跨层查询必须显式声明至少一个目标节点",
            summary="目标集合被拒绝",
            source=source,
        )
    if len(set(declared)) != len(declared):
        _fail(
            "DUPLICATE_TARGET",
            "目标集合包含重复目标",
            summary="目标集合被拒绝",
            targets=list(declared),
        )
    unknown = [target for target in declared if target not in version.out_edges]
    if unknown:
        _fail(
            "TARGET_NOT_IN_VERSION",
            "目标不在冻结版本中: " + ", ".join(sorted(unknown)),
            summary="目标集合被拒绝",
            targets=sorted(unknown),
        )

    weights = {target: closure.weight(source, target) for target in declared}
    path_counts = {target: closure.path_count(source, target) for target in declared}
    total = sum(weights.values(), ZERO)
    caliber = closure.root_caliber(source)
    overlaps = overlap_pairs(version, declared, source=source)
    reasons: list[ValidationIssue] = []
    if overlaps:
        reasons.append(
            ValidationIssue(
                "CLOSURE_TARGET_LEVELS_MIXED",
                "目标集合含祖先与后代交叠，普通闭包会让同一路径多次命中: "
                + ", ".join(f"{first}>{later}" for first, later in overlaps),
                {"source": source, "overlaps": [list(pair) for pair in overlaps]},
            )
        )
    if total != ONE:
        reasons.append(
            ValidationIssue(
                "CLOSURE_SUM_NOT_ONE",
                f"起点 {source} 的闭包权重和为 {fraction_text(total)}，必须为 1；禁止事后归一化",
                {"source": source, "total": fraction_text(total), "targets": list(declared)},
            )
        )
    return ClosureQuery(
        source=source,
        targets=declared,
        weights=weights,
        path_counts=path_counts,
        total=total,
        root_caliber=caliber,
        overlaps=overlaps,
        direct_allowed=not reasons,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# 整数分归集：最大余数、目标键并列、负费用绝对值分配
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Charge:
    fact_id: str
    node: str
    amount_cents: int

    def as_dict(self) -> dict[str, Any]:
        return {"fact_id": self.fact_id, "node": self.node, "amount_cents": self.amount_cents}


def load_charges(path: str | Path) -> tuple[list[Charge], str]:
    rows, digest = _read_csv(Path(path))
    charges: list[Charge] = []
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        fact_id = (row.get("fact_id") or "").strip()
        node = (row.get("node") or "").strip()
        if not fact_id:
            issues.append(ValidationIssue("INVALID_FACT_ID", f"第 {row_number} 行 fact_id 为空", {"row": row_number}))
            continue
        if fact_id in seen_ids:
            issues.append(
                ValidationIssue("DUPLICATE_FACT_ID", f"fact_id 重复: {fact_id}", {"row": row_number, "fact_id": fact_id})
            )
            continue
        seen_ids.add(fact_id)
        if not node:
            issues.append(ValidationIssue("INVALID_FACT_NODE", f"第 {row_number} 行 node 为空", {"row": row_number}))
            continue
        try:
            amount = parse_cents(row.get("amount_cents"))
        except ValueError as error:
            issues.append(
                ValidationIssue(
                    "INVALID_AMOUNT",
                    str(error),
                    {"row": row_number, "fact_id": fact_id, "raw": row.get("amount_cents")},
                )
            )
            continue
        charges.append(Charge(fact_id=fact_id, node=node, amount_cents=amount))
    if issues:
        raise BridgeInputError(issues, summary="费用事实被拒绝")
    return charges, digest


@dataclass(frozen=True)
class RoundingLine:
    key: str
    weight: Fraction
    exact_abs_cents: Fraction
    floor_cents: int
    remainder: Fraction
    bonus: int
    allocated_cents: int
    sign: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "weight": fraction_text(self.weight),
            "exact_abs_cents": fraction_text(self.exact_abs_cents),
            "floor_cents": self.floor_cents,
            "remainder": fraction_text(self.remainder),
            "bonus": self.bonus,
            "allocated_cents": self.allocated_cents,
            "sign": self.sign,
        }


def largest_remainder_allocate(
    amount_cents: int, weights: Mapping[str, Fraction], *, tie_break: str = SUPPORTED_TIE_BREAK
) -> tuple[dict[str, int], list[RoundingLine], int]:
    """把一笔整数分按权重拆成整数分；最大余数法 + 目标键升序并列规则。

    * 负费用按绝对值分配，再逐桶恢复符号（``allocate_absolute_then_restore_sign``）；
    * 每个桶先取精确绝对值的下取整，剩余分按余数降序、键升序派发；
    * 返回值保证 ``sum(allocated) == amount_cents``，且各桶符号一致或为零。
    """

    if tie_break != SUPPORTED_TIE_BREAK:
        _fail(
            "UNSUPPORTED_TIE_BREAK",
            f"不支持并列规则: {tie_break!r}",
            summary="舍入配置被拒绝",
            value=tie_break,
        )
    keys = list(weights)
    duplicate = len(keys) != len(set(keys))
    if duplicate:
        _fail("DUPLICATE_ALLOCATION_KEY", "归集桶键重复", summary="舍入被拒绝", keys=keys)
    if not keys:
        _fail("EMPTY_ALLOCATION", "没有可归集的桶", summary="舍入被拒绝")

    weight_sum = sum(weights.values(), ZERO)
    if weight_sum != ONE:
        _fail(
            "ALLOCATION_WEIGHTS_NOT_ONE",
            f"归集权重和为 {fraction_text(weight_sum)}，必须为 1；禁止隐式归一化",
            summary="舍入被拒绝",
            total=fraction_text(weight_sum),
            keys=keys,
        )
        weight_sum = sum(weights.values(), ZERO)
    if weight_sum != ONE:
        _fail(
            "ALLOCATION_WEIGHTS_NOT_ONE",
            f"归集权重和为 {fraction_text(weight_sum)}，必须为 1；禁止隐式归一化",
            summary="舍入被拒绝",
            total=fraction_text(weight_sum),
            keys=keys,
        )
    if any(weights[key] < ZERO for key in keys):
        _fail("NEGATIVE_WEIGHT", "归集权重必须非负", summary="舍入被拒绝", keys=keys)

    magnitude = abs(amount_cents)
    sign = 1 if amount_cents >= 0 else -1
    exact = {key: magnitude * weights[key] for key in keys}
    floors = {key: math.floor(exact[key]) for key in keys}
    remainders = {key: exact[key] - floors[key] for key in keys}
    residual = magnitude - sum(floors.values())
    # 并列时目标键升序；UNASSIGNED 是桶名而不是目标键，永远排在真实目标之后。
    ranking = sorted(
        keys,
        key=lambda key: (-remainders[key], key == UNASSIGNED_KEY, key),
    )
    bonuses = {key: 0 for key in keys}
    for key in ranking[:residual]:
        bonuses[key] += 1

    lines: list[RoundingLine] = []
    allocated: dict[str, int] = {}
    for key in keys:
        cents = (floors[key] + bonuses[key]) * sign
        allocated[key] = cents
        lines.append(
            RoundingLine(
                key=key,
                weight=weights[key],
                exact_abs_cents=exact[key],
                floor_cents=floors[key],
                remainder=remainders[key],
                bonus=bonuses[key],
                allocated_cents=cents,
                sign=sign,
            )
        )
    if sum(allocated.values()) != amount_cents:
        _fail("ROUNDING_NOT_CONSERVED", "舍入后整数分总和不等于原金额", summary="舍入不守恒")
    return allocated, lines, sign


@dataclass(frozen=True)
class FactAllocation:
    fact_id: str
    node: str
    amount_cents: int
    mode: str
    target_set: tuple[str, ...]
    buckets: Mapping[str, int]
    sign: int
    lines: tuple[RoundingLine, ...]
    bridge_kind: str
    basis_digests: Mapping[str, str]
    frozen_digest: str
    frozen_at: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "node": self.node,
            "amount_cents": self.amount_cents,
            "mode": self.mode,
            "target_set": list(self.target_set),
            "buckets": dict(self.buckets),
            "sign": self.sign,
            "rounding": [line.as_dict() for line in self.lines],
            "bridge_kind": self.bridge_kind,
            "basis_digests": dict(self.basis_digests),
            "frozen_digest": self.frozen_digest,
            "frozen_at": self.frozen_at,
        }


def _frozen_payload(
    charge: Charge,
    mode: str,
    target_set: Sequence[str],
    bridge_kind: str,
    buckets: Mapping[str, int],
    lines: Sequence[RoundingLine],
    basis_digests: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "fact_id": charge.fact_id,
        "node": charge.node,
        "amount_cents": charge.amount_cents,
        "mode": mode,
        "target_set": list(target_set),
        "bridge_kind": bridge_kind,
        "buckets": dict(buckets),
        "rounding": [
            {
                "key": line.key,
                "weight": fraction_text(line.weight),
                "exact_abs_cents": fraction_text(line.exact_abs_cents),
                "floor_cents": line.floor_cents,
                "remainder": fraction_text(line.remainder),
                "bonus": line.bonus,
                "allocated_cents": line.allocated_cents,
                "sign": line.sign,
            }
            for line in lines
        ],
        "basis_digests": dict(basis_digests),
    }


def allocate_charge(
    version: Version,
    charge: Charge,
    *,
    mode: str,
    target_set: Sequence[str] = (),
    closure: ClosureTable | None = None,
    frontier: FrontierTable | None = None,
    tie_break: str = SUPPORTED_TIE_BREAK,
    frozen_at: int = 0,
) -> FactAllocation:
    """单笔事实归集并立即冻结；冻结后才允许进入汇总。

    ``closure`` 模式只在目标层次不交叠且闭包权重和为 1 时允许，
    否则必须改用 ``frontier``。两种模式都要求桶权重和为 1。
    """

    if charge.node not in version.out_edges:
        _fail(
            "CHARGE_NODE_NOT_IN_VERSION",
            f"费用节点 {charge.node} 不在冻结版本中",
            summary="费用事实被拒绝",
            fact_id=charge.fact_id,
            node=charge.node,
        )
    if mode == "closure":
        if closure is None:
            _fail("MISSING_CLOSURE", "closure 模式需要闭包桥表", summary="归集模式被拒绝")
        query = check_closure_query(version, closure, charge.node, target_set)
        if not query.direct_allowed:
            _fail(
                "CLOSURE_QUERY_REJECTED",
                f"普通闭包不能直接归集事实 {charge.fact_id}；请改用 frontier",
                summary="闭包归集被拒绝",
                fact_id=charge.fact_id,
                source=charge.node,
                reasons=[reason.as_dict() for reason in query.reasons],
            )
        weights = dict(query.weights)
        bridge_kind = "closure"
        basis_digests = {
            "query": _digest(query.as_dict()),
            "closure_cells": _digest(
                {
                    target: closure.cell(charge.node, target).evidence_digest for target in target_set
                }
            ),
        }
    elif mode == "frontier":
        if frontier is None:
            _fail("MISSING_FRONTIER", "frontier 模式需要 frontier 桥表", summary="归集模式被拒绝")
        if tuple(target_set) != frontier.target_set:
            _fail(
                "FRONTIER_TARGET_SET_MISMATCH",
                "事实目标集合与 frontier 桥表的目标集合不一致",
                summary="frontier 归集被拒绝",
                fact_id=charge.fact_id,
                expected=list(frontier.target_set),
                actual=list(target_set),
            )
        weights = {target: frontier.cell(charge.node, target).weight for target in queue_targets(frontier)}
        bridge_kind = "frontier"
        basis_digests = {
            "frontier_cells": _digest(
                {
                    target: frontier.cell(charge.node, target).evidence_digest
                    for target in queue_targets(frontier)
                }
            ),
        }
    else:
        _fail("UNSUPPORTED_MODE", f"未知归集模式: {mode!r}", summary="归集模式被拒绝", mode=mode)

    buckets, lines, sign = largest_remainder_allocate(
        charge.amount_cents, weights, tie_break=tie_break
    )
    payload = _frozen_payload(charge, mode, target_set, bridge_kind, buckets, lines, basis_digests)
    return FactAllocation(
        fact_id=charge.fact_id,
        node=charge.node,
        amount_cents=charge.amount_cents,
        mode=mode,
        target_set=tuple(target_set),
        buckets=buckets,
        sign=sign,
        lines=tuple(lines),
        bridge_kind=bridge_kind,
        basis_digests=basis_digests,
        frozen_digest=_digest(payload),
        frozen_at=frozen_at,
    )


def queue_targets(frontier: FrontierTable) -> tuple[str, ...]:
    return frontier.target_set + (UNASSIGNED_KEY,)


@dataclass(frozen=True)
class Aggregate:
    target: str
    total_cents: int
    fact_count: int
    fact_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "total_cents": self.total_cents,
            "fact_count": self.fact_count,
            "fact_ids": list(self.fact_ids),
        }


def freeze_and_aggregate(facts: Sequence[FactAllocation], *, aggregate_version: str = "") -> dict[str, Any]:
    """仅接受已冻结事实；按桶求和并再次校验整数分守恒。"""

    seen: set[str] = set()
    for fact in facts:
        if not fact.frozen_digest:
            _fail("UNFROZEN_FACT", f"事实 {fact.fact_id} 未冻结", summary="汇总被拒绝")
        if fact.fact_id in seen:
            _fail("DUPLICATE_FROZEN_FACT", f"事实 {fact.fact_id} 重复汇总", summary="汇总被拒绝")
        seen.add(fact.fact_id)
        recomputed = _digest(
            _frozen_payload(
                Charge(fact.fact_id, fact.node, fact.amount_cents),
                fact.mode,
                fact.target_set,
                fact.bridge_kind,
                fact.buckets,
                fact.lines,
                fact.basis_digests,
            )
        )
        if recomputed != fact.frozen_digest:
            _fail(
                "FROZEN_DIGEST_MISMATCH",
                f"事实 {fact.fact_id} 的冻结摘要与内容不符，疑似冻结后被篡改",
                summary="汇总被拒绝",
                fact_id=fact.fact_id,
                expected=fact.frozen_digest,
                recomputed=recomputed,
            )

    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    identifiers: dict[str, list[str]] = {}

    # 按冻结顺序重组，再逐桶整数相加；不按路径条数复制事实。
    for fact in sorted(facts, key=lambda item: (item.frozen_at, item.fact_id)):
        expected = sum(fact.buckets.values())
        if expected != fact.amount_cents:
            _fail(
                "FROZEN_FACT_NOT_CONSERVED",
                f"事实 {fact.fact_id} 的桶和 {expected} 不等于原金额 {fact.amount_cents}",
                summary="汇总被拒绝",
                fact_id=fact.fact_id,
            )
        for target, cents in fact.buckets.items():
            totals[target] = totals.get(target, 0) + cents
            counts[target] = counts.get(target, 0) + 1
            identifiers.setdefault(target, []).append(fact.fact_id)

    source_total = sum(fact.amount_cents for fact in facts)
    aggregated_total = sum(totals.values())
    if source_total != aggregated_total:
        _fail(
            "AGGREGATE_NOT_CONSERVED",
            f"汇总桶和 {aggregated_total} 不等于事实金额和 {source_total}",
            summary="汇总不守恒",
            source_total=source_total,
            aggregated_total=aggregated_total,
        )

    return {
        "aggregate_version": aggregate_version,
        "fact_count": len(facts),
        "source_total_cents": source_total,
        "aggregated_total_cents": aggregated_total,
        "buckets": [
            Aggregate(
                target=target,
                total_cents=totals[target],
                fact_count=counts[target],
                fact_ids=tuple(identifiers[target]),
            ).as_dict()
            for target in sorted(totals)
        ],
        "frozen_facts": [fact.as_dict() for fact in sorted(facts, key=lambda item: (item.frozen_at, item.fact_id))],
    }


# ---------------------------------------------------------------------------
# 组合产物：闭包桥、frontier 桥、叶事实联接与报告
# ---------------------------------------------------------------------------


def build_closure_artifact(version: Version, closure: ClosureTable) -> dict[str, Any]:
    rows = [
        closure.cells[(source, target)].as_dict()
        for source in sorted(closure.sources)
        for target in sorted({cell.target for key, cell in closure.cells.items() if key[0] == source})
    ]
    return {
        "bridge_kind": "closure",
        "version": version.version_id,
        "version_digest": version.digest,
        "semantics": "sum_over_all_directed_path_weight_products",
        "roots": list(version.roots),
        "root_caliber_rule": "sum(weight(source, root) for root in roots)",
        "root_caliber": {
            source: fraction_text(closure.root_caliber(source)) for source in sorted(closure.sources)
        },
        "cells": rows,
    }


def build_frontier_artifact(version: Version, frontier: FrontierTable) -> dict[str, Any]:
    rows = [
        frontier.cells[(source, target)].as_dict()
        for source in sorted(frontier.sources)
        for target in queue_targets(frontier)
    ]
    distributions = {}
    for source in sorted(frontier.sources):
        distribution = frontier.distribution(source)
        distributions[source] = {
            "buckets": {target: fraction_text(weight) for target, weight in distribution},
            "total": fraction_text(sum((weight for _target, weight in distribution), ZERO)),
        }
    return {
        "bridge_kind": "frontier",
        "version": version.version_id,
        "version_digest": version.digest,
        "semantics": "first_hit_along_each_path_to_root",
        "target_set": list(frontier.target_set),
        "unmatched_bucket": UNASSIGNED_KEY,
        "cells": rows,
        "distributions": distributions,
    }


def build_query_records(
    version: Version, closure: ClosureTable, config: HierarchyConfig
) -> list[dict[str, Any]]:
    """逐 (起点, 目标集合) 声明口径；闭包与 frontier 结果并列但权重绝不混加。"""

    records: list[dict[str, Any]] = []
    for source in sorted(closure.sources):
        for target_set in config.target_sets:
            query = check_closure_query(version, closure, source, target_set)
            records.append(
                {
                    "source": source,
                    "target_set": list(target_set),
                    "closure": query.as_dict(),
                    "root_caliber_declared_separately": fraction_text(query.root_caliber),
                    "frontier_reference": {
                        target: fraction_text(weight)
                        for target, weight in build_frontier_bridges(version, target_set, [source])
                        .distribution(source)
                    },
                }
            )
    return records


def _choose_mode(version: Version, closure: ClosureTable, source: str, targets: Sequence[str]) -> str:
    query = check_closure_query(version, closure, source, targets)
    return "closure" if query.direct_allowed else "frontier"


def build_fact_ledger(
    version: Version,
    charges: Sequence[Charge],
    closure: ClosureTable,
    config: HierarchyConfig,
    *,
    charges_digest: str,
) -> dict[str, Any]:
    """叶事实联接示例：每笔事实单独冻结，再按目标集合分别汇总。"""

    frontier_tables = {
        target_set: build_frontier_bridges(version, target_set)
        for target_set in config.target_sets
    }
    allocations: list[FactAllocation] = []
    groups: list[dict[str, Any]] = []
    for set_index, target_set in enumerate(config.target_sets):
        facts_for_set: list[FactAllocation] = []
        modes = set()
        for charge_index, charge in enumerate(charges):
            mode = _choose_mode(version, closure, charge.node, target_set)
            modes.add(mode)
            frozen_at = set_index * len(charges) + charge_index
            fact = allocate_charge(
                version,
                charge,
                mode=mode,
                target_set=target_set,
                closure=closure,
                frontier=frontier_tables[target_set],
                tie_break=config.tie_break,
                frozen_at=frozen_at,
            )
            facts_for_set.append(fact)
            allocations.append(fact)
        aggregate = freeze_and_aggregate(
            facts_for_set,
            aggregate_version=f"{version.version_id}:{set_index}:{json.dumps(list(target_set), ensure_ascii=False)}",
        )
        groups.append(
            {
                "target_set": list(target_set),
                "modes": sorted(modes),
                "buckets": aggregate["buckets"],
                "source_total_cents": aggregate["source_total_cents"],
                "aggregated_total_cents": aggregate["aggregated_total_cents"],
                "fact_ids": [fact.fact_id for fact in facts_for_set],
            }
        )

    return {
        "version": version.version_id,
        "version_digest": version.digest,
        "charges_digest": charges_digest,
        "negative_rounding": config.negative_rounding,
        "tie_break": config.tie_break,
        "allocations": [
            fact.as_dict() for fact in sorted(allocations, key=lambda item: (item.frozen_at, item.fact_id))
        ],
        "aggregates_by_target_set": groups,
    }


def build_report(
    version: Version,
    config: HierarchyConfig,
    closure: ClosureTable,
    charges: Sequence[Charge],
    *,
    charges_digest: str,
) -> dict[str, Any]:
    return {
        "version": version.as_dict(),
        "config": config.as_dict(),
        "closure": build_closure_artifact(version, closure),
        "frontier": [
            build_frontier_artifact(version, build_frontier_bridges(version, target_set))
            for target_set in config.target_sets
        ],
        "queries": build_query_records(version, closure, config),
        "facts": build_fact_ledger(
            version, charges, closure, config, charges_digest=charges_digest
        ),
        "invariants": {
            "path_count_never_multiplies_facts": True,
            "closure_never_normalized_post_hoc": True,
            "root_caliber_separate_from_target_weights": True,
            "negative_amounts_absolute_then_sign": config.negative_rounding,
            "rounding_tie_break": config.tie_break,
        },
    }


def _write_json(path: Path, value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_artifacts(
    nodes_path: str | Path,
    edges_path: str | Path,
    charges_path: str | Path,
    hierarchy_path: str | Path,
    output_dir: str | Path,
    *,
    expected_version_digest: str | None = None,
) -> dict[str, Any]:
    version = load_version(nodes_path, edges_path, expected_digest=expected_version_digest)
    config = load_hierarchy_config(hierarchy_path)
    charges, charges_digest = load_charges(charges_path)
    unknown_nodes = [charge.node for charge in charges if charge.node not in version.out_edges]
    if unknown_nodes:
        _fail(
            "CHARGE_NODE_NOT_IN_VERSION",
            "费用节点不在冻结版本中: " + ", ".join(sorted(set(unknown_nodes))),
            summary="费用事实被拒绝",
            nodes=sorted(set(unknown_nodes)),
        )
    for target_set in config.target_sets:
        unknown = [target for target in target_set if target not in version.out_edges]
        if unknown:
            _fail(
                "TARGET_NOT_IN_VERSION",
                "目标不在冻结版本中: " + ", ".join(sorted(unknown)),
                summary="层级配置被拒绝",
                targets=sorted(unknown),
            )

    closure = build_closure_bridges(version)
    report = build_report(version, config, closure, charges, charges_digest=charges_digest)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_entries: dict[str, str] = {}
    artifacts = {
        "version.json": version.as_dict(),
        "closure_bridges.json": build_closure_artifact(version, closure),
        "frontier_bridges.json": [
            build_frontier_artifact(version, build_frontier_bridges(version, target_set))
            for target_set in config.target_sets
        ],
        "facts_ledger.json": report["facts"],
        "report.json": report,
    }
    for name, value in artifacts.items():
        manifest_entries[name] = _write_json(output / name, value)
    manifest = {
        "inputs": {"nodes": version.source_digests["nodes"], "edges": version.source_digests["edges"],
                   "charges": charges_digest,
                   "hierarchy": _file_digest(Path(hierarchy_path))},
        "version_digest": version.digest,
        "outputs": manifest_entries,
    }
    manifest_entries["manifest.json"] = _write_json(output / "manifest.json", manifest)

    return {
        "version": version.version_id,
        "version_digest": version.digest,
        "roots": list(version.roots),
        "target_sets": [list(target_set) for target_set in config.target_sets],
        "fact_count": len(charges),
        "outputs": {name: str(output / name) for name in sorted(artifacts) + ["manifest.json"]},
        "output_digests": manifest_entries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="构建层级归集桥表：闭包桥、frontier 桥、叶事实联接与汇总"
    )
    parser.add_argument("--nodes", default="fixtures/nodes.json")
    parser.add_argument("--edges", default="fixtures/edges.csv")
    parser.add_argument("--charges", default="fixtures/charges.csv")
    parser.add_argument("--hierarchy", default="fixtures/hierarchy.json")
    parser.add_argument("--output", default=".workspace-output/bridge")
    parser.add_argument("--expect-version-digest", default=None)
    args = parser.parse_args(argv)

    try:
        result = build_artifacts(
            args.nodes,
            args.edges,
            args.charges,
            args.hierarchy,
            args.output,
            expected_version_digest=args.expect_version_digest,
        )
    except BridgeInputError as error:
        print(json.dumps(error.as_dict(), ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



