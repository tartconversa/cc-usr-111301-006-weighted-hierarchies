import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from src.bridge_builder import (
    BridgeInputError,
    Charge,
    UNASSIGNED_KEY,
    allocate_charge,
    build_closure_bridges,
    build_frontier_bridges,
    check_closure_query,
    freeze_and_aggregate,
    largest_remainder_allocate,
    load_version,
)


def write_version(directory: Path, nodes, edges, version="v") -> None:
    (directory / "nodes.json").write_text(
        json.dumps({"version": version, "nodes": list(nodes)}, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["child,parent,weight"]
    lines.extend(edges)
    (directory / "edges.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_fixture_edges(directory: Path):
    write_version(
        directory,
        ["a", "b", "c", "r"],
        ["a,b,1/2", "a,c,1/2", "b,r,1", "c,r,1"],
    )
    return load_version(directory / "nodes.json", directory / "edges.csv")


class VersionValidationTests(unittest.TestCase):
    def reject(self, nodes, edges) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            write_version(Path(directory), nodes, edges)
            with self.assertRaises(BridgeInputError) as caught:
                load_version(Path(directory) / "nodes.json", Path(directory) / "edges.csv")
            return [issue.code for issue in caught.exception.issues]

    def test_cycle_rejected(self):
        codes = self.reject(["a", "b"], ["a,b,1", "b,a,1"])
        self.assertIn("CYCLE", codes)

    def test_dangling_reference_rejected(self):
        codes = self.reject(["a", "b"], ["a,ghost,1"])
        self.assertIn("DANGLING_REFERENCE", codes)

    def test_weight_not_conserved_rejected(self):
        codes = self.reject(["a", "b", "r"], ["a,b,1/2", "a,r,1/4", "b,r,1"])
        self.assertIn("WEIGHT_NOT_CONSERVED", codes)

    def test_negative_and_non_finite_weights_rejected(self):
        self.assertIn("INVALID_WEIGHT", self.reject(["a", "r"], ["a,r,-1"]))
        self.assertIn("INVALID_WEIGHT", self.reject(["a", "r"], ["a,r,1/0"]))
        self.assertIn("INVALID_WEIGHT", self.reject(["a", "r"], ["a,r,0.1x"]))

    def test_valid_version_roots_and_digest_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            version = load_fixture_edges(Path(directory))
            reloaded = load_version(Path(directory) / "nodes.json", Path(directory) / "edges.csv")
        self.assertEqual(version.roots, ("r",))
        self.assertEqual(version.leaves, ("a",))
        self.assertEqual(version.digest, reloaded.digest)


class ClosureBridgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.version = load_fixture_edges(Path(self.directory.name))
        self.closure = build_closure_bridges(self.version)

    def tearDown(self):
        self.directory.cleanup()

    def test_diamond_weight_and_path_evidence(self):
        cell = self.closure.cell("a", "r")
        self.assertEqual(cell.weight, Fraction(1, 1))
        self.assertEqual(cell.path_count, 2)
        # 证据只保存两条首步转移，而不是枚举路径本身
        self.assertEqual([step.parent for step in cell.steps], ["b", "c"])
        self.assertEqual([step.contribution for step in cell.steps], [Fraction(1, 2), Fraction(1, 2)])

    def test_self_bridge_is_one_even_for_non_root(self):
        for node in self.version.nodes:
            self.assertEqual(self.closure.weight(node, node), Fraction(1, 1), node)

    def test_root_caliber_is_one_but_not_target_weights(self):
        self.assertEqual(self.closure.root_caliber("a"), Fraction(1, 1))
        query = check_closure_query(self.version, self.closure, "a", ["b", "r"])
        # 普通闭包：b=1/2、r=1，混加两层会得到 3/2
        self.assertEqual(query.weights, {"b": Fraction(1, 2), "r": Fraction(1, 1)})
        self.assertEqual(query.total, Fraction(3, 2))
        self.assertFalse(query.direct_allowed)
        self.assertEqual(query.overlaps, (("b", "r"),))

    def test_zero_edge_path_count_is_kept_but_contributes_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            write_version(Path(directory), ["a", "b", "r"], ["a,b,1", "b,r,1", "a,r,0"])
            version = load_version(Path(directory) / "nodes.json", Path(directory) / "edges.csv")
            closure = build_closure_bridges(version)
        cell = closure.cell("a", "r")
        self.assertEqual(cell.weight, Fraction(1, 1))
        self.assertEqual(cell.path_count, 2)
        self.assertEqual(cell.steps[1].contribution, Fraction(0, 1))

    def test_cross_level_query_requires_explicit_targets(self):
        with self.assertRaises(BridgeInputError) as caught:
            check_closure_query(self.version, self.closure, "a", [])
        self.assertEqual(caught.exception.issues[0].code, "EMPTY_TARGET_SET")


class FrontierBridgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.version = load_fixture_edges(Path(self.directory.name))

    def tearDown(self):
        self.directory.cleanup()

    def test_first_hit_does_not_mix_levels(self):
        frontier = build_frontier_bridges(self.version, ["b", "r"])
        buckets = dict(frontier.distribution("a"))
        self.assertEqual(buckets["b"], Fraction(1, 2))
        self.assertEqual(buckets["r"], Fraction(1, 2))  # 不是普通闭包的 r=1
        r_cell = frontier.cell("a", "r")
        self.assertEqual([step.parent for step in r_cell.steps], ["c"])
        self.assertEqual(buckets[UNASSIGNED_KEY], Fraction(0, 1))
        self.assertEqual(sum(buckets.values()), Fraction(1, 1))

    def test_unmatched_flows_to_unassigned(self):
        frontier = build_frontier_bridges(self.version, ["b"])
        buckets = dict(frontier.distribution("a"))
        self.assertEqual(buckets["b"], Fraction(1, 2))
        self.assertEqual(buckets[UNASSIGNED_KEY], Fraction(1, 2))
        self.assertEqual(sum(buckets.values()), Fraction(1, 1))

    def test_target_at_root_catches_everything(self):
        frontier = build_frontier_bridges(self.version, ["r"])
        self.assertEqual(dict(frontier.distribution("a"))["r"], Fraction(1, 1))


class RoundingTests(unittest.TestCase):
    def test_positive_largest_remainder_with_key_tie_break(self):
        allocated, lines, sign = largest_remainder_allocate(
            101, {"b": Fraction(1, 2), "c": Fraction(1, 2)}
        )
        self.assertEqual(allocated, {"b": 51, "c": 50})
        self.assertEqual(sign, 1)
        self.assertEqual([line.bonus for line in lines], [1, 0])

    def test_negative_allocates_absolute_then_restores_sign(self):
        allocated, _lines, sign = largest_remainder_allocate(
            -101, {"b": Fraction(1, 2), "c": Fraction(1, 2)}
        )
        self.assertEqual(allocated, {"b": -51, "c": -50})
        self.assertEqual(sign, -1)
        self.assertEqual(sum(allocated.values()), -101)

    def test_unassigned_loses_ties_to_real_targets(self):
        allocated, _lines, _sign = largest_remainder_allocate(
            1, {"z": Fraction(1, 2), UNASSIGNED_KEY: Fraction(1, 2)}
        )
        self.assertEqual(allocated["z"], 1)
        self.assertEqual(allocated[UNASSIGNED_KEY], 0)

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(BridgeInputError) as caught:
            largest_remainder_allocate(10, {"b": Fraction(1, 3), "c": Fraction(1, 3)})
        self.assertEqual(caught.exception.issues[0].code, "ALLOCATION_WEIGHTS_NOT_ONE")


class FactFreezingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.version = load_fixture_edges(Path(self.directory.name))
        self.closure = build_closure_bridges(self.version)
        self.frontier = build_frontier_bridges(self.version, ["b", "r"])

    def tearDown(self):
        self.directory.cleanup()

    def test_facts_not_multiplied_by_path_count(self):
        # a->r 有 2 条路径，但 101 分只归集一次，不会变成 202 分
        fact = allocate_charge(
            self.version,
            Charge("f1", "a", 101),
            mode="frontier",
            target_set=["b", "r"],
            frontier=self.frontier,
        )
        self.assertEqual(fact.buckets, {"b": 51, "r": 50, UNASSIGNED_KEY: 0})
        self.assertEqual(sum(fact.buckets.values()), 101)

    def test_frozen_facts_aggregate_and_net_to_zero(self):
        facts = [
            allocate_charge(
                self.version, Charge("f1", "a", 101), mode="frontier", target_set=["b", "r"],
                frontier=self.frontier, frozen_at=0,
            ),
            allocate_charge(
                self.version, Charge("f2", "a", -101), mode="frontier", target_set=["b", "r"],
                frontier=self.frontier, frozen_at=1,
            ),
        ]
        aggregate = freeze_and_aggregate(facts)
        self.assertEqual(aggregate["source_total_cents"], 0)
        self.assertEqual(aggregate["aggregated_total_cents"], 0)
        totals = {bucket["target"]: bucket["total_cents"] for bucket in aggregate["buckets"]}
        self.assertEqual(totals, {"UNASSIGNED": 0, "b": 0, "r": 0})

    def test_tampered_fact_is_rejected_before_aggregation(self):
        fact = allocate_charge(
            self.version, Charge("f1", "a", 101), mode="frontier",
            target_set=["b", "r"], frontier=self.frontier,
        )
        tampered = type(fact)(**{**fact.__dict__, "buckets": {"b": 101, "r": 0, UNASSIGNED_KEY: 0}})
        with self.assertRaises(BridgeInputError) as caught:
            freeze_and_aggregate([tampered])
        self.assertIn(
            {issue.code for issue in caught.exception.issues},
            [{"FROZEN_DIGEST_MISMATCH"}, {"FROZEN_FACT_NOT_CONSERVED"}],
        )

    def test_closure_mode_rejects_mixed_levels(self):
        with self.assertRaises(BridgeInputError) as caught:
            allocate_charge(
                self.version, Charge("f1", "a", 101), mode="closure",
                target_set=["b", "r"], closure=self.closure,
            )
        self.assertEqual(caught.exception.issues[0].code, "CLOSURE_QUERY_REJECTED")


if __name__ == "__main__":
    unittest.main()
