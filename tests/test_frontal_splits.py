"""Patient membership is an independent acceptance condition for training."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.frontal.splits import group, restrict


class SplitBoundaryTests(unittest.TestCase):
    def fixture(self, path):
        path.write_text(json.dumps({"schema": "anycxr_eval_group_exclusion_v1",
            "d3_validation_groups": ["t_1"], "nw_groups": ["train_2"],
            "excluded_patient_groups": ["t_1", "train_2"]}))
        d3 = SimpleNamespace(rows=[{"case_id": f"train_{n}_a_1", "view": view}
                                  for n in [2, 3] for view in ["AP", "PA"]])
        val = SimpleNamespace(rows=[{"case_id": "t_1_a_1"}])
        ct = SimpleNamespace(cases=[{"case_id": s} for s in
            ["train_2_b_2", "train_3_a_1", "train_3_b_2", "valid_1_a_1", "valid_9_a_1"]],
            meta={}, slot_names=["AP", "PA", "LA", "OB"])
        return d3, val, ct

    def test_excludes_patient_across_reconstructions_and_views(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "exclusions.json"; train, val, ct = self.fixture(p)
            membership, _ = restrict(train, val, ct, p)
            self.assertEqual({group(r["case_id"]) for r in train.rows}, {"train_3"})
            self.assertEqual(len(train.rows), 2)
            self.assertEqual(membership["ct_case_ids"], ["train_3_a_1", "train_3_b_2"])

    def test_missing_nw_exclusion_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "exclusions.json"; train, val, ct = self.fixture(p)
            j = json.loads(p.read_text()); j["excluded_patient_groups"] = ["t_1"]
            p.write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError, "NW"):
                restrict(train, val, ct, p)

    def test_missing_validation_exclusion_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "exclusions.json"; train, val, ct = self.fixture(p)
            j = json.loads(p.read_text()); j["excluded_patient_groups"] = ["train_2"]
            p.write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError, "validation"):
                restrict(train, val, ct, p)

    def test_ct_validation_alias_and_unknown_identity(self):
        self.assertEqual(group("valid_47_b_2"), group("t_47_a_1"))
        with self.assertRaises(ValueError):
            group("untraceable_image")


if __name__ == "__main__":
    unittest.main()
