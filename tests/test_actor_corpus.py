"""Verify TRAIN-only supervision uses real production host prefixes."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from decision_factory.training.actor_corpus import build_actor_corpus
from decision_factory.training.multi_tool_corpus import build


class ActorCorpusTests(unittest.TestCase):
    def test_train_only_trajectories_preserve_order_completion_and_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            build(root / "original")
            (root / "train-only").mkdir()
            for name in ("train.states.jsonl", "train.labels.jsonl"):
                shutil.copyfile(root / "original" / name, root / "train-only" / name)
            manifest = build_actor_corpus(root / "train-only", root / "actor")
            self.assertEqual(manifest["rows"], 312)
            self.assertEqual(manifest["target_counts"], {"ACT": 192, "STOP": 96, "REVIEW": 24})
            self.assertEqual(len(manifest["files_read"]), 2)
            rows = [json.loads(line) for line in (root / "actor/rows.jsonl").read_text().splitlines()]
            traces = [json.loads(line) for line in (root / "actor/teacher-traces.jsonl").read_text().splitlines()]
            self.assertEqual(Counter(row["domain"] for row in rows), {"files": 78, "messaging": 78, "records": 78, "commerce": 78})
            for row in rows:
                self.assertNotIn("source_train_scenario_id", row["input"])
                self.assertNotIn("source_train_group_id", row["input"])
                self.assertIn("ORIGINAL USER TASK TO COMPLETE NOW:", row["input"])
                self.assertIn("host-progress", row["input"])
            completed = reviewed = 0
            for trace in traces:
                if trace["terminal"] == "qwen_stop":
                    completed += 1
                    self.assertEqual(len(trace["committed_effects"]), 2)
                    a, b = [e["arguments"]["units"] for e in trace["committed_effects"]]
                    self.assertNotEqual(a, b)
                    task = trace["steps"][0]["guard_request"]["user_request"]["text"]
                    self.assertIn(f"First process {a} unit(s), then process {b} unit(s)", task)
                else:
                    reviewed += 1
                    self.assertEqual(trace["terminal"], "qwen_review")
                    self.assertEqual(trace["committed_effects"], [])
            self.assertEqual((completed, reviewed), (96, 24))


if __name__ == "__main__":
    unittest.main()
