"""Producer-prefix distribution matching keeps source split ownership intact."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from decision_factory.training.actor_corpus import build_actor_corpus
from decision_factory.training.multi_tool_corpus import build, reference_verdict
from decision_factory.training.producer_corpus import build_producer_corpora, build_split_prefixes


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class ProducerCorpusTests(unittest.TestCase):
    def test_actual_prefixes_and_layout_variants_preserve_source_group_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build(root / "original")
            build_actor_corpus(root / "original", root / "actor-original")
            result = build_producer_corpora(root / "actor-original", root / "actor-new", root / "prefix-train")
            self.assertEqual(result["actor"]["rows"], 312)
            self.assertEqual(result["guard"]["rows"], 576)
            # All TRAIN trajectory effects remain unchanged despite menu and wording changes.
            old = {t["case_id"]: t for t in read(root / "actor-original/teacher-traces.jsonl")}
            new_rows = read(root / "actor-new/rows.jsonl")
            metadata = {r["trajectory_id"]: r for r in new_rows}
            for trace in read(root / "actor-new/teacher-traces.jsonl"):
                source = old[metadata[trace["case_id"]]["source_train_trajectory_id"]]
                self.assertEqual(trace["committed_effects"], source["committed_effects"])
                self.assertEqual(trace["terminal"], source["terminal"])
            episodes = read(root / "actor-new/train-episodes.jsonl")
            self.assertEqual({len(e["action_menu"]) for e in episodes}, {3, 5})
            self.assertTrue(any("one" in e["request"]["user_request"]["text"] for e in episodes))
            split_manifest = build_split_prefixes(root / "original", root / "prefix-train", root / "prefix-splits")
            self.assertEqual({s: d["rows"] for s, d in split_manifest["splits"].items()}, {"train": 576, "calibration": 144, "dev": 144})
            owners = {}
            for split in ("train", "calibration", "dev"):
                original_labels = read(root / "original" / f"{split}.labels.jsonl")
                original_groups = {r["group_id"] for r in original_labels}
                original_ids = {r["scenario_id"] for r in original_labels}
                labels = read(root / "prefix-splits" / f"{split}.labels.jsonl")
                states = {r["scenario_id"]: r["model_state"] for r in read(root / "prefix-splits" / f"{split}.states.jsonl")}
                for label in labels:
                    self.assertIn(label["group_id"], original_groups)
                    self.assertIn(label["source_train_scenario_id"], original_ids)
                    self.assertEqual(owners.setdefault(label["group_id"], split), split)
                    self.assertEqual(reference_verdict(states[label["scenario_id"]]), label["verdict"])
                self.assertEqual(len({l["group_id"] for l in labels}), len(original_groups))


if __name__ == "__main__":
    unittest.main()
