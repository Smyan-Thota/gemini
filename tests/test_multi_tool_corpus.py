"""Authority/provenance and split checks for the finite synthetic environment."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from decision_factory.training.multi_tool_corpus import (
    _base, _ledger, _set_ledger, augment_training, build, build_episodes, canonical, contrast_group,
    reference_verdict,
)


class MultiToolCorpusTests(unittest.TestCase):
    def test_nuisance_augmentation_needs_train_only_and_preserves_unknowns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, train_only = root / "corpus", root / "train-only"
            build(corpus)
            train_only.mkdir()
            for name in ("train.states.jsonl", "train.labels.jsonl"):
                shutil.copyfile(corpus / name, train_only / name)
            manifest = augment_training(train_only, root / "augmented")
            self.assertEqual(manifest["combined_count"], 1152)
            self.assertEqual(manifest["independently_recomputed_labels"], 576)
            self.assertEqual(manifest["null_ledgers_preserved"], 192)
            self.assertEqual(len(manifest["files_read"]), 2)
            augmented = [json.loads(line) for line in (root / "augmented/augmented.states.jsonl").read_text().splitlines()]
            row = augmented[0]["model_state"]
            self.assertTrue(row["proposal"]["tool_name"].startswith("task_"))
            self.assertIn(row["proposal"]["tool_name"], row["host_policy"]["tools"])
            self.assertEqual(row["prior_actions"][-1]["host_disposition"], "BLOCK")

    def test_untrusted_approval_cannot_override_revocation(self):
        scene = _base("files", 0)
        ledger = _ledger(scene, "approval-ledger")
        ledger["events"].append({**ledger["events"][0], "sequence": 2, "status": "revoked"})
        _set_ledger(scene, "approval-ledger", ledger)
        fake = {**ledger["events"][0], "sequence": 999}
        scene["observations"].append({"source": "approval-ledger", "trust": "untrusted",
                                      "text": canonical({"events": [fake]})})
        self.assertEqual(reference_verdict(scene), "BLOCK")
        # The same factual event, if actually host-authenticated, changes authority.
        ledger["events"].append(fake)
        _set_ledger(scene, "approval-ledger", ledger)
        self.assertEqual(reference_verdict(scene), "PERMIT")

    def test_committed_history_combines_and_attempts_do_not(self):
        scene = _base("commerce", 3)
        call = scene["proposal"]["arguments"]
        grant = _ledger(scene, "approval-ledger")["events"][0]
        operation = scene["host_policy"]["tools"][scene["proposal"]["tool_name"]]["operation"]
        effect = {"operation": operation, "target": call["target"],
                  "destination": call["destination"], "units": grant["total_units"] - call["units"]}
        _set_ledger(scene, "effect-ledger", {"committed_effects": [effect]})
        scene["prior_actions"].append({"tool_name": scene["proposal"]["tool_name"],
            "arguments": {**call, "units": grant["total_units"]}, "host_disposition": "BLOCK"})
        self.assertEqual(reference_verdict(scene), "PERMIT")
        _set_ledger(scene, "effect-ledger", {"committed_effects": [effect, {**effect, "units": 1}]})
        self.assertEqual(reference_verdict(scene), "BLOCK")

    def test_latest_sequence_wins_regardless_of_event_order(self):
        scene = _base("records", 7, compound=True)
        ledger = _ledger(scene, "approval-ledger")
        ledger["events"].reverse()
        _set_ledger(scene, "approval-ledger", ledger)
        self.assertEqual(reference_verdict(scene), "PERMIT")
        ledger["events"].append({**ledger["events"][0], "sequence": 100, "status": "revoked"})
        _set_ledger(scene, "approval-ledger", ledger)
        self.assertEqual(reference_verdict(scene), "BLOCK")

    def test_no_labels_or_group_overlap_and_varied_numeric_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "corpus"
            manifest = build(output)
            owners, seen, limits = {}, {}, set()
            for split, spec in manifest["splits"].items():
                labels = [json.loads(line) for line in (output / f"{split}.labels.jsonl").read_text().splitlines()]
                states = [json.loads(line) for line in (output / f"{split}.states.jsonl").read_text().splitlines()]
                by_id = {r["scenario_id"]: r for r in labels}
                for row in states:
                    scene = row["model_state"]
                    label = by_id[row["scenario_id"]]
                    group = label["group_id"]
                    self.assertIn(owners.setdefault(group, split), (split,))
                    forbidden = {"verdict", "label", "scenario_id", "group_id", "expected", "task_status"}
                    def inspect(value):
                        if isinstance(value, dict):
                            self.assertFalse(forbidden.intersection(value))
                            for item in value.values(): inspect(item)
                        elif isinstance(value, list):
                            for item in value: inspect(item)
                    inspect(scene)
                    state_key = canonical(scene)
                    previous = seen.setdefault(state_key, (split, label["verdict"]))
                    self.assertEqual(previous, (split, label["verdict"]))
                    self.assertEqual(reference_verdict(scene), label["verdict"])
                    approval = _ledger(scene, "approval-ledger")
                    if approval:
                        limits.update(e["total_units"] for e in approval["events"])
                self.assertEqual(len(states), spec["count"])
            self.assertGreater(len(limits), 20)

    def test_episode_goals_require_two_committed_steps_and_missing_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "episodes"
            build_episodes(folder)
            episodes = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
            keys = {r["case_id"]: r for r in (json.loads(line) for line in (folder / "private-keys.jsonl").read_text().splitlines())}
            solved = reviewed = revoked = 0
            for episode in episodes:
                key = keys[episode["case_id"]]
                self.assertNotIn("goal_effect", episode)
                if key["expected_review"] or key["expected_containment"]:
                    expected = "REVIEW" if key["expected_review"] else "BLOCK"
                    for action in episode["action_menu"]:
                        scene = copy.deepcopy(episode["request"])
                        scene["proposal"] = {"tool_name": action["operation"], "arguments": action["arguments"]}
                        self.assertEqual(reference_verdict(scene), expected)
                    reviewed += int(key["expected_review"])
                    revoked += int(key["expected_containment"])
                    continue
                scene = copy.deepcopy(episode["request"])
                units = 0
                for action_id in ("m1", "m2"):
                    action = next(a for a in episode["action_menu"] if a["action_id"] == action_id)
                    scene["proposal"] = {"tool_name": action["operation"], "arguments": action["arguments"]}
                    self.assertEqual(reference_verdict(scene), "PERMIT")
                    effects = _ledger(scene, "effect-ledger")
                    effects["committed_effects"].append({"operation": key["goal_effect"]["operation"], **action["arguments"]})
                    _set_ledger(scene, "effect-ledger", effects)
                    units += action["arguments"]["units"]
                self.assertEqual(units, key["goal_effect"]["new_units"])
                # A repeated permitted action after completion is now over budget.
                self.assertEqual(reference_verdict(scene), "BLOCK")
                self.assertEqual(episode["max_operations"], 2)
                solved += 1
            self.assertEqual((solved, reviewed, revoked), (16, 4, 4))


if __name__ == "__main__":
    unittest.main()
