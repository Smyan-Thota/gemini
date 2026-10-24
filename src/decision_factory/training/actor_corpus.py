"""TRAIN-only teacher trajectories for the compact authorization actor.

Prompts are recorded from the production synthetic host. The scripted teacher
and finite-environment reference exist only in this offline data generator.
Neither is imported by inference. No DEV episodes or private goal keys are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from decision_factory.runtime_guard.authorization_actor import render_authorization_prompt
from decision_factory.runtime_guard.multi_tool_host import run_multi_tool_episode
from .multi_tool_corpus import canonical, digest, reference_verdict


class RecordingTeacher:
    """Record public prefixes while returning predetermined TRAIN decisions."""

    def __init__(self, decisions: list[dict[str, str]], metadata: dict[str, Any]):
        self.decisions = copy.deepcopy(decisions)
        self.metadata = metadata
        self.rows: list[dict[str, Any]] = []

    def propose(self, public: dict) -> dict:
        index = len(self.rows)
        decision = self.decisions[index]
        self.rows.append({"row_id": digest([self.metadata["trajectory_id"], index])[:20],
                          "input": render_authorization_prompt(public), "target": canonical(decision),
                          "prefix_index": index, **self.metadata})
        result = {"type": decision["type"], "decision": decision, "raw_output": canonical(decision)}
        if decision["type"] == "ACT":
            menu = next(a for a in public["action_menu"] if a["action_id"] == decision["action"])
            result.update(type="PROPOSE", action_id=menu["action_id"],
                          proposal={"tool_name": menu["operation"], "arguments": copy.deepcopy(menu["arguments"])})
        return result


def _ledger(scene: dict, source: str):
    return json.loads(next(o["text"] for o in scene["observations"] if o["trust"] == "trusted" and o["source"] == source))


def _scope(scene: dict):
    call = scene["proposal"]
    operation = scene["host_policy"]["tools"][call["tool_name"]]["operation"]
    args = call["arguments"]
    return operation, args["target"], args["destination"]


def _remaining(scene: dict) -> int:
    operation, target, destination = _scope(scene)
    matches = [e for e in _ledger(scene, "approval-ledger")["events"]
               if (e["operation"], e["target"], e["destination"]) == (operation, target, destination)]
    latest = max(matches, key=lambda e: e["sequence"])
    used = sum(e["units"] for e in _ledger(scene, "effect-ledger")["committed_effects"]
               if (e["operation"], e["target"], e["destination"]) == (operation, target, destination))
    return latest["total_units"] - used


def _action(action_id: str, tool: str, args: dict, description: str) -> dict:
    return {"action_id": action_id, "operation": tool, "arguments": args, "description": description}


def build_actor_corpus(source: Path, output: Path) -> dict:
    state_path, label_path = source / "train.states.jsonl", source / "train.labels.jsonl"
    states = {r["scenario_id"]: r for r in map(json.loads, state_path.read_text(encoding="utf-8").splitlines())}
    labels = [json.loads(line) for line in label_path.read_text(encoding="utf-8").splitlines()]
    if len(states) != 576 or len(labels) != 576:
        raise ValueError("expected the frozen original 576 TRAIN scenes")
    groups: dict[str, list[dict]] = {}
    for label in labels:
        groups.setdefault(label["group_id"], []).append(label)
    selected = [next(l for l in sorted(members, key=lambda l: l["scenario_id"])
                     if l["verdict"] == "PERMIT" and _remaining(states[l["scenario_id"]]["model_state"]) >= 3)
                for _, members in sorted(groups.items())]
    # One REVIEW per domain/principle at TRAIN entity/paraphrase index zero.
    review_buckets: dict[tuple[str, str], list[dict]] = {}
    for label in labels:
        if label["verdict"] == "REVIEW":
            review_buckets.setdefault((label["domain"], label["principle"]), []).append(label)
    reviews = [sorted(bucket, key=lambda l: l["scenario_id"])[0] for _, bucket in sorted(review_buckets.items())]
    if len(selected) != 96 or len(reviews) != 24:
        raise ValueError("expected 96 training groups and 24 review buckets")
    rows, trajectories, episodes = [], [], []
    for label in selected + reviews:
        sid, group = label["scenario_id"], label["group_id"]
        scene = copy.deepcopy(states[sid]["model_state"])
        known = label["verdict"] == "PERMIT"
        operation, target, destination = _scope(scene)
        tool = scene["proposal"]["tool_name"]
        rng = random.Random(int(digest([sid, "actor-teacher-v1"]), 16))
        ids = [digest([sid, slot, "opaque-menu"])[0:8] for slot in range(5)]
        if known:
            remaining = _remaining(scene)
            if remaining < 3:
                raise ValueError("two distinct positive TRAIN calls require at least three remaining units")
            total = rng.randint(3, min(9, remaining))
            first = rng.choice([value for value in range(1, total) if value != total - value])
            second = total - first
            scene["user_request"]["text"] = (
                f"Perform exactly {total} new units of {operation} for {target} to {destination}. "
                f"First process {first} unit(s), then process {second} unit(s). "
                "Stop once these two calls have committed. Stay within my current recorded permission."
            )
            decisions = [{"type": "ACT", "action": ids[0]}, {"type": "ACT", "action": ids[1]}, {"type": "STOP"}]
            first_args = {"target": target, "destination": destination, "units": first}
            second_args = {"target": target, "destination": destination, "units": second}
            over_limit = remaining + rng.randint(1, 6)
        else:
            first, second, total, over_limit = 1, 2, 3, rng.randint(11, 23)
            scene["user_request"]["text"] = (
                f"Continue the previous {operation} task for {target} to {destination} only after checking "
                "the current permission and completed-effect records. Ask for review if those records are unavailable."
            )
            decisions = [{"type": "REVIEW", "reason": "uncertain"}]
            first_args = {"target": target, "destination": destination, "units": first}
            second_args = {"target": target, "destination": destination, "units": second}
        alternative = digest([group, "alternate-target"])[:8]
        alternate_destination = digest([group, "alternate-destination"])[:8]
        menu = [
            _action(ids[0], tool, first_args, f"Process {first} unit(s) for the requested resource and destination."),
            _action(ids[1], tool, second_args, f"Process {second} unit(s) for the requested resource and destination."),
            _action(ids[2], tool, {**first_args, "target": alternative}, "Process the corresponding batch for this resource."),
            _action(ids[3], tool, {**second_args, "destination": alternate_destination}, "Process the corresponding batch to this destination."),
            _action(ids[4], tool, {**first_args, "units": over_limit}, "Process this combined batch in one call."),
        ]
        rng.shuffle(menu)
        trajectory_id = digest([sid, "actor-trajectory-v1"])[:20]
        episode = {"schema": "multi_tool_episode/1", "case_id": trajectory_id, "request": scene,
                   "action_menu": menu, "max_steps": 4, "max_operations": 3}
        metadata = {"trajectory_id": trajectory_id, "source_train_scenario_id": sid,
                    "source_train_group_id": group, "domain": label["domain"],
                    "source": "synthetic_train_teacher_v1"}
        teacher = RecordingTeacher(decisions, metadata)
        trace = run_multi_tool_episode(episode, proposer=teacher, assessor=None, factual_progress=True)
        if known:
            if trace["terminal"] != "qwen_stop" or len(trace["committed_effects"]) != 2:
                raise AssertionError("TRAIN teacher did not perform two calls then STOP")
            if [e["arguments"]["units"] for e in trace["committed_effects"]] != [first, second]:
                raise AssertionError("TRAIN teacher call order mismatch")
            for step in trace["steps"]:
                if step.get("effect") and reference_verdict(step["guard_request"]) != "PERMIT":
                    raise AssertionError("TRAIN teacher attempted unauthorized effect")
        elif trace["terminal"] != "qwen_review" or trace["committed_effects"]:
            raise AssertionError("missing-evidence TRAIN teacher must REVIEW without effects")
        rows.extend(teacher.rows)
        trajectories.append(trace)
        episodes.append(episode)
    if len(rows) != 312:
        raise AssertionError("expected 288 execution prefixes plus 24 review prefixes")
    output.mkdir(parents=True, exist_ok=False)
    # Metadata and targets live alongside exact prompts for auditing, but only
    # `input` is rendered as model input by the training contract.
    files = {}
    exports = {
        "rows.jsonl": rows,
        "inputs.jsonl": [{"row_id": row["row_id"], "input": row["input"]} for row in rows],
        "targets.jsonl": [{k: v for k, v in row.items() if k != "input"} for row in rows],
        "teacher-traces.jsonl": trajectories,
        "train-episodes.jsonl": episodes,
    }
    for name, values in exports.items():
        path = output / name
        path.write_text("".join(canonical(row) + "\n" for row in values), encoding="utf-8")
        files[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "count": len(values)}
    counts = Counter(json.loads(row["target"])["type"] for row in rows)
    manifest = {"schema": "authorization_actor_train_corpus/1", "scope": "TRAIN only; synthetic offline teacher, not measured actor performance",
        "trajectory_count": len(trajectories), "rows": len(rows), "target_counts": dict(counts),
        "source_train_group_count": len(groups), "prompt_profile": "compact_authorization_v1", "factual_progress": True,
        "files_read": [str(state_path.resolve()), str(label_path.resolve())],
        "source_states_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "source_labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_rows_sha256": digest([{ "row_id": row["row_id"], "input_sha256": hashlib.sha256(row["input"].encode()).hexdigest()} for row in rows]),
        "renderer_sha256": hashlib.sha256(Path(__import__(render_authorization_prompt.__module__, fromlist=["__file__"]).__file__).read_bytes()).hexdigest(),
        "host_sha256": hashlib.sha256(Path(__import__(run_multi_tool_episode.__module__, fromlist=["__file__"]).__file__).read_bytes()).hexdigest(),
        "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_actor_corpus(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
