"""Label-separated corpus protocol and deterministic sampling helpers."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

CHOICES = ("PERMIT", "BLOCK", "REVIEW")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(row)
    return rows


def normalize_corpus(state_rows: Iterable[Mapping[str, Any]],
                     request_rows: Iterable[Mapping[str, Any]],
                     label_rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Join scenario_id/model_state, guard_request, and verdict files by id."""
    states: dict[str, dict[str, Any]] = {}
    for line, raw in enumerate(state_rows, 1):
        row = dict(raw)
        if not {"scenario_id", "model_state"} <= set(row) or set(row) - {"scenario_id", "model_state", "source"}:
            raise ValueError(f"state row {line} must contain scenario_id, model_state, and optional source")
        scenario_id = row["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id.strip() or scenario_id in states:
            raise ValueError(f"state row {line} has an empty or duplicate scenario_id")
        if not isinstance(row["model_state"], dict):
            raise ValueError(f"state row {line} model_state must be an object")
        source = row.get("source", "default")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"state row {line} source must be nonempty text")
        states[scenario_id] = {"sample_id": scenario_id, "source": source,
                               "state": row["model_state"]}
    if not states:
        raise ValueError("state JSONL must contain at least one row")

    requests: dict[str, dict[str, Any]] = {}
    for line, raw in enumerate(request_rows, 1):
        row = dict(raw)
        if set(row) != {"scenario_id", "guard_request"}:
            raise ValueError(f"request row {line} must contain scenario_id and guard_request")
        scenario_id = row["scenario_id"]
        if not isinstance(scenario_id, str) or scenario_id not in states or scenario_id in requests:
            raise ValueError(f"request row {line} has an unknown or duplicate scenario_id")
        if not isinstance(row["guard_request"], dict):
            raise ValueError(f"request row {line} guard_request must be an object")
        requests[scenario_id] = row["guard_request"]
    if set(requests) != set(states):
        raise ValueError("request JSONL must match every state scenario_id exactly")

    labels: dict[str, dict[str, Any]] = {}
    for line, raw in enumerate(label_rows, 1):
        row = dict(raw)
        allowed = {"scenario_id", "verdict", "pair_id", "source", "group_id"}
        if not {"scenario_id", "verdict"} <= set(row) or set(row) - allowed:
            raise ValueError(f"label row {line} must contain scenario_id, verdict, and optional pair_id/source")
        scenario_id, verdict = row["scenario_id"], row["verdict"]
        if not isinstance(scenario_id, str) or scenario_id not in states or scenario_id in labels:
            raise ValueError(f"label row {line} has an unknown or duplicate scenario_id")
        if verdict not in CHOICES:
            raise ValueError(f"label row {line} verdict must be PERMIT, BLOCK, or REVIEW")
        pair_id = row.get("pair_id")
        if pair_id is not None and (not isinstance(pair_id, str) or not pair_id.strip()):
            raise ValueError(f"label row {line} pair_id must be nonempty text")
        group_id = row.get("group_id")
        if group_id is not None and (not isinstance(group_id, str) or not group_id.strip()):
            raise ValueError(f"label row {line} group_id must be nonempty text")
        if "source" in row:
            if not isinstance(row["source"], str) or not row["source"].strip():
                raise ValueError(f"label row {line} source must be nonempty text")
            if states[scenario_id]["source"] != "default" and states[scenario_id]["source"] != row["source"]:
                raise ValueError(f"source mismatch for scenario_id {scenario_id!r}")
            states[scenario_id]["source"] = row["source"]
        labels[scenario_id] = row
    if set(labels) != set(states):
        raise ValueError("label JSONL must match every state scenario_id exactly")
    result = []
    for scenario_id in states:
        sample = states[scenario_id]
        sample["request"] = requests[scenario_id]
        result.append(sample)
    return result, labels


def balanced_source_indices(rows: list[Mapping[str, Any]], *, seed: int, epoch: int) -> list[int]:
    """Return a deterministic epoch with equal draws from every source bucket."""
    if not rows:
        return []
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        source = row.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("every state row needs a source bucket")
        groups.setdefault(source, []).append(index)
    rng = random.Random((seed << 32) ^ epoch)
    sources = sorted(groups)
    for source in sources:
        rng.shuffle(groups[source])
    per_source = max(len(indices) for indices in groups.values())
    sampled = {source: [indices[i % len(indices)] for i in range(per_source)]
               for source, indices in groups.items()}
    return [sampled[source][offset] for offset in range(per_source) for source in sources]


def pair_indices(rows: list[Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Find explicit two-row pairs with one PERMIT and one BLOCK verdict."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        pair_id = labels[row["sample_id"]].get("pair_id")
        if pair_id is not None:
            groups.setdefault(pair_id, []).append(index)
    pairs: list[tuple[int, int]] = []
    for pair_id, members in sorted(groups.items()):
        if len(members) != 2:
            raise ValueError(f"pair_id {pair_id!r} must identify exactly two rows")
        by_verdict = {labels[rows[i]["sample_id"]]["verdict"]: i for i in members}
        if set(by_verdict) != {"PERMIT", "BLOCK"}:
            raise ValueError(f"pair_id {pair_id!r} must contain one PERMIT and one BLOCK")
        pairs.append((by_verdict["PERMIT"], by_verdict["BLOCK"]))
    return pairs


def balanced_accumulation_groups(rows: list[Mapping[str, Any]],
                                 labels: Mapping[str, Mapping[str, Any]], *,
                                 seed: int, examples_per_update: int = 24,
                                 use_pairs: bool = True) -> list[list[tuple[int, ...]]]:
    """Pack each row once into source-balanced singleton-forward update groups.

    Each unit is a singleton row or an explicitly labeled PERMIT/BLOCK pair.
    With two equal source buckets, every update contains half of its rows from
    each source; pairs never cross source buckets or optimizer updates.
    """
    if examples_per_update <= 0:
        raise ValueError("examples_per_update must be positive")
    if not rows:
        return []
    by_source: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        source = row.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("every state row needs a source bucket")
        by_source.setdefault(source, []).append(index)
    if len(by_source) != 2 or len({len(indices) for indices in by_source.values()}) != 1:
        raise ValueError("balanced accumulation requires exactly two equal-sized source buckets")
    if examples_per_update < 2 or examples_per_update % 2:
        raise ValueError("examples_per_update must be an even number for two-source balance")
    pairs = pair_indices(rows, labels) if use_pairs else []
    paired: set[int] = set()
    pairs_by_source: dict[str, list[tuple[int, int]]] = {source: [] for source in by_source}
    for positive, negative in pairs:
        pos_source = rows[positive]["source"]
        neg_source = rows[negative]["source"]
        if pos_source != neg_source:
            raise ValueError("a contrast pair must stay within one source bucket")
        pairs_by_source[pos_source].append((positive, negative))
        paired.update((positive, negative))

    rng = random.Random((seed << 32) ^ examples_per_update)
    source_groups: dict[str, list[list[tuple[int, ...]]]] = {}
    per_source_examples = examples_per_update // 2
    for source in sorted(by_source):
        pairs_for_source = pairs_by_source[source]
        singles = [i for i in by_source[source] if i not in paired]
        rng.shuffle(pairs_for_source)
        rng.shuffle(singles)
        pair_cursor = single_cursor = 0
        while pair_cursor < len(pairs_for_source) or single_cursor < len(singles):
            remaining = per_source_examples
            units: list[tuple[int, ...]] = []
            while remaining:
                if remaining == 1:
                    if single_cursor >= len(singles):
                        break
                    units.append((singles[single_cursor],))
                    single_cursor += 1
                    remaining -= 1
                elif pair_cursor < len(pairs_for_source):
                    pair = pairs_for_source[pair_cursor]
                    pair_cursor += 1
                    units.append(pair)
                    remaining -= 2
                elif single_cursor < len(singles):
                    units.append((singles[single_cursor],))
                    single_cursor += 1
                    remaining -= 1
                else:
                    break
            if not units:
                raise ValueError("could not pack remaining rows into a singleton accumulation group")
            source_groups.setdefault(source, []).append(units)
    result = []
    sources = sorted(source_groups)
    if len({len(source_groups[source]) for source in sources}) != 1:
        raise RuntimeError("equal source buckets produced different update counts")
    for update_index in range(len(source_groups[sources[0]])):
        group = source_groups[sources[0]][update_index] + source_groups[sources[1]][update_index]
        rng.shuffle(group)
        result.append(group)
    rng.shuffle(result)
    flattened = [index for group in result for unit in group for index in unit]
    if sorted(flattened) != list(range(len(rows))):
        raise RuntimeError("accumulation schedule did not include each row exactly once")
    return result


def assert_gemini_module_origins(source_root: str | Path) -> None:
    """Fail if loaded runtime or vendored projection modules resolve elsewhere."""
    import sys

    root = Path(source_root).resolve()
    vendored = {"native_state_v1", "alpha_bind_v1", "compact_state_v3", "ledger_summary",
                "native_state_v2", "alpha_bind_v2", "bev_sequence_native_state",
                "bev_sequence_alpha_bind", "bev_sequence_native_state_v2",
                "bev_sequence_alpha_bind_v2"}
    for name, module in tuple(sys.modules.items()):
        if not (name.startswith("decision_factory") or name in vendored):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            raise RuntimeError(f"loaded module {name!r} has no source path")
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"module {name!r} resolved outside Gemini source: {path}")
