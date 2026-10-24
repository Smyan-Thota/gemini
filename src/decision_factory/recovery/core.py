"""Finite explicit-world simulator, solver, runner, and grouped data builder."""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


class SolverExhausted(RuntimeError):
    """The configured exact-search node limit was reached."""


@dataclass(frozen=True)
class World:
    schema: str
    task_id: str
    states: tuple[str, ...]
    initial_belief: tuple[str, ...]
    safe_states: frozenset[str]
    goal_states: frozenset[str]
    action_limit: int
    cost_limit: int
    actions: tuple[dict[str, Any], ...]

    @property
    def action_map(self) -> dict[str, dict[str, Any]]:
        return {a["id"]: a for a in self.actions}


@dataclass(frozen=True)
class Solution:
    status: str
    winning_first_actions: tuple[str, ...]
    nodes: int


@dataclass
class Step:
    status: str
    raw_input: dict[str, Any] | None = None
    raw_output: str | None = None
    decision: dict[str, str] | None = None
    parse_error: str | None = None
    observation: str | None = None
    remaining_actions: int = 0
    remaining_cost: int = 0
    actions_before: int = 0
    cost_before: int = 0
    effect: dict[str, Any] | None = None
    latency_s: float | None = None


@dataclass
class Episode:
    status: str
    trajectory: list[Step]
    elapsed_latency: float
    metrics: dict[str, int]
    seed: int = 0


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _parse_world(mapping: Mapping[str, Any]) -> World:
    if not isinstance(mapping, Mapping):
        raise ValueError("world must be a JSON object")
    w = dict(mapping)
    if w.get("schema") != "decision-factory.recovery.v1":
        raise ValueError("unsupported world schema")
    if not _identifier(w.get("task_id")):
        raise ValueError("task_id must be a nonempty string")
    states = w.get("states")
    if not isinstance(states, list) or not states or any(not _identifier(s) for s in states) or len(set(states)) != len(states):
        raise ValueError("states must be a nonempty list of unique nonempty strings")
    state_set = set(states)
    initial = w.get("initial_belief")
    if not isinstance(initial, list) or not initial or any(s not in state_set for s in initial) or len(set(initial)) != len(initial):
        raise ValueError("initial_belief must be a nonempty unique subset of states")
    safe, goals = w.get("safe_states"), w.get("goal_states")
    if not isinstance(safe, list) or not isinstance(goals, list):
        raise ValueError("safe_states and goal_states must be lists")
    if any(s not in state_set for s in safe + goals) or len(set(safe)) != len(safe) or len(set(goals)) != len(goals):
        raise ValueError("safe_states and goal_states must be unique state subsets")
    if not set(goals) <= set(safe):
        raise ValueError("every goal state must be safe")
    if not _nonnegative_int(w.get("action_limit")) or not _nonnegative_int(w.get("cost_limit")):
        raise ValueError("action_limit and cost_limit must be nonnegative integers")
    rows = w.get("actions")
    if not isinstance(rows, list):
        raise ValueError("actions must be a list")
    ids: list[str] = []
    normalized: list[dict[str, Any]] = []
    for a in rows:
        if not isinstance(a, dict) or not _identifier(a.get("id")):
            raise ValueError("each action must have a nonempty string id")
        aid = a["id"]
        ids.append(aid)
        if a.get("kind") not in ("READ", "WRITE"):
            raise ValueError("action kind must be READ or WRITE")
        if not _nonnegative_int(a.get("cost")):
            raise ValueError("action cost must be a nonnegative integer")
        outs = a.get("outcomes")
        if not isinstance(outs, dict) or set(outs) != state_set:
            raise ValueError("each action must define outcomes for every state")
        clean_outs: dict[str, list[dict[str, str]]] = {}
        for s in states:
            options = outs[s]
            if not isinstance(options, list) or not options:
                raise ValueError("each state outcome list must be nonempty")
            clean: list[dict[str, str]] = []
            for o in options:
                if not isinstance(o, dict) or set(o) != {"next", "observation"}:
                    raise ValueError("outcome must contain exactly next and observation")
                if o["next"] not in state_set or not _identifier(o["observation"]):
                    raise ValueError("outcome next state or observation is invalid")
                clean.append({"next": o["next"], "observation": o["observation"]})
            clean_outs[s] = clean
        normalized.append({"id": aid, "kind": a["kind"], "cost": a["cost"], "outcomes": clean_outs})
    if len(ids) != len(set(ids)):
        raise ValueError("action ids must be unique")
    return World(w["schema"], w["task_id"], tuple(states), tuple(initial), frozenset(safe),
                 frozenset(goals), w["action_limit"], w["cost_limit"], tuple(normalized))


def load_world(mapping_or_path: Mapping[str, Any] | str | Path) -> World:
    if isinstance(mapping_or_path, str) and mapping_or_path.lstrip().startswith("{"):
        try:
            raw = json.loads(mapping_or_path)
        except json.JSONDecodeError as exc:
            raise ValueError(f"cannot parse world JSON: {exc}") from exc
    elif isinstance(mapping_or_path, (str, Path)):
        try:
            raw = json.loads(Path(mapping_or_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load world: {exc}") from exc
    else:
        raw = mapping_or_path
    return _parse_world(raw)


def public_input(world: World | Mapping[str, Any] | str | Path,
                 history: Iterable[Mapping[str, str]] = ()) -> dict[str, Any]:
    w = world if isinstance(world, World) else load_world(world)
    hist = []
    for item in history:
        if not isinstance(item, Mapping) or set(item) != {"action", "observation"}:
            raise ValueError("each history item must contain action and observation")
        if not _identifier(item["action"]) or not _identifier(item["observation"]):
            raise ValueError("history action and observation must be nonempty strings")
        hist.append({"action": item["action"], "observation": item["observation"]})
    return {"schema": w.schema, "task_id": w.task_id, "states": list(w.states),
            "initial_belief": list(w.initial_belief),
            "public_rules": {"goal_states": sorted(w.goal_states), "safe_states": sorted(w.safe_states)},
            "action_limit": w.action_limit, "cost_limit": w.cost_limit,
            "actions": [{"id": a["id"], "kind": a["kind"], "cost": a["cost"],
                         "outcomes": {s: [dict(o) for o in a["outcomes"][s]] for s in w.states}}
                        for a in w.actions], "history": hist}


def belief(world: World | Mapping[str, Any] | str | Path,
           history: Iterable[Mapping[str, str]] = ()) -> tuple[str, ...]:
    w = world if isinstance(world, World) else load_world(world)
    current = set(w.initial_belief)
    actions = w.action_map
    for item in history:
        if not isinstance(item, Mapping) or set(item) != {"action", "observation"}:
            raise ValueError("invalid history item")
        aid, obs = item["action"], item["observation"]
        if aid not in actions:
            raise ValueError(f"unknown history action: {aid!r}")
        a = actions[aid]
        current = {o["next"] for s in current for o in a["outcomes"][s] if o["observation"] == obs}
        if not current:
            raise ValueError("history is inconsistent with the world")
    return tuple(sorted(current))


def solve(world: World | Mapping[str, Any] | str | Path, *, max_nodes: int | None = None) -> Solution:
    w = world if isinstance(world, World) else load_world(world)
    if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 0):
        raise ValueError("max_nodes must be a nonnegative integer or None")
    safe, goals, initial = w.safe_states, w.goal_states, frozenset(w.initial_belief)
    count = 0

    def branches(b: frozenset[str], action: dict[str, Any]) -> tuple[frozenset[str], ...]:
        by_obs: dict[str, set[str]] = {}
        for state in b:
            for o in action["outcomes"][state]:
                by_obs.setdefault(o["observation"], set()).add(o["next"])
        return tuple(frozenset(by_obs[k]) for k in sorted(by_obs))

    @lru_cache(maxsize=None)
    def win(b: frozenset[str], actions_left: int, cost_left: int) -> bool:
        nonlocal count
        count += 1
        if max_nodes is not None and count > max_nodes:
            raise SolverExhausted(f"solver exceeded max_nodes={max_nodes}")
        if not b or not b <= safe:
            return False
        if b <= goals:
            return True
        if actions_left <= 0:
            return False
        for a in w.actions:
            if a["cost"] > cost_left:
                continue
            if all(win(br, actions_left - 1, cost_left - a["cost"]) for br in branches(b, a)):
                return True
        return False

    def first(a: dict[str, Any]) -> bool:
        if w.action_limit <= 0 or a["cost"] > w.cost_limit or any(s not in safe for s in initial):
            return False
        return all(win(br, w.action_limit - 1, w.cost_limit - a["cost"]) for br in branches(initial, a))

    initially_winning = win(initial, w.action_limit, w.cost_limit)
    winning = tuple(sorted(a["id"] for a in w.actions if first(a)))
    if initially_winning:
        status = "solvable"
    elif all(win(frozenset((s,)), w.action_limit, w.cost_limit) for s in initial):
        status = "information_gap"
    else:
        status = "impossible"
    return Solution(status, winning, count)


def _strict_decision(raw: str) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(raw, str):
        return None, "response must be text"
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    try:
        obj = json.loads(raw, object_pairs_hook=pairs,
                         parse_constant=lambda v: (_ for _ in ()).throw(ValueError("invalid JSON constant")))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return None, str(exc)
    if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
        return None, "decision must be a JSON object with string type"
    typ = obj["type"]
    if typ == "ACT" and set(obj) == {"type", "action"} and isinstance(obj["action"], str):
        return {"type": "ACT", "action": obj["action"]}, None
    if typ == "STOP" and set(obj) == {"type"}:
        return {"type": "STOP"}, None
    if typ == "REVIEW" and set(obj) == {"type", "reason"} and isinstance(obj["reason"], str):
        return {"type": "REVIEW", "reason": obj["reason"]}, None
    return None, "decision shape or field types are invalid"


class Simulator:
    def __init__(self, world: World | Mapping[str, Any] | str | Path, actual_state: str, seed: int = 0):
        self.world = world if isinstance(world, World) else load_world(world)
        if actual_state not in self.world.states:
            raise ValueError("actual_state must be a state in the world")
        if actual_state not in self.world.initial_belief:
            raise ValueError("actual_state must belong to initial_belief")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.state = actual_state
        self.seed = seed
        self.rng = random.Random(seed)
        self.remaining_actions = self.world.action_limit
        self.remaining_cost = self.world.cost_limit
        self.initial_unsafe = actual_state not in self.world.safe_states
        self.terminal: str | None = None
        self.history: list[dict[str, str]] = []

    def decide(self, raw_json: str) -> Step:
        before_a, before_c = self.remaining_actions, self.remaining_cost
        if self.terminal is not None:
            return Step("terminal", raw_output=raw_json if isinstance(raw_json, str) else None,
                        remaining_actions=before_a, remaining_cost=before_c,
                        actions_before=before_a, cost_before=before_c)
        if self.initial_unsafe:
            self.terminal = "unsafe_initial"
            return Step("unsafe_initial", raw_output=raw_json if isinstance(raw_json, str) else None,
                        remaining_actions=before_a, remaining_cost=before_c,
                        actions_before=before_a, cost_before=before_c)
        decision, error = _strict_decision(raw_json)
        if error:
            self.terminal = "malformed"
            return Step(self.terminal, raw_output=raw_json if isinstance(raw_json, str) else None,
                        parse_error=error, remaining_actions=before_a, remaining_cost=before_c,
                        actions_before=before_a, cost_before=before_c)
        assert decision is not None
        typ = decision["type"]
        if typ == "STOP":
            status = "success" if self.state in self.world.safe_states and self.state in self.world.goal_states else "false_stop"
            self.terminal = status
            return Step(status, raw_output=raw_json, decision=decision, remaining_actions=before_a,
                        remaining_cost=before_c, actions_before=before_a, cost_before=before_c)
        if typ == "REVIEW":
            self.terminal = "review"
            return Step("review", raw_output=raw_json, decision=decision, remaining_actions=before_a,
                        remaining_cost=before_c, actions_before=before_a, cost_before=before_c)
        action = self.world.action_map.get(decision["action"])
        if action is None:
            self.terminal = "invalid_action"
            return Step(self.terminal, raw_output=raw_json, decision=decision, remaining_actions=before_a,
                        remaining_cost=before_c, actions_before=before_a, cost_before=before_c)
        if self.remaining_actions < 1 or action["cost"] > self.remaining_cost:
            self.terminal = "over_budget"
            return Step(self.terminal, raw_output=raw_json, decision=decision, remaining_actions=before_a,
                        remaining_cost=before_c, actions_before=before_a, cost_before=before_c)
        prior = self.state
        selected = action["outcomes"][prior][self.rng.randrange(len(action["outcomes"][prior]))]
        self.state = selected["next"]
        self.remaining_actions -= 1
        self.remaining_cost -= action["cost"]
        self.history.append({"action": action["id"], "observation": selected["observation"]})
        effect = {"action": action["id"], "prior_state": prior, "next_state": self.state,
                  "observation": selected["observation"], "cost_charged": action["cost"],
                  "remaining_actions": self.remaining_actions, "remaining_cost": self.remaining_cost}
        return Step("executed", raw_output=raw_json, decision=decision, observation=selected["observation"],
                    remaining_actions=self.remaining_actions, remaining_cost=self.remaining_cost,
                    actions_before=before_a, cost_before=before_c, effect=effect)


def _episode_metrics(trajectory: list[Step]) -> dict[str, int]:
    proposals = sum(1 for x in trajectory if x.decision and x.decision.get("type") == "ACT")
    executed = sum(1 for x in trajectory if x.status == "executed")
    return {"act_proposals": proposals, "executed_actions": executed, "actual_effects": executed}


def run_episode(world: World | Mapping[str, Any] | str | Path, actual_state: str,
                model: Callable[[dict[str, Any]], str], seed: int = 0,
                max_latency_s: float | None = None) -> Episode:
    w = world if isinstance(world, World) else load_world(world)
    sim = Simulator(w, actual_state, seed=seed)
    trajectory: list[Step] = []
    t0 = time.perf_counter()
    if sim.initial_unsafe:
        return Episode("unsafe_initial", trajectory, 0.0, _episode_metrics(trajectory), seed)
    while True:
        prompt = public_input(w, sim.history)
        start = time.perf_counter()
        try:
            raw = model(prompt)
            latency = time.perf_counter() - start
            # Synchronous callbacks cannot be cancelled; classify only after return.
            if max_latency_s is not None and latency > max_latency_s:
                step = Step("timeout", raw_input=prompt,
                            raw_output=raw if isinstance(raw, str) else repr(raw),
                            remaining_actions=sim.remaining_actions, remaining_cost=sim.remaining_cost,
                            actions_before=sim.remaining_actions, cost_before=sim.remaining_cost,
                            latency_s=latency)
                trajectory.append(step)
                return Episode("timeout", trajectory, time.perf_counter() - t0, _episode_metrics(trajectory), seed)
            step = sim.decide(raw)
            step.raw_input = prompt
            step.latency_s = latency
        except TimeoutError:
            latency = time.perf_counter() - start
            trajectory.append(Step("timeout", raw_input=prompt, remaining_actions=sim.remaining_actions,
                                   remaining_cost=sim.remaining_cost, actions_before=sim.remaining_actions,
                                   cost_before=sim.remaining_cost, latency_s=latency))
            return Episode("timeout", trajectory, time.perf_counter() - t0, _episode_metrics(trajectory), seed)
        except KeyboardInterrupt:
            latency = time.perf_counter() - start
            trajectory.append(Step("interrupted", raw_input=prompt, remaining_actions=sim.remaining_actions,
                                   remaining_cost=sim.remaining_cost, actions_before=sim.remaining_actions,
                                   cost_before=sim.remaining_cost, latency_s=latency))
            return Episode("interrupted", trajectory, time.perf_counter() - t0, _episode_metrics(trajectory), seed)
        trajectory.append(step)
        if step.status != "executed":
            return Episode(step.status, trajectory, time.perf_counter() - t0, _episode_metrics(trajectory), seed)


def replay(world: World | Mapping[str, Any] | str | Path, actual_state: str,
           raw_decisions: Iterable[str], seed: int = 0) -> Episode:
    sim = Simulator(world, actual_state, seed=seed)
    trajectory: list[Step] = []
    for raw in raw_decisions:
        step = sim.decide(raw)
        trajectory.append(step)
        if step.status != "executed":
            return Episode(step.status, trajectory, 0.0, _episode_metrics(trajectory), seed)
    return Episode("incomplete", trajectory, 0.0, _episode_metrics(trajectory), seed)


def export_trajectory(episode: Episode) -> dict[str, Any]:
    return {"status": episode.status, "seed": episode.seed,
            "trajectory": [asdict(step) for step in episode.trajectory],
            "elapsed_latency": episode.elapsed_latency, "metrics": dict(episode.metrics)}


def _canonical_body(world: World, *, include_rules: bool) -> dict[str, Any]:
    """Canonicalize neutral state/action/observation names and list order.

    Tiny decision worlds are deliberately finite and small. Exhaustive state
    labelings make the fingerprint invariant to arbitrary neutral state names.
    Above eight states, a deterministic name-sorted fallback avoids factorial
    work; generated benchmark worlds are expected to remain tiny.
    """
    if len(world.states) <= 8:
        import itertools
        labelings = itertools.permutations(world.states)
    else:
        labelings = (tuple(sorted(world.states)),)
    best: str | None = None
    for labeling in labelings:
        ix = {state: i for i, state in enumerate(labeling)}
        initial = sorted(ix[s] for s in world.initial_belief)
        body: dict[str, Any] = {
            "schema": world.schema, "initial_belief": initial,
            "action_limit": world.action_limit, "cost_limit": world.cost_limit,
            "actions": [],
        }
        if include_rules:
            body["public_rules"] = {
                "goal_states": sorted(ix[s] for s in world.goal_states),
                "safe_states": sorted(ix[s] for s in world.safe_states),
            }
        for action in world.actions:
            obs_profiles: dict[str, tuple[Any, ...]] = {}
            for obs in {o["observation"] for rows in action["outcomes"].values() for o in rows}:
                profile = []
                for state in labeling:
                    profile.append(tuple(sorted(ix[o["next"]] for o in action["outcomes"][state]
                                                if o["observation"] == obs)))
                obs_profiles[obs] = tuple(profile)
            obs_rank = {obs: rank for rank, (_, obs) in enumerate(
                sorted((profile, obs) for obs, profile in obs_profiles.items()))}
            action_body = {
                "kind": action["kind"], "cost": action["cost"],
                "outcomes": [
                    sorted((obs_rank[o["observation"]], ix[o["next"]])
                           for o in action["outcomes"][state])
                    for state in labeling
                ],
            }
            body["actions"].append(action_body)
        body["actions"].sort(key=lambda a: json.dumps(a, sort_keys=True, separators=(",", ":")))
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if best is None or encoded < best:
            best = encoded
    assert best is not None
    return json.loads(best)


def _fingerprint(world: World, *, include_rules: bool = True) -> str:
    blob = json.dumps(_canonical_body(world, include_rules=include_rules),
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _normalized_targets(world: World, solution: Solution) -> tuple[str, tuple[str, ...]]:
    """Represent valid actions by their renamed-world transition signatures."""
    winning = set(solution.winning_first_actions)
    signatures = []
    for action in world.actions:
        one = World(world.schema, world.task_id, world.states, world.initial_belief,
                    world.safe_states, world.goal_states, world.action_limit,
                    world.cost_limit, (action,))
        signatures.append((_fingerprint(one, include_rules=False), action["id"] in winning))
    return solution.status, tuple(sorted(sig for sig, is_winning in signatures if is_winning))


def _remap_world(world: Mapping[str, Any], rng: random.Random, task_id: str) -> dict[str, Any]:
    """Rename neutral symbols consistently and permute semantically neutral lists."""
    source = json.loads(json.dumps(world))
    # A common positive scale preserves every cost-fit comparison while
    # creating varied legal integer costs and bounds.
    cost_scale = rng.randint(1, 3)
    source["cost_limit"] *= cost_scale
    for action in source["actions"]:
        action["cost"] *= cost_scale
    states = list(source["states"])
    actions = list(source["actions"])
    observations = sorted({o["observation"] for a in actions for rows in a["outcomes"].values() for o in rows})
    state_names = [f"st-{rng.getrandbits(64):016x}" for _ in states]
    action_names = [f"act-{rng.getrandbits(64):016x}" for _ in actions]
    observation_names = [f"obs-{rng.getrandbits(64):016x}" for _ in observations]
    state_map = dict(zip(states, state_names))
    action_map = {a["id"]: action_names[i] for i, a in enumerate(actions)}
    observation_map = dict(zip(observations, observation_names))
    generated = {
        "schema": source["schema"], "task_id": task_id,
        "states": [state_map[s] for s in states],
        "initial_belief": [state_map[s] for s in source["initial_belief"]],
        "safe_states": [state_map[s] for s in source["safe_states"]],
        "goal_states": [state_map[s] for s in source["goal_states"]],
        "action_limit": source["action_limit"], "cost_limit": source["cost_limit"],
        "actions": [],
    }
    for action in actions:
        outcomes = {}
        for state in states:
            outcomes[state_map[state]] = [
                {"next": state_map[o["next"]], "observation": observation_map[o["observation"]]}
                for o in action["outcomes"][state]
            ]
            rng.shuffle(outcomes[state_map[state]])
        generated["actions"].append({
            "id": action_map[action["id"]], "kind": action["kind"], "cost": action["cost"],
            "outcomes": outcomes,
        })
    rng.shuffle(generated["states"])
    rng.shuffle(generated["initial_belief"])
    rng.shuffle(generated["safe_states"])
    rng.shuffle(generated["goal_states"])
    rng.shuffle(generated["actions"])
    return generated


def generate_groups(spec: Mapping[str, Any], seed: int) -> dict[str, Any]:
    if type(seed) is not int or not isinstance(spec, Mapping) or not isinstance(spec.get("groups"), list):
        raise ValueError("spec must contain groups and seed must be an integer")
    rng = random.Random(seed)
    group_splits: dict[str, str] = {}
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    labels: dict[str, tuple[str, tuple[str, ...]]] = {}
    structural_labels: dict[str, tuple[str, tuple[str, ...]]] = {}
    group_indices: dict[str, int] = {}
    for group in spec["groups"]:
        if (not isinstance(group, Mapping) or not _identifier(group.get("group"))
                or group.get("split") not in ("train", "dev") or not isinstance(group.get("worlds"), list)):
            raise ValueError("each group needs group, train/dev split, and worlds list")
        gid, split = group["group"], group["split"]
        if gid in group_splits and group_splits[gid] != split:
            raise ValueError("group overlap across splits")
        group_splits[gid] = split
        worlds = list(group["worlds"])
        rng.shuffle(worlds)
        for raw_world in worlds:
            world = load_world(raw_world)
            solution = solve(world)
            fingerprint = _fingerprint(world, include_rules=True)
            structural_fp = _fingerprint(world, include_rules=False)
            target = _normalized_targets(world, solution)
            if fingerprint in labels and labels[fingerprint] != target:
                raise ValueError("duplicate meaningful public task has conflicting valid targets")
            if structural_fp in structural_labels and structural_labels[structural_fp] != target:
                raise ValueError("duplicate transition task has conflicting public goal/safety targets")
            labels[fingerprint] = target
            structural_labels[structural_fp] = target
            example_index = group_indices.get(gid, 0)
            group_indices[gid] = example_index + 1
            id_material = f"{seed}:{gid}:{example_index}".encode("utf-8")
            generated_id = "task-" + hashlib.sha256(id_material).hexdigest()[:20]
            local_seed = int.from_bytes(hashlib.sha256(id_material + fingerprint.encode()).digest()[:8], "big")
            generated = _remap_world(raw_world, random.Random(local_seed), generated_id)
            generated_solution = solve(load_world(generated))
            rows[split].append({"group": gid, "world": generated, "fingerprint": fingerprint,
                                "oracle_status": generated_solution.status,
                                "valid_first_actions": list(generated_solution.winning_first_actions)})
    train_groups = sorted({x["group"] for x in rows["train"]})
    dev_groups = sorted({x["group"] for x in rows["dev"]})
    if set(train_groups) & set(dev_groups):
        raise ValueError("train and dev groups must be disjoint")
    train_fp = {x["fingerprint"] for x in rows["train"]}
    dev_fp = {x["fingerprint"] for x in rows["dev"]}
    if train_fp & dev_fp:
        raise ValueError("normalized public tasks must be disjoint between train and dev")
    for split_rows in rows.values():
        split_rows.sort(key=lambda x: (x["group"], x["fingerprint"]))
    return {"train": rows["train"], "dev": rows["dev"],
            "train_groups": train_groups, "dev_groups": dev_groups}
