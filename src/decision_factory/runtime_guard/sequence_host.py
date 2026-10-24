"""Narrow no-oracle virtual host helpers retained from the audited episode runner."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import PurePosixPath
from typing import Any

def copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False))

def build_request(episode: dict, prior_actions: list[dict], observations: list[dict], proposal: dict) -> dict:
    return {"schema": "bev_guard_request/1", "user_request": copy_json(episode["task"]),
            "host_policy": copy_json(episode["host_policy"]), "prior_actions": copy_json(prior_actions),
            "observations": copy_json(observations), "proposal": copy_json(proposal)}

def qwen_observations(observations: list[dict], prior_actions: list[dict]) -> list[dict]:
    result = copy_json(observations)
    for index, action in enumerate(prior_actions, 1):
        result.append({"source": "host-action-history", "trust": "trusted",
                      "text": f"Actual prior action {index}: " + json.dumps(
                          {key: action[key] for key in ("tool_name", "arguments", "host_disposition") if key in action},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))})
        observation = action.get("observation")
        if isinstance(observation, dict) and observation.get("trust") in ("trusted", "untrusted"):
            result.append(copy_json(observation))
    return result

def qwen_visible_files(prior_actions: list[dict]) -> dict[str, str]:
    """Expose only file contents actually returned by prior successful reads."""
    files: dict[str, str] = {}
    for action in prior_actions:
        if (action.get("tool_name") != "read_file"
                or action.get("host_disposition") != "ALLOW"):
            continue
        args = action.get("arguments")
        observation = action.get("observation")
        if (isinstance(args, dict) and isinstance(args.get("path"), str)
                and isinstance(observation, dict)
                and observation.get("source") == "virtual-files"
                and observation.get("trust") == "untrusted"
                and isinstance(observation.get("text"), str)):
            files[args["path"]] = observation["text"]
    return files

def hard_validate(episode: dict, proposal: dict, executed_ops: int) -> str | None:
    policy = episode["host_policy"]
    name = proposal.get("tool_name")
    args = proposal.get("arguments")
    if name not in policy.get("allowed_operations", []) or not isinstance(args, dict):
        return "hard_schema_or_operation"
    schema = policy.get("tool_schemas", {}).get(name)
    if not isinstance(schema, dict):
        return "hard_missing_tool_schema"
    if not set(schema.get("required", [])).issubset(args):
        return "hard_missing_required_argument"
    if not set(args).issubset(set(schema.get("properties", {}))):
        return "hard_unknown_argument"
    if executed_ops >= policy.get("max_operations", 0):
        return "hard_operation_budget"
    if name in {"read_file", "upload_file"}:
        path = args.get("path")
        if not isinstance(path, str):
            return "hard_invalid_path"
        root = PurePosixPath(policy.get("workspace_root", "/workspace"))
        target = PurePosixPath(path)
        if not target.is_absolute() or ".." in target.parts or target == root or root not in target.parents:
            return "hard_path_outside_workspace"
    if name == "upload_file":
        if not isinstance(args.get("path"), str) or not isinstance(args.get("destination"), str):
            return "hard_invalid_upload_arguments"
        if "data_units" in args and (not isinstance(args["data_units"], int) or isinstance(args["data_units"], bool) or args["data_units"] <= 0):
            return "hard_invalid_data_units"
        if "data_fields" in args and (not isinstance(args["data_fields"], list) or not all(isinstance(x, str) for x in args["data_fields"])):
            return "hard_invalid_data_fields"
    return None

def host_execute(episode: dict, proposal: dict, decision: str, executed_ops: int) -> tuple[dict, bool]:
    """Apply only synthetic schemas/bounds and exact simulated tool effects."""
    tool = proposal["tool_name"]
    args = copy_json(proposal["arguments"])
    hard_error = hard_validate(episode, proposal, executed_ops)
    if hard_error:
        return {"tool_name": tool, "arguments": args, "host_disposition": "BLOCK",
                "observation": {"source": "host-runtime", "trust": "trusted", "text": hard_error}}, False
    if decision in ("BLOCK", "REVIEW"):
        observation = None
        if tool == "upload_file":
            receipt = {"status": "blocked_no_effect", "effect": args}
            observation = {"source": "effect-ledger", "trust": "trusted",
                           "text": json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
        entry = {"tool_name": tool, "arguments": args, "host_disposition": decision}
        if observation:
            entry["observation"] = observation
        return entry, False
    if decision != "PERMIT":
        raise ValueError(f"invalid guard decision: {decision}")
    if tool == "read_file":
        path = args["path"]
        files = episode.get("virtual_files", {})
        content = files.get(path)
        if not isinstance(content, str):
            return {"tool_name": tool, "arguments": args, "host_disposition": "BLOCK",
                    "observation": {"source": "host-runtime", "trust": "trusted",
                                    "text": "Synthetic virtual file result: not found."}}, False
        text = content
        return {"tool_name": tool, "arguments": args, "host_disposition": "ALLOW",
                "observation": {"source": "virtual-files", "trust": "untrusted", "text": text}}, True
    if tool == "upload_file":
        effect = {key: args[key] for key in ("path", "destination", "data_units", "data_fields", "grant_ref", "artifact_revision") if key in args}
        receipt = {"status": "committed", "effect": effect}
        return {"tool_name": tool, "arguments": args, "host_disposition": "ALLOW",
                "observation": {"source": "effect-ledger", "trust": "trusted",
                                "text": json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}}, True
    raise ValueError(f"no synthetic executor for listed operation {tool}")
