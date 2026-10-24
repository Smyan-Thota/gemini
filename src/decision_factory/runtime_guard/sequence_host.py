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
    seen = {json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in result if isinstance(item, dict)}
    for index, action in enumerate(prior_actions, 1):
        result.append({"source": "host-action-history", "trust": "trusted",
                      "text": f"Actual prior action {index}: " + json.dumps(
                          {key: action[key] for key in ("tool_name", "arguments", "host_disposition") if key in action},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))})
        observation = action.get("observation")
        if isinstance(observation, dict) and observation.get("trust") in ("trusted", "untrusted"):
            canonical = json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if canonical not in seen:
                result.append(copy_json(observation))
                seen.add(canonical)
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
                and observation.get("source") in {"virtual-file", "virtual-files"}
                and observation.get("trust") == "untrusted"
                and isinstance(observation.get("text"), str)):
            files[args["path"]] = observation["text"]
    return files


def run_sequence_episode(case: dict, *, proposer: Any, host: Any,
                         max_steps: int | None = None) -> dict:
    """Run the production Qwen -> exact-request Bev -> bounded host loop.

    The only effects are held in ``host.files``. Future menus or episode
    metadata are never included in a proposer or guard request.
    """
    if not isinstance(case, dict) or case.get("schema") != "sequence_host_demo_case/1":
        raise ValueError("expected sequence_host_demo_case/1")
    task = case.get("task")
    policy = case.get("host_policy")
    if not isinstance(task, dict) or not isinstance(policy, dict):
        raise ValueError("episode requires public task and policy objects")
    static_menu = case.get("action_menu")
    scheduled_menus = case.get("turn_action_menus")
    if scheduled_menus is not None:
        if not isinstance(scheduled_menus, list) or any(not isinstance(menu, list) for menu in scheduled_menus):
            raise ValueError("turn_action_menus must be a list of action menus")
        menu_limit = len(scheduled_menus)
    elif isinstance(static_menu, list) and static_menu:
        menu_limit = policy.get("max_steps")
        if type(menu_limit) is not int or menu_limit < 1:
            menu_limit = 4
    else:
        raise ValueError("episode requires a nonempty action_menu or turn_action_menus")
    if max_steps is None:
        max_steps = menu_limit
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if max_steps > 32:
        raise ValueError("max_steps exceeds the runtime safety bound of 32")
    if scheduled_menus is not None:
        max_steps = min(max_steps, len(scheduled_menus))
    steps: list[dict[str, Any]] = []
    terminal = "step_limit"
    for turn in range(max_steps):
        if scheduled_menus is not None:
            menu = scheduled_menus[turn]
            if not menu:
                terminal = "menu_exhausted"
                break
        else:
            menu = static_menu
        operation_budget = policy.get("max_operations")
        if (type(operation_budget) is int and operation_budget >= 1
                and host.operations >= operation_budget):
            terminal = "operation_budget"
            break
        model_menu = []
        for item in menu:
            if not isinstance(item, dict):
                raise ValueError("action menu items must be objects")
            proposal = item.get("proposal")
            if isinstance(proposal, dict):
                operation = proposal.get("tool_name")
                arguments = proposal.get("arguments")
            else:
                operation = item.get("operation", item.get("tool_name"))
                arguments = item.get("arguments")
            if (not isinstance(item.get("action_id"), str)
                    or not isinstance(item.get("description"), str)
                    or not isinstance(operation, str) or not isinstance(arguments, dict)):
                raise ValueError("action menu items require id, description, operation, and arguments")
            model_menu.append({"action_id": item["action_id"], "description": item["description"],
                               "operation": operation, "arguments": copy_json(arguments)})
        qwen_input = {
            "user_request": copy_json(task),
            "host_policy": copy_json(host.base_request["host_policy"]),
            "action_menu": model_menu,
            "observations": qwen_observations(host.observations, host.prior_actions),
            "initial_files": qwen_visible_files(host.prior_actions),
        }
        prompt_before = len(getattr(proposer, "calls", []))
        try:
            qwen_result = proposer.propose(qwen_input)
        except Exception as exc:
            call = proposer.calls[-1] if len(getattr(proposer, "calls", [])) > prompt_before else {}
            prompt = call.get("prompt") if isinstance(call, dict) else None
            steps.append({"turn": turn + 1, "qwen": {"type": "ERROR"},
                          "qwen_error": {"stage": "proposal", "code": "qwen_proposal_error",
                                         "exception_type": type(exc).__name__},
                          "qwen_input": qwen_input,
                          "qwen_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                          if isinstance(prompt, str) else None,
                          "guard_request": None, "guard_assessment": None,
                          "host_result": None})
            terminal = "qwen_error"
            break
        call = proposer.calls[-1] if len(getattr(proposer, "calls", [])) > prompt_before else {}
        prompt = call.get("prompt") if isinstance(call, dict) else None
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if isinstance(prompt, str) else None
        if qwen_result.get("type") in {"STOP", "REVIEW"}:
            terminal = "qwen_stop" if qwen_result["type"] == "STOP" else "qwen_review"
            steps.append({"turn": turn + 1, "qwen": copy_json(qwen_result),
                          "qwen_input": qwen_input, "qwen_prompt_sha256": prompt_hash,
                          "guard_request": None, "guard_assessment": None,
                          "host_result": {"final_disposition": qwen_result["type"], "effect": None}})
            break
        if qwen_result.get("type") != "PROPOSE" or not isinstance(qwen_result.get("proposal"), dict):
            terminal = "qwen_error"
            steps.append({"turn": turn + 1, "qwen": copy_json(qwen_result),
                          "qwen_input": qwen_input, "qwen_prompt_sha256": prompt_hash,
                          "guard_request": None, "guard_assessment": None,
                          "qwen_error": {"stage": "proposal_protocol", "code": "invalid_qwen_result"},
                          "host_result": None})
            break
        proposal = copy_json(qwen_result["proposal"])
        host_result = host.evaluate(proposal)
        host_record = host_result.as_dict()
        assessment = host_record.get("classifier_assessment")
        evidence = assessment.get("evidence", {}) if isinstance(assessment, dict) else {}
        guard_request = evidence.get("request") if isinstance(evidence, dict) else None
        steps.append({"turn": turn + 1, "qwen": copy_json(qwen_result),
                      "qwen_input": qwen_input, "qwen_prompt_sha256": prompt_hash,
                      "proposal": proposal, "guard_request": guard_request,
                      "request_sha256": host_record.get("request_sha256"),
                      "guard_assessment": assessment,
                      "assessment_snapshot_sha256": host_record.get("assessment_snapshot_sha256"),
                      "host_result": host_record})
        if host_result.final_disposition == "REVIEW":
            terminal = "guard_review"
            break
    if terminal == "step_limit" and host.operations >= policy.get("max_operations", 0):
        terminal = "operation_budget"
    return {"schema": "sequence_host_trace/1", "case_id": case.get("case_id"),
            "steps": steps, "terminal": terminal, "steps_completed": len(steps),
            "max_steps": max_steps, "executed_operations": host.operations,
            "final_files": copy_json(host.files)}

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
