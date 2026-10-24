"""H17 action-ID proposal generation using the existing portable Qwen loader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class QwenProposalError(RuntimeError):
    pass


class QwenProposer:
    """Resolve an H17 ACT action ID to an exact tool call from a public menu."""

    def __init__(self, adapter_dir: str | Path, *, device: str = "cuda",
                 cache_root: str | Path | None = None, max_new_tokens: int = 128,
                 base_model_path: str | Path | None = None):
        adapter = Path(adapter_dir).resolve(strict=True)
        cfg_path = adapter / "adapter_config.json"
        if not cfg_path.is_file() or not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError("H17 adapter files are missing")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        configured_base = cfg.get("base_model_name_or_path")
        if not isinstance(configured_base, str) or not configured_base:
            raise ValueError("adapter does not identify its pinned base snapshot")
        # adapter_config.json may have been authored on Windows even when a
        # local copy is used from another OS.
        self.revision = Path(configured_base.replace("\\", "/")).name
        base = (Path(base_model_path).expanduser() if base_model_path is not None
                else Path(configured_base))
        if not base.is_dir() or not (base / "config.json").is_file():
            raise ValueError("base model must be an existing local model snapshot with config.json")
        # The adapter is tied to the immutable HF snapshot revision encoded by
        # its original cache directory. Accept a relocated cache only when its
        # snapshot directory still carries that same revision identifier.
        if base.name != self.revision:
            raise ValueError(f"base model snapshot revision mismatch: expected {self.revision}")
        self.base_snapshot = str(base.resolve())
        self.adapter_dir = str(adapter)
        self.device = device
        self.adapter_sha256 = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        self.adapter_config_sha256 = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        from decision_factory.recovery.computer_safety import load_text_tool_model
        self.callback = load_text_tool_model(
            "Qwen/Qwen3.5-2B", self.revision, adapter_dir=adapter, device=device,
            cache_root=cache_root or base, max_new_tokens=max_new_tokens)
        self.calls: list[dict[str, Any]] = []

    @property
    def provenance(self) -> dict[str, Any]:
        return {"model_id": "Qwen/Qwen3.5-2B", "revision": self.revision,
                "base_snapshot": self.base_snapshot, "adapter_dir": self.adapter_dir,
                "adapter_sha256": self.adapter_sha256,
                "adapter_config_sha256": self.adapter_config_sha256,
                "device": self.device}

    def propose(self, task_input: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(task_input, Mapping):
            raise TypeError("task_input must be an object")
        request = task_input.get("user_request")
        menu = task_input.get("action_menu")
        observations = task_input.get("observations", [])
        if (not isinstance(request, Mapping) or not isinstance(request.get("text"), str)
                or request.get("trust") != "trusted"):
            raise ValueError("task_input requires user_request.text")
        if not isinstance(menu, list) or not menu:
            raise ValueError("task_input requires a nonempty action_menu")
        from decision_factory.recovery.computer_safety import TextToolCase, ToolAction
        actions = []
        action_ids: set[str] = set()
        for row in menu:
            if not isinstance(row, Mapping):
                raise ValueError("action_menu rows must be objects")
            action_id = row.get("action_id")
            description = row.get("description")
            operation = row.get("operation", row.get("tool_name"))
            arguments = row.get("arguments")
            if (not isinstance(action_id, str) or not isinstance(description, str)
                    or not isinstance(operation, str) or not isinstance(arguments, Mapping)):
                raise ValueError("action_menu row requires action_id, description, operation, arguments")
            if action_id in action_ids:
                raise ValueError("action_menu action_id values must be unique")
            action_ids.add(action_id)
            actions.append(ToolAction(action_id, description, operation, dict(arguments)))
        if not isinstance(observations, list):
            raise ValueError("observations must be a list")
        policy = task_input.get("host_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("task_input requires host_policy")
        files = task_input.get("initial_files", {})
        if not isinstance(files, Mapping) or not all(
                isinstance(path, str) and isinstance(content, str)
                for path, content in files.items()):
            raise ValueError("initial_files must map virtual paths to text")
        observations = list(observations) + [
            {"source": "host-policy", "trust": "trusted",
             "text": "Host-enforced virtual workspace policy: " + __import__("json").dumps(
                 dict(policy), ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
        ]
        # Successful read results are rendered once in observations. Preserve a
        # path index without copying the same untrusted bytes into a second block.
        observation_texts = [item.get("text", "") for item in observations
                             if isinstance(item, Mapping) and isinstance(item.get("text"), str)]
        indexed_files: dict[str, str] = {}
        for path, content in files.items():
            repeated = (any(content and content in text for text in observation_texts)
                        or (content == "" and any(
                            isinstance(item, Mapping) and item.get("source") in {"virtual-file", "virtual-files"}
                            for item in observations)))
            indexed_files[path] = ("[contents shown once in the untrusted read history]"
                                   if repeated else content)
        if indexed_files:
            observations.append({"source": "virtual-file-index", "trust": "untrusted",
                                 "text": "Previously read virtual file index (contents may be shown above): "
                                         + __import__("json").dumps(
                                             indexed_files, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":"))})
        case = TextToolCase(request["text"], actions, observations,
                            str(task_input.get("prompt_version", "bev-qwen-runtime-v1")))
        from decision_factory.recovery.computer_safety import render_text_tool_prompt
        from decision_factory.recovery.core import _strict_decision
        prompt = render_text_tool_prompt(case, observations)
        record: dict[str, Any] = {"prompt": prompt, "raw_output": None,
                                  "decision": None, "proposal": None, "error": None}
        self.calls.append(record)
        try:
            raw = self.callback(prompt)
            record["raw_output"] = raw
            generation_audit = None
            audit_getter = getattr(self.callback, "get_model_calls", None)
            if callable(audit_getter):
                model_calls = audit_getter()
                if model_calls:
                    generation_audit = model_calls[-1]
            decision, error = _strict_decision(raw)
            record["decision"] = decision
            if decision is None:
                raise ValueError(f"H17 output failed strict decision parsing: {error}")
            if decision["type"] == "STOP":
                return {"type": "STOP", "raw_output": raw, "decision": decision,
                        "generation_audit": generation_audit,
                        "model_provenance": self.provenance}
            if decision["type"] == "REVIEW":
                return {"type": "REVIEW", "raw_output": raw, "decision": decision,
                        "generation_audit": generation_audit,
                        "model_provenance": self.provenance}
            by_id = {action.action_id: action for action in actions}
            action = by_id.get(decision.get("action"))
            if action is None:
                raise ValueError("H17 selected an action ID outside the public menu")
            proposal = {"tool_name": action.operation, "arguments": dict(action.arguments)}
            record["proposal"] = proposal
            return {"type": "PROPOSE", "proposal": proposal,
                    "action_id": action.action_id, "raw_output": raw,
                    "decision": decision, "generation_audit": generation_audit,
                    "model_provenance": self.provenance}
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise QwenProposalError("H17 Qwen proposal failed closed") from exc

