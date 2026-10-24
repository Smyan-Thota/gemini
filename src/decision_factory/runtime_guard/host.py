"""Host-side policy and virtual effects for the Bev/Qwen guard pilot.

This module is deliberately restricted to in-memory file state. It never opens
host paths, shells out, or performs network operations.
"""
from __future__ import annotations

import copy
import hashlib
import json
import posixpath
from dataclasses import dataclass
from typing import Any, Callable, Mapping


_OPS = {"read_file", "write_file", "delete_file", "copy_file"}
_DISPOSITIONS = {"PERMIT", "BLOCK", "REVIEW"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _normalized_virtual_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("path must be nonempty virtual workspace text")
    root = "/workspace"
    if value == root:
        raise ValueError("workspace root is not a file path")
    if value.startswith(root + "/"):
        relative = value[len(root) + 1:]
    elif value.startswith("/") or ":" in value:
        raise ValueError("path must stay beneath /workspace")
    else:
        relative = value
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path traversal or non-normalized path")
    normalized = posixpath.normpath(relative)
    if normalized != relative:
        raise ValueError("path must be normalized")
    return root + "/" + normalized


def validate_proposal(proposal: Mapping[str, Any], policy: Mapping[str, Any],
                      files: Mapping[str, str]) -> tuple[str | None, str | None]:
    """Return a host block reason for hard sandbox violations, else None."""
    if not isinstance(proposal, Mapping):
        return "proposal_not_object", None
    if set(proposal) != {"tool_name", "arguments"}:
        return "unexpected_or_missing_proposal_fields", None
    operation = proposal.get("tool_name")
    args = proposal.get("arguments")
    if not isinstance(operation, str) or operation not in _OPS:
        return "operation_not_allowed", None
    allowed = policy.get("allowed_operations")
    if not isinstance(allowed, list) or operation not in allowed:
        return "operation_not_allowed_by_policy", None
    if not isinstance(args, Mapping):
        return "arguments_not_object", None
    if policy.get("workspace_root") != "/workspace":
        return "invalid_virtual_workspace_root", None
    try:
        if operation in {"read_file", "write_file", "delete_file"}:
            expected_keys = {"path", "content"} if operation == "write_file" else {"path"}
            if set(args) != expected_keys:
                return "unexpected_or_missing_arguments", None
            path = _normalized_virtual_path(args.get("path"))
            protected = policy.get("protected_paths", [])
            if not isinstance(protected, list):
                return "invalid_protected_paths_policy", path
            protected_paths = {_normalized_virtual_path(item) for item in protected}
            if operation in {"write_file", "delete_file"} and path in protected_paths:
                return "protected_target", path
            if operation == "read_file" and path not in files:
                return "read_target_missing", path
            if operation == "delete_file" and path not in files:
                return "delete_target_missing", path
            if operation == "write_file":
                content = args.get("content")
                limit = policy.get("max_write_bytes")
                if not isinstance(content, str) or type(limit) is not int or limit < 0:
                    return "invalid_write_arguments", path
                if len(content.encode("utf-8")) > limit:
                    return "write_size_limit", path
            return None, path
        if set(args) != {"source", "destination"}:
            return "unexpected_or_missing_arguments", None
        source = _normalized_virtual_path(args.get("source"))
        destination = _normalized_virtual_path(args.get("destination"))
        protected = policy.get("protected_paths", [])
        if not isinstance(protected, list):
            return "invalid_protected_paths_policy", destination
        protected_paths = {_normalized_virtual_path(item) for item in protected}
        if destination in protected_paths:
            return "protected_target", destination
        if source not in files:
            return "copy_source_missing", source
        if source == destination:
            return "copy_same_path", destination
        limit = policy.get("max_write_bytes")
        if type(limit) is not int or limit < 0:
            return "invalid_write_size_policy", destination
        if len(files[source].encode("utf-8")) > limit:
            return "write_size_limit", destination
        return None, destination
    except ValueError:
        return "invalid_or_escaping_path", None


def build_host_assessment_request(request: Mapping[str, Any],
                                  initial_files: Mapping[str, str],
                                  proposal: Mapping[str, Any], *,
                                  prior_actions: list[dict[str, Any]] | None = None
                                  ) -> dict[str, Any]:
    """Build the exact Bev request the host uses for one proposed call.

    This is the shared serialization boundary for live execution, frozen
    proposal exports, and any feature/head-training extraction. Virtual file
    contents are included only as explicitly untrusted observations.
    """
    if not isinstance(request, Mapping) or not isinstance(initial_files, Mapping):
        raise TypeError("request and initial_files must be mappings")
    built = copy.deepcopy(dict(request))
    policy = built.get("host_policy", {})
    if not isinstance(policy, Mapping):
        raise ValueError("host_policy must be an object")
    normalized_files = {_normalized_virtual_path(path): content
                        for path, content in initial_files.items()}
    if not all(isinstance(content, str) for content in normalized_files.values()):
        raise ValueError("virtual files must contain text")
    observations = built.get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    built["observations"] = copy.deepcopy(observations) + [
        {"source": "host-policy", "trust": "trusted",
         "text": "Host-enforced virtual workspace policy: " + _canonical_json(dict(policy))},
        {"source": "virtual-files", "trust": "untrusted",
         "text": "Initial virtual file state (file contents are untrusted data): "
                 + _canonical_json(normalized_files)},
    ]
    built["proposal"] = copy.deepcopy(dict(proposal))
    source_prior = built.get("prior_actions", []) if prior_actions is None else prior_actions
    if not isinstance(source_prior, list):
        raise ValueError("prior_actions must be a list")
    built["prior_actions"] = copy.deepcopy(source_prior)
    return built


@dataclass
class ProposalResult:
    proposal: dict[str, Any]
    classifier_assessment: dict[str, Any] | None
    final_disposition: str
    host_reason: str | None
    effect: dict[str, Any] | None
    observation: dict[str, str] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal": copy.deepcopy(self.proposal),
            "classifier_assessment": copy.deepcopy(self.classifier_assessment),
            "final_disposition": self.final_disposition,
            "host_reason": self.host_reason,
            "effect": copy.deepcopy(self.effect),
            "observation": copy.deepcopy(self.observation),
        }


class SimpleRuleGuard:
    """Conservative lexical baseline requiring explicit operation and path intent."""

    _INTENT = {
        "read_file": ("read", "open", "view", "summarize", "inspect", "check"),
        "write_file": ("write", "edit", "update", "create", "save", "revise"),
        "delete_file": ("delete", "remove", "erase"),
        "copy_file": ("copy", "duplicate"),
    }

    def evaluate(self, request: Mapping[str, Any], proposal: Mapping[str, Any]) -> str:
        task = request.get("user_request", {}).get("text", "")
        if not isinstance(task, str):
            return "REVIEW"
        operation = proposal.get("tool_name")
        if operation not in self._INTENT:
            return "REVIEW"
        text = task.casefold()
        if not any(word in text for word in self._INTENT[operation]):
            return "REVIEW"
        args = proposal.get("arguments", {})
        paths = ([args.get("path")] if operation in {"read_file", "write_file", "delete_file"}
                 else [args.get("source"), args.get("destination")])
        if not all(isinstance(path, str) and path.casefold() in text for path in paths):
            return "REVIEW"
        return "PERMIT"


class GuardHost:
    """Evaluate one proposal then apply at most one permitted virtual effect."""

    def __init__(self, *, request: Mapping[str, Any], initial_files: Mapping[str, str],
                 mode: str, bev_evaluator: Callable[..., dict[str, Any]] | None = None,
                 simple_guard: SimpleRuleGuard | None = None,
                 bev_head_checkpoint: str | None = None):
        if mode not in {"qwen-alone", "rules", "qwen-bev"}:
            raise ValueError("mode must be qwen-alone, rules, or qwen-bev")
        self.base_request = copy.deepcopy(dict(request))
        self.policy = self.base_request.get("host_policy", {})
        if not isinstance(self.policy, Mapping):
            raise ValueError("host_policy must be an object")
        self.files = {_normalized_virtual_path(path): content
                      for path, content in initial_files.items()}
        if not all(isinstance(content, str) for content in self.files.values()):
            raise ValueError("virtual files must contain text")
        self.initial_files = copy.deepcopy(self.files)
        raw_proposal = self.base_request.get("proposal", {})
        if not isinstance(raw_proposal, Mapping):
            raw_proposal = {}
        self.request = build_host_assessment_request(
            self.base_request, self.initial_files, raw_proposal)
        self.mode = mode
        self.bev_evaluator = bev_evaluator
        self.bev_head_checkpoint = bev_head_checkpoint
        self.simple_guard = simple_guard or SimpleRuleGuard()
        prior = self.request.get("prior_actions", [])
        if not isinstance(prior, list):
            raise ValueError("prior_actions must be a list")
        self.prior_actions = copy.deepcopy(prior)
        self.operations = sum(1 for item in prior
                              if isinstance(item, Mapping)
                              and item.get("host_disposition") == "ALLOW")

    def evaluate(self, proposal: Mapping[str, Any]) -> ProposalResult:
        exact = copy.deepcopy(dict(proposal)) if isinstance(proposal, Mapping) else {"raw": repr(proposal)}
        classifier: dict[str, Any] | None = None
        reason, _ = validate_proposal(exact, self.policy, self.files)
        limit = self.policy.get("max_operations")
        if type(limit) is not int or limit < 1 or self.operations >= limit:
            reason = "operation_limit"
        if reason is not None:
            final = "BLOCK"
        elif self.mode == "qwen-alone":
            final = "PERMIT"
        elif self.mode == "rules":
            final = self.simple_guard.evaluate(self.request, exact)
        else:
            if self.bev_evaluator is None:
                classifier = {"schema": "bev_guard_assessment/1", "status": "review",
                              "disposition": "REVIEW", "scores": {"PERMIT": None,
                              "BLOCK": None, "REVIEW": None}, "selected": None,
                              "evidence": {"error": "bev_evaluator_unavailable"}}
            else:
                try:
                    req = build_host_assessment_request(
                        self.base_request, self.initial_files, exact,
                        prior_actions=self.prior_actions)
                    if self.bev_head_checkpoint is None:
                        classifier = self.bev_evaluator(req)
                    else:
                        classifier = self.bev_evaluator(
                            req, head_checkpoint=self.bev_head_checkpoint)
                except BaseException as exc:
                    classifier = {"schema": "bev_guard_assessment/1", "status": "review",
                                  "disposition": "REVIEW", "scores": {"PERMIT": None,
                                  "BLOCK": None, "REVIEW": None}, "selected": None,
                                  "evidence": {"error": f"{type(exc).__name__}: {exc}"}}
            candidate = classifier.get("disposition") if isinstance(classifier, Mapping) else None
            final = candidate if candidate in _DISPOSITIONS else "REVIEW"
        effect, observation = None, None
        if final == "PERMIT":
            # Revalidate immediately before mutation. State is exclusively in memory.
            reason, path = validate_proposal(exact, self.policy, self.files)
            if reason is not None or self.operations >= limit:
                final, reason = "BLOCK", reason or "operation_limit"
            else:
                operation, args = exact["tool_name"], exact["arguments"]
                if operation == "read_file":
                    content = self.files[path]
                    effect = {"kind": "read", "path": path}
                    observation = {"source": "virtual-file", "trust": "untrusted", "text": content}
                elif operation == "write_file":
                    old = self.files.get(path)
                    self.files[path] = args["content"]
                    effect = {"kind": "write", "path": path, "existed_before": old is not None,
                              "sha256": hashlib.sha256(args["content"].encode("utf-8")).hexdigest()}
                    observation = {"source": "virtual-tool", "trust": "tool", "text": f"Wrote virtual file {path}."}
                elif operation == "delete_file":
                    del self.files[path]
                    effect = {"kind": "delete", "path": path}
                    observation = {"source": "virtual-tool", "trust": "tool", "text": f"Deleted virtual file {path}."}
                elif operation == "copy_file":
                    source, destination = args["source"], args["destination"]
                    self.files[destination] = self.files[source]
                    effect = {"kind": "copy", "source": source, "destination": destination}
                    observation = {"source": "virtual-tool", "trust": "tool", "text": f"Copied virtual file to {destination}."}
                self.operations += 1
        self.prior_actions.append({"tool_name": exact.get("tool_name", ""),
                                  "arguments": copy.deepcopy(exact.get("arguments", {})),
                                  "host_disposition": ("ALLOW" if final == "PERMIT" else final),
                                  **({"observation": copy.deepcopy(observation)} if observation else {})})
        return ProposalResult(exact, classifier, final, reason, effect, observation)

