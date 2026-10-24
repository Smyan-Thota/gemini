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
from pathlib import Path
from typing import Any, Callable, Mapping


_OPS = {"read_file", "write_file", "delete_file", "copy_file", "upload_file"}
_DISPOSITIONS = {"PERMIT", "BLOCK", "REVIEW"}
_PUBLIC_CONTRACT_PATH = Path(__file__).resolve().parents[3] / "docs" / "PUBLIC_DECISION_SEMANTICS.txt"


def public_decision_contract() -> str:
    text = _PUBLIC_CONTRACT_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("public decision contract is empty")
    return text


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
    if not isinstance(allowed, list) or any(not isinstance(x, str) for x in allowed):
        return "review_invalid_allowed_operations", None
    if operation not in allowed:
        return "operation_not_allowed_by_policy", None
    if not isinstance(args, Mapping):
        return "arguments_not_object", None
    schemas = policy.get("tool_schemas")
    schema = schemas.get(operation) if isinstance(schemas, Mapping) else None
    if (not isinstance(schema, Mapping) or not isinstance(schema.get("required"), list)
            or any(not isinstance(x, str) for x in schema.get("required", []))
            or not isinstance(schema.get("properties"), Mapping)):
        return "review_missing_or_malformed_tool_schema", None
    if policy.get("workspace_root") != "/workspace":
        return "review_missing_or_malformed_workspace_root", None
    if not set(schema["required"]).issubset(args) or not set(args).issubset(schema["properties"]):
        return "unexpected_or_missing_arguments", None
    max_operations = policy.get("max_operations")
    if type(max_operations) is not int or max_operations < 1:
        return "review_missing_or_malformed_operation_budget", None
    try:
        if operation in {"read_file", "write_file", "delete_file"}:
            expected_keys = {"path", "content"} if operation == "write_file" else {"path"}
            if set(args) != expected_keys:
                return "unexpected_or_missing_arguments", None
            path = _normalized_virtual_path(args.get("path"))
            protected = policy.get("protected_paths", [])
            if not isinstance(protected, list):
                return "review_invalid_protected_paths_policy", path
            protected_paths = {_normalized_virtual_path(item) for item in protected}
            if operation in {"write_file", "delete_file"} and path in protected_paths:
                return "protected_target", path
            if operation == "delete_file" and path not in files:
                return "delete_target_missing", path
            if operation == "write_file":
                content = args.get("content")
                limit = policy.get("max_write_bytes")
                if type(limit) is not int or limit < 0:
                    return "review_invalid_write_size_policy", path
                if not isinstance(content, str):
                    return "invalid_write_arguments", path
                if len(content.encode("utf-8")) > limit:
                    return "write_size_limit", path
            return None, path
        if operation == "copy_file" and set(args) != {"source", "destination"}:
            return "unexpected_or_missing_arguments", None
        if operation == "upload_file":
            path = _normalized_virtual_path(args.get("path"))
            destination = args.get("destination")
            if not isinstance(destination, str) or not destination:
                return "invalid_upload_destination", path
            units = args.get("data_units", 1)
            fields = args.get("data_fields", [])
            if type(units) is not int or units <= 0 or not isinstance(fields, list) or not all(
                    isinstance(item, str) for item in fields):
                return "invalid_upload_effect", path
            return None, path
        source = _normalized_virtual_path(args.get("source"))
        destination = _normalized_virtual_path(args.get("destination"))
        protected = policy.get("protected_paths", [])
        if not isinstance(protected, list):
            return "review_invalid_protected_paths_policy", destination
        protected_paths = {_normalized_virtual_path(item) for item in protected}
        if destination in protected_paths:
            return "protected_target", destination
        if source not in files:
            return "copy_source_missing", source
        if source == destination:
            return "copy_same_path", destination
        limit = policy.get("max_write_bytes")
        if type(limit) is not int or limit < 0:
            return "review_invalid_write_size_policy", destination
        if len(files[source].encode("utf-8")) > limit:
            return "write_size_limit", destination
        return None, destination
    except ValueError:
        return "invalid_or_escaping_path", None


def build_sequence_assessment_request(request: Mapping[str, Any], proposal: Mapping[str, Any], *,
                                      prior_actions: list[dict[str, Any]],
                                      observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct the exact public sequence request used by host, guard, and trace."""
    built = copy.deepcopy(dict(request))
    policy = built.get("host_policy")
    if not isinstance(policy, dict):
        raise ValueError("host_policy must be an object")
    policy["decision_contract"] = public_decision_contract()
    # Keep each actual observation in one model-visible location. Read results
    # belong in observations (as untrusted text); effect receipts remain nested
    # in their action record because the public history contract binds them to
    # the exact executed upload. The host retains its richer internal history.
    action_records = copy.deepcopy(prior_actions)
    nested_effect_receipts = {
        _canonical_json(action["observation"])
        for action in action_records
        if (isinstance(action, Mapping) and action.get("tool_name") != "read_file"
            and isinstance(action.get("observation"), Mapping))
    }
    model_observations = [copy.deepcopy(item) for item in observations
                          if not isinstance(item, Mapping)
                          or _canonical_json(item) not in nested_effect_receipts]
    for action in action_records:
        if action.get("tool_name") == "read_file":
            action.pop("observation", None)
    built["prior_actions"] = action_records
    built["observations"] = model_observations
    built["proposal"] = copy.deepcopy(dict(proposal))
    return built


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
    request_sha256: str | None = None
    assessment_snapshot_sha256: str | None = None
    assessment_error: dict[str, Any] | None = None
    profile: str = "sequence"
    requested_device: str | None = None
    effective_device: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal": copy.deepcopy(self.proposal),
            "classifier_assessment": copy.deepcopy(self.classifier_assessment),
            "final_disposition": self.final_disposition,
            "host_reason": self.host_reason,
            "effect": copy.deepcopy(self.effect),
            "observation": copy.deepcopy(self.observation),
            "request_sha256": self.request_sha256,
            "assessment_snapshot_sha256": self.assessment_snapshot_sha256,
            "assessment_error": copy.deepcopy(self.assessment_error),
            "profile": self.profile,
            "requested_device": self.requested_device,
            "effective_device": self.effective_device,
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
    """Assess one exact proposal and apply at most one in-memory virtual effect."""

    def __init__(self, *, request: Mapping[str, Any], initial_files: Mapping[str, str],
                 mode: str, bev_evaluator: Callable[..., dict[str, Any]] | None = None,
                 simple_guard: SimpleRuleGuard | None = None,
                 bev_head_checkpoint: str | None = None,
                 bev_device: str = "cuda", profile: str = "sequence"):
        if mode not in {"qwen-alone", "rules", "qwen-bev"}:
            raise ValueError("mode must be qwen-alone, rules, or qwen-bev")
        if profile not in {"sequence", "legacy_v4_cpu"}:
            raise ValueError("profile must be sequence or legacy_v4_cpu")
        if profile == "sequence" and bev_device != "cuda":
            raise ValueError("the sequence Bev profile requires CUDA FP32")
        if profile == "legacy_v4_cpu" and bev_device != "cpu":
            raise ValueError("the legacy V4 profile is CPU-only")
        self.base_request = copy.deepcopy(dict(request))
        if profile == "sequence":
            policy_copy = self.base_request.get("host_policy")
            if not isinstance(policy_copy, dict):
                raise ValueError("host_policy must be an object")
            policy_copy["decision_contract"] = public_decision_contract()
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
        self.observations = copy.deepcopy(self.base_request.get("observations", []))
        self.mode = mode
        self.profile = profile
        self.bev_device = bev_device
        self.bev_evaluator = bev_evaluator
        self.bev_head_checkpoint = bev_head_checkpoint
        self.simple_guard = simple_guard or SimpleRuleGuard()
        prior = self.base_request.get("prior_actions", [])
        if not isinstance(prior, list):
            raise ValueError("prior_actions must be a list")
        self.prior_actions = copy.deepcopy(prior)
        self.operations = sum(1 for item in prior
                              if isinstance(item, Mapping)
                              and item.get("host_disposition") == "ALLOW")

    def evaluate(self, proposal: Mapping[str, Any]) -> ProposalResult:
        exact = copy.deepcopy(dict(proposal)) if isinstance(proposal, Mapping) else {"raw": repr(proposal)}
        classifier: dict[str, Any] | None = None
        bound_request: dict[str, Any] | None = None
        request_digest: str | None = None
        assessment_error: dict[str, Any] | None = None
        try:
            if self.profile == "sequence":
                bound_request = build_sequence_assessment_request(
                    self.base_request, exact, prior_actions=self.prior_actions,
                    observations=self.observations)
            else:
                bound_request = build_host_assessment_request(
                    self.base_request, self.initial_files, exact,
                    prior_actions=self.prior_actions)
            request_digest = hashlib.sha256(_canonical_json(bound_request).encode("utf-8")).hexdigest()
        except Exception as exc:
            assessment_error = {"failure_stage": "request_construction",
                                "error_code": "request_construction_failed",
                                "error": f"{type(exc).__name__}: {exc}"}
        reason, _ = validate_proposal(exact, self.policy, self.files)
        limit = self.policy.get("max_operations")
        if reason is not None and reason.startswith("review_"):
            final = "REVIEW"
        elif reason is not None:
            final = "BLOCK"
        elif type(limit) is not int or limit < 1:
            reason, final = "review_missing_or_malformed_operation_budget", "REVIEW"
        elif self.operations >= limit:
            reason, final = "operation_limit", "BLOCK"
        elif self.mode == "qwen-alone":
            final = "PERMIT"
        elif self.mode == "rules":
            final = self.simple_guard.evaluate(self.base_request, exact)
        else:
            if self.bev_evaluator is None:
                classifier = {"schema": "bev_guard_assessment/1", "status": "review",
                              "disposition": "REVIEW", "scores": {"PERMIT": None,
                              "BLOCK": None, "REVIEW": None}, "selected": None,
                              "evidence": {"request": bound_request,
                                           "request_sha256": request_digest,
                                           "requested_device": self.bev_device,
                                           "effective_device": None,
                                           "error": "bev_evaluator_unavailable",
                                           "error_code": "assessor_unavailable",
                                           "failure_stage": "assessment"}}
                final = "REVIEW"
            else:
                state_before = self._snapshot_state()
                try:
                    if bound_request is None or request_digest is None:
                        raise ValueError("could not construct a canonical request for assessment")
                    req = bound_request
                    kwargs: dict[str, Any] = {}
                    if self.bev_head_checkpoint is not None:
                        kwargs["head_checkpoint"] = self.bev_head_checkpoint
                    if self.profile == "sequence":
                        kwargs["device"] = self.bev_device
                    classifier = self.bev_evaluator(req, **kwargs)
                    changed_state = self._state_digest() != state_before["digest"]
                    if changed_state:
                        self._restore_state(state_before)
                        reason = "assessment_snapshot_state_changed"
                        final = "REVIEW"
                    else:
                        evidence = classifier.get("evidence", {}) if isinstance(classifier, Mapping) else {}
                        echoed = evidence.get("request") if isinstance(evidence, Mapping) else None
                        echoed_hash = evidence.get("request_sha256") if isinstance(evidence, Mapping) else None
                        if echoed_hash != request_digest or not isinstance(echoed, Mapping) or _canonical_json(echoed) != _canonical_json(req):
                            reason = "assessment_request_binding_mismatch"
                        elif self._assessment_request_hash(exact) != request_digest:
                            reason = "assessment_snapshot_changed_before_execution"
                            final = "REVIEW"
                        else:
                            candidate = classifier.get("disposition") if isinstance(classifier, Mapping) else None
                            status = classifier.get("status") if isinstance(classifier, Mapping) else None
                            if status == "assessed" and candidate in _DISPOSITIONS:
                                final = candidate
                            else:
                                final = "REVIEW"
                except Exception as exc:
                    classifier = {"schema": "bev_guard_assessment/1", "status": "review",
                                  "disposition": "REVIEW", "scores": {"PERMIT": None,
                                  "BLOCK": None, "REVIEW": None}, "selected": None,
                                  "evidence": {"error": f"{type(exc).__name__}: {exc}",
                                               "error_code": "assessor_exception",
                                               "failure_stage": "assessment",
                                               "request": bound_request,
                                               "request_sha256": request_digest,
                                               "requested_device": self.bev_device,
                                               "effective_device": None}}
                    final = "REVIEW"
            if reason in {"assessment_snapshot_state_changed", "assessment_request_binding_mismatch",
                          "assessment_snapshot_changed_before_execution"}:
                final = "REVIEW"
        effect, observation = None, None
        if final == "PERMIT":
            # Revalidate bounds and assessment binding immediately before mutation.
            reason, path = validate_proposal(exact, self.policy, self.files)
            if reason is not None or self.operations >= limit:
                final = "REVIEW" if reason and reason.startswith("review_") else "BLOCK"
                reason = reason or "operation_limit"
            else:
                operation, args = exact["tool_name"], exact["arguments"]
                if operation == "read_file":
                    if path in self.files:
                        content = self.files[path]
                        effect = {"kind": "read", "path": path, "found": True}
                        observation = {"source": "virtual-file", "trust": "untrusted", "text": content}
                    else:
                        effect = {"kind": "read", "path": path, "found": False}
                        observation = {"source": "virtual-file-not-found", "trust": "untrusted",
                                       "text": f"The virtual file {path} was not found."}
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
                elif operation == "upload_file":
                    receipt = {"status": "committed", "effect": copy.deepcopy(dict(args))}
                    effect = {"kind": "upload", **copy.deepcopy(dict(args))}
                    observation = {"source": "effect-ledger", "trust": "trusted",
                                   "text": _canonical_json(receipt)}
                self.operations += 1
        if (exact.get("tool_name") == "upload_file" and final in {"BLOCK", "REVIEW"}
                and observation is None):
            observation = {"source": "effect-ledger", "trust": "trusted",
                           "text": _canonical_json({"status": "blocked_no_effect",
                                                    "effect": copy.deepcopy(exact.get("arguments", {}))})}
        entry = {"tool_name": exact.get("tool_name", ""),
                 "arguments": copy.deepcopy(exact.get("arguments", {})),
                 "host_disposition": ("ALLOW" if final == "PERMIT" else final),
                 **({"observation": copy.deepcopy(observation)} if observation else {})}
        self.prior_actions.append(entry)
        if observation is not None:
            self.observations.append(copy.deepcopy(observation))
        evidence = classifier.get("evidence", {}) if isinstance(classifier, Mapping) else {}
        if assessment_error is None and reason is not None and classifier is None:
            assessment_error = {"failure_stage": "host_validation",
                                "error_code": reason,
                                "error": reason}
        if assessment_error is None and isinstance(evidence, Mapping) and evidence.get("error"):
            assessment_error = {"failure_stage": evidence.get("failure_stage"),
                                "error_code": evidence.get("error_code"),
                                "error": evidence.get("error")}
        return ProposalResult(exact, classifier, final, reason, effect, observation,
                              evidence.get("request_sha256") if isinstance(evidence, Mapping) else request_digest,
                              evidence.get("request_sha256") if isinstance(evidence, Mapping) else request_digest,
                              assessment_error, self.profile, self.bev_device,
                              evidence.get("effective_device") if isinstance(evidence, Mapping) else None)

    def _assessment_request_hash(self, proposal: Mapping[str, Any]) -> str:
        if self.profile == "sequence":
            request = build_sequence_assessment_request(
                self.base_request, proposal, prior_actions=self.prior_actions,
                observations=self.observations)
        else:
            request = build_host_assessment_request(self.base_request, self.initial_files,
                                                    proposal, prior_actions=self.prior_actions)
        return hashlib.sha256(_canonical_json(request).encode("utf-8")).hexdigest()

    def _state_digest(self) -> str:
        value = {"base_request": self.base_request, "files": self.files,
                 "prior_actions": self.prior_actions, "observations": self.observations,
                 "operations": self.operations}
        return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()

    def _snapshot_state(self) -> dict[str, Any]:
        return {"base_request": copy.deepcopy(self.base_request), "files": copy.deepcopy(self.files),
                "prior_actions": copy.deepcopy(self.prior_actions),
                "observations": copy.deepcopy(self.observations), "operations": self.operations,
                "digest": self._state_digest()}

    def _restore_state(self, snapshot: Mapping[str, Any]) -> None:
        self.base_request = copy.deepcopy(snapshot["base_request"])
        self.policy = self.base_request.get("host_policy", {})
        self.files = copy.deepcopy(snapshot["files"])
        self.prior_actions = copy.deepcopy(snapshot["prior_actions"])
        self.observations = copy.deepcopy(snapshot["observations"])
        self.operations = snapshot["operations"]

