"""Focused host-boundary regressions for Stage B; never loads model weights."""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import io
import json
import sys
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_factory.runtime_guard import bev, contract_reference  # noqa: E402
from decision_factory.runtime_guard.host import GuardHost  # noqa: E402
from decision_factory.runtime_guard.qwen import QwenProposer  # noqa: E402
from decision_factory.runtime_guard.sequence import (  # noqa: E402
    evaluate_routed_proposal,
    evaluate_sequence_proposal,
)
from decision_factory.runtime_guard.sequence import sequence_state  # noqa: E402
from decision_factory.runtime_guard.sequence_host import (  # noqa: E402
    qwen_observations,
    qwen_visible_files,
    run_sequence_episode,
)


FIXTURES = ROOT / "tests" / "contract_fixtures.jsonl"
CONTRACT_TEXT = (ROOT / "docs" / "PUBLIC_DECISION_SEMANTICS.txt").read_text(encoding="utf-8").strip()


def _fixture(fixture_id: str) -> dict:
    for line in FIXTURES.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["fixture_id"] == fixture_id:
            return copy.deepcopy(row["guard_request"])
    raise AssertionError(f"missing frozen fixture {fixture_id}")


def _host_request(*, prior_actions: list[dict] | None = None) -> dict:
    return {
        "schema": "bev_guard_request/1",
        "user_request": {"text": "Read /workspace/report.txt", "trust": "trusted"},
        "host_policy": {
            "decision_contract": CONTRACT_TEXT,
            "allowed_operations": ["read_file"],
            "max_operations": 2,
            "max_write_bytes": 1024,
            "workspace_root": "/workspace",
            "tool_schemas": {"read_file": {"required": ["path"], "properties": {"path": "virtual path"}}},
            "workflow_rules": {"rule_type": "recipient_allowlist",
                               "missing_evidence_disposition": "REVIEW",
                               "recipient_directory_status": "verified",
                               "approved_recipients": ["archive-good"]},
        },
        "prior_actions": copy.deepcopy(prior_actions or []),
        "observations": [],
        "proposal": {"tool_name": "read_file", "arguments": {"path": "/workspace/report.txt"}},
    }


def _assessment(request: dict, *, disposition: str = "PERMIT", request_sha: str | None = None) -> dict:
    serialized = json.dumps(request, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = request_sha or hashlib.sha256(serialized).hexdigest()
    return {
        "schema": bev.ASSESSMENT_SCHEMA,
        "status": "assessed",
        "disposition": disposition,
        "scores": {"PERMIT": 1.0, "BLOCK": 0.0, "REVIEW": 0.0},
        "selected": disposition,
        "evidence": {"request": copy.deepcopy(request), "request_sha256": digest,
                     "requested_device": "cuda", "effective_device": "cuda"},
    }


class StageBRuntimeContractTests(unittest.TestCase):
    def test_sequence_api_defaults_to_cuda_and_cpu_fails_closed_with_bound_evidence(self) -> None:
        self.assertEqual(inspect.signature(evaluate_sequence_proposal).parameters["device"].default, "cuda")
        self.assertEqual(inspect.signature(evaluate_routed_proposal).parameters["sequence_device"].default, "cuda")
        request = _fixture("approval_complete_empty_active")
        for result in (
            evaluate_sequence_proposal(request, device="cpu"),
            evaluate_routed_proposal(request, sequence_device="cpu"),
        ):
            self.assertEqual(result["status"], "review")
            self.assertEqual(result["disposition"], "REVIEW")
            evidence = result["evidence"]
            self.assertEqual(evidence["requested_device"], "cpu")
            self.assertIsNone(evidence["effective_device"])
            self.assertEqual(evidence["failure_stage"], "device_validation")
            self.assertEqual(evidence["error_code"], "unsupported_device")
            self.assertEqual(evidence["request"], request)
            canonical = json.dumps(request, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.assertEqual(evidence["request_sha256"], hashlib.sha256(canonical).hexdigest())

    def test_successful_read_is_rendered_once_as_untrusted_and_only_from_history(self) -> None:
        path = "/workspace/returned.txt"
        marker = "READ_CONTENT_MARKER_4827"
        prior = [{
            "tool_name": "read_file", "arguments": {"path": path}, "host_disposition": "ALLOW",
            "observation": {"source": "virtual-file", "trust": "untrusted", "text": marker},
        }, {
            "tool_name": "read_file", "arguments": {"path": "/workspace/blocked.txt"},
            "host_disposition": "BLOCK",
        }]
        self.assertEqual(qwen_visible_files(prior), {path: marker})
        observations = qwen_observations([], prior)
        self.assertEqual(sum(item.get("text") == marker for item in observations), 1)

        prompt_capture: list[str] = []
        proposer = object.__new__(QwenProposer)
        proposer.revision = "stub-revision"
        proposer.base_snapshot = "stub-snapshot"
        proposer.adapter_dir = "stub-adapter"
        proposer.adapter_sha256 = "stub-hash"
        proposer.adapter_config_sha256 = "stub-config-hash"
        proposer.device = "cpu"
        proposer.calls = []
        proposer.callback = lambda prompt: prompt_capture.append(prompt) or '{"type":"STOP"}'
        task = {
            "user_request": {"text": "Summarize the read file.", "trust": "trusted"},
            "host_policy": {"allowed_operations": ["read_file"]},
            "action_menu": [{"action_id": "stop", "description": "Finish", "operation": "read_file",
                             "arguments": {"path": path}}],
            "observations": observations,
            "initial_files": qwen_visible_files(prior),
        }
        proposer.propose(task)
        prompt = prompt_capture[0]
        self.assertEqual(prompt.count(marker), 1)
        self.assertIn("trust=untrusted", prompt)
        self.assertIn(path, prompt)
        self.assertNotIn("FUTURE_FILE_MARKER", prompt)

        # Exercise the real episode host path with the model callback stubbed.
        turn_prompts: list[str] = []
        markers = ["READ_CONTENT_MARKER_A", "READ_CONTENT_MARKER_B", "READ_CONTENT_MARKER_C"]
        paths = ["/workspace/read-a.txt", "/workspace/read-b.txt", "/workspace/read-c.txt"]

        def two_turn_callback(prompt: str) -> str:
            turn_prompts.append(prompt)
            if len(turn_prompts) <= len(markers):
                return json.dumps({"type": "ACT", "action": f"read-{len(turn_prompts)}"})
            return '{"type":"STOP"}'

        loop_proposer = object.__new__(QwenProposer)
        loop_proposer.revision = "stub-revision"
        loop_proposer.base_snapshot = "stub-snapshot"
        loop_proposer.adapter_dir = "stub-adapter"
        loop_proposer.adapter_sha256 = "stub-hash"
        loop_proposer.adapter_config_sha256 = "stub-config-hash"
        loop_proposer.device = "cpu"
        loop_proposer.calls = []
        loop_proposer.callback = two_turn_callback
        file_state = dict(zip(paths, markers))
        file_state["/workspace/not-yet-read.txt"] = "FUTURE_FILE_MARKER"
        loop_request = _host_request()
        loop_request["host_policy"]["max_operations"] = 4
        loop_host = GuardHost(
            request=loop_request, initial_files=file_state,
            mode="qwen-bev", bev_evaluator=lambda req, **kwargs: _assessment(req),
            bev_device="cuda",
        )
        menus = [[{"action_id": f"read-{index + 1}", "description": "Read a file",
                   "operation": "read_file", "arguments": {"path": file_path}}]
                 for index, file_path in enumerate(paths)]
        menus.append([{"action_id": "done", "description": "Finish", "operation": "read_file",
                       "arguments": {"path": paths[0]}}])
        loop_case = {
            "schema": "sequence_host_demo_case/1", "case_id": "one-read-rendering",
            "task": {"text": "Read and summarize the file.", "trust": "trusted"},
            "host_policy": loop_host.base_request["host_policy"],
            "action_menu": menus[0], "turn_action_menus": menus,
            "virtual_files": file_state,
        }
        trace = run_sequence_episode(loop_case, proposer=loop_proposer, host=loop_host, max_steps=4)
        self.assertEqual(trace["terminal"], "qwen_stop")
        self.assertEqual(trace["executed_operations"], 3)
        self.assertEqual(len(turn_prompts), 4)
        for turn, prompt in enumerate(turn_prompts[1:], start=1):
            for marker_index, value in enumerate(markers):
                expected_count = 1 if marker_index < turn else 0
                self.assertEqual(prompt.count(value), expected_count)
            self.assertIn("trust=untrusted", prompt)
            self.assertNotIn("FUTURE_FILE_MARKER", prompt)

        assessment_step = trace["steps"][2]
        assessment_request = assessment_step["guard_request"]
        self.assertEqual(assessment_request["proposal"], assessment_step["proposal"])
        assessment_model_state = sequence_state(assessment_request)
        serialized_state = json.dumps(assessment_model_state, ensure_ascii=False, sort_keys=True)
        self.assertEqual(serialized_state.count(markers[0]), 1)
        self.assertEqual(serialized_state.count(markers[1]), 1)
        self.assertEqual(serialized_state.count(markers[2]), 0)
        self.assertNotIn("FUTURE_FILE_MARKER", serialized_state)
        # The internal host audit retains the tool receipt while the Bev state
        # contains each read result once in its untrusted context observations.
        self.assertEqual(loop_host.prior_actions[0]["observation"]["text"], markers[0])
        independent_hash = hashlib.sha256(json.dumps(
            assessment_request, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        self.assertEqual(assessment_step["request_sha256"], independent_hash)
        self.assertEqual(assessment_step["assessment_snapshot_sha256"], independent_hash)

    def test_host_rejects_assessment_for_changed_snapshot_before_any_effect(self) -> None:
        host_box: dict[str, GuardHost] = {}

        def assessor(request: dict, **kwargs) -> dict:
            assessment = _assessment(request)
            host_box["host"].prior_actions.append({
                "tool_name": "read_file", "arguments": {"path": "/workspace/earlier.txt"},
                "host_disposition": "ALLOW",
            })
            return assessment

        host = GuardHost(request=_host_request(), initial_files={"/workspace/report.txt": "contents"},
                         mode="qwen-bev", bev_evaluator=assessor, bev_device="cuda")
        host_box["host"] = host
        result = host.evaluate(_host_request()["proposal"])
        self.assertEqual(result.final_disposition, "REVIEW")
        self.assertIsNone(result.effect)
        self.assertNotIn("ALLOW", [row["host_disposition"] for row in host.prior_actions])

        stale_host = GuardHost(
            request=_host_request(), initial_files={"/workspace/report.txt": "contents"},
            mode="qwen-bev",
            bev_evaluator=lambda req, **kwargs: _assessment(req, request_sha="0" * 64),
            bev_device="cuda",
        )
        stale = stale_host.evaluate(_host_request()["proposal"])
        self.assertEqual(stale.final_disposition, "REVIEW")
        self.assertIsNone(stale.effect)

    def test_host_does_not_use_independent_reference_as_a_permit_veto(self) -> None:
        host = GuardHost(
            request=_host_request(), initial_files={"/workspace/report.txt": "contents"},
            mode="qwen-bev", bev_evaluator=lambda request, **kwargs: _assessment(request),
            bev_device="cuda",
        )
        with patch.object(contract_reference, "decide_request", side_effect=AssertionError("reference called")) as scorer:
            result = host.evaluate(_host_request()["proposal"])
        scorer.assert_not_called()
        self.assertEqual(result.final_disposition, "PERMIT")
        self.assertIsNotNone(result.effect)

    def test_only_executed_allow_calls_consume_host_budget_and_steps_are_separate(self) -> None:
        request = _host_request()
        request["host_policy"]["max_operations"] = 1
        review_host = GuardHost(
            request=request, initial_files={}, mode="qwen-bev",
            bev_evaluator=lambda req, **kwargs: _assessment(req, disposition="REVIEW"),
            bev_device="cuda",
        )
        result = review_host.evaluate(request["proposal"])
        self.assertEqual(result.final_disposition, "REVIEW")
        self.assertEqual(review_host.operations, 0)

        assessor_calls = []
        missing_read_host = GuardHost(
            request=request, initial_files={}, mode="qwen-bev",
            bev_evaluator=lambda req, **kwargs: (assessor_calls.append(req) or _assessment(req)),
            bev_device="cuda",
        )
        missing_read = missing_read_host.evaluate(request["proposal"])
        self.assertEqual(missing_read.final_disposition, "PERMIT")
        self.assertEqual(missing_read.effect["found"], False)
        self.assertEqual(missing_read.observation["trust"], "untrusted")
        self.assertEqual(missing_read_host.operations, 1)
        second_read = missing_read_host.evaluate(request["proposal"])
        self.assertEqual(second_read.final_disposition, "BLOCK")
        self.assertEqual(len(assessor_calls), 1)
        self.assertEqual(missing_read_host.operations, 1)

        class StubProposer:
            def __init__(self):
                self.calls = []

            def propose(self, task_input):
                self.calls.append(copy.deepcopy(task_input))
                return {"type": "PROPOSE", "proposal": {
                    "tool_name": "read_file", "arguments": {"path": "/workspace/../escape.txt"}
                }}

        host = GuardHost(request=request, initial_files={}, mode="qwen-bev",
                         bev_evaluator=lambda req, **kwargs: _assessment(req), bev_device="cuda")
        proposer = StubProposer()
        case = {
            "schema": "sequence_host_demo_case/1", "case_id": "steps-are-not-effects",
            "task": request["user_request"], "host_policy": request["host_policy"],
            "action_menu": [{"action_id": "bad-read", "description": "Try the invalid path",
                             "operation": "read_file", "arguments": {"path": "/workspace/../escape.txt"}}],
            "virtual_files": {"/workspace/future.txt": "FUTURE_FILE_MARKER"},
            "evaluator": {"expected": "hidden"},
        }
        trace = run_sequence_episode(case, proposer=proposer, host=host, max_steps=2)
        self.assertEqual(trace["steps_completed"], 2)
        self.assertEqual(trace["max_steps"], 2)
        self.assertEqual(trace["executed_operations"], 0)
        self.assertEqual(len(proposer.calls), 2)
        self.assertNotIn("FUTURE_FILE_MARKER", json.dumps(proposer.calls))
        self.assertNotIn("evaluator", json.dumps(proposer.calls))

    def test_actual_cli_scripts_point_at_closed_loop_and_explicit_legacy_command(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = project["project"]["scripts"]
        self.assertEqual(scripts["sequence-guard"], "decision_factory.runtime_guard.sequence_cli:main")
        self.assertIn("sequence-guard-assess-legacy", scripts)
        self.assertEqual(scripts["sequence-guard-demo"], scripts["sequence-guard-assess-legacy"])
        self.assertEqual(scripts["sequence-guard-host-demo"], scripts["sequence-guard"])
        self.assertNotEqual(scripts["sequence-guard-assess-legacy"], scripts["sequence-guard"])
        for entrypoint in (scripts["sequence-guard"], scripts["sequence-guard-assess-legacy"]):
            module_name, function_name = entrypoint.split(":", maxsplit=1)
            self.assertTrue(callable(getattr(importlib.import_module(module_name), function_name)))
        cli = importlib.import_module("decision_factory.runtime_guard.sequence_cli")
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as exit_info:
            cli.main(["--help"])
        self.assertEqual(exit_info.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("--max-steps", help_text)
        self.assertIn("--qwen-device {cuda,cpu}", help_text)
        self.assertIn("--bev-device {cuda}", help_text)


if __name__ == "__main__":
    unittest.main()
