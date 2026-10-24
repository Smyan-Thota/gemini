"""CPU regressions for the shared training and assessment input builder."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_factory.runtime_guard import bev, sequence_candidate, sequence_v2  # noqa: E402
from decision_factory.runtime_guard.contract_reference import decide_request  # noqa: E402
from decision_factory.runtime_guard.host import public_decision_contract  # noqa: E402
from decision_factory.training.trainer import _tokenize_rows  # noqa: E402


class _FakeTensor:
    dtype = "fake-u8"

    def __init__(self, value: bytes, shape: tuple[int, ...]):
        self.value = value
        self.shape = shape

    def detach(self):
        return self

    def to(self, *, device):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def view(self, _dtype):
        return self

    def reshape(self, *_shape):
        return self

    def numpy(self):
        return self.value


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class _FakeEncoder:
    tok = _FakeTokenizer()

    def encode(self, state: dict, question: dict) -> dict:
        serialized = json.dumps([state, question], ensure_ascii=False, indent=2).encode("utf-8")
        token_ids = list(hashlib.sha256(serialized).digest())
        return {
            "input_ids": token_ids,
            "attention_mask": _FakeTensor(bytes([1]) * len(token_ids), (1, len(token_ids))),
            "position_ids": list(range(len(token_ids))),
            "choice_read_idx": [2, 4, 6],
            "answer_end_idx": len(token_ids) - 1,
            "task_type": 0,
            "keys": list(bev.CHOICES),
        }


class _FakeDecider:
    encoder = _FakeEncoder()


def _fixture_request() -> dict:
    fixtures = ROOT / "tests" / "contract_fixtures.jsonl"
    for line in fixtures.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("fixture_id") == "approval_complete_empty_active":
            return copy.deepcopy(row["guard_request"])
    raise AssertionError("frozen approval fixture missing")


def _reverse_mapping_order(value):
    if isinstance(value, dict):
        return {key: _reverse_mapping_order(child)
                for key, child in reversed(list(value.items()))}
    if isinstance(value, list):
        return [_reverse_mapping_order(child) for child in value]
    return value


class CandidateInputParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_torch = sys.modules.get("torch")
        fake_torch = types.ModuleType("torch")
        fake_torch.uint8 = object()
        sys.modules["torch"] = fake_torch

        self._previous_bev_decider = sys.modules.get("bev_decider")
        self._previous_encode = sys.modules.get("bev_decider.encode")
        package = types.ModuleType("bev_decider")
        package.__path__ = []
        encode_module = types.ModuleType("bev_decider.encode")

        def question_options(question):
            keys = list(question["criteria"])
            return ([{"key": key, "description": question["criteria"][key]} for key in keys], keys)

        encode_module.question_options = question_options
        sys.modules["bev_decider"] = package
        sys.modules["bev_decider.encode"] = encode_module

    def tearDown(self) -> None:
        for name, previous in (("torch", self._previous_torch),
                               ("bev_decider", self._previous_bev_decider),
                               ("bev_decider.encode", self._previous_encode)):
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def test_raw_order_and_contract_presence_share_all_candidate_fingerprints(self) -> None:
        raw_missing = _fixture_request()
        raw_provided = copy.deepcopy(raw_missing)
        raw_provided["host_policy"]["decision_contract"] = public_decision_contract()
        variants = [raw_missing, _reverse_mapping_order(raw_missing),
                    raw_provided, _reverse_mapping_order(raw_provided)]
        prepared = [sequence_candidate.encode_candidate_input(request, _FakeDecider())
                    for request in variants]

        fingerprints = (
            "request_sha256", "model_state_text_sha256", "question_sha256", "options_sha256",
            "input_ids_sha256", "attention_mask_sha256", "encoded_input_sha256",
        )
        baseline = prepared[0]
        for result in prepared[1:]:
            for key in fingerprints:
                self.assertEqual(result[key], baseline[key], key)
            self.assertEqual(result["model_state"], baseline["model_state"])
            self.assertEqual(result["model_state_text"], baseline["model_state_text"])
            self.assertEqual(result["question_text"], baseline["question_text"])
            self.assertEqual(result["rendered_options"], baseline["rendered_options"])
            self.assertEqual(result["rendered_option_keys"], baseline["rendered_option_keys"])
            self.assertEqual(result["input_ids"], baseline["input_ids"])
        self.assertEqual(baseline["clean_request"]["host_policy"]["decision_contract"],
                         public_decision_contract())
        self.assertEqual(
            {decide_request(result["clean_request"], context="fixed_proposal")["decision"]
             for result in prepared},
            {"PERMIT"},
        )
        self.assertEqual(baseline["question_sha256"], hashlib.sha256(
            baseline["question_text"].encode("utf-8")).hexdigest())

    def test_trainer_consumes_same_shared_encoding_and_hashes(self) -> None:
        request = _fixture_request()
        expected = sequence_candidate.encode_candidate_input(request, _FakeDecider())
        row = {"sample_id": "contract-fixture", "source": "test", "request": request,
               "state": expected["model_state"]}
        labels = {"contract-fixture": {"verdict": "PERMIT"}}
        samples, _ = _tokenize_rows([row], labels, _FakeDecider(), bev, sequence_v2,
                                    sequence_candidate, ROOT / "src")
        sample = samples[0]
        self.assertEqual(sample["state_text_sha256"], expected["model_state_text_sha256"])
        self.assertEqual(sample["question_sha256"], expected["question_sha256"])
        self.assertEqual(sample["options_sha256"], expected["options_sha256"])
        self.assertEqual(sample["encoded_sha256"], expected["encoded_input_sha256"])
        self.assertEqual(sample["input_ids_sha256"], expected["input_ids_sha256"])
        self.assertEqual(sample["attention_mask_sha256"], expected["attention_mask_sha256"])


if __name__ == "__main__":
    unittest.main()
