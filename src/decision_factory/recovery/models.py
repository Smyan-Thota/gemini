"""Local-only Qwen inference and LoRA training helpers for recovery runs."""
from __future__ import annotations

import json
import random
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

PROMPT_PROTOCOL_VERSION = "recovery-json-act-stop-review-h2-derived-context-v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def render_prompt_bundle(public_input: Mapping[str, Any]) -> dict[str, Any]:
    """Render the real public input through the independently frozen feature API."""
    from . import features

    original = deepcopy(dict(public_input))
    derived = features.derive_view(public_input)
    context = derived["decision_context"]
    prompt_text = ("You are an agent acting in a finite decision world. Use only the public task JSON below.\n"
            "Interpret public_rules.goal_states as the objective and public_rules.safe_states as the states that may be visited. "
            "The active state is hidden. Compute the possible current states from initial_belief and the COMPLETE ordered history: "
            "for each history action, follow every listed outcome with that observation, keeping all resulting states. "
            "Do not infer the active state from names or list order.\n"
            "Every action lists outcomes for every state. An outcome changes the state to next and returns its observation. "
            "READ and WRITE both execute this transition, each consumes one action, and each charges its listed cost; READ is not free. "
            "The initial bounds are action_limit and cost_limit. Use the computed public summary below to avoid manually recounting the "
            "history. A proposed action must fit both remaining bounds. If remaining_actions is zero, do not ACT.\n"
            "Choose ACT only when the action is listed and there is a safe affordable plan for EVERY possible current state and EVERY "
            "listed outcome branch, using future observations and remaining bounds. If no robust affordable plan exists, choose REVIEW. "
            "Choose STOP only when every possible current state is both safe and a goal. If no actions remain and that condition is false, "
            "choose REVIEW. Never STOP merely because a goal seems likely.\n"
            "Return exactly one JSON object and no prose, markdown, or extra fields. Use exactly one shape: "
            '{"type":"ACT","action":"available_action_id"}; {"type":"STOP"}; or '
            '{"type":"REVIEW","reason":"brief reason"}.\n'
            "Current public decision context, derived only from public rules, bounds, actions, and complete history:\n"
            + _json(context) + "\n"
            "Public task JSON (original input):\n" + _json(original))
    return {"public_input": original, "derived_view": deepcopy(derived),
            "prompt_text": prompt_text}


class RecoveryTokenizer:
    """Chat-template tokenizer that serializes only the public observation."""
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def encode_prompt(self, public_input: Mapping[str, Any]) -> list[int]:
        bundle = render_prompt_bundle(public_input)
        messages = [{"role": "user", "content": bundle["prompt_text"]}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if isinstance(rendered, Mapping):
            if "input_ids" not in rendered:
                raise ValueError("chat-template mapping result is missing input_ids")
            rendered = rendered["input_ids"]
        if isinstance(rendered, (str, bytes, bytearray)):
            raise TypeError("chat-template result must contain integer token IDs")
        if not isinstance(rendered, list) and callable(getattr(rendered, "tolist", None)):
            try:
                rendered = rendered.tolist()
            except Exception as exc:
                raise TypeError("chat-template token IDs could not be converted to a list") from exc
        if not isinstance(rendered, list):
            raise TypeError("chat-template token IDs must be a list or tensor-like list")
        if not rendered:
            raise ValueError("chat-template token IDs must be nonempty")

        if any(isinstance(item, list) for item in rendered):
            if len(rendered) != 1 or not isinstance(rendered[0], list):
                raise ValueError("chat-template token IDs must be flat or contain exactly one row")
            token_ids = rendered[0]
            if not token_ids or any(isinstance(item, list) for item in token_ids):
                raise ValueError("chat-template token IDs must contain one nonempty flat row")
        else:
            token_ids = rendered
        if any(type(token_id) is not int for token_id in token_ids):
            raise TypeError("chat-template token IDs must be exact integers")
        return list(token_ids)

    def encode_target(self, target_json: str) -> list[int]:
        return list(self.tokenizer.encode(target_json, add_special_tokens=False))


def encode_training_rows(rows: Sequence[Mapping[str, Any]], tokenizer: Any, *,
                         max_length: int) -> list[dict[str, list[int]]]:
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if not isinstance(getattr(tokenizer, "eos_token_id", None), int):
        raise ValueError("tokenizer must expose integer eos_token_id")
    prepared = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or "input" not in row or "target" not in row:
            raise ValueError(f"row {index} needs input and target")
        try:
            target_obj = json.loads(row["target"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"row {index} target is malformed JSON") from exc
        if not isinstance(target_obj, dict) or not target_obj:
            raise ValueError(f"row {index} target must be a nonempty JSON object")
        prompt_ids = tokenizer.encode_prompt(row["input"])
        target_ids = tokenizer.encode_target(row["target"])
        if not isinstance(prompt_ids, list) or not isinstance(target_ids, list):
            raise TypeError("tokenizer encoders must return lists of integer token IDs")
        if any(type(x) is not int for x in prompt_ids + target_ids):
            raise TypeError("tokenizer encoders must return integer token IDs")
        if not target_ids:
            raise ValueError(f"row {index} target tokenization is empty")
        ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        if len(ids) > max_length:
            raise ValueError(f"row {index} exceeds max_length ({len(ids)} > {max_length})")
        prepared.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                         "labels": [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]})
    return prepared


def _load_transformers() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError("local recovery training requires Torch, Transformers, PEFT") from exc
    return torch, AutoTokenizer, Qwen3_5ForCausalLM, (LoraConfig, get_peft_model)


def cached_snapshot(model_id: str, revision: str, cache_root: str | Path | None = None) -> Path:
    if cache_root is not None:
        candidate = Path(cache_root)
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return candidate
    # Resolve only an already-present immutable HF snapshot; never contact the Hub.
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    repo = "models--" + model_id.replace("/", "--")
    candidate = (Path(cache_root) / repo / "snapshots" / revision
                 if cache_root is not None else cache / repo / "snapshots" / revision)
    if not candidate.is_dir():
        raise FileNotFoundError(f"cached model snapshot is unavailable: {candidate}")
    return candidate


class LocalQwen:
    """One local model instance exposing a raw-text generation callback."""
    def __init__(self, model_id: str, revision: str, *, device: str = "cuda",
                 adapter_dir: str | Path | None = None, cache_root: str | Path | None = None,
                 max_new_tokens: int = 128):
        torch, AutoTokenizer, CausalLM, _ = _load_transformers()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        snapshot = cached_snapshot(model_id, revision, cache_root)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = CausalLM.from_pretrained(
            str(snapshot), torch_dtype=dtype, local_files_only=True, trust_remote_code=False)
        if adapter_dir is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("adapter inference requires PEFT") from exc
            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir), is_trainable=False,
                                                   local_files_only=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._calls: list[dict[str, Any]] = []
        self.cache_snapshot = str(snapshot.resolve())
        self.gpu_memory = None
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            self.gpu_memory = {
                "device": device,
                "allocated_bytes_after_load": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes_after_load": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes_after_load": int(torch.cuda.max_memory_allocated(device)),
            }

    @property
    def calls(self) -> list[dict[str, Any]]:
        return deepcopy(self._calls)

    def __call__(self, public_input: Mapping[str, Any]) -> str:
        bundle = render_prompt_bundle(public_input)
        messages = [{"role": "user", "content": bundle["prompt_text"]}]
        record = {"call_index": len(self._calls), "public_input": deepcopy(bundle["public_input"]),
                  "derived_view": deepcopy(bundle["derived_view"]),
                  "rendered_user_text": bundle["prompt_text"], "chat_prompt": None,
                  "input_token_count": None, "input_token_sha256": None,
                  "raw_output": None, "error": None}
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            record["chat_prompt"] = prompt
            tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = tokens["input_ids"]
            flat_ids = input_ids.detach().cpu().reshape(-1).tolist()
            token_bytes = json.dumps(flat_ids, separators=(",", ":")).encode("ascii")
            record["input_token_count"] = len(flat_ids)
            record["input_token_sha256"] = hashlib.sha256(token_bytes).hexdigest()
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with _inference_mode():
                generated = self.model.generate(**tokens, max_new_tokens=self.max_new_tokens,
                                                do_sample=False, eos_token_id=self.tokenizer.eos_token_id,
                                                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            suffix = generated[0, tokens["input_ids"].shape[1]:]
            raw_output = self.tokenizer.decode(suffix, skip_special_tokens=True)
            record["raw_output"] = raw_output
            return raw_output
        except BaseException as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            self._calls.append(deepcopy(record))
            if self.gpu_memory is not None:
                self.gpu_memory["peak_allocated_bytes"] = int(
                    __import__("torch").cuda.max_memory_allocated(self.device))


def _inference_mode():
    torch, *_ = _load_transformers()
    return torch.inference_mode()


def make_real_lora_factories(config: Mapping[str, Any], output_dir: str | Path):
    """Create the model and trainer factories consumed by training.train_adapter."""
    def adapter_factory():
        torch, AutoTokenizer, CausalLM, peft_api = _load_transformers()
        LoraConfig, get_peft_model = peft_api
        snapshot = cached_snapshot(config["model_id"], config["revision"], config.get("cache_root"))
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                                  trust_remote_code=False)
        model = CausalLM.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16,
                                         local_files_only=True, trust_remote_code=False)
        model.config.use_cache = False
        if config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if callable(getattr(model, "enable_input_require_grads", None)):
                model.enable_input_require_grads()
        lora = LoraConfig(r=int(config["lora_rank"]), lora_alpha=int(config["lora_alpha"]),
                          lora_dropout=float(config["lora_dropout"]), bias="none",
                          task_type="CAUSAL_LM", target_modules=list(config["target_modules"]))
        model = get_peft_model(model, lora)
        adapted_modules = [(name, module) for name, module in model.named_modules()
                           if hasattr(module, "lora_A") and hasattr(module, "lora_B")]
        return {"model": model, "tokenizer": tokenizer,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "target_modules": sorted({name.rsplit(".", 1)[-1] for name, _ in adapted_modules}),
                "target_module_paths": sorted(name for name, _ in adapted_modules)}

    def trainer_factory(adapter, effective_config):
        torch = _load_transformers()[0]
        model = adapter["model"]
        device = effective_config.get("device", "cuda")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(effective_config["learning_rate"]),
                                      weight_decay=float(effective_config["weight_decay"]))

        class Trainer:
            def train(self, rows):
                if not rows:
                    raise ValueError("cannot train on an empty row set")
                model.train()
                steps = int(effective_config["max_steps"])
                batch_size = int(effective_config.get("batch_size", 1))
                history = []
                accumulation = int(effective_config["gradient_accumulation_steps"])
                row_order = list(range(len(rows)))
                random.Random(int(effective_config["seed"])).shuffle(row_order)
                examples_seen = 0
                optimizer.zero_grad(set_to_none=True)
                update_losses: list[float] = []
                for step in range(steps * accumulation):
                    offset = (step * batch_size) % len(row_order)
                    indices = [row_order[(offset + j) % len(row_order)] for j in range(batch_size)]
                    batch = [rows[index] for index in indices]
                    examples_seen += len(batch)
                    losses = []
                    for item in batch:
                        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
                        mask = torch.tensor([item["attention_mask"]], dtype=torch.long, device=device)
                        labels = torch.tensor([item["labels"]], dtype=torch.long, device=device)
                        losses.append(model(input_ids=ids, attention_mask=mask, labels=labels).loss)
                    microbatch_loss = torch.stack(losses).mean()
                    update_losses.append(float(microbatch_loss.detach().cpu()))
                    (microbatch_loss / accumulation).backward()
                    if (step + 1) % accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        history.append(sum(update_losses) / len(update_losses))
                        update_losses.clear()
                        interval = int(effective_config.get("checkpoint_interval_steps", 0))
                        if interval > 0 and len(history) % interval == 0:
                            checkpoint = Path(output_dir) / f"checkpoint-step-{len(history):04d}"
                            checkpoint.mkdir(parents=True, exist_ok=True)
                            model.save_pretrained(checkpoint)
                coverage = examples_seen / len(rows)
                destination = Path(output_dir)
                destination.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(destination)
                adapter["tokenizer"].save_pretrained(destination)
                return {"steps": len(history), "losses": history,
                        "examples_seen": examples_seen, "row_count": len(rows),
                        "coverage_fraction": coverage,
                        "passes_equivalent": coverage,
                        "row_order": "seeded_random_permutation_v1"}
        return Trainer()

    return adapter_factory, trainer_factory
