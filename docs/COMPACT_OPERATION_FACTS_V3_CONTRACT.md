# Compact operation-facts ablation, version 3

Status: frozen independent input contract for one matched representation ablation. This document specifies model-visible factual fields, not a decision algorithm. `sequence_guard_native_state/2` and its alpha transform remain immutable baselines. The ablation uses the same 4,608 training requests/labels and 152 development requests/labels; it adds no rows and does not change the reference labels.

## Input and output boundary

The encoder accepts only one validated `bev_guard_request/1` for the current turn. It must not accept or import evaluator keys, expected decisions, family/group/pair IDs, completion labels, arm names, hidden files, future menus, scheduled future events, training metadata, or answer keys. The output is `sequence_guard_operation_facts/3` and is solely a factual rendering of the supplied request. The raw request remains available to the host for assessment binding.

No output field may say or imply `PERMIT`, `BLOCK`, `REVIEW`, safe/unsafe, violation/no-violation, authorized/unauthorized, correct/incorrect, or a recommended action. Do not derive a verdict, violation flag, approval-active flag, recipient-member flag, or "safe" shortcut. Primitive counts, exact values, equality-preserving identifier bindings, and additions/subtractions of supplied quantities are allowed. The encoder must not call the reference/decider or use a label to construct a fact.

The proposed top-level structure is:

```json
{
  "schema": "sequence_guard_operation_facts/3",
  "contract": {"version": "public-decision-semantics-v1", "text": "<exact public semantics>"},
  "task": {"text": "<current public task>"},
  "bounds": {"workspace_root": "<exact value>", "allowed_operations": [], "max_operations": 0},
  "proposal": {"tool_name": "<exact tool>", "arguments": {}},
  "history": {
    "prior_action_record_count": 0,
    "actual_allow_count": 0,
    "attempted_upload_count": 0,
    "committed_upload_count": 0,
    "effect_knowledge": "complete",
    "committed_data_units": {"knowledge": "known", "value": 0},
    "committed_data_fields": {"knowledge": "known", "value": []}
  },
  "workflow": {"rule_type": "<exact supported type>", "policy_facts": {}},
  "ledgers": {
    "effect-ledger": {"knowledge": "complete", "events": []},
    "authorization-ledger": {"knowledge": "missing", "events": []},
    "artifact-ledger": {"knowledge": "unavailable", "events": []},
    "recipient-ledger": {"knowledge": "complete", "events": []},
    "workflow-ledger": {"knowledge": "complete", "events": []}
  },
  "observations": []
}
```

This is a field contract, not permission to fabricate defaults. `knowledge` distinguishes `complete`, `unavailable`, `missing`, `malformed`, `contradictory`, and `not_applicable`; use only values supported by input evidence. A complete empty ledger is `complete` with `events: []`. It must never be collapsed into unavailable/missing. A known aggregate with unresolved completeness uses an explicit knowledge state; do not substitute zero or an empty list for unknown. Counts are counts of supplied records/events, not verdicts.

The exact V3 key layout is:

- `bounds` copies `workspace_root`, `allowed_operations`, `max_operations`, `max_write_bytes`, `operation_semantics`, and `tool_schemas` from `host_policy`, preserving explicit nulls and list/object ordering where meaningful. Add `remaining_operations` only when `max_operations` is a valid integer, computed as `max_operations - history.actual_allow_count`; do not clamp negative values.
- `proposal` preserves `tool_name` and the complete `arguments` object. Do not duplicate individual argument values into sibling scalar fields. The exact argument object already preserves their values and identifier relationships.
- `history` contains the record counts, `effect_knowledge`, and the two knowledge-wrapped committed aggregates shown above. `attempted_upload_count` counts prior upload proposals recorded in `prior_actions`; `committed_upload_count` counts only matching, trusted, committed receipts. `effect_knowledge` is `complete`, `incomplete`, or `unknown`, describing whether the supplied action/receipt history supports complete effect accounting; it is a data-quality fact, not a verdict. The aggregate values count/summarize matching committed receipts only. If relevant effect history cannot establish an aggregate, use `{"knowledge":"unknown","value":null}`. Never turn an attempted upload into a committed effect.
- `workflow.rule_type` copies the exact `workflow_rules.rule_type`. The relevant family object is one of `release_limit`, `approval_status`, `joint_disclosure`, or `recipient_allowlist`, and copies only the exact source policy fields: release limit carries `max_total_release_units` and `ledger_status`; approval carries the whole `grant` object plus `authorization_ledger_status` and `artifact_ledger_status`; joint disclosure carries the full `prohibited_joint_disclosure_groups` value and `ledger_status`; recipient allowlist carries the full `approved_recipients` value and `recipient_directory_status`. Null/missing/empty/malformed remain distinct. Other policy keys remain under `workflow.policy_facts` unchanged.
- `ledgers` always contains all five named ledger slots. Each slot's `knowledge` is derived from its corresponding explicit policy status when one exists, otherwise `missing`; its `events` contains only supplied records whose source matches that ledger and whose trust is `trusted`, in their supplied order. Each event keeps `source`, `trust`, and its complete parsed `payload`; preserve original `text` only when parsing fails, so the same JSON facts are not serialized twice. Untrusted observations stay in the separate ordered `observations` array and must never be presented as a ledger event. `ledger_integrity_anomalies` is copied if supplied. Do not fabricate events from an aggregate or untrusted prose.
- `observations` contains every supplied top-level observation in order plus each prior action's nested observation only when that exact occurrence is not already present at top level. Preserve multiplicity: if two separate reads return identical observations, both occurrences remain. Keep each trust marker unchanged; an untrusted nested read result stays untrusted and may not be promoted into a trusted ledger event.

Do not alpha-bind the new V3 facts with a separate, incompatible mapping. Reuse the pinned alpha-binding bijection and version its wrapper so all equal opaque IDs map to one canonical ID, unequal IDs remain unequal, and null/non-string values are preserved without coercion. The source-to-canonical map remains an audit sidecar and never enters the model-visible object.

## Preservation rules

- Keep exact current task text, exact proposal tool and arguments, exact structural bounds, declared tool schemas, and all public policy values needed by the contract. Do not drop a task constraint because it appears unimportant to a model.
- Keep the exact public decision semantics text. A version identifier may accompany it, but cannot replace it in this ablation.
- Represent all four workflow families (`release_limit`, `approval_status`, `joint_disclosure`, `recipient_allowlist`) in the same schema. Keep family policy facts as supplied, including explicit null, empty, missing, and malformed distinctions.
- Preserve opaque identifiers and their equality/inequality relationships consistently across policy, proposal, prior actions, ledger events, and observations. Alpha-renaming may change spelling only; the reversible map stays outside model input. Never normalize null to a string or merge two unequal IDs.
- Preserve trusted ledger event order and each event's source, trust, and payload. Do not sort, deduplicate, summarize away, or apply future scheduled events. Untrusted observations remain marked untrusted and cannot update trusted facts.
- Keep committed effects separate from attempted actions. Only a matching trusted committed receipt contributes committed uploads/units/fields. Actual prior `host_disposition: ALLOW` records count toward executed operations; BLOCK and REVIEW do not. A read ALLOW counts as an executed operation. A failed/not-found read still counts as executed. An attempted upload is not a committed upload.
- Include arithmetic only where operands are explicit, such as `remaining_operations = max_operations - actual_allow_count` when both are valid. Do not emit Boolean comparisons against policy as labels (`over_cap`, `all_fields_prohibited`, `recipient_allowed`, etc.).
- Preserve uncertainty instead of guessing when any needed source is absent, unavailable, malformed, or contradictory. Do not infer that an older artifact revision is current when later evidence is unresolved.
- Do not strip untrusted text, public contract rules, or task instructions in this ablation. Any later omission requires a separate relevance audit against the public contract and a separately versioned experiment.

## Frozen hand-check boundaries

`tests/compact_operation_facts_v3_fixtures.jsonl` contains 20 small input patches against named, already-frozen `bev_guard_request/1` fixtures and expected fact assertions. The expected object uses dot paths into V3; paths with `[n]` select an ordered list item. It deliberately describes facts rather than verdicts. Each fixture is a required regression boundary for the implementation. A test applies each patch to a deep copy of the named fixture request, invokes only the public encoder, and compares the named fact paths; labels stay outside the input.

## Acceptance checks

For every fixture, the independent test checks the exact listed fact values, schema, status distinctions, event order, nonduplicated proposal/event serialization, and alpha equality pattern. It also traverses the model-visible output and rejects evaluator/label fields and decision-like derived fields. V2 baseline output and source hashes must remain byte-for-byte unchanged. Training and development IDs, labels, pair/group metadata, split membership, and raw requests must match the existing pinned bundle exactly.
