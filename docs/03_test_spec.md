# Test Specification — RET-C2-005

**Template ID:** RET-C2-005
**Template Name:** ShrinkageInventoryQAAgent
**Category:** Cat 2 (nested RAG)

This document defines the test cases for the template (state, nodes,
inner/outer graphs, manifest, server). The test code lives in `tests/unit/` +
`tests/proof_of_boundary/`; this spec is the contract those tests implement.

## 1. Scope & Invocation Conventions

- Per-node unit tests for the 5 inner domain nodes + the 2 outer gate nodes.
- Manifest/config consistency (`config/agent.yaml` + `config/config.yaml` ↔ code)
  and seeded-KB integrity.
- Caller-data contract (`input_context`) + external output schema enforcement.
- Retrieval quality (golden queries over `config/kb/shrinkage_kb.json`).
- Inner-graph (`DomainWorkflowGraph`) and outer-graph
  (`ShrinkageInventoryQAAgent`) composition / integration.
- Proof-of-Boundary (PoB): import isolation, State msgpack safety, invoke
  order (PB-6), HITL propagation (PB-7, conditional), server boot + entry-point
  auth, end-to-end `/invoke` behaviour.

**Trust-gate invocation canon.** Every per-node test invokes the node via
`node(state)` — through `BaseNode.__call__`, which runs the trust gate → input
mask → `execute()` → output gate — never a bare `node.execute(state)`. The
state builder sets `caller_trust_level` to
`TrustLevel.VERIFIED_EXTERNAL.value` for the outer ingest slot
(PreProcessNode — the manifest's declared caller level) and
`TrustLevel.ANONYMOUS.value` for the five inner domain nodes and the
post_process output gate (trust is enforced once, at the ingest boundary).

**No config-parameter carve-out.** Nodes do not accept a `config` parameter:
`execute(self, state)` is the only signature. Every test in this suite,
without exception, invokes exclusively through `node(state)`. Retrieval tuning
(`top_k` / `score_threshold` / `kb_path`) is exercised by seeding the SCALAR
state keys `retrieval_top_k` / `retrieval_score_threshold` /
`retrieval_kb_path` (the same keys `DomainWorkflowGraph._extra_initial_state()`
forwards at runtime) and still invoking through `node(state)`.

**Input-mask expectations.** The framework input gate masks
`user_input`/`validated_input`/`llm_response` (e-mail, phone/SSN/CC digit
groups, Title-Case name bigrams) to `[MASKED]` before `execute()` runs.
Positive-path payloads are therefore lowercase, PII-free store-ops phrasing;
intentional-PII tests assert the raw identifier is gone and `[MASKED]` is
present. This template's `PreProcessNode` adds no domain-specific identifier
screen of its own (no IBAN/account-number strip — that is a different
template's node behaviour), so only the framework-level patterns are
exercised. Domain fields (`grounded_answer`, `formatted_answer`,
`retrieved_documents`, …) are not input-mask scan targets.

**Audit muting.** `shared.*` is never sys.modules-stubbed (the framework
imports `shared.security` at load time). The domain audit emitter is muted via
an autouse fixture patching `src.nodes.<mod>.emit_trace_event`; audit
assertion tests re-patch the same attribute with a spy and assert on
`call.args[1]` (the event payload).

## 2. Unit Test Cases

### 2.1 PreProcessNode (outer pre_process slot) — `test_pre_process_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid query | lowercase store-ops question | `status=SUCCESS`, `validated_input` set, `enriched_context` (JSON string) carries channel/source |
| PRE-02 | Empty input | `""` / whitespace | `status=ERROR`, `error_log` non-empty, no `validated_input` |
| PRE-03 | Missing / non-string / over-long input | `user_input` absent; dict payload; > 2000 chars | `status=ERROR` |
| PRE-04 | Framework input mask | e-mail / 4-4-4 digit groups | raw identifier absent from `validated_input`; `[MASKED]` present |
| PRE-05 | input_context: channel | non-identifier / non-string / over-long | `status=ERROR` naming `input_context.channel`; value never echoed |
| PRE-06 | input_context: category | non-identifier / non-string | `status=ERROR` naming `input_context.category` |
| PRE-07 | input_context: top_k | `"NaN"`/`"Infinity"`/raw NaN/Inf/float/bool/0/21 | `status=ERROR` naming `input_context.top_k` (fail CLOSED, full non-finite matrix) |
| PRE-08 | Audit | valid query; rejected context | `pre_process_complete` payload carries `input_chars`; `pre_process_validation_failed` payload names the reason class, never the value |

### 2.2 InputValidateNode (inner node 1) — `test_input_validate_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| VAL-01 | Plain text | free-text query | whole string becomes `search_query`; filters `{category: None, top_k: None}` |
| VAL-02 | Whitespace | ragged spacing/newlines | collapsed to single spaces |
| VAL-03 | Caller parameters | `input_context` with valid `category` + `top_k` | both re-validated and published in `query_filters` |
| VAL-04 | Absent / unknown context | no `input_context`; unknown keys | no filters; unknown keys ignored |
| VAL-05 | top_k out of contract | 99 / −5 / 0 / non-int / non-finite / bool | `status=ERROR` naming `input_context.top_k` (fail CLOSED) |
| VAL-06 | Rejected value never echoed | invalid top_k / category | offending value absent from `error_log` |
| VAL-07 | Boundary top_k | 1 / 20 | accepted |
| VAL-08 | Oversize query | > 2000 chars | truncated to 2000 + note |
| VAL-09 | Empty request | `""` | `search_query=""` + "empty request" note (non-fatal) |
| — | JSON-string contract | any | `query_filters` is a JSON string, never a bare dict |

### 2.3 RetrieveNode (inner node 2) — `test_retrieve_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RET-01 | Happy path | anomaly-patterns query | top-1 candidate is `kb-001` |
| RET-02 | Ordering | anomaly-patterns query | scores strictly sorted desc; all > 0 |
| RET-03 | Entry shape | any hit | keys `{id,title,category,source,score,excerpt}`; excerpt ≤ 400 chars |
| RET-04 | Category filter | `query_filters.category="investigation_procedures"` | only that category's entry; top-1 `kb-002` |
| RET-05 | Empty query | `""` | no candidates |
| RET-06 | State `retrieval_kb_path` override, via `node(state)` | bogus path | `[]` + "not readable" note |
| RET-07 | Malformed `retrieval_top_k` seed | non-numeric / non-finite | degrades to module default, never crashes |
| RET-08 | Notes accumulation | prior `intake_notes` | appended, never clobbered |

### 2.4 RerankFilterNode (inner node 3) — `test_rerank_filter_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RRF-01 | Relevance floor | scores 0.9 / 0.1 | 0.1 dropped (default 0.25 floor) |
| RRF-02 | State `retrieval_score_threshold` seed | `0.5` | 0.3 dropped |
| RRF-03 | State `retrieval_top_k` seed | `1` | one survivor, highest score |
| RRF-04 | Category boost | matching category | +0.1, re-ranked ahead |
| RRF-05 | Boost cap | 0.95 + boost | capped at 1.0 |
| RRF-06 | Caller top_k | stricter (1) wins; looser (10) does not widen past the seeded `retrieval_top_k` | enforced |
| RRF-07 | Garbage entries | non-dict / uncoercible score | skipped / treated as 0.0 and dropped |
| RRF-08 | Tie-break | equal scores | deterministic id-ascending order |

### 2.5 GenerateAnswerNode (inner node 4) — `test_generate_answer_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| GEN-01 | Citation markers | 2 ranked passages | `[1]`/`[2]` markers with titles |
| GEN-02 | No caller text | query smuggling a fake `[n]` marker | question never embedded in the answer body |
| GEN-03 | Citations list | ranked passages | refs 1..n mirror ranked order; id/title/source carried |
| GEN-04 | Groundedness | single passage | answer body traces to ranked passages only |
| GEN-05 | No coverage | empty/missing `ranked_documents` | escalation answer; `citations=[]` |

### 2.6 OutputFormatNode (inner node 5, terminal) — `test_output_format_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| FMT-01 | Full compose | body + citations | header + body + `## Sources` rows + scope disclaimer; `status=SUCCESS` |
| FMT-02 | Blank source | citation without source | no `()` suffix |
| FMT-03 | Disclaimer | every input | disclaimer rides with every answer |
| FMT-04 | No citations | empty list | explicit "- none (…)" sources line |
| FMT-05 | Missing body | no `grounded_answer` | fallback text; `status=SUCCESS` |

### 2.7 PostProcessNode (outer post_process slot; output gate) — `test_post_process_node.py`

| ID | Case | Input (`result`) | Expected |
|----|------|------------------|----------|
| POST-01 | Clean output | normal KB answer | `formatted_output=result`, `status=SUCCESS` |
| POST-02 | Empty result | `""` | fallback "no answer content" message, `status=SUCCESS` (non-fatal); `shrinkage_answer` used when `result` absent |
| POST-03..06 | Credential leak | `sk-` API key / `password=` assignment / JWT (built at runtime) / Bearer token | `formatted_output` + `result` replaced with the sanitised stub, `status=ERROR`, raw secret absent from both |
| POST-07 | Caller-text redaction | verbatim question embedded in `result` | replaced with `[REDACTED]`; short incidental overlaps untouched |
| POST-08 | Precision grid | off-grid monetary token in `result` | snapped to the 1,000 grid; on-grid and structural tokens byte-identical |

### 2.8 Manifest / config consistency — `test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Template id | `id` = `RET-C2-005`; `namespace` = `ret`; `enabled` |
| CFG-02 | Class-name contract | manifest `class` = `src.graph.graph.ShrinkageInventoryQAAgent` (the graph.py class); `name` matches the agent |
| CFG-03 | Classification | Cat 2 / RET / RAGAgent / deterministic; `requires.secrets`/`extras` empty (code-derived) |
| CFG-04 | Trust level | manifest `VERIFIED_EXTERNAL` == PreProcessNode `required_trust_level` |
| CFG-05 | max_retry | int, `0 ≤ v < 10` (framework ceiling); hitl not enabled (PB-7 waiver contract) — from `config/config.yaml` |
| CFG-06 | Retrieval block | `config/config.yaml` `retrieval.*` mirror the owning node's module-level defaults (`RetrieveNode._DEFAULT_TOP_K`/`_DEFAULT_KB_PATH`, `RerankFilterNode._DEFAULT_TOP_K`/`_DEFAULT_SCORE_THRESHOLD`) |
| CFG-07 | `_parent_config()` | forwards the declared retrieval + llm blocks from `config/config.yaml` |
| — | KB integrity | JSON list ≥ 5 entries; unique ids; required keys per entry |

### 2.9 Retrieval quality (golden queries) — `test_retrieval_quality.py`

| ID | Case | Expected |
|----|------|----------|
| QUAL-01 | 6 golden domain queries | expected KB entry is top-1 (kb-001/005/006/009/007/010) |
| QUAL-02 | Relevance floor | every survivor ≥ 0.25 |
| QUAL-03 | Citation integrity | every survivor id exists in the seeded KB |
| QUAL-04 | Precision | internal-theft query keeps ONLY `kb-005` |
| QUAL-05 | Category filter | `investigation_procedures` filter → only that entry, top-1 `kb-002` |
| QUAL-06 | No coverage | out-of-domain query → zero survivors |
| QUAL-07 | Escalation answer | no-coverage → explicit escalation text, no citations |

### 2.10 Trust matrix — `test_trust_gate.py`

| ID | Case | Expected |
|----|------|----------|
| TRUST-01 | ANONYMOUS on inner node | passes (inner nodes declare ANONYMOUS) |
| TRUST-02 | ANONYMOUS on pre_process | denial dict: `status=ERROR`, "trust gate denied", no execute-only keys |
| TRUST-03 | VERIFIED_EXTERNAL on pre_process | passes; `validated_input` written |
| TRUST-04 | post_process posture | ANONYMOUS passes (trust enforced once, at ingest); declared matrix pinned (pre VERIFIED_EXTERNAL; post + 5 inner nodes ANONYMOUS) |

### 2.11 Caller-data contract + output schema — `test_caller_data_and_output_schema.py`

| ID | Case | Expected |
|----|------|----------|
| CDC-01 | `_validate_input_context` | full valid/invalid matrix per field (channel / category / top_k); rejected values never echoed |
| CDC-02 | `_finite_in_range` | rejects bools/non-numerics/NaN/±Inf/out-of-range; accepts finite in range |
| CDC-03 | Seeded-scalar fail-safes | malformed `retrieval_top_k` / `retrieval_score_threshold` / candidate scores fall back safely (never weaken the floor) |
| CDC-04 | Caller top_k narrows only | stricter caller cap wins; wider never overrides the seeded cap |
| CDC-05 | Runtime-config plumbing | `_runtime_config()` reads `config/config.yaml`; missing file degrades; declared values reach the inner state seeding |
| CDC-06 | Context bridge e2e | caller `top_k` changes the number of cited sources through the FULL nested graph (threshold opened) |
| CDC-07 | Precision gate forms | every leak form snaps (marker/value both orders, symbols incl. fullwidth, signed, any-whitespace delimiters, comma-grouped, 5+-digit runs); structural + on-grid tokens byte-identical |
| CDC-07a | Precision gate exclusions | decimals and domain identifiers byte-identical (`8.512345`, `99.99999%`, `JPY 1234.56m`, `SKU-48210`, `sku_48210`, `PO-2026-004821`, `LP-2026-00318`, `POS-07`, `EAS-1024`, `ABC-3`, `SOP-2026-014`, `kb-001`); marker/value delimiter never crosses a paragraph break; off-grid amounts still snap as one number (`JPY 1234.56` → `JPY 1,000`); attached marker restricted to ISO-4217 (`JPY-9999` snaps, `SKU-48210` survives) while a separated marker stays unrestricted |
| CDC-08 | Blocked-field redaction + schema note | verbatim caller text redacted; schema note rendered iff money renders |

## 3. Integration / Composition

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| INT-01 | Composition | inherits `BaseGraph`; registers exactly the 5 domain nodes; no initialize/finalize |
| INT-02 | Config forwarding | `_extra_initial_state()` republishes the retrieval block as SCALAR keys (only the keys present in the passed config) and always seeds the bridged `input_context` (stashed caller context read back; `{}` when none) |
| INT-03 | Output shape | `get_output()` emits `formatted_answer`/`citations`/`status`/… (the merge contract); `route()` → END on error |
| INT-04 | Inner e2e | full inner `invoke()` → SUCCESS; formatted answer + disclaimer + `kb-001` citation; inner `node_history` = the 5 domain nodes in linear order |

### 3.2 Outer graph + e2e — `test_graph_composition.py`

| ID | Case | Expected |
|----|------|----------|
| INT-05 | Outer composition | inherits `AgentBaseGraph` (direct framework inheritance); `Graph` alias; `add_edges()` NOT overridden |
| INT-06 | Backbone slots | compile() fills all 5; pre/main/post are PreProcessNode / ShrinkageQAGraphNode / PostProcessNode |
| INT-07 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded retrieval config |
| INT-08 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| INT-09 | `merge_output()` | inner `formatted_answer` → outer `shrinkage_answer` AND `result`; `citations`/`status` mapped; changed keys only |
| INT-10 | Runtime config | `_parent_config()` reads `config/config.yaml`; missing file degrades to node defaults; malformed values (non-finite / out-of-range / mistyped) are never forwarded |
| INT-11 | e2e happy path | VERIFIED_EXTERNAL invoke → SUCCESS; `output` = gated formatted answer; PostProcessNode traversed |
| INT-12 | e2e trust denial | ANONYMOUS invoke → ERROR; empty `output`; PostProcessNode NOT traversed |
| — | JSON helpers | `to_json`/`from_json` round-trip; None/malformed handling |

## 4. Proof-of-Boundary

| ID | Case | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no platform-SDK import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no `BaseModel` / `InvocationContext` annotations |
| PB-6 | `test_pb_invoke_order.py` | full `Graph().invoke()` with `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` (never `for_internal()`) over the payload byte-equal to `deploy/invoke_payload.json`'s `input` → SUCCESS with outer `node_history` exactly `[InitializeNode, PreProcessNode, ShrinkageQAGraphNode, PostProcessNode, FinalizeNode]` |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — non-HITL** (no graph class declares `propagate_hitl=True`); conditional skip-stub retained |
| PB-BOOT | `test_server_boot.py` | `import src.api.server` does not raise; module-level agent is this template's class, compiled with the declared runtime config; fresh ctor→`compile()` fills the 5 backbone slots; `/health` reports the agent; **entry-point auth boundary**: missing/wrong/malformed/empty/non-ASCII Bearer → generic 401; correct Bearer → VERIFIED_EXTERNAL; unset/empty token → ANONYMOUS (never elevated); middleware trust never demoted; oversized `input_context` → 413 |
| PB-E2E | `test_invoke_e2e.py` | REAL compiled agent through the real ASGI `/invoke` (Bearer auth): grounded question → cited answer + disclaimer; no-coverage → explicit decline as SUCCESS; category filter + top_k override reach the inner retrieval; empty input / invalid channel / invalid category / full non-finite top_k matrix → error with no output; fake `[n]` marker never surfaces; every rendered monetary-form numeric on the 1,000 grid |

> **Gate checklist:** PB-IMPORT, PB-STATE, PB-6, PB-BOOT and PB-E2E are
> mandatory. PB-7 applies only to templates that opt into cross-boundary HITL
> propagation — this template does not, so PB-7 is **Auto-waived — non-HITL**
> and its skip must not block the gate.

## 5. Test Execution Summary

- Execution date: 2026-08-25
- Runner: real SDK wheel (`agenticstar-agentcore==1.0.1`), `python -m pytest tests/`
- Full suite: 318 total — Pass: 317 / Fail: 0 / Skip: 1 (PB-7, auto-waived non-HITL)
- Determinism: no LLM, no network; retrieval + answer assembly are rule-based
