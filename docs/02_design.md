# Template Design Specification — RET-C2-005

**Template ID:** RET-C2-005
**Template Name:** ShrinkageInventoryQAAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** RET

## Position in AgentCore Architecture

- **Agent Class:** `ShrinkageInventoryQAAgent` (alias `Graph`)
- **L1 Base (framework base class):** `AgentBaseGraph` — direct framework inheritance
- **Inner graph base:** `BaseGraph` — `DomainWorkflowGraph`
- **Pattern:** Cat 2 two-layer nested architecture (outer fixed 5-node backbone +
  `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow)
- **Three-Layer Separation:**
  - State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
    structured fields stored as JSON strings via `to_json()` / `from_json()`;
    retrieval tuning knobs are plain scalars (see the config-forwarding note below)
  - Node: framework inheritance via `FunctionNode` (override
    `execute(self, state: AgentState) -> dict` ONLY — no `config` parameter)
  - Graph: composition (`register_nodes()` for node substitution); outer
    `add_edges()` is NOT overridden

## Purpose

Shrinkage & inventory-anomaly Q&A agent: store-ops and loss-prevention
questions are answered over a seeded loss-prevention knowledge base
(anomaly patterns, investigation procedures, prevention playbooks) —
retrieve → rerank/filter → grounded answer with citations + a standing
scope disclaimer. The current build is fully deterministic (keyword
retrieval + rule-based grounded answer assembly; no live LLM call — see the
Implementation Note below).

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max 3)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | InitializeNode (framework default) | session_id, trust_level, schema_version | — (framework) |
| pre_process | `PreProcessNode` | **Trust gate (VERIFIED_EXTERNAL)** + question validation + the `input_context` caller-data contract → `validated_input`, `enriched_context` | `TrustLevel.VERIFIED_EXTERNAL` |
| main | `ShrinkageQAGraphNode` (`GraphNode`) | delegates to inner `DomainWorkflowGraph`; bridges `input_context`; maps inner `formatted_answer` → outer `result` | — (GraphNode delegation) |
| post_process | `PostProcessNode` | **Output gate** (credential scan + caller-text redaction + monetary precision grid) → `formatted_output` | `TrustLevel.ANONYMOUS` |
| finalize | FinalizeNode (framework default) | response_metadata, total_time_ms | — (framework) |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer pre_process slot; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime).

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|--------------|
| `InputValidateNode` | Normalise the question (whitespace, length cap); re-validate the bridged caller parameters (`category` identifier, `top_k` 1–20 — fail CLOSED) into structured filters | `TrustLevel.ANONYMOUS` | `validated_input` \| `user_input`, `input_context` | `search_query`, `query_filters`, `intake_notes` |
| `RetrieveNode` | Deterministic keyword retrieval over the seeded KB (`config/kb/shrinkage_kb.json`): tokenise query, score title/tags/content overlap, apply category filter | `TrustLevel.ANONYMOUS` | `search_query`, `query_filters`, `retrieval_top_k`, `retrieval_kb_path` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Rerank candidates (category-match boost), drop entries below `score_threshold`, cap at `top_k` (a stricter caller override narrows, never widens) | `TrustLevel.ANONYMOUS` | `retrieved_documents`, `query_filters`, `retrieval_top_k`, `retrieval_score_threshold` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based grounded answer assembly from the ranked KB passages only, with numbered citation markers; the caller's question is never embedded (deterministic — LLM synthesis seam documented below) | `TrustLevel.ANONYMOUS` | `ranked_documents` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Compose the final answer: body + Sources list + the standing scope disclaimer (disclaimer is part of this node, NOT post_process); prints the monetary schema note whenever money renders, using the gate's own grammar | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data Flow

```
user_input, input_context
  → PreProcessNode (trust gate + caller-data contract) → validated_input
  → ShrinkageQAGraphNode.extract_input                 → stashes input_context (context bridge);
                                                         inner DomainWorkflowGraph.invoke(validated_input)
        → input_validate                               → search_query / query_filters
        → retrieve                                     → retrieved_documents
        → rerank_filter                                → ranked_documents
        → generate_answer                              → grounded_answer / citations
        → output_format                                → formatted_answer (+ scope disclaimer)
     get_output() → {formatted_answer, citations, status, ...}
  → ShrinkageQAGraphNode.merge_output                  → result = formatted_answer, shrinkage_answer
  → PostProcessNode (output gate)                      → formatted_output (gated)
```

### Runtime configuration

`config/agent.yaml` is the **flat registration manifest** (root-level keys only —
identity, entry point, trust level, compile-time requires). Every runtime
parameter lives in `config/config.yaml`, which the platform registry loads and
passes as `Graph(config=...)`; the standalone server (`src/api/server.py`) reads
the same file via `_runtime_config()` so both deployments see identical
configuration. The outer `AgentBaseGraph` consumes `max_retry` from that config
(retry routing); the `retrieval`/`llm` blocks are validated (type, finiteness,
range) in `_parent_config()` before being forwarded — a malformed file can
neither crash graph construction nor weaken the retrieval relevance floor.

**Config → node flow (state-seeded scalars).** No node `execute()` accepts a
`config` parameter. `ShrinkageQAGraphNode._parent_config()` forwards the
validated `retrieval` + `llm` blocks under `config["configurable"]` into
`DomainWorkflowGraph(config=...)` (a graph constructor argument — part of the
`BaseGraph` contract, not a node signature). The inner graph's
`_extra_initial_state()` republishes the `retrieval` block into the inner
initial state as three SCALAR fields — `retrieval_top_k`,
`retrieval_score_threshold`, `retrieval_kb_path` — so config knobs travel
through State. `RetrieveNode` / `RerankFilterNode` read these keys directly off
`state`, re-check them (finite + bounded + strictly typed), and fall back to
module-level defaults that mirror `config/config.yaml` when a key is unseeded
(e.g. a node instantiated directly in a unit test) or malformed.

### Caller-data contract (`input_context`)

`POST /invoke` accepts an optional `input_context` object (adapter-capped at
256 KB serialized). Every declared field is validated in PreProcessNode before
the domain workflow runs; violations fail CLOSED with an error naming the field
— never echoing the value. Undeclared keys are ignored. InputValidateNode
re-validates the bridged parameters with the same rules, so a direct
inner-graph invocation gets the same fail-closed contract.

| Field | Type / bounds | Effect |
|-------|---------------|--------|
| `channel` | str, `^[a-z0-9_]{1,32}$` | recorded in `enriched_context` audit metadata (absent → `"unknown"`) |
| `category` | str, `^[a-z0-9_]{1,32}$` | narrows retrieval to one KB category (absent → no filter) |
| `top_k` | int, 1..20 | narrows retrieval depth for this invocation (absent → configured `retrieval.top_k`; a wider value never overrides a stricter configured cap) |

The relevance floor (`retrieval.score_threshold`) is deliberately **not**
caller-controllable — a caller can never lower the grounding bar.

**Context bridge.** The framework's `GraphNode.execute()` does not forward the
outer state's `input_context` into `subgraph.invoke()` (SDK 1.0.1), so the
outer graph stashes it in a ContextVar (`ShrinkageQAGraphNode.extract_input`)
and the inner graph re-seeds it (`DomainWorkflowGraph._extra_initial_state`) —
see `src/graph/context_bridge.py`. Verified end-to-end: a caller `top_k`
override changes the number of cited sources through the full nested graph.

### External output schema

The answer is built from retrieved knowledge-base content only — **no
caller-supplied text is embedded** (a question smuggling fake `[n]` markers
cannot surface them as pseudo-citations). The output gate (PostProcessNode)
enforces three independent layers:

1. **Credential scan** — API-key/JWT/Bearer/assignment patterns anywhere in the
   answer withhold the output entirely (sanitised stub, `status=error`).
2. **Verbatim caller-text redaction** — a verbatim embedding of the caller's raw
   or normalised question (or request metadata) is replaced with `[REDACTED]`.
3. **Monetary precision grid** — monetary figures are expressed in units of
   1,000; every monetary-form token (comma-grouped, 5+-digit runs, or short
   values in currency context — code or symbol, either side, horizontal
   whitespace or at most one newline, signed) is snapped onto the grid, with an
   audit event. The renderer prints the schema note whenever money renders,
   using the same grammar as the gate.

   Three rules keep the grid off everything that is **not** an amount, because
   this answer renders retail identifiers next to its numbers:

   - **Decimal absorption.** Each value alternative takes its own fraction into
     the same token (`(?:\.\d+|(?!\.\d))`), so a cycle-count KPI
     (`99.99999%`) or a shrink rate (`1.42%`) is never read as a bare digit run
     and rewritten, and an off-grid decimal amount snaps as ONE number
     (`JPY 1234.56` → `JPY 1,000`, never `JPY 1,000.56`). The `(?!\.\d)` arm
     is required: a plain optional fraction lets the engine backtrack out of it
     and re-snap the integer part alone (`JPY 1234.56m`).
   - **Identifier guards** on both ends of the match, over this template's real
     render alphabet `[A-Za-z0-9_-]` (plus `.` on the leading end only, so an
     amount ending a sentence cannot escape). Hyphen covers the record
     references this domain names — `SKU-48210`, `PO-2026-004821`,
     `LP-2026-00318`, and the KB citation ids (`kb-001`); underscore covers the
     inert identifier alphabet the template declares for caller and config
     identifiers (`^[a-z0-9_]{1,32}$`) and its own KB category keys, i.e.
     `sku_48210`.
   - **Attached markers are restricted to ISO-4217 codes.** `<3 uppercase
     letters>-<digits>` is the shape of a retail identifier, and the answer
     renders five such words today (SKU, POS, SOP, EAS, ABC) — none of them a
     currency. Attached, only a real currency code snaps (`JPY-9999`);
     separated, any 3-letter uppercase word still counts as a marker
     (`SKU 48210` → `SKU 48,000`), because a false snap fails safe and a missed
     leak does not.

### State Definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `Optional[str]` | validated, normalised question | outer |
| `enriched_context` | `Optional[str]` (JSON) | channel audit metadata | outer |
| `shrinkage_answer` | `Optional[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `search_query` | `Optional[str]` | normalised search query | inner |
| `query_filters` | `Optional[str]` (JSON) | validated caller parameters (`category`, `top_k`) | inner |
| `retrieval_top_k` | `int` | declared `retrieval.top_k` (scalar) | inner |
| `retrieval_score_threshold` | `float` | declared `retrieval.score_threshold` (scalar) | inner |
| `retrieval_kb_path` | `str` | declared `retrieval.kb_path` (scalar) | inner |
| `retrieved_documents` | `Optional[str]` (JSON) | scored KB candidates | inner |
| `ranked_documents` | `Optional[str]` (JSON) | reranked + threshold-filtered passages | inner |
| `grounded_answer` | `Optional[str]` | rule-assembled grounded answer body | inner |
| `citations` | `Optional[str]` (JSON) | `[{ref, id, title, source}]` | inner |
| `formatted_answer` | `Optional[str]` | final answer + sources + scope disclaimer | inner |
| `intake_notes` | `Optional[str]` (JSON) | validation / parse notes (no PII, no employee-identifying detail) | inner |
| `trace_id` / `correlation_id` | `str` | framework-managed tracing (inherited from `AgentState`, not re-declared) | both |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serialisable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer (checkpoint
  msgpack safety). Retrieval tuning knobs are plain scalars, not JSON.
- `formatted_output` is NOT re-declared (backbone field stays framework-owned).
- No JWT, API keys, credentials, or employee-identifying detail in State.
- `InvocationContext` via `config["configurable"]` only (never in State).
- No Pydantic models / dataclasses / arbitrary Python objects.

## Security Gates

- **Trust gate:** every node declares `required_trust_level` (see tables
  above); `PreProcessNode` (VERIFIED_EXTERNAL) rejects empty / non-string /
  over-long `user_input` and every out-of-contract `input_context` field
  before the inner workflow runs. The standalone server elevates
  authenticated Bearer callers to VERIFIED_EXTERNAL (`INVOKE_AUTH_TOKEN`).
  The framework `BaseNode.__call__` default PII scan additionally masks
  `user_input` / `validated_input` at every node boundary.
- **Output gate:** the canonical `_security_gate_output` is a method on the
  agent class `ShrinkageInventoryQAAgent` (graph.py), delegating to the
  module-level scanner in `post_process_node.py` that the post_process slot
  applies at runtime — one source of truth for the pattern set. The runtime
  gate applies the three layers described under *External output schema*.
  No `_extra_security_gate_input` / `_extra_security_gate_output` instance
  methods are defined on any node, and the inner FunctionNodes do NOT
  override `_security_gate_output` (framework `@final`). `merge_output()` and
  both graphs' `get_output()` surface only vetted SCALAR fields
  (`formatted_answer` / `citations` / `status` — strings, never a raw
  request/response dict), so the gate's top-level-string scan on `result`
  has nothing nested to miss.
- **Audit logging:** every node's `execute()` emits domain-specific
  `emit_trace_event(event, {small non-PII payload}, state)` calls (free
  function, positional args). Nodes do NOT emit `node_start` /
  `node_complete` / `node_error` — `BaseNode.__call__()` emits those. Domain
  event names (also listed in the operation guide):
  - `pre_process_complete` / `pre_process_validation_failed`
  - `input_validate_complete` / `input_validate_failed`
  - `retrieve_complete`
  - `rerank_filter_complete`
  - `generate_answer_complete`
  - `output_format_complete`
  - `post_process_complete` / `post_process_credential_violation` /
    `post_process_blocked_field_redaction` / `post_process_precision_redaction`

## Scope Disclaimer

Every answer carries a standing scope disclaimer: the content is
informational, describes general anomaly patterns and procedures, never
identifies or accuses a specific associate, and is not a substitute for a
formal investigation. It is appended by `OutputFormatNode` as part of the
domain output contract — NOT injected by `post_process` (post_process only
gates).

## Implementation Note — LLM synthesis

The current build of this template is **deterministic end-to-end**: retrieval
is keyword scoring over the seeded KB and `GenerateAnswerNode` assembles the
grounded answer rule-based from the ranked passages (lead sentence + cited
passage excerpts). There is NO live LLM call and no LLM client dependency —
the `llm` config block is forwarded through `_parent_config()` for
forward-compatibility but is not consumed by any current node, and no
`system_prompt` is read at runtime. The LLM synthesis upgrade seam is
documented in `config/prompts/answer_synthesis_prompt.md`: an upgraded
`GenerateAnswerNode` swaps the rule-based assembly for an LLM call that
synthesises over the same `ranked_documents` input and emits the same
`grounded_answer` / `citations` state contract, so no other node changes.
Live vector-store integration is likewise deferred behind the store-agnostic
`retrieved_documents` contract.

## Composition Pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation strategy:** `propagate` (inner errors re-raised as `SubgraphError`).
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; the outer pre_process
  ingest gate runs at `TrustLevel.VERIFIED_EXTERNAL`.

## Import Isolation Confirmation
- [x] Template does not import the agenticstar platform SDK.
- [x] Import targets: `framework/` and `shared/` only.

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG workflow, no autonomous loop |
| Composition pattern | Standalone Cat 1 slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| Answer synthesis | Rule-based assembly | LLM call | **Rule-based (current build)** | The framework ships no LLM client; deterministic assembly is testable; a later build swaps the LLM in at the documented seam |
| KB storage | External vector store | Seeded JSON KB | **Seeded JSON KB** | Self-contained, deterministic CI; the retrieval contract (`retrieved_documents` JSON) is store-agnostic for a later vector-store upgrade |
| Config→node flow | `execute(state, config)` param | State-seeded scalars | **State-seeded scalars** | The node contract is `execute(self, state)` only; tuning knobs flow via `_extra_initial_state()` into State, never a node ctor or execute() param |
| Structured caller params | JSON envelope inside the input string | `input_context` + context bridge | **`input_context` + context bridge** | First-class, field-validated contract at the ingest boundary; the bridge closes the framework's input_context forwarding gap |
| Query echo in answer | Echo the question | Never embed caller text | **Never embed caller text** | The answer's value is that every statement traces to a retrieved source; caller text could smuggle pseudo-citations |
