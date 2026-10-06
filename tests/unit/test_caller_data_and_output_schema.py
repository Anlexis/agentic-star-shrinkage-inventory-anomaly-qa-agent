# RET-C2-005 — Unit Tests: caller-data contract + external output schema
#
# The two hostile boundaries, tested together because they are two ends of the
# same rule: every caller input is validated against explicit bounds before it
# can influence the run (fail CLOSED, never echo), and the external answer
# enforces its documented schema for every representation (monetary values on
# the 1,000 grid; no verbatim caller text).
#
# Sections:
#   1. input_context validation (the ingest boundary contract)
#   2. _finite_in_range (the untrusted-numeric parser)
#   3. Retrieve/Rerank fail-safes (state-seeded scalars, candidate scores)
#   4. Runtime-config plumbing (config/config.yaml -> inner graph, end-to-end)
#   5. Precision gate + blocked-field redaction + schema-note rendering
#
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import re

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.graph.context_bridge import set_caller_input_context
from src.nodes.output_format_node import _SCHEMA_NOTE, OutputFormatNode
from src.nodes.post_process_node import (
    PostProcessNode,
    _enforce_precision,
    _redact_blocked_fields,
)
from src.nodes.pre_process_node import _validate_input_context
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import _finite_in_range, from_json, to_json

_GROUNDED_QUERY = (
    "what common shrinkage anomaly patterns should a store watch for, and "
    "what is the standard procedure for investigating them?"
)


def _inner_state(**fields) -> dict:
    state = {
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(fields)
    return state


# ── 1. input_context validation (ingest boundary) ─────────────────────────────


class TestInputContextValidation:
    def test_absent_context_defaults_channel_unknown(self):
        context, error = _validate_input_context(None)
        assert error is None
        assert context == {"channel": "unknown"}

    def test_valid_full_context(self):
        context, error = _validate_input_context({"channel": "store_ops", "category": "anomaly_patterns", "top_k": 3})
        assert error is None
        assert context == {"channel": "store_ops", "category": "anomaly_patterns", "top_k": 3}

    def test_unknown_keys_are_ignored(self):
        context, error = _validate_input_context({"unexpected": object()})
        assert error is None
        assert "unexpected" not in context

    def test_non_mapping_context_is_rejected(self):
        context, error = _validate_input_context(["not", "a", "dict"])
        assert error == "input_context must be an object"
        assert context == {}

    @pytest.mark.parametrize(
        "bad_channel",
        ["Store-Ops!", "a" * 33, "", 7, ["web"]],
        ids=["punctuated", "too-long", "empty", "int", "list"],
    )
    def test_invalid_channel_fails_closed(self, bad_channel):
        _, error = _validate_input_context({"channel": bad_channel})
        assert error is not None and "channel" in error

    def test_rejected_channel_value_is_never_echoed(self):
        _, error = _validate_input_context({"channel": "Store-Ops!"})
        assert "Store-Ops!" not in error

    @pytest.mark.parametrize(
        "bad_category",
        ["Anomaly Patterns", "b" * 33, "", 3.5, {"k": "v"}],
        ids=["spaces-upper", "too-long", "empty", "float", "dict"],
    )
    def test_invalid_category_fails_closed(self, bad_category):
        _, error = _validate_input_context({"category": bad_category})
        assert error is not None and "category" in error

    @pytest.mark.parametrize(
        "bad_top_k",
        ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), 3.5, True, 0, 21, "3"],
        ids=["str-nan", "str-inf", "str-neginf", "raw-nan", "raw-inf", "float", "bool", "zero", "over", "numeric-str"],
    )
    def test_invalid_top_k_fails_closed(self, bad_top_k):
        _, error = _validate_input_context({"top_k": bad_top_k})
        assert error is not None and "top_k" in error

    def test_rejected_top_k_value_is_never_echoed(self):
        _, error = _validate_input_context({"top_k": 12345})
        assert "12345" not in error

    @pytest.mark.parametrize("good_top_k", [1, 20])
    def test_valid_top_k_bounds(self, good_top_k):
        context, error = _validate_input_context({"top_k": good_top_k})
        assert error is None
        assert context["top_k"] == good_top_k


# ── 2. _finite_in_range (the untrusted-numeric parser) ────────────────────────


class TestFiniteInRange:
    @pytest.mark.parametrize(
        "value",
        [
            "NaN",
            "Infinity",
            "-Infinity",
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
            None,
            [],
            {},
            "abc",
            21.0,
            -1.0,
        ],
        ids=[
            "str-nan",
            "str-inf",
            "str-neginf",
            "raw-nan",
            "raw-inf",
            "raw-neginf",
            "true",
            "false",
            "none",
            "list",
            "dict",
            "text",
            "above",
            "below",
        ],
    )
    def test_rejects_non_finite_and_out_of_range(self, value):
        assert _finite_in_range(value, 0, 20) is None

    @pytest.mark.parametrize("value,expected", [(0, 0.0), (20, 20.0), ("5", 5.0), (2.5, 2.5)])
    def test_accepts_finite_in_range(self, value, expected):
        assert _finite_in_range(value, 0, 20) == expected


# ── 3. Retrieve/Rerank fail-safes ─────────────────────────────────────────────


class TestRetrieveScalarFailSafe:
    @pytest.mark.parametrize(
        "bad_top_k",
        [float("nan"), float("inf"), "lots", True, -3, 99],
        ids=["nan", "inf", "str", "bool", "negative", "over"],
    )
    def test_malformed_seeded_top_k_falls_back_to_default(self, bad_top_k):
        """A malformed retrieval_top_k in (possibly checkpoint-restored) state
        must fall back to the module default, never crash or run unbounded."""
        result = RetrieveNode()(_inner_state(search_query=_GROUNDED_QUERY, retrieval_top_k=bad_top_k))
        candidates = from_json(result["retrieved_documents"])
        # default top_k=4 -> pool = max(4*3, 10) = 12, capped by matches.
        assert 0 < len(candidates) <= 12

    def test_valid_seeded_top_k_is_used(self):
        result = RetrieveNode()(_inner_state(search_query=_GROUNDED_QUERY, retrieval_top_k=1))
        # top_k=1 -> pool = max(1*3, 10) = 10, capped by matching entries.
        assert len(from_json(result["retrieved_documents"])) <= 10


class TestRerankThresholdFailSafe:
    @pytest.mark.parametrize(
        "bad_threshold",
        [float("nan"), float("inf"), "0.1", True, -0.1, 1.5],
        ids=["nan", "inf", "str", "bool", "below", "above"],
    )
    def test_malformed_threshold_never_weakens_the_gate(self, bad_threshold):
        """A malformed seeded threshold falls back to the default (0.25) — a
        NaN would otherwise compare False against every score and drop every
        passage silently (or, inverted, pass every low-relevance one)."""
        low_relevance = [{"id": "kb-x", "title": "t", "category": "c", "source": "s", "score": 0.1, "excerpt": "e"}]
        result = RerankFilterNode()(
            _inner_state(
                retrieved_documents=to_json(low_relevance),
                retrieval_score_threshold=bad_threshold,
            )
        )
        assert from_json(result["ranked_documents"]) == []

    def test_non_finite_candidate_score_never_clears_the_gate(self):
        chunks = [{"id": "kb-x", "title": "t", "category": "c", "source": "s", "score": float("nan"), "excerpt": "e"}]
        result = RerankFilterNode()(_inner_state(retrieved_documents=to_json(chunks)))
        assert from_json(result["ranked_documents"]) == []

    def test_caller_top_k_narrows_but_never_widens(self):
        docs = [
            {"id": f"kb-{i}", "title": "t", "category": "c", "source": "s", "score": 0.9, "excerpt": "e"}
            for i in range(6)
        ]
        narrowed = RerankFilterNode()(
            _inner_state(
                retrieved_documents=to_json(docs),
                retrieval_top_k=4,
                query_filters=to_json({"category": None, "top_k": 2}),
            )
        )
        assert len(from_json(narrowed["ranked_documents"])) == 2

        widened = RerankFilterNode()(
            _inner_state(
                retrieved_documents=to_json(docs),
                retrieval_top_k=4,
                query_filters=to_json({"category": None, "top_k": 6}),
            )
        )
        assert len(from_json(widened["ranked_documents"])) == 4


# ── 4. Runtime-config plumbing ────────────────────────────────────────────────


class TestRuntimeConfigPlumbing:
    def test_runtime_config_reads_repo_config(self):
        import src.graph.graph as graph_module

        cfg = graph_module._runtime_config()
        assert cfg.get("max_retry") == 3
        assert cfg.get("retrieval", {}).get("top_k") == 4
        assert cfg.get("retrieval", {}).get("score_threshold") == 0.25

    def test_missing_config_file_degrades_to_empty(self, monkeypatch, tmp_path):
        import src.graph.graph as graph_module

        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", tmp_path / "absent.yaml")
        assert graph_module._runtime_config() == {}

    def test_declared_config_reaches_inner_state_seeding(self):
        from src.graph.graph import ShrinkageInventoryQAAgent

        set_caller_input_context(None)
        agent = ShrinkageInventoryQAAgent()
        agent.compile()
        subgraph = agent._nodes["main"].get_subgraph()
        extra = subgraph._extra_initial_state()
        assert extra["retrieval_top_k"] == 4
        assert extra["retrieval_score_threshold"] == 0.25
        assert extra["retrieval_kb_path"] == "config/kb/shrinkage_kb.json"

    def test_caller_top_k_reaches_inner_retrieval_end_to_end(self, monkeypatch, tmp_path):
        """The context bridge, proven through the FULL nested graph: with the
        relevance floor opened (threshold 0), the number of cited sources in
        the final answer equals the caller's top_k override."""
        import src.graph.graph as graph_module
        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel as TL

        open_gate = tmp_path / "config.yaml"
        open_gate.write_text("max_retry: 3\nretrieval:\n  top_k: 5\n  score_threshold: 0.0\n")
        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", open_gate)
        agent = graph_module.ShrinkageInventoryQAAgent(config=graph_module._runtime_config())
        agent.compile()
        ctx = InvocationContext(caller_trust_level=TL.VERIFIED_EXTERNAL)
        for requested, expected in ((1, 1), (2, 2), (None, 5)):
            input_context = {"top_k": requested} if requested else {}
            result = agent.invoke(_GROUNDED_QUERY, ctx=ctx, input_context=input_context)
            assert result["status"] == AgentStatus.SUCCESS.value
            markers = set(re.findall(r"^- \[(\d+)\]", result["output"], re.M))
            assert (
                len(markers) == expected
            ), f"top_k={requested}: expected {expected} cited sources, got {sorted(markers)}"


# ── 5. External output schema enforcement ─────────────────────────────────────


class TestPrecisionGate:
    @pytest.mark.parametrize(
        "leak,expected",
        [
            ("JPY 9999", "JPY 10,000"),
            ("9999 JPY", "10,000 JPY"),
            ("JPY 1,234", "JPY 1,000"),
            ("1,234 JPY", "1,000 JPY"),
            ("JPY 1,234,567", "JPY 1,235,000"),
            ("¥9999", "¥10,000"),
            ("￥9999", "￥10,000"),
            ("9999円", "10,000円"),
            ("JPY -9999", "JPY -10,000"),
            ("JPY +9999", "JPY +10,000"),
            ("+9999 JPY", "+10,000 JPY"),
            ("JPY\t9999", "JPY\t10,000"),
            ("JPY  9999", "JPY  10,000"),
            ("JPY\n9999", "JPY\n10,000"),
            ("9,999", "10,000"),
            ("123456", "123,000"),
            ("-12345", "-12,000"),
            ("1,234,567", "1,235,000"),
        ],
        ids=[
            "marker-value",
            "value-marker",
            "marker-grouped",
            "grouped-marker",
            "marker-grouped-millions",
            "symbol",
            "fullwidth-symbol",
            "yen-suffix",
            "signed-neg",
            "signed-pos",
            "signed-pos-before",
            "tab",
            "double-space",
            "newline",
            "grouped-bare",
            "long-run",
            "signed-run",
            "grouped-millions",
        ],
    )
    def test_every_leak_form_snaps(self, leak, expected):
        sanitised, redactions = _enforce_precision(leak)
        assert sanitised == expected
        assert redactions == 1

    @pytest.mark.parametrize(
        "structural",
        [
            "register 4",
            "sku v12",
            "in 2026",
            "STAR 2026",
            "90d",
            "10,000",
            "JPY 1,000",
            "3 sources",
            "year 2026",
            "12 units",
        ],
    )
    def test_structural_and_on_grid_tokens_byte_identical(self, structural):
        sanitised, redactions = _enforce_precision(structural)
        assert sanitised == structural
        assert redactions == 0

    # ── Precision-gate regression suite ───────────────────────────────────────
    # The published grammar snapped the FRACTION of a decimal, backtracked out
    # of an absorbed fraction, and read every 3-letter uppercase word as a
    # currency marker. On this template that rewrote the identifiers the answer
    # exists to name: "SKU-48210" -> "SKU-48,000", "POS-07" -> "POS0",
    # "LP-2026-00318" -> "LP-20260", and the cycle-count KPI "99.99999%" ->
    # "99.100,000%". Each case below is one of those failures, pinned.

    @pytest.mark.parametrize(
        "text",
        [
            # Decimals are not monetary tokens and must survive byte-identical.
            "8.512345",
            "cycle-count accuracy 99.99999%",
            "shrink rate of 1.42%",
            "ratio 0.123456",
            # The fraction is absorbed into the token, so the engine cannot
            # backtrack out of it and re-snap the integer part alone.
            "JPY 1234.56m",
            # Retail identifiers this template renders (config/kb/shrinkage_kb.json
            # vocabulary: SKU, POS, EAS, ABC, SOP, PO number, employee ID,
            # carrier seal number, LP case file) plus this repo's own KB
            # citation ids.
            "SKU-48210",
            "sku_48210",
            "SKU_48210",
            "kb-001",
            "PO-2026-004821",
            "LP-2026-00318",
            "carrier seal 88421-B",
            "employee ID-40912",
            "EAS-1024 tag",
            "POS-07 register",
            "ABC-3 tier",
            "SOP-2026-014",
            # Bundled-KB prose that must not be touched.
            "A-tier SKUs are counted every 30 days",
            "rolling 30-day baseline",
            "exceeds 2% of unit count",
            "within 24 hours",
            # The marker/value delimiter never crosses a paragraph break, so a
            # numbered heading after a 3-letter code keeps its own number.
            "Currency: JPY\n\n3. Cash Position",
            "SKU\n\n2026 units short",
            # CJK word boundaries. 万 円 条 日 人 are all `\\w`, so the ASCII
            # identifier guards ADD to the grammar's `\\b` assertions and never
            # replace them: with an ASCII-only trailing guard the 円 below binds
            # to "+600" and a fixed figure is rewritten to "+1,000万円".
            "（3000万円+600万円×法定相続人の数）",
            "控除額3000万円",
        ],
    )
    def test_decimals_and_domain_identifiers_are_byte_identical(self, text):
        sanitised, redactions = _enforce_precision(text)
        assert sanitised == text
        assert redactions == 0

    @pytest.mark.parametrize(
        "leak,expected",
        [
            # An off-grid decimal amount in currency context snaps as ONE
            # number - never integer-snapped with the fraction left dangling.
            ("JPY 1234.56", "JPY 1,000"),
            ("USD-1234.56", "USD-1,000"),
            ("JPY 1,234", "JPY 1,000"),
            # An amount that ends a sentence must not escape: `.` is in the
            # leading guard only.
            ("The quarter totals JPY 9999.", "The quarter totals JPY 10,000."),
            # ATTACHED marker: restricted to real ISO-4217 codes, so a currency
            # still snaps while SKU-/POS-/EAS- identifiers survive (above).
            ("JPY-9999", "JPY-10,000"),
            # SEPARATED marker stays unrestricted - a 3-letter word plus
            # whitespace plus a number is currency context, and a false snap
            # fails safe.
            ("SKU 48210", "SKU 48,000"),
            # Restoring `\\b` must not cost the CJK leak forms the grammar
            # already caught: a 円-suffixed amount still snaps.
            ("在庫差異 9999円", "在庫差異 10,000円"),
        ],
    )
    def test_off_grid_amounts_still_snap(self, leak, expected):
        sanitised, redactions = _enforce_precision(leak)
        assert sanitised == expected
        assert redactions == 1

    def test_on_grid_amount_is_byte_identical(self):
        """The control: an amount already on the grid is never rewritten."""
        sanitised, redactions = _enforce_precision("JPY 1,000")
        assert sanitised == "JPY 1,000"
        assert redactions == 0


class TestBlockedFieldRedaction:
    def test_verbatim_caller_text_is_redacted(self):
        question = "did register four void nineteen transactions this shift"
        doc = f"HEADER\n{question}\nFOOTER"
        sanitised, fields = _redact_blocked_fields(doc, {"validated_input": question})
        assert question not in sanitised
        assert "[REDACTED]" in sanitised
        assert fields == ["validated_input"]

    def test_short_incidental_overlap_is_not_redacted(self):
        doc = "Voided transactions cluster on one register."
        sanitised, fields = _redact_blocked_fields(doc, {"validated_input": "register"})
        assert sanitised == doc
        assert fields == []


class TestPostProcessGateLayers:
    def test_off_grid_money_is_snapped_on_success_path(self):
        report = "# Shrinkage & Inventory Anomaly Q&A Result\nestimated loss JPY 123,456 per quarter."
        result = PostProcessNode()(
            {
                "result": report,
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "node_history": [],
                "error_log": [],
                "session_id": "unit-session",
                "execution_time": {},
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "123,456" not in result["formatted_output"]
        assert "JPY 123,000" in result["formatted_output"]

    def test_embedded_caller_question_is_redacted(self):
        question = "how much shrink can one register hide per quarter"
        report = f"# Shrinkage & Inventory Anomaly Q&A Result\n{question}\nanswer body"
        result = PostProcessNode()(
            {
                "result": report,
                "validated_input": question,
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "node_history": [],
                "error_log": [],
                "session_id": "unit-session",
                "execution_time": {},
            }
        )
        assert question not in result["formatted_output"]
        assert "[REDACTED]" in result["formatted_output"]
        assert result["status"] == AgentStatus.SUCCESS.value


class TestSchemaNoteRendering:
    def test_note_rendered_when_money_renders(self):
        result = OutputFormatNode()(
            _inner_state(
                grounded_answer="[1] Escalate when the estimated loss exceeds JPY 10,000.",
                citations=to_json([]),
            )
        )
        assert _SCHEMA_NOTE in result["formatted_answer"]

    def test_note_absent_when_no_money_renders(self):
        result = OutputFormatNode()(
            _inner_state(
                grounded_answer="[1] Escalate repeat patterns to the regional lead.",
                citations=to_json([]),
            )
        )
        assert _SCHEMA_NOTE not in result["formatted_answer"]
