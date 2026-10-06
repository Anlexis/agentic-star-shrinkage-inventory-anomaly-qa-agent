# RET-C2-005 — Unit Tests: InputValidateNode (inner domain node 1)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# (inner Cat-2 domain node). Payloads are lowercase / PII-free so the
# framework input mask leaves them untouched.
#
# The caller's structured parameters (category, top_k) arrive via
# input_context — seeded into inner state by the context bridge — and are
# re-validated here with the same fail-closed rules as the ingest boundary.
#
# Mirrors docs/03_test_spec.md section 2.2 (VAL-01..VAL-09).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json


def _make_state(payload, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPlainTextParsing:
    def test_val_01_plain_text_becomes_query(self):
        result = InputValidateNode()(_make_state("shrinkage anomaly patterns"))
        assert result["search_query"] == "shrinkage anomaly patterns"
        filters = from_json(result["query_filters"])
        assert filters == {"category": None, "top_k": None}

    def test_val_02_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  shrinkage   anomaly\n patterns "))
        assert result["search_query"] == "shrinkage anomaly patterns"

    def test_query_filters_is_json_string(self):
        # Structured State fields travel as JSON strings, never dicts
        # (checkpoint serialization safety).
        result = InputValidateNode()(_make_state("shrinkage anomaly patterns"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestInputContextParameters:
    """VAL-03/04: category + top_k arrive via the bridged input_context."""

    def test_val_03_category_and_top_k_from_input_context(self):
        result = InputValidateNode()(
            _make_state(
                "investigation escalation steps",
                input_context={"category": "investigation_procedures", "top_k": 2},
            )
        )
        assert result["search_query"] == "investigation escalation steps"
        filters = from_json(result["query_filters"])
        assert filters == {"category": "investigation_procedures", "top_k": 2}

    def test_val_04_absent_input_context_means_no_filters(self):
        result = InputValidateNode()(_make_state("cycle count recount triggers"))
        assert from_json(result["query_filters"]) == {"category": None, "top_k": None}

    def test_unknown_context_keys_are_ignored(self):
        result = InputValidateNode()(_make_state("pos exception checks", input_context={"unexpected": "x"}))
        assert from_json(result["query_filters"]) == {"category": None, "top_k": None}


class TestTopKGuard:
    """VAL-05..07: the caller-supplied top_k is untrusted and fails CLOSED."""

    @pytest.mark.parametrize(
        "bad_top_k",
        [99, -5, 0, "many", "NaN", "Infinity", "-Infinity", float("nan"), float("inf"), 3.5, True],
        ids=[
            "over-range",
            "negative",
            "zero",
            "str",
            "str-nan",
            "str-inf",
            "str-neginf",
            "raw-nan",
            "raw-inf",
            "float",
            "bool",
        ],
    )
    def test_val_05_invalid_top_k_fails_closed(self, bad_top_k):
        result = InputValidateNode()(_make_state("shrinkage anomaly patterns", input_context={"top_k": bad_top_k}))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("input_context.top_k" in e for e in result["error_log"])
        assert "search_query" not in result

    def test_val_06_rejected_top_k_value_is_never_echoed(self):
        result = InputValidateNode()(_make_state("shrinkage anomaly patterns", input_context={"top_k": 99}))
        assert "99" not in " ".join(result["error_log"])

    @pytest.mark.parametrize("good_top_k", [1, 20])
    def test_val_07_boundary_top_k_accepted(self, good_top_k):
        result = InputValidateNode()(_make_state("shrinkage anomaly patterns", input_context={"top_k": good_top_k}))
        assert from_json(result["query_filters"])["top_k"] == good_top_k


class TestCategoryGuard:
    """The caller-supplied category is locked to an inert identifier."""

    @pytest.mark.parametrize(
        "bad_category",
        ["POS Exceptions", "a" * 33, "", "café", 7, ["investigation_procedures"]],
        ids=["spaces-upper", "too-long", "empty", "non-ascii", "int", "list"],
    )
    def test_invalid_category_fails_closed(self, bad_category):
        result = InputValidateNode()(_make_state("pos exception checks", input_context={"category": bad_category}))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("input_context.category" in e for e in result["error_log"])

    def test_rejected_category_value_is_never_echoed(self):
        result = InputValidateNode()(_make_state("pos exception checks", input_context={"category": "POS Exceptions!"}))
        assert "POS Exceptions!" not in " ".join(result["error_log"])

    def test_valid_category_is_forwarded(self):
        result = InputValidateNode()(
            _make_state("pos exception checks", input_context={"category": "prevention_playbooks"})
        )
        assert from_json(result["query_filters"])["category"] == "prevention_playbooks"


class TestSizeAndEmptyGuards:
    def test_val_08_oversize_query_is_truncated(self):
        payload = "shrinkage " * 300  # ~3000 chars after collapse
        result = InputValidateNode()(_make_state(payload))
        assert len(result["search_query"]) == 2000
        notes = from_json(result.get("intake_notes"), [])
        assert any("truncated" in n for n in notes)

    def test_val_09_empty_request_yields_note_not_error(self):
        result = InputValidateNode()(_make_state(""))
        assert result["search_query"] == ""
        notes = from_json(result.get("intake_notes"), [])
        assert any("empty request" in n for n in notes)
