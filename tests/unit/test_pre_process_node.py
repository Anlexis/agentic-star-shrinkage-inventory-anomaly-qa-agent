# RET-C2-005 — Unit Tests: PreProcessNode (outer pre_process slot)
#
# Invocation canon: every test invokes the node via node(state) —
# BaseNode.__call__ → trust gate → input mask → execute() → output gate —
# never a bare node.execute(state). PreProcessNode requires
# VERIFIED_EXTERNAL, so its behavioural tests build the state at that level
# (the ANONYMOUS rejection lives in test_trust_gate.py).
#
# Input-mask layering note: the FRAMEWORK's default input gate
# (framework.nodes.function_node.FunctionNode._security_gate_input, backed by
# shared.security.pii_detector.detect_pii) masks user_input / validated_input
# / llm_response BEFORE execute() runs — e-mail, SSN/phone/CC digit groups,
# and 2+ consecutive Title-Case words ("name") surface as [MASKED]. This
# node's own execute() adds no further identifier screening (no IBAN /
# account-number strip — that is a finance-specific node behaviour this
# template does not implement; see src/nodes/pre_process_node.py). Positive-
# path payloads are therefore lowercase, PII-free store-ops phrasing;
# intentional-PII tests assert the raw identifier is gone and [MASKED] is
# present.
#
# Mirrors docs/03_test_spec.md section 2.1 (PRE-01..PRE-08).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.pre_process_node
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import from_json

# Lowercase store-ops phrasing on purpose: PII-free (no Title-Case bigram,
# no @, no digit run), so the framework input mask leaves the payload untouched.
_VALID_QUERY = (
    "what common shrinkage anomaly patterns should a store watch for, and "
    "what is the standard procedure for investigating them?"
)


def _make_state(user_input=_VALID_QUERY, **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPreProcessSuccess:
    def test_pre_01_valid_query_accepted(self):
        result = PreProcessNode()(_make_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert type(result["status"]) is str  # noqa: E721
        assert result["validated_input"] == _VALID_QUERY

    def test_enriched_context_carries_channel(self):
        result = PreProcessNode()(_make_state(input_context={"channel": "web"}))
        context = from_json(result["enriched_context"])
        assert context["channel"] == "web"
        assert context["source"] == "ShrinkageInventoryQAAgent"

    def test_enriched_context_is_json_string(self):
        # Structured State fields travel as JSON strings, never dicts
        # (checkpoint serialization safety).
        result = PreProcessNode()(_make_state())
        assert isinstance(result["enriched_context"], str)

    def test_missing_channel_defaults_to_unknown(self):
        result = PreProcessNode()(_make_state())
        assert from_json(result["enriched_context"])["channel"] == "unknown"

    def test_control_characters_are_stripped(self):
        result = PreProcessNode()(_make_state(user_input="voided\x00 transactions\x1b per shift"))
        assert result["validated_input"] == "voided transactions per shift"


class TestPreProcessRejection:
    def test_pre_02_empty_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input=""))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"]
        # No validated_input is produced on the reject path.
        assert "validated_input" not in result

    def test_whitespace_only_is_error(self):
        result = PreProcessNode()(_make_state(user_input="   \n\t "))
        assert result["status"] == AgentStatus.ERROR.value

    def test_pre_03_missing_user_input_is_error(self):
        state = _make_state()
        del state["user_input"]
        result = PreProcessNode()(state)
        assert result["status"] == AgentStatus.ERROR.value

    def test_non_string_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input={"malicious": "dict"}))
        assert result["status"] == AgentStatus.ERROR.value

    def test_over_long_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input="q" * 2001))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("2000" in e for e in result["error_log"])


class TestInputContextContract:
    """PRE-05..07: every declared input_context field is bounds-checked and
    fails CLOSED; rejected values are never echoed."""

    def test_non_mapping_context_is_rejected(self):
        result = PreProcessNode()(_make_state(input_context="not-an-object"))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("input_context" in e for e in result["error_log"])

    @pytest.mark.parametrize(
        "bad_channel",
        ["Store-Ops!", "A" * 33, "", 7, ["web"]],
        ids=["punctuated", "too-long", "empty", "int", "list"],
    )
    def test_invalid_channel_fails_closed(self, bad_channel):
        result = PreProcessNode()(_make_state(input_context={"channel": bad_channel}))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("input_context.channel" in e for e in result["error_log"])

    def test_rejected_channel_value_is_never_echoed(self):
        result = PreProcessNode()(_make_state(input_context={"channel": "Store-Ops!"}))
        assert "Store-Ops!" not in " ".join(result["error_log"])

    @pytest.mark.parametrize(
        "bad_category",
        ["Anomaly Patterns", "b" * 33, "", 3.5],
        ids=["spaces-upper", "too-long", "empty", "float"],
    )
    def test_invalid_category_fails_closed(self, bad_category):
        result = PreProcessNode()(_make_state(input_context={"category": bad_category}))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("input_context.category" in e for e in result["error_log"])

    @pytest.mark.parametrize(
        "bad_top_k",
        ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), 3.5, True, 0, 21],
        ids=["str-nan", "str-inf", "str-neginf", "raw-nan", "raw-inf", "float", "bool", "zero", "over-range"],
    )
    def test_non_finite_or_out_of_contract_top_k_fails_closed(self, bad_top_k):
        result = PreProcessNode()(_make_state(input_context={"top_k": bad_top_k}))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("input_context.top_k" in e for e in result["error_log"])

    def test_valid_full_context_is_accepted(self):
        result = PreProcessNode()(
            _make_state(input_context={"channel": "store_ops", "category": "anomaly_patterns", "top_k": 3})
        )
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_unknown_context_keys_are_ignored(self):
        result = PreProcessNode()(_make_state(input_context={"unexpected": object()}))
        assert result["status"] == AgentStatus.SUCCESS.value


class TestPreProcessFrameworkInputMask:
    """PRE-04: the FRAMEWORK's default input gate masks PII before execute() runs.

    This node adds no domain-specific identifier screen of its own, so only
    the framework-level patterns (email, grouped digits) are exercised here.
    """

    def test_email_masked_by_framework_input_gate(self):
        raw = "escalate the anomaly review to lp.coordinator@example.com today"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "lp.coordinator@example.com" not in vi
        assert "[MASKED]" in vi

    def test_grouped_digits_masked(self):
        # 4-4-4 digit groups match the framework's my_number_jp pattern.
        raw = "case 1234 5678 9012 shows a pending variance flag"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "1234 5678 9012" not in vi
        assert "[MASKED]" in vi


class TestPreProcessAudit:
    def test_pre_08_domain_audit_payload(self, monkeypatch):
        """The accepted request emits pre_process_complete; the assertion
        targets call.args[1] — the event payload — never the whole call repr."""
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.pre_process_node, "emit_trace_event", spy)
        PreProcessNode()(_make_state())
        events = [call.args[0] for call in spy.call_args_list]
        assert "pre_process_complete" in events
        payload = spy.call_args_list[events.index("pre_process_complete")].args[1]
        assert payload["input_chars"] == len(_VALID_QUERY)

    def test_rejected_context_emits_validation_failed_audit(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.pre_process_node, "emit_trace_event", spy)
        PreProcessNode()(_make_state(input_context={"top_k": 99}))
        events = [call.args[0] for call in spy.call_args_list]
        assert "pre_process_validation_failed" in events
        payload = spy.call_args_list[events.index("pre_process_validation_failed")].args[1]
        # The audit payload names the reason class, never the rejected value.
        assert payload == {"reason": "invalid_input_context"}
