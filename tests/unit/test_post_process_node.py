# RET-C2-005 — Unit Tests: PostProcessNode (outer post_process slot; output gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is
# declared ANONYMOUS — trust is enforced once, at the pre_process boundary —
# so these behavioural tests exercise the gate layers, not the trust matrix
# (that lives in test_trust_gate.py).
#
# Gate layering: the node's own module-level _security_gate_output() scan runs
# INSIDE execute() and replaces a violating answer with the sanitised stub
# (returned dict — no exception). The framework's FunctionNode output scan
# then sees only the clean stub. Intentional-credential tests assert the raw
# secret never survives into formatted_output OR result. The verbatim
# caller-text redaction and the monetary precision grid are covered here and
# in test_caller_data_and_output_schema.py.
#
# Mirrors docs/03_test_spec.md section 2.7 (POST-01..POST-06).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import PostProcessNode

_CLEAN_REPORT = (
    "# Shrinkage & Inventory Anomaly Q&A Result\n\n"
    "[1] recurring void clusters on a single register are a common shrinkage anomaly signal.\n"
)

# JWT-shaped token built at runtime so no credential-shaped literal ever sits
# in the repository (credential-scan hygiene).
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert type(result["status"]) is str  # noqa: E721
        assert result["formatted_output"] == _CLEAN_REPORT

    def test_post_02_empty_result_yields_fallback_message(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "No answer content generated" in result["formatted_output"]

    def test_shrinkage_answer_is_the_fallback_source(self):
        state = _make_state("")
        del state["result"]
        state["shrinkage_answer"] = _CLEAN_REPORT
        result = PostProcessNode()(state)
        assert result["formatted_output"] == _CLEAN_REPORT


class TestPostProcessCredentialGate:
    def _assert_blocked(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("credential pattern detected" in str(e) for e in result["error_log"])
        # The raw secret must not survive into either surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "[ANSWER REDACTED" in result["formatted_output"]

    def test_post_03_api_key_is_blocked(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"# Report\n\n<!-- debug api_key={secret} -->\n"))
        self._assert_blocked(result, secret)

    def test_post_04_credential_assignment_is_blocked(self):
        secret = "password=super_secret_value_123"
        result = PostProcessNode()(_make_state(f"# Report\n\ninternal note: {secret}\n"))
        self._assert_blocked(result, "super_secret_value_123")

    def test_post_05_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nsession token {_FAKE_JWT}\n"))
        self._assert_blocked(result, _FAKE_JWT)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"# Report\n\nauthorization: {secret}\n"))
        self._assert_blocked(result, secret)


class TestBlockedFieldRedaction:
    def test_verbatim_caller_text_is_redacted(self):
        question = "did register four void nineteen transactions this shift"
        result = PostProcessNode()(_make_state(f"# Report\n\nanswering: {question}\n", validated_input=question))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert question not in result["formatted_output"]
        assert "[REDACTED]" in result["formatted_output"]

    def test_short_incidental_overlap_is_not_redacted(self):
        # <= 10 chars: incidental overlap, not a verbatim embedding.
        result = PostProcessNode()(_make_state(_CLEAN_REPORT, validated_input="register"))
        assert "[REDACTED]" not in result["formatted_output"]


class TestPrecisionGrid:
    def test_off_grid_monetary_value_is_snapped(self):
        result = PostProcessNode()(_make_state("# Report\n\nestimated loss JPY 123,456 this quarter\n"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "123,456" not in result["formatted_output"]
        assert "JPY 123,000" in result["formatted_output"]

    def test_on_grid_and_structural_tokens_are_byte_identical(self):
        text = "# Report\n\nJPY 1,000 threshold; 12 units on register 4 in 2026\n"
        result = PostProcessNode()(_make_state(text))
        assert result["formatted_output"] == text
