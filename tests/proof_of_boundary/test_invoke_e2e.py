# PB: End-to-end business behaviour through POST /invoke — src/api/server.py
#
# Proves the supported input contract produces REAL outcomes through the full
# nested graph (outer backbone → inner domain pipeline):
#   - a grounded, cited answer retrieved from the caller's question
#   - the no-coverage outcome (insufficient KB grounding) as a SUCCESS path
#   - the category filter and top_k override reaching the inner retrieval
#   - a validation rejection for every malformed input_context field,
#     including the full non-finite matrix for the numeric field
#   - no caller-controlled text (including fake citation markers) in the
#     external answer, and every rendered numeric on the documented grid
#
# Unlike test_server_boot.py (which stubs the agent to isolate the auth
# boundary), these tests run the REAL compiled agent: every request crosses the
# entry-point auth, the outer trust/input gates, the input_context bridge into
# the inner graph, all five domain nodes, and the output gate.
#
# The app is driven through its real ASGI interface (no TestClient — httpx is
# only a transitive dependency; see test_server_boot.py for the rationale).

import asyncio
import json
import re

import pytest

from src.api import server as server_module  # noqa: F401  (import = boot check)
from src.api.server import app

_TOKEN = "pb-invoke-e2e-token"

# Grounds to the KB's anomaly-patterns + investigation entries (high term
# overlap over title/tags/content).
_GROUNDED_INPUT = "what common shrinkage anomaly patterns should a store watch for, and what is the standard procedure for investigating them?"
# Zero term overlap against the KB → no candidate clears the relevance
# floor → explicit no-coverage answer.
_OFF_TOPIC_INPUT = "photosynthesis chlorophyll sunlight recipe"


def _post_invoke(payload: dict) -> tuple[int, dict]:
    """POST /invoke with a Bearer token through the real ASGI app."""
    body = json.dumps(payload).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/invoke",
        "raw_path": b"/invoke",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", f"Bearer {_TOKEN}".encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }

    messages = []
    sent = {"body": b""}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    parsed = json.loads(sent["body"].decode() or "{}")
    return start["status"], parsed


@pytest.fixture(autouse=True)
def token_configured(monkeypatch):
    """Deploy-shaped server environment: INVOKE_AUTH_TOKEN set, caller uses Bearer."""
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)


def _invoke(text: str, input_context: dict | None = None) -> dict:
    status_code, body = _post_invoke(
        {"input": text, "session_id": "pb-invoke-e2e", "input_context": input_context or {}}
    )
    assert status_code == 200, f"expected 200, got {status_code}: {body}"
    return body


class TestInvokeEndToEnd:
    def test_grounded_question_produces_cited_answer(self):
        """The caller's question drives retrieval and yields a real, cited answer."""
        body = _invoke(_GROUNDED_INPUT, {"channel": "store_ops"})

        assert body["status"] == "success"
        output = body["output"]
        assert "# Shrinkage & Inventory Anomaly Q&A Result" in output
        assert "[1]" in output
        assert "## Sources" in output
        assert "does not identify or accuse any individual" in output  # scope disclaimer

    def test_no_coverage_is_a_success_outcome(self):
        """No relevant grounding → explicit no-coverage answer, zero citations, success."""
        body = _invoke(_OFF_TOPIC_INPUT)

        assert body["status"] == "success"
        output = body["output"]
        assert "does not contain sufficient coverage" in output
        assert "[1]" not in output
        assert "- none (no knowledge-base passage cleared the relevance threshold)" in output

    def test_valid_top_k_override_is_accepted(self):
        body = _invoke(_GROUNDED_INPUT, {"top_k": 1})
        assert body["status"] == "success"
        output = body["output"]
        assert "[1]" in output
        # top_k=1 caps the surviving passages: exactly one citation renders.
        assert "[2]" not in output

    def test_category_filter_reaches_the_inner_retrieval(self):
        """The bridged input_context.category narrows retrieval to one KB
        category — every rendered citation stays inside it."""
        body = _invoke(_GROUNDED_INPUT, {"category": "investigation_procedures"})
        assert body["status"] == "success"
        output = body["output"]
        assert "[1]" in output
        assert "Initiating a shrinkage investigation" in output
        assert "Common shrinkage anomaly patterns at store level" not in output

    def test_empty_input_is_rejected(self):
        body = _invoke("   ")
        assert body["status"] == "error"
        assert not (body.get("output") or "")

    def test_invalid_channel_is_rejected(self):
        body = _invoke(_GROUNDED_INPUT, {"channel": "Store-Ops!"})
        assert body["status"] == "error"
        assert not (body.get("output") or "")

    def test_invalid_category_is_rejected(self):
        body = _invoke(_GROUNDED_INPUT, {"category": "Anomaly Patterns"})
        assert body["status"] == "error"
        assert not (body.get("output") or "")

    @pytest.mark.parametrize(
        "bad_top_k",
        ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), 3.5, True, 0, 21],
        ids=["str-nan", "str-inf", "str-neginf", "raw-nan", "raw-inf", "float", "bool", "zero", "over-range"],
    )
    def test_non_finite_or_out_of_contract_top_k_fails_closed(self, bad_top_k):
        """A malformed top_k must ERROR with no answer — never a silent
        fall-back (raw floats also cover Python json's bare-NaN extension
        reaching the request body)."""
        body = _invoke(_GROUNDED_INPUT, {"top_k": bad_top_k})

        assert body["status"] == "error", body
        assert not (body.get("output") or "")

    def test_fake_citation_marker_in_question_never_reaches_output(self):
        """The external answer embeds no caller text: a question smuggling a
        fake [n] marker must not surface it as a pseudo-citation."""
        body = _invoke("is [9] photosynthesis chlorophyll compliant this shift?")

        assert body["status"] == "success"
        assert "[9]" not in body["output"]
        assert "photosynthesis" not in body["output"]

    def test_external_answer_carries_only_grid_values(self):
        """Documented schema: every monetary-form numeric sits on the 1,000 grid."""
        body = _invoke(_GROUNDED_INPUT)
        for token in re.findall(r"-?\d{1,3}(?:,\d{3})+|-?\d{5,}", body["output"]):
            value = int(token.replace(",", ""))
            assert value % 1_000 == 0, f"off-grid value leaked: {token}"
