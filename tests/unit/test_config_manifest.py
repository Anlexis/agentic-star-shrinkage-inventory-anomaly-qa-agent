# RET-C2-005 — Unit Tests: manifest / config consistency
#
# config/agent.yaml is the flat registry manifest (identity + entry point);
# config/config.yaml carries the runtime parameters that
# ShrinkageQAGraphNode._parent_config() forwards into the inner graph. The
# class-name contract requires the declared entry point to BE the
# src/graph/graph.py agent class. These tests pin manifest <-> code
# consistency so a config drift fails fast in CI.
#
# Mirrors docs/03_test_spec.md section 2.8 (CFG-01..CFG-07).
# Deterministic — no LLM, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import ShrinkageInventoryQAAgent, ShrinkageQAGraphNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_template_id_is_consistent(self):
        assert _MANIFEST["id"] == "RET-C2-005"
        assert _MANIFEST["namespace"] == "ret"
        assert _MANIFEST["enabled"] is True

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # Class-name contract: manifest entry point == graph.py class == server import.
        assert _MANIFEST["class"] == "src.graph.graph.ShrinkageInventoryQAAgent"
        module_path, _, class_name = _MANIFEST["class"].rpartition(".")
        assert class_name == ShrinkageInventoryQAAgent.__name__
        assert _MANIFEST["name"] == ShrinkageInventoryQAAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "RET"
        assert _MANIFEST["base_type"] == "RAGAgent"
        assert _MANIFEST["generation_mode"] == "deterministic"

    def test_requires_matches_code(self):
        # No ctx.secrets.require() calls and no LLM client construction in
        # src/ — the declared compile-time gates must stay empty (a declared
        # secret/extra that is not provisioned fails at deploy compile).
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared

    def test_cfg_05_max_retry_within_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # AgentBaseGraph MAX_RETRY_CEILING

    def test_hitl_is_not_enabled(self):
        # PB-7 auto-waiver contract: this template declares no HITL.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False


class TestRetrievalBlock:
    def test_cfg_06_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror the declared runtime settings — a drift
        # silently changes tuning. RerankFilterNode/RetrieveNode each own only
        # the constants their own logic reads (state-seeded; no shared
        # config-param object).
        retrieval = _RUNTIME["retrieval"]
        from src.nodes.retrieve_node import _DEFAULT_KB_PATH, _DEFAULT_TOP_K as _RETRIEVE_TOP_K
        from src.nodes.rerank_filter_node import (
            _DEFAULT_SCORE_THRESHOLD,
            _DEFAULT_TOP_K as _RERANK_TOP_K,
        )

        assert retrieval["top_k"] == _RETRIEVE_TOP_K == _RERANK_TOP_K
        assert retrieval["score_threshold"] == _DEFAULT_SCORE_THRESHOLD
        assert retrieval["kb_path"] == _DEFAULT_KB_PATH
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_07_parent_config_forwards_runtime_blocks(self):
        cfg = ShrinkageQAGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"] == _RUNTIME["retrieval"]
        assert cfg["configurable"]["llm"] == _RUNTIME["llm"]
        assert cfg["configurable"]["retrieval"], "_parent_config() must forward the declared retrieval block"


class TestSeededKnowledgeBase:
    def test_kb_is_a_well_formed_entry_list(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded KB must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "category", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))
