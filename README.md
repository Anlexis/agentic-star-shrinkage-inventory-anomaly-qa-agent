# Shrinkage & Inventory Anomaly Q&A Agent

AI agent for answering retail shrinkage and inventory anomaly questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Retail
> **Template ID**: RET-C2-005

## Overview

Answers natural-language questions about retail shrinkage and inventory anomalies —
which anomaly patterns to watch for, how to run an investigation, and which prevention
playbooks apply — with a retrieval-augmented pipeline over a loss-prevention knowledge
base. Each question is validated and normalised, matching knowledge-base passages are
retrieved, reranked and filtered by a relevance threshold, and the answer is assembled
**only** from passages that clear that threshold, with numbered `[n]` citations and a
standing scope disclaimer. When no passage is relevant enough, the agent explicitly says
the knowledge base has insufficient coverage instead of fabricating procedure — and the
answer never names or accuses any individual.

The bundled knowledge base is a small sample of loss-prevention playbook material so the
pipeline runs and tests end-to-end out of the box; a real deployment replaces it with its
own SOP and anomaly-log corpus behind the same node contract.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent
fails at graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration and the sample knowledge base
docs/         design and operational documentation
```

See `docs/` for the design specification and test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the sample knowledge base (`config/kb/`) with your own loss-prevention corpus.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
