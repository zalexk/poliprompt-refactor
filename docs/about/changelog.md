# Changelog

All notable changes to PoliPrompt are documented here.

This project follows [Semantic Versioning](https://semver.org/). Format based on [Keep a Changelog](https://keepachangelog.com/).

**Tips:**
- Write entries for humans, not machines. Be descriptive but concise.
- One bullet = one logical change. Don't bundle unrelated changes.
- Omit sections that have no entries for that release.
- Reference version numbers where relevant.

---

## [Unreleased]

### Added
- Anthropic provider support (`langchain-anthropic`); `claude-*` models auto-routed via `create_llm`.
- `PROVIDERS` registry in `llm_contribs.py` — single source of truth for all four providers (OpenAI, Anthropic, Google, Qwen): env vars, default URLs, inference and embedding model lists.
- `langgraph-checkpoint-sqlite` as a core dependency (split from `langgraph` in recent releases).
- Full unit test suite across all modules (`tests/poliprompt/`), including hypothesis-based property tests and `monkeypatch` env-var fixtures.
- Complete documentation under `docs/`: quick start, API reference, configuration reference, architecture concepts, changelog, roadmap.
- Streamlit UI section in `README.md` and `docs/getting-started/quickstart.md`.

### Changed
- `env_path` made optional in `BaseClassifier.__init__`; `python-dotenv` auto-discovers `.env` by walking up from CWD when omitted.
- Streamlit sidebar replaced hardcoded vendor sections with four role-based expanders (Embedding, Primary, Secondary, Expert), each with provider dropdown, model dropdown, API key, and optional base URL — driven by `PROVIDERS`.
- `KMeansExemplarSelector` raises `RuntimeError` on empty cluster instead of silently skipping.
- `get_universal_embeddings` sets `dashscope.api_key` before spawning threads, fixing a race condition.
- Dead `_l1_router` conditional edge replaced with a plain `add_edge`; `_model_name_from` static method fixes config key lookup for inline model dicts.
- Column validation in `load_and_validate_data` reports all missing columns at once instead of stopping at the first.
- `notebooks/TopicExperiment.ipynb` rewritten: platform-agnostic paths via `pathlib`, no hardcoded OS paths, two self-contained examples (text + multimodal).
- `demo/demo_multimodel.ipynb` and `demo/hitl_demo.ipynb` updated to use `pathlib.Path` throughout; removed unused imports and hardcoded paths.

### Fixed
- `options` and `k_shots` validation added to `_parse_config_to_self`; empty options or non-positive `k_shots` now raise `ValueError` at init time.
- Inline model dict key corrected from `name:` to `model:` in configuration docs and `_model_name_from`.

---

## [0.2.1]

### Added
- Multimodal classification support via `MultiModalClassifier` — classifies text and images together using base64-encoded image payloads.
- Evaluation module with accuracy, per-class F1, and confusion matrix.
- LangGraph-based stateful inference pipeline with checkpointing and resume support.
- Human-in-the-Loop (HITL) active learning — pauses on low-confidence rows and injects human labels into the exemplar pool.
- Map-Reduce rule synthesis (`optimize_task_description`) to extract reasoning rules from exemplars.
- FAISS-backed exemplar pool with Maximal Marginal Relevance (MMR) retrieval.
- Streamlit annotation UI for interactive HITL labeling.
- Langfuse observability integration for LLM call tracing.

### Changed
- Refactored codebase into `src/poliprompt/` package layout.
- Unified `TextClassifier` and `MultiModalClassifier` under a shared `BaseClassifier` abstract class.
- Replaced linear inference loop with a LangGraph compiled graph.
