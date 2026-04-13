# Tests

Unit tests for the `poliprompt` package. The folder structure mirrors `src/poliprompt/`.

```
tests/
├── conftest.py                          # Shared fixtures (embeddings, FAISS index, project dirs)
└── poliprompt/
    ├── test_utils.py                    # Data loading, config parsing, response parsing
    ├── test_selectors.py                # KMeans and Random exemplar selectors
    ├── test_retrieves.py                # MMR, class-balanced retrieval, select_kshots
    ├── test_llm_contribs.py             # LLM factory and embedding engine
    ├── test_base_classifier.py          # Config parsing, message formatting, evaluate()
    ├── test_text_classifier.py          # Text-only data pipeline and prompt construction
    └── test_multimodal_classifier.py    # Multimodal data pipeline and image path resolution
```

## Running tests

Install the package with the `dev` dependency group, then run pytest:

```bash
uv sync --group dev
uv run pytest
```

Run a specific file:

```bash
uv run pytest tests/poliprompt/test_utils.py -v
```

Run only fast offline tests:

```bash
uv run pytest -m unit
```

Skip integration tests (default):

```bash
uv run pytest -m "not integration"
```

## Coverage

```bash
uv run pytest --cov=src --cov-report=term-missing
```

## Markers

| Marker | Meaning |
|---|---|
| `unit` | Fast, fully offline tests |
| `integration` | Requires live API keys; excluded by default |
| `slow` | Tests with non-trivial runtime |

## Design notes

- **No real API calls.** LLM clients are mocked via `unittest.mock.patch`. Tests that exercise
  embedding or inference code patch `create_llm`, `_fetch_openai_batch`, and `_fetch_qwen_batch`.
- **Shared fixtures** (embeddings, FAISS index, sample DataFrames, project directories) live in
  `conftest.py` and are available to all test files without explicit import.
- **Property-based tests** in `test_utils.py` use [Hypothesis](https://hypothesis.readthedocs.io/)
  to verify that `parse_llm_response_generic` never raises on arbitrary string input.
