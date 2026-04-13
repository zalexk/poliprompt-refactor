# Architecture

## Package Structure

```
src/poliprompt/
├── __init__.py                  # Public exports: TextClassifier, MultiModalClassifier
├── base_classifier.py           # Abstract base: LangGraph agent, all three phases
├── text_classifier.py           # Concrete: text-only data pipeline
├── multimodal_classifier.py     # Concrete: text + image data pipeline
├── llm_contribs.py              # LLM factory and embedding engine
├── retrieves.py                 # MMR few-shot retrieval
├── selectors.py                 # Exemplar pool selection strategies
├── utils.py                     # Data loading, image encoding, response parsing
└── configs/                     # Default model hyperparameters
```

---

## Class Design

`BaseClassifier` implements all three phases and the LangGraph state machine. The two concrete subclasses only override data-pipeline methods:

| Method | `TextClassifier` | `MultiModalClassifier` |
|---|---|---|
| `_convert_df_to_docs()` | Extracts text column | Extracts text + resolves image paths |
| `_get_embeddings()` | Text embeddings via OpenAI | Multimodal embeddings via Qwen |
| `_prepare_agent_messages()` | `[SystemMessage, HumanMessage(text)]` | `[SystemMessage, HumanMessage([text blocks + image_url blocks])]` |
| `_display_item_for_hitl()` | Prints text to terminal | Renders image inline (Jupyter) or prints path |

---

## Three-Layer Inference Pipeline

The inference loop is implemented as a LangGraph compiled graph. State is passed between nodes as `AgentState`:

```python
class AgentState(TypedDict):
    pending_indices: List[int]       # rows left to process
    current_index: Optional[int]
    l1_res: Optional[ClassificationResult]
    l2_res: Optional[ClassificationResult]
    l3_res_list: List[ClassificationResult]
    rag_distances: List[float]
    rag_labels: List[str]
    results: Dict[str, dict]         # accumulated output
```

### Routing Logic

| Condition | Next node |
|---|---|
| L1 and L2 labels match | Accept result |
| L1 and L2 disagree | Route to L3 (runs twice) |
| Both L3 runs agree | Accept result |
| L3 runs disagree | Route to HITL |
| Human provides label | Accept result, inject into exemplar pool |

### HITL Injection

When a human labels a row, the sample is appended to the FAISS index and exemplar pool. All subsequent rows immediately include the new human-labeled example as a candidate in k-shot retrieval.

---

## Few-Shot Retrieval (MMR)

For each row at inference time, `select_kshots()` retrieves examples using **Maximal Marginal Relevance**:

```
score(c) = λ · sim(query, c) − (1 − λ) · max_{s ∈ selected} sim(c, s)
```

Candidates are first filtered to ensure at least one example per class (balance pre-filter). Then the highest-scoring candidate is selected greedily and removed from the pool, repeating until `k_shots` examples are chosen.

`lambda_param = 0.5` (default) balances relevance against diversity, which tends to outperform pure nearest-neighbor retrieval on imbalanced datasets.

---

## Map-Reduce Rule Synthesis

`optimize_task_description()` runs in two steps:

- **Map**: The expert LLM processes each exemplar and explains why it has its label. Results are stored in `rules.json`.
- **Reduce**: All per-exemplar explanations are grouped by class and synthesized by the expert LLM into a compact set of discriminative rules per class. The output (`enhanced_rules.txt`) is prepended to the system prompt for every inference call.

This gives each LLM layer a distilled understanding of the decision boundary without increasing per-call token costs significantly.

---

## LLM Factory

`create_llm()` in `llm_contribs.py` centralizes all LLM instantiation:

- Accepts a model name string or a dict with `name` + override fields.
- Auto-detects vendor by model name prefix (`gpt-*` → OpenAI, `qwen-*` → DashScope, `gemini-*` → Google).
- Wraps each client with `tenacity` retry: 5 attempts, exponential backoff starting at 4 s.

All three pipeline layers are instantiated once at constructor time and reused across all rows.

---

## Embedding Engine

`get_universal_embeddings()` in `llm_contribs.py` handles two backends:

| Backend | Trigger | Notes |
|---|---|---|
| DashScope `MultiModalEmbedding` | model name contains `"qwen"` | Accepts text and image together; recommended batch size ≤ 5 |
| OpenAI `Embeddings` | all other models | Text only; batch size up to 32 |

Both backends run in a `ThreadPoolExecutor`. Failed embedding slots are filled with zero vectors and logged as warnings — a single API failure does not abort the pipeline.
