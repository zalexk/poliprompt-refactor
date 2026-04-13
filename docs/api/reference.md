# API Reference

Both classifiers share the same public interface. Import from the package root:

```python
from poliprompt import TextClassifier
from poliprompt import MultiModalClassifier
```

---

## Constructor

```python
TextClassifier(config_path, prompt_path, env_path=None)
MultiModalClassifier(config_path, prompt_path, env_path=None)
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `config_path` | `str \| Path` | Yes | Path to `config.yaml`. |
| `prompt_path` | `str \| Path` | Yes | Path to a plain-text task description used as the system prompt. |
| `env_path` | `str \| Path` | No | Path to a `.env` file. When omitted, `python-dotenv` auto-discovers `.env` by walking up from the current working directory. |

Initializes LLM clients and checks for existing phase outputs. No API calls are made.

---

## `create_few_shot_pool()`

**Phase 1.** Embeds all documents and selects an elite exemplar pool.

```python
clf.create_few_shot_pool()
```

Embeds every document using the configured `embedding_llm` (parallel, batched), builds a FAISS index, and selects `n_exemplars_pool` representatives via KMeans clustering. Skips automatically if outputs already exist.

**Outputs written to `outfiles_dir`:**

| File | Description |
|---|---|
| `embeddings.index` | FAISS index of all document embeddings. |
| `exemplar_indices.json` | Integer indices of selected exemplars. |

---

## `optimize_task_description()`

**Phase 2.** Extracts reasoning rules from the exemplar pool via Map-Reduce.

```python
clf.optimize_task_description()
```

Runs the expert LLM on each exemplar to extract a reasoning explanation (Map), then synthesizes per-class rules (Reduce). Skips automatically if outputs already exist.

**Outputs written to `outfiles_dir`:**

| File | Description |
|---|---|
| `rules.json` | Per-exemplar reasoning strings, keyed by index. |
| `enhanced_rules.txt` | Synthesized class-level rules prepended to every inference prompt. |

---

## `annotate()`

**Phase 3.** Classifies all rows through the three-layer pipeline.

```python
clf.annotate()
```

For each row: retrieves `k_shots` examples via MMR, sends the assembled prompt through L1 → L2 → L3 → HITL, and logs the result. Writes incrementally so progress is preserved across interruptions.

**Outputs written to `outfiles_dir`:**

| File | Description |
|---|---|
| `{project_name}_backup.csv` | Predictions with `predicted_label`, `path_taken`, `rag_distances`, `rag_labels`. |
| `observability_logs.jsonl` | Full per-row inference trace including prompts and responses. |

**`path_taken` values:**

| Value | Meaning |
|---|---|
| `"L2_Heterogeneous_Match"` | L1 and L2 agreed. |
| `"L3_Expert_Consensus"` | L3 resolved after L1/L2 disagreement. |
| `"HITL_Manual"` | Human label was required. |
| `"L1_Initial_Exit"` | L1 produced a result but L2 was not reached (rare failure path). |

---

## `evaluate()`

Computes classification metrics against ground-truth labels.

```python
metrics = clf.evaluate()
```

Reads the backup CSV and compares predictions to the `answer_col`. Rows without a ground-truth label are excluded.

**Returns:**

```python
{
    "total":            228,          # total rows in the backup CSV
    "n_no_gt":          12,           # rows excluded (no ground-truth label)
    "n_infer_fail":     3,            # rows excluded (inference produced no valid label)
    "n_evaluated":      213,          # rows used for metric computation
    "accuracy":         0.923,
    "labels":           ["0", "1"],   # sorted unique labels
    "confusion_matrix": pd.DataFrame, # rows = true, columns = predicted
    "per_class": {
        "0": {"precision": 0.91, "recall": 0.94, "f1-score": 0.92, "support": 110},
        "1": {"precision": 0.94, "recall": 0.91, "f1-score": 0.92, "support": 118},
    },
    "averages": {
        "macro avg":    {"precision": ..., "recall": ..., "f1-score": ..., "support": ...},
        "weighted avg": {"precision": ..., "recall": ..., "f1-score": ..., "support": ...},
    },
}
```

**Requires:** `answer_col` must be set in `config.yaml` (not `"None"`).
