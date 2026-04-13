# Configuration Reference

All settings are defined in a single `config.yaml` file. The repository root contains an annotated template.

---

## `project`

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Project identifier. Used as a prefix for output file names. |
| `modality` | `"text"` \| `"multimodal"` | Yes | Selects `TextClassifier` or `MultiModalClassifier`. |
| `work_station` | string | Yes | Absolute path to your project directory. All relative paths resolve from here. |
| `data_path` | string | Yes | Dataset path relative to `work_station`. CSV or JSONL. |
| `image_dir` | string | Multimodal only | Directory containing image files, relative to `work_station`. |
| `outfiles_dir` | string | Yes | Output directory name, relative to `work_station`. Created automatically. |

---

## `column_mapping`

| Field | Type | Description |
|---|---|---|
| `text_col` | string | Column containing the text to classify. |
| `image_col` | string | Column with image filenames. Set to `"None"` for text-only tasks. |
| `answer_col` | string | Column with ground-truth labels. Set to `"None"` for inference-only mode. |

---

## `user_settings`

| Field | Type | Default | Description |
|---|---|---|---|
| `options` | list of strings | — | All valid class labels. Must match the values in your dataset exactly. **Cannot be empty.** |
| `k_shots` | int | `3` | Number of few-shot examples retrieved per inference call. **Must be ≥ 1.** |
| `lambda_param` | float | `0.5` | MMR balance parameter. `1.0` = pure similarity; `0.0` = pure diversity. |
| `testing` | bool | `false` | If `true`, only processes the first `testing_size` rows. |
| `testing_size` | int | `128` | Number of rows to process when `testing` is `true`. |

---

## `models`

| Field | Description | Example values |
|---|---|---|
| `embedding_llm` | Model used to embed documents for FAISS. Use a Qwen model for multimodal. | `"text-embedding-3-small"`, `"qwen-vl-max"` |
| `primary_llm` | L1 — fast, cost-efficient first-pass model. | `"gpt-4o-mini"` |
| `secondary_llm` | L2 — cross-validation model; ideally a different vendor than L1. | `"qwen-vl-max"` |
| `expert_llm` | L3 — most capable model; invoked only on L1/L2 disagreements. | `"gpt-4o"` |

Each field accepts either a model name string or an inline dict to override defaults:

```yaml
models:
  primary_llm:
    model: "gpt-4o-mini"        # use "model" (not "name")
    temperature: 0.2
    max_tokens: 512
    base_url: "https://my-proxy.example.com/v1"
    api_key: "sk-..."           # optional: overrides the env-var key
```

---

## `retrieval`

| Field | Type | Default | Description |
|---|---|---|---|
| `n_exemplars_pool` | int | `256` | Number of elite exemplars selected from the full dataset for the few-shot pool. |

---

## `parallel`

| Field | Type | Default | Description |
|---|---|---|---|
| `embedding_workers` | int | `8` | Parallel threads for embedding API calls. |
| `inference_workers` | int | `2` | Parallel threads for inference. Keep low (1–4) when HITL is enabled. |
| `embedding_batch_size` | int | `5` | Documents per API request during embedding. Qwen multimodal recommends ≤ 5. |

---

## `observability`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable LLM call tracing via Langfuse. |
| `provider` | string | — | Tracing backend. Only `"langfuse"` is currently supported. |

When enabled, set these environment variables:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Full Example

```yaml
project:
  name: "HateSpeech"
  modality: "multimodal"
  work_station: "/data/projects/hate-speech"
  data_path: "train.jsonl"
  image_dir: "images/"
  outfiles_dir: "outfiles"

column_mapping:
  text_col: "text"
  image_col: "img"
  answer_col: "label"

user_settings:
  options: ["0", "1"]
  k_shots: 4
  lambda_param: 0.5
  testing: false

models:
  embedding_llm: "qwen-vl-max"
  primary_llm: "gpt-4o-mini"
  secondary_llm: "qwen-vl-max"
  expert_llm: "gpt-4o"

retrieval:
  n_exemplars_pool: 256

parallel:
  embedding_workers: 8
  inference_workers: 2
  embedding_batch_size: 5

observability:
  enabled: false
```
