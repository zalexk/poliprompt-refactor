# Quick Start

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- At least one API key: `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, or `GOOGLE_API_KEY`

## Installation

Clone the repository and install with uv, adding the extra for your LLM provider:

```bash
git clone https://github.com/geshijoker/PoliPrompt.git
cd PoliPrompt
uv sync --extra openai        # OpenAI provider
uv sync --extra qwen          # Alibaba / Qwen provider (required for multimodal)
uv sync --extra google        # Google Gemini provider
uv sync --extra all           # all providers + Streamlit UI + observability
```

## 1. Set Up Environment

Create a `.env` file in your project directory (or any parent directory):

```env
OPENAI_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...
```

PoliPrompt auto-discovers `.env` by walking up from the current working directory, so no explicit path is needed.

## 2. Prepare Your Files

Your project directory needs three files before you start:

```
my_project/
├── .env
├── config.yaml
├── prompt.txt
└── train.csv
```

**`prompt.txt`** — a plain-text task description that is used as the system prompt:

```
Classify the following political statement as "positive" or "negative" in sentiment
toward the incumbent government. Respond in JSON with keys "label" and "reason".
```

**`config.yaml`** — minimal setup for a text task:

```yaml
project:
  name: "MySentiment"
  modality: "text"
  work_station: "/absolute/path/to/my_project"
  data_path: "train.csv"
  outfiles_dir: "outfiles"

column_mapping:
  text_col: "text"
  image_col: "None"
  answer_col: "label"

user_settings:
  options: ["positive", "negative"]
  k_shots: 3

models:
  embedding_llm: "text-embedding-3-small"
  primary_llm: "gpt-4o-mini"
  secondary_llm: "gpt-4o-mini"
  expert_llm: "gpt-4o"
```

See [Configuration](../configuration/configuration.md) for all available fields.

## 3. Run Classification

```python
from poliprompt import TextClassifier

clf = TextClassifier(
    config_path="config.yaml",
    prompt_path="prompt.txt",
    # env_path=".env"  # optional — omit to auto-discover .env
)

clf.create_few_shot_pool()       # Phase 1: embed + select exemplars
clf.optimize_task_description()  # Phase 2: extract reasoning rules
clf.annotate()                   # Phase 3: classify all rows
```

Each phase saves its outputs to `outfiles/`. Re-running a phase that already has outputs skips it automatically.

## 4. Evaluate Results

If your dataset includes ground-truth labels:

```python
metrics = clf.evaluate()
print(metrics["accuracy"])
print(metrics["averages"])
```

## Output Files

| File | Description |
|---|---|
| `outfiles/embeddings.index` | FAISS vector index |
| `outfiles/exemplar_indices.json` | Indices of selected exemplars |
| `outfiles/rules.json` | Per-exemplar reasoning rules |
| `outfiles/enhanced_rules.txt` | Synthesized class-level rules |
| `outfiles/{project}_backup.csv` | Predictions with path metadata |
| `outfiles/observability_logs.jsonl` | Full per-row inference trace |

## Multimodal Quick Start

For datasets with images, use `MultiModalClassifier` and update `config.yaml`:

```yaml
project:
  modality: "multimodal"
  data_path: "train.jsonl"
  image_dir: "images/"

column_mapping:
  image_col: "img"

models:
  embedding_llm: "qwen-vl-max"
  primary_llm: "gpt-4o"
  secondary_llm: "qwen-vl-max"
  expert_llm: "gpt-4o"
```

```python
from poliprompt import MultiModalClassifier

clf = MultiModalClassifier(config_path="config.yaml", prompt_path="prompt.txt")
clf.create_few_shot_pool()
clf.optimize_task_description()
clf.annotate()
```

The dataset must be JSONL with one object per line containing `text`, `img` (filename), and optionally `label`.

## Human-in-the-Loop

When models disagree and cannot resolve a row, the pipeline pauses:

```
[PAUSED] Row 42 needs manual intervention.
```

**Terminal mode** — enter the correct label at the prompt.

**Streamlit mode** — launch the UI first, then run `annotate()`. A labeling card appears inline when intervention is needed:

```bash
streamlit run streamlit_app.py
```

Add `hitl_mode: "streamlit"` to `config.yaml`. Human labels are automatically injected into the exemplar pool.

---

## Streamlit UI

The Streamlit app provides a no-code interface for the entire pipeline.

### Launch

```bash
uv sync --extra all
streamlit run streamlit_app.py
```

### Sidebar — Model Configuration

One expander per LLM role:

| Role | Purpose |
|---|---|
| Embedding LLM | Embeds documents for FAISS; use **qwen** for multimodal tasks |
| Primary LLM (L1) | Fast first-pass classifier |
| Secondary LLM (L2) | Cross-validates L1; ideally a different vendor |
| Expert LLM (L3) | Resolves L1/L2 disagreements |

For each role: choose a **Provider** (OpenAI, Anthropic, Google, Qwen), select a **Model** from the dropdown, enter an **API Key**, and optionally override the **Base URL**. Click **Save Credentials to .env** to persist keys locally.

### Main Panel

1. **project / column_mapping / user_settings** — fill in dataset path, label options, and k-shot count.
2. **Inference Hyperparameters** — set temperature and max tokens per agent.
3. **Seed Prompt** — paste your task description; saved to `work_station/prompt.txt`.
4. Click **Save Config & Initialise Classifier**.

### Running the Pipeline

Three buttons appear after initialisation. Run them in order:

| Step | Button | What it does |
|---|---|---|
| 1 | Build Pool | Embeds data, runs KMeans, builds FAISS index |
| 2 | Optimise Rules | Map-Reduce rule synthesis over the exemplar pool |
| 3 | Annotate | Runs the L1 → L2 → L3 → HITL pipeline; live log streams |

Steps already completed (output files present) are skipped automatically.

After annotation, click **Run Evaluation** to view accuracy, per-class F1, and a confusion matrix, with a downloadable report.
