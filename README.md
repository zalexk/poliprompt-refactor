
<p align="center">
  <img src="https://github.com/geshijoker/PoliPrompt/blob/main/poliprompt_logo.png" width="800" />
</p>

---

[![PyPI - Python](https://img.shields.io/badge/python-v3.10+-blue.svg)](https://pypi.org/project/PoliPrompt/)
[![Build](https://img.shields.io/github/actions/workflow/status/geshijoker/PoliPrompt/ci.yaml?branch=main)](https://github.com/geshijoker/PoliPrompt/actions)
[![docs](https://img.shields.io/badge/docs-Passing-green.svg)](https://poliprompt-tutorial.readthedocs.io/en/latest/)
[![PyPI - PyPi](https://img.shields.io/pypi/v/PoliPrompt)](https://pypi.org/project/poliprompt/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/geshijoker/PoliPrompt/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2409.01466-<COLOR>.svg)](https://arxiv.org/pdf/2409.01466)

# PoliPrompt: Multimodal Agentic Framework

PoliPrompt is a Python framework for automated text and multimodal classification using a three-layer LLM inference pipeline with Human-in-the-Loop (HITL) active learning. It was originally designed for political science research and is general enough for any labeling task that benefits from few-shot learning.

## How the System Works

```mermaid
flowchart TD
    A[Input Data\nCSV / JSONL] --> B

    subgraph PREP ["Phase 1 & 2: Preparation"]
        B[Embed Documents\nParallel workers] --> C[FAISS Index]
        C --> D[Select Exemplar Pool\nKMeans / Random]
        D --> E[Map-Reduce\nRule Synthesis]
        E --> F[enhanced_rules.txt\nrules.json]
    end

    F --> G

    subgraph INFER ["Phase 3: Inference Loop"]
        G[Retrieve k-shot Examples\nMMR from FAISS] --> H

        H[L1: Primary LLM\ne.g. gpt-4o-mini] -->|Send to L2| I

        I[L2: Secondary LLM\ne.g. qwen-vl-max] -->|L1 == L2| M
        I -->|Disagree| J

        J[L3: Expert LLM x2\ne.g. gpt-4o] -->|Consensus| M
        J -->|Still Conflict| K

        K[Human-in-the-Loop\nStreamlit / Terminal] --> L[Inject into\nExemplar Pool]
        L --> M
    end

    M[Log Result\nbackup.csv + JSONL] -->|Next row| G
    M --> N[Evaluation\nAccuracy / F1 / Confusion Matrix]
```

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (skip if already installed)
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone the repository
git clone -b multimodal-feature-refinement https://github.com/cora0413/PoliPrompt-Refactor.git
cd PoliPrompt-Refactor

# 3. Install dependencies — choose what you need:
uv sync --extra openai        # OpenAI (GPT-4o, embeddings)
uv sync --extra qwen          # Alibaba / Qwen (multimodal embeddings)
uv sync --extra google        # Google Gemini
uv sync --extra anthropic     # Anthropic / Claude
uv sync --extra all           # All providers + Streamlit UI + observability
```

## API Keys

Obtain API keys from your provider before running:

| Provider | Environment Variable | Where to Get |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Alibaba / Qwen | `DASHSCOPE_API_KEY` | https://dashscope.console.aliyun.com/ |
| Google | `GOOGLE_API_KEY` | https://aistudio.google.com/apikey |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com/ |

## Streamlit UI

The easiest way to run PoliPrompt — no coding required.

> Requires `uv sync --extra all`

```bash
uv run streamlit run streamlit_app.py
```

**Steps in the UI:**
1. Enter your API keys in the sidebar → click **Save Credentials to .env**
2. Fill in project settings (dataset path, label options, models)
3. Run in order: **Build Pool → Optimise Rules → Annotate**
4. Click **Run Evaluation** to view accuracy, F1, and confusion matrix

### Quick Demo with Included Datasets

Two example datasets are included under `examples/`:

**Multimodal — Hateful Memes detection (0 = not hateful, 1 = hateful):**
- `work_station`: absolute path to `examples/HarmfulMemes-tiny`
- `modality`: `multimodal`
- `data_path`: `train-tiny.jsonl`
- `image_dir`: `.` *(the JSONL file already contains the subfolder prefix `img/`, so set this to `.` to use `work_station` as the image root)*
- `options`: `0, 1`

**Text — BBC News topic classification:**
- `work_station`: absolute path to `examples/BBCNews-tiny`
- `modality`: `text`
- `data_path`: `BBCNews-tiny.csv`
- `options`: `politics, business, sport, technology, entertainment`

## Python API

For developers and researchers who prefer code:

```python
from poliprompt import TextClassifier, MultiModalClassifier

clf = TextClassifier(
    config_path="path/to/config.yaml",
    prompt_path="path/to/prompt.txt",
)

clf.create_few_shot_pool()       # Phase 1: embed + select exemplars
clf.optimize_task_description()  # Phase 2: extract reasoning rules
clf.annotate()                   # Phase 3: classify all rows

metrics = clf.evaluate()
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

A complete end-to-end example (both text and multimodal) is available in
`notebooks/TopicExperiment.ipynb`. Select the `.venv` kernel and run all cells.

## Documentation

| Section | Contents |
|---|---|
| [Getting Started](docs/getting-started/introduction.md) | What PoliPrompt is and when to use it |
| [Quick Start](docs/getting-started/quickstart.md) | Step-by-step installation and first run |
| [API Reference](docs/api/reference.md) | Public methods of `TextClassifier` and `MultiModalClassifier` |
| [Configuration](docs/configuration/configuration.md) | All `config.yaml` fields explained |
| [Architecture](docs/concepts/architecture.md) | Inference pipeline and component design |
| [Changelog](docs/about/changelog.md) | Version history |

## Citation

```bibtex
@article{poliprompt2024,
  title={PoliPrompt: A High-Performance Cost-Effective LLM-Based Text Classification Framework for Political Science},
  url={https://arxiv.org/abs/2409.01466}
}
```
