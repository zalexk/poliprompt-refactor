
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

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.

```bash
git clone https://github.com/geshijoker/PoliPrompt.git
cd PoliPrompt
uv sync                       # core dependencies only
```

Install the LLM provider you need:

```bash
uv sync --extra openai        # OpenAI (GPT-4o, embeddings)
uv sync --extra qwen          # Alibaba / Qwen (multimodal embeddings)
uv sync --extra google        # Google Gemini
uv sync --extra anthropic     # Anthropic (Claude)
uv sync --extra all           # all providers + Streamlit UI + observability
```

## Streamlit UI

A no-code interface for configuring and running the full pipeline:

```bash
uv sync --extra all
streamlit run streamlit_app.py
```

The sidebar has one expander per LLM role (**Embedding**, **Primary L1**, **Secondary L2**, **Expert L3**). For each role, select a provider (OpenAI, Anthropic, Google, Qwen), pick a model from the dropdown, and enter your API key and optional base URL. Click **Save Credentials to .env** to persist them locally.

In the main panel, fill in project settings, column mapping, and inference hyperparameters, then click **Save Config & Initialise Classifier**. Run the three pipeline phases in order — live logs stream as each phase executes. When models disagree, a Human-in-the-Loop card appears inline for manual labeling.

---

## Documentation

| Section | Contents |
|---|---|
| [Getting Started](./docs/getting-started/introduction.md) | What PoliPrompt is and how to run your first classification |
| [API](./docs/api/reference.md) | Public methods of `TextClassifier` and `MultiModalClassifier` |
| [Concepts](./docs/concepts/architecture.md) | Inference pipeline, LangGraph state machine, and component design |
| [Configuration](./docs/configuration/configuration.md) | All `config.yaml` fields explained |
| [About](./docs/about/about-us.md) | Project background, changelog, and roadmap |

## Citation

To cite the [PoliPrompt](https://arxiv.org/abs/2409.01466) paper:

```bibtex
@article{liu2024poliprompt,
  title={PoliPrompt: A High-Performance Cost-Effective LLM-Based Text Classification Framework for Political Science},
  author={Liu, Menglin and Shi, Ge},
  journal={arXiv preprint arXiv:2409.01466},
  year={2024}
}
```
