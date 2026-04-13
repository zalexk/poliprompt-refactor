# Introduction

PoliPrompt is a Python framework for automated classification of text and multimodal (text + image) data using large language models. It was originally designed for political science research and is general enough for any labeling task that benefits from few-shot learning.

## Key Features

- **Three-layer inference pipeline** — a fast primary model, a cross-validating secondary model, and an expert model invoked only on disagreement. This reduces cost while maintaining accuracy.
- **Human-in-the-Loop (HITL)** — when models cannot reach consensus, the pipeline pauses for a human label. Human-labeled samples are immediately added to the exemplar pool.
- **Multimodal support** — classifies text alongside images using base64-encoded image payloads. No image preprocessing required.
- **FAISS-backed retrieval** — few-shot examples are selected per query using Maximal Marginal Relevance (MMR), balancing relevance and diversity.
- **Map-Reduce rule synthesis** — the system automatically extracts reasoning rules from labeled examples and injects them into every inference prompt.
- **Resumable** — each phase saves its outputs to disk. Re-running after an interruption picks up where it left off.

## Supported Models

| Vendor | Example models | Required env var |
|---|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `text-embedding-3-small` | `OPENAI_API_KEY` |
| Alibaba / Qwen | `qwen-vl-max`, `qwen-max` | `DASHSCOPE_API_KEY` |
| Google | `gemini-pro`, `gemini-1.5-flash` | `GOOGLE_API_KEY` |

## When to Use PoliPrompt

PoliPrompt is a good fit when:

- You have a **labeled dataset** (or a partially labeled one) you want to annotate at scale.
- Your task benefits from **few-shot examples** — nuanced categories, domain-specific jargon, or ambiguous boundaries.
- You want **cost control** — cheaper models handle easy cases; expensive models are reserved for hard ones.
- Your data includes **images** alongside text (memes, policy documents with figures, social media posts).

## Next Step

Follow the [Quick Start](./quickstart.md) guide to install PoliPrompt and run your first classification.
