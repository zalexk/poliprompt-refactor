# Roadmap

Planned improvements and known gaps. Items are not committed to any release schedule.

## Planned

- **Async inference** — replace `ThreadPoolExecutor` with `asyncio` to improve throughput and reduce latency under concurrent workloads.
- **Additional embedding backends** — first-class support for Cohere, Voyage, and local sentence-transformers models.
- **Batch HITL** — queue multiple uncertain rows and present them together in the Streamlit UI rather than one at a time.
- **CLI entry point** — a `poliprompt` command to run the three phases without writing Python.
- **Structured output enforcement** — use vendor-native structured output APIs (e.g., OpenAI JSON mode) to eliminate regex fallback parsing.

## Known Limitations

- `inference_workers > 1` can cause ordering issues when HITL is triggered; keep at `1` or `2` for tasks expected to generate many HITL rows.
- Qwen multimodal embeddings require `DASHSCOPE_API_KEY`; there is no local fallback for multimodal embedding.
- The Streamlit HITL mode requires the app to be running before `annotate()` is called; it does not launch automatically.
