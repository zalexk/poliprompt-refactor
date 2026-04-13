import os
import logging
import numpy as np
import httpx
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

try:
    from langchain_openai import ChatOpenAI
    import openai
except ImportError:
    ChatOpenAI = None  # type: ignore
    openai = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None  # type: ignore

try:
    import dashscope
    from dashscope import MultiModalEmbedding
except ImportError:
    dashscope = None  # type: ignore
    MultiModalEmbedding = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
#
# Single source of truth for all LLM / embedding providers.
# Used by create_llm for vendor detection and by the Streamlit UI for
# building provider / model dropdown menus.
#
# Keys under each provider:
#   label            – human-readable display name
#   api_key_env      – environment variable that holds the API key
#   base_url_env     – environment variable for an optional custom base URL
#   default_url      – official endpoint; None means the SDK uses its own default
#   inference_models – ordered list of chat/completion model IDs
#   embedding_models – ordered list of embedding model IDs (empty = not supported)
# ---------------------------------------------------------------------------
PROVIDERS: dict = {
    "openai": {
        "label": "OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_url": "https://api.openai.com/v1",
        "inference_models": ["gpt-4o", "gpt-4o-mini", "o3-mini", "o4-mini"],
        "embedding_models": ["text-embedding-3-small", "text-embedding-3-large"],
    },
    "anthropic": {
        "label": "Anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "default_url": None,   # SDK manages its own default; no OpenAI-compat endpoint
        "inference_models": [
            "claude-opus-4-5",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022",
        ],
        "embedding_models": [],  # Anthropic has no public embedding API
    },
    "google": {
        "label": "Google",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url_env": "GOOGLE_BASE_URL",
        "default_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "inference_models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "embedding_models": [],  # Google embedding not yet wired into get_universal_embeddings
    },
    "qwen": {
        "label": "Alibaba / Qwen",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "inference_models": ["qwen-vl-max", "qwen-vl-plus", "qwen-max", "qwen-turbo"],
        "embedding_models": ["qwen3-vl-embedding", "qwen-vl-max"],
    },
}

# Retry configuration: 5 attempts with exponential backoff
llm_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)


def create_llm(model_data: Any, default_config: Dict = None):
    """
    Unified LLM factory supporting OpenAI, Anthropic, Google, and Alibaba vendors.

    Vendor is auto-detected from the model name. API keys and base URLs are
    read from environment variables; an inline config dict can override them.
    """
    cfg = default_config or {}

    if isinstance(model_data, dict):
        m_name   = model_data.get('model') or model_data.get('model_name')
        temp     = model_data.get('temperature', cfg.get('temperature', 0.0))
        tokens   = model_data.get('max_tokens',  cfg.get('max_tokens',  1000))
        dict_key = model_data.get('api_key')
        dict_url = model_data.get('base_url')
    else:
        m_name   = model_data
        temp     = cfg.get('temperature', 0.0)
        tokens   = cfg.get('max_tokens',  1000)
        dict_key = None
        dict_url = None

    name_lower = (m_name or "").lower()

    if "claude" in name_lower:
        if ChatAnthropic is None:
            raise ImportError(
                "langchain-anthropic is required for Anthropic models. "
                "Install it with: uv sync --extra anthropic"
            )
        return ChatAnthropic(
            model=m_name,
            api_key=dict_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=dict_url or os.getenv("ANTHROPIC_BASE_URL") or None,
            temperature=float(temp),
            max_tokens=int(tokens),
            max_retries=5,
            timeout=120,
        )

    if ChatOpenAI is None:
        raise ImportError(
            "langchain-openai is required for LLM inference. "
            "Install it with: uv sync --extra openai"
        )

    if any(k in name_lower for k in ("gpt", "o1", "o3", "o4")):
        default_key = os.getenv("OPENAI_API_KEY")
        default_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    elif "gemini" in name_lower:
        default_key = os.getenv("GOOGLE_API_KEY")
        default_url = (os.getenv("GOOGLE_BASE_URL")
                       or "https://generativelanguage.googleapis.com/v1beta/openai")

    elif "qwen" in name_lower:
        default_key = os.getenv("DASHSCOPE_API_KEY")
        default_url = (os.getenv("DASHSCOPE_BASE_URL")
                       or "https://dashscope.aliyuncs.com/compatible-mode/v1")

    else:
        logger.warning(f"Unknown vendor for '{m_name}', falling back to OpenAI config.")
        default_key = os.getenv("OPENAI_API_KEY")
        default_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    return ChatOpenAI(
        model=m_name,
        api_key=dict_key or default_key,
        base_url=dict_url or default_url,
        temperature=float(temp),
        max_tokens=int(tokens),
        max_retries=5,
        timeout=120,
        http_client=httpx.Client(),
        http_async_client=httpx.AsyncClient(),
    )


@llm_retry
def _fetch_openai_batch(
    client: openai.OpenAI,
    texts: List[str],
    model: str,
) -> List[List[float]]:
    """Fetch embeddings for a batch of texts via the OpenAI API."""
    resp = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


@llm_retry
def _fetch_qwen_batch(
    batch: List[Dict],
    model: str,
) -> List[List[float]]:
    """Fetch multimodal embeddings for a batch via the Qwen DashScope SDK. Recommended batch size <= 5."""
    if dashscope is None:
        raise ImportError(
            "dashscope is required for Qwen embeddings. "
            "Install it with: uv sync --extra qwen"
        )

    input_data = []
    for doc in batch:
        item = {"text": doc["text"]}
        img  = doc.get("image_path")
        if img and str(img).strip() and Path(img).exists():
            item["image"] = Path(img).resolve().as_posix()
        input_data.append(item)

    resp = MultiModalEmbedding.call(model=model, input=input_data)
    if resp.status_code == 200:
        return [e["embedding"] for e in resp.output["embeddings"]]
    raise RuntimeError(f"Qwen Batch Embedding Error: {resp.message}")


def get_universal_embeddings(
    docs: List[Dict],
    model_name: str,
    max_workers: int = 8,
    batch_size: int = 5,
) -> np.ndarray:
    """
    Dispatch embedding calls to the appropriate backend (batched, multi-threaded).

    - Model names containing 'qwen': Qwen DashScope SDK, supports text + image.
    - All other models: OpenAI-compatible endpoint, text only.

    Failed slots are filled with zero vectors and a warning is logged.
    """
    total   = len(docs)
    results = [None] * total
    service = "qwen" if "qwen" in model_name.lower() else "openai"

    batches = [
        (start, docs[start: start + batch_size])
        for start in range(0, total, batch_size)
    ]
    n_batches = len(batches)
    batch_sizes = {start: len(batch) for start, batch in batches}

    client = None
    if service == "openai":
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    elif service == "qwen" and dashscope is not None:
        # Set the API key once here, before spawning threads, to avoid a race
        # condition from multiple workers mutating the module-level attribute.
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    logger.info(
        "Dispatching %d docs in %d batches "
        "(batch=%d, workers=%d, service=%s | %s)",
        total, n_batches, batch_size, max_workers, service, model_name,
    )

    sample_emb = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if service == "openai":
            future_map = {
                executor.submit(
                    _fetch_openai_batch,
                    client,
                    [doc["text"] for doc in batch],
                    model_name,
                ): start
                for start, batch in batches
            }
        else:
            future_map = {
                executor.submit(_fetch_qwen_batch, batch, model_name): start
                for start, batch in batches
            }

        for f in tqdm(as_completed(future_map), total=n_batches, desc="Embedding Progress"):
            start = future_map[f]
            try:
                embs = f.result()
                expected = batch_sizes[start]
                if len(embs) != expected:
                    logger.warning(
                        "Batch at %d: expected %d embeddings, got %d; "
                        "trailing slots will be zero-filled.",
                        start, expected, len(embs),
                    )
                for j, emb in enumerate(embs):
                    results[start + j] = emb
                    if sample_emb is None:
                        sample_emb = emb
            except Exception as e:
                logger.error(f"Batch starting at {start} failed: {e}")

    dim = len(sample_emb) if sample_emb is not None else None
    for i in range(total):
        if results[i] is None:
            if dim is None:
                raise RuntimeError("FATAL: All embedding batches failed. Check API keys and network.")
            logger.warning(f"Slot {i} failed, filling with zeros.")
            results[i] = np.zeros(dim).tolist()

    return np.array(results, dtype="float32")
