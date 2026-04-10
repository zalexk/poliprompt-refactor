import os
import logging
import numpy as np
import dashscope
import httpx
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
import openai
from dashscope import MultiModalEmbedding
from tqdm import tqdm

logger = logging.getLogger(__name__)

# --- 重试配置 ---
llm_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)


# ──────────────────────────────────────────────
# LLM 工厂
# ──────────────────────────────────────────────
def create_llm(model_data: Any, default_config: Dict = None) -> ChatOpenAI:
    """
    统一模型工厂，支持 OpenAI / Google / Alibaba 三个厂商。
    每个厂商读取各自的环境变量，不填 Base URL 则使用官方默认地址。
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

    if any(k in name_lower for k in ("gpt", "o1", "o3")):
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


# ──────────────────────────────────────────────
# Embedding：批量调用（OpenAI + Qwen 均支持批量）
# ──────────────────────────────────────────────

@llm_retry
def _fetch_openai_batch(
    client: openai.OpenAI,
    texts: List[str],
    model: str,
) -> List[List[float]]:
    """OpenAI 批量 Embedding，一次调用返回多条向量。"""
    resp = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


@llm_retry
def _fetch_qwen_batch(
    batch: List[Dict],
    model: str,
) -> List[List[float]]:
    """Qwen 批量多模态 Embedding，batch_size 建议 ≤ 5。"""
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    input_data = []
    for doc in batch:
        item = {"text": doc["text"]}
        img  = doc.get("image_path")
        if img and str(img).strip() and os.path.exists(img):
            item["image"] = os.path.abspath(img).replace("\\", "/")
        input_data.append(item)

    resp = MultiModalEmbedding.call(model=model, input=input_data)
    if resp.status_code == 200:
        return [e["embedding"] for e in resp.output["embeddings"]]
    raise Exception(f"Qwen Batch Embedding Error: {resp.message}")


# ──────────────────────────────────────────────
# 统一多线程 Embedding 引擎
# ──────────────────────────────────────────────

def get_universal_embeddings(
    docs: List[Dict],
    model_name: str,
    max_workers: int = 8,
    batch_size: int = 5,
) -> np.ndarray:
    """
    Embedding 分发引擎（批量 + 多线程）：
    - model_name 含 'qwen' → Qwen 官方 SDK，支持图文多模态
    - 其他               → OpenAI 兼容端点，纯文本
    """
    total   = len(docs)
    results = [None] * total
    service = "qwen" if "qwen" in model_name.lower() else "openai"

    batches = [
        (start, docs[start: start + batch_size])
        for start in range(0, total, batch_size)
    ]
    n_batches = len(batches)

    client = None
    if service == "openai":
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    print(f"--- Dispatching {total} docs in {n_batches} batches "
          f"(batch={batch_size}, workers={max_workers}, service={service} | {model_name}) ---")

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
                raise RuntimeError("FATAL: All embedding batches failed! Check API Keys and Network.")
            logger.warning(f"Slot {i} failed, filling with zeros.")
            results[i] = np.zeros(dim).tolist()

    return np.array(results, dtype="float32")
