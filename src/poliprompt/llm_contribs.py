import os
import logging
import numpy as np
import dashscope
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
import openai
from dashscope import MultiModalEmbedding
from tqdm import tqdm
import httpx

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
    统一模型工厂，支持 5 个厂商：OpenAI / Google / Alibaba / Anthropic / Mistral。
    URL 优先级：dict 里的 base_url → 厂商专属环境变量 → 官方默认地址
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

    # 中转站 / relay station 兜底：只要某厂商的专属 Key 没填，就用 OpenAI Key + Base URL
    relay_key = os.getenv("OPENAI_API_KEY")
    relay_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    def _vendor(vendor_key_env: str, vendor_url_env: str, official_url: str):
        """
        返回 (api_key, base_url)。
        如果用户没有填写该厂商的专属 Key，则认为使用了中转站，
        直接回退到 OPENAI_API_KEY + OPENAI_BASE_URL。
        """
        vk = os.getenv(vendor_key_env)
        vu = os.getenv(vendor_url_env)
        if not vk:
            # 没有厂商专属 Key → 中转站模式，全用 OpenAI 配置
            return relay_key, relay_url
        # 有专属 Key → 使用厂商配置（Base URL 可选覆盖）
        return vk, vu or official_url

    if any(k in name_lower for k in ("gpt", "o1", "o3")):
        default_key, default_url = relay_key, relay_url
    elif "gemini" in name_lower:
        default_key, default_url = _vendor(
            "GOOGLE_API_KEY", "GOOGLE_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai")
    elif "qwen" in name_lower:
        default_key, default_url = _vendor(
            "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1")
    elif "claude" in name_lower:
        default_key, default_url = _vendor(
            "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", relay_url)
    elif any(k in name_lower for k in ("mistral", "ministral", "pixtral")):
        default_key, default_url = _vendor(
            "MISTRAL_API_KEY", "MISTRAL_BASE_URL",
            "https://api.mistral.ai/v1")
    else:
        logger.warning(f"Unknown vendor for '{m_name}', falling back to OpenAI/relay config.")
        default_key, default_url = relay_key, relay_url

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
    """
    OpenAI 批量 Embedding。
    API 的 input 字段原生支持 list[str]，一次调用返回多条向量。
    """
    resp = client.embeddings.create(input=texts, model=model)
    # resp.data 按 index 排序，直接取 embedding
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


@llm_retry
def _fetch_qwen_batch(
    batch: List[Dict],
    model: str,
) -> List[List[float]]:
    """
    Qwen 批量多模态 Embedding（调用时懒加载 Key）。
    每个 doc = {"text": str, "image_path": str | None}
    官方 SDK 单次调用支持多条 input，batch_size 建议 ≤ 5。
    """
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
        # 返回顺序与 input 一致
        return [e["embedding"] for e in resp.output["embeddings"]]
    raise Exception(f"Qwen Batch Embedding Error: {resp.message}")


# ──────────────────────────────────────────────
# 统一多线程 Embedding 引擎
# ──────────────────────────────────────────────

def get_universal_embeddings(
    docs: List[Dict],
    model_name: str,
    max_workers: int = 8,
    batch_size: int  = 5,      # Qwen 推荐 5，OpenAI 可以更大（32~100）
) -> np.ndarray:
    """
    Embedding 分发引擎（批量 + 多线程）：
    - model_name 含 'qwen' → Qwen 官方 SDK，批量多模态
    - 其他               → OpenAI 兼容端点，批量文本

    batch_size：每次 API 调用包含的样本数，越大调用次数越少但单次延迟越高。
    max_workers：并发线程数，决定同时进行多少个批次调用。
    """
    total   = len(docs)
    results = [None] * total
    service = "qwen" if "qwen" in model_name.lower() else "openai"

    # 按 batch_size 切分，保留起始下标用于结果回填
    batches: List[tuple[int, List[Dict]]] = [
        (start, docs[start: start + batch_size])
        for start in range(0, total, batch_size)
    ]
    n_batches = len(batches)

    # OpenAI 客户端（Qwen 不需要）
    client = None
    if service == "openai":
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        # OpenAI 批量可以更大，覆盖外部传入的 batch_size
        # （外部若传 5 也能用，只是调用次数多一些）

    print(f"--- Dispatching {total} docs in {n_batches} batches "
          f"(batch={batch_size}, workers={max_workers}, service={service} | {model_name}) ---")

    sample_emb = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if service == "openai":
            future_map = {
                executor.submit(
                    _fetch_openai_batch,
                    client,
                    [doc["text"] for doc in batch],  # OpenAI 只用 text
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
                embs = f.result()          # list of vectors, len == batch size
                for j, emb in enumerate(embs):
                    results[start + j] = emb
                    if sample_emb is None:
                        sample_emb = emb
            except Exception as e:
                logger.error(f"Batch starting at {start} failed: {e}")

    # 失败的 slot 用零向量填充
    dim = len(sample_emb) if sample_emb is not None else None
    for i in range(total):
        if results[i] is None:
            if dim is None:
                raise RuntimeError("FATAL: All embedding batches failed! Check API Keys and Network.")
            logger.warning(f"Slot {i} failed, filling with zeros.")
            results[i] = np.zeros(dim).tolist()

    return np.array(results, dtype="float32")
