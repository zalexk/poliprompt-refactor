import numpy as np

import openai
# import voyageai
# import anthropic

# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
# from langchain_mistralai import ChatMistralAI
from langchain_core.language_models.chat_models import BaseChatModel

from ratelimit import limits, sleep_and_retry
from typing import List, Dict
import os
import dashscope
from dashscope import MultiModalEmbedding
import base64
from langchain_core.messages import HumanMessage, SystemMessage


import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .utils import get_base64_image





def create_llm(llm_name, model_config: Dict) -> BaseChatModel:
    # 逻辑：如果是 qwen 模型，使用通义千问的兼容端点；否则使用中转站端点
    if "qwen" in llm_name.lower():
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = os.getenv("DASHSCOPE_API_KEY")
    else:
        base_url = "https://api2.aigcbest.top/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        
    return ChatOpenAI(
        model_name=llm_name, 
        max_tokens=model_config["max_tokens"], 
        temperature=model_config["temperature"],
        base_url=base_url,
        openai_api_key=api_key
    )



def call_llm_wrapper(calls, period):
    @sleep_and_retry
    @limits(calls=calls, period=period)
    def call_llm(func, *args, **kwargs):
        """
        万能限速包装器：不再限制参数格式，直接执行传入的函数。
        """
        # func 就是你传进来的 model.invoke 或者 chain.invoke
        # *args 和 **kwargs 会原封不动地传给那个函数
        response = func(*args, **kwargs)
        return response

    return call_llm


def get_llm_embeddings(service, docs, batch_size, model,**kwargs):
    """
    Fetches embeddings for a batch of texts using the voyageai model.

    Parameters:
        - docs (list of str): A list of docs to embed.
        - batch_size (int): The number of texts to process in each batch.
        - model: The embedding model to use.

    Returns:
        - numpy array of vectors of float: The embeddings for the input docs.
    """
    if service == "openai":
        embeddings = get_openai_embeddings(docs, batch_size, model)
    elif service == "qwen":
        embeddings = get_qwen_embeddings(docs, batch_size, model,**kwargs)
    # elif service == "voyageai":
    #     embeddings = get_voyageai_embeddings(docs, batch_size, model)
    else:
        raise ValueError(f"Unsupported service: {service}. Choose 'voyageai' or 'openai'.")
    return embeddings




def get_openai_embeddings(docs, batch_size, model="text-embedding-3-small"):
    """
    Fetches embeddings for a batch of texts using the openai model.

    Parameters:
        - docs (list of str): A list of docs to embed.
        - batch_size (int): The number of texts to process in each batch.
        - model: The embedding model to use.

    Returns:
        - numpy array of vectors of float: The embeddings for the input docs.
    """

    client = openai.OpenAI(base_url="https://api2.aigcbest.top/v1",api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = []

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_ebs = client.embeddings.create(input=batch_docs, model=model).data
        # 返回一个列表里面每一个元素是一个对象，包含embedding对应的向量
        embeddings.extend([eb.embedding for eb in batch_ebs])
        # 对于刚才拿到的对象（eb），把embedding那一串数字向量找出来，得到一个组干净的向量列表
        # extend作用是把新列表里的每一个向量拆开，一个个加紧原本的embeddings大列表里，extend是全部展开还是原来的维度，但是append会增加维度，但是后面的降维等函数只认二维数组
    return np.array(embeddings)
    # 转换为Numpy数组


# 1. 初始化日志 (确保能看到每个线程的进度)
logger = logging.getLogger(__name__)

# 2. 从 .env 文件加载 API 密钥
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 3. 定义安全的 API 调用 (带指数退避重试)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def call_qwen_api_safe(input_data: List[Dict], model: str) -> List[np.ndarray]:
    """
    线程安全地调用 Qwen API，带指数退避和重试。
    """
    try:
        resp = MultiModalEmbedding.call(
            model=model,
            input=input_data,
            parameters={"dimension": 1024}
        )
        if resp.status_code == 200:
            embeddings = [np.array(eb['embedding']) for eb in resp.output['embeddings']]
            return embeddings
        else:
            raise Exception(f"API Error: {resp.code} - {resp.message}")
    except Exception as e:
        logger.error(f"API 调用失败: {e}")
        raise

# 4. 定义核心的 Embedding 函数 (多线程并发)
def get_qwen_embeddings(
    docs: List[Dict],
    batch_size: int = 5,
    model="qwen3-vl-embedding",
    max_workers: int = 8  # 调整并发线程数
) -> np.ndarray:
    """
    多线程并发获取 Qwen-VL Embedding 向量。

    参数:
        docs: 包含 "text" 和 "image_path" 字段的文档列表
        batch_size: 每个 API 请求的样本数量 (Qwen 限制为 5)
        max_workers: 并发线程数 (根据 API 限流调整)

    返回:
        所有文档的 Embedding 向量 (N, 1024)
    """
    total_docs = len(docs)
    all_embeddings = [None] * total_docs  # 预分配空间，保证顺序

    # 内部函数：处理单个批次
    def process_batch(start_idx: int) -> None:
        end_idx = min(start_idx + batch_size, total_docs)
        batch_docs = docs[start_idx:end_idx]
        input_data = [{"text": item["text"], "image": item["image_path"]} for item in batch_docs]

        try:
            embeddings = call_qwen_api_safe(input_data, model)   # 传入 model
            for i, emb in enumerate(embeddings):
                all_embeddings[start_idx + i] = emb
            logger.info(f"线程 {os.getpid()}：已完成 {start_idx + 1} - {end_idx} / {total_docs} 的 Embedding")
        except Exception as e:
            logger.error(f"线程 {os.getpid()}：批次 {start_idx} 发生错误: {e}")
            raise   # 重新抛出，让 future.result() 能捕获

    # 创建线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有批次任务
        futures = [executor.submit(process_batch, i) for i in range(0, total_docs, batch_size)]
        
        # 等待所有任务完成 (as_completed 保证了即使有异常也能正常退出)
        for future in as_completed(futures):
            try:
                future.result()  # 如果有异常，在这里会被抛出
            except Exception as e:
                logger.error(f"线程池中发生未处理的异常: {e}")
                raise 

    # 转换为 NumPy 数组
    return np.array(all_embeddings)



def _prepare_multimodal_message(system_prompt, text_content, image_path):
    # 使用带缓存的读取
    b64_image = get_base64_image(image_path)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": text_content},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "low"}}
        ])
    ]
    return messages


MODEL_PRICES = {
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
    "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
}

# 2026年最新参考价 (美元/百万Token)
def calculate_token_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """计算单次调用的美元成本"""
    # 模糊匹配 model_name
    matched_price = None
    for key in MODEL_PRICES:
        if key in model_name.lower():
            matched_price = MODEL_PRICES[key]
            break
    
    # 如果没匹配到，默认按 gpt-4o-mini 计费（或者报错）
    if not matched_price:
        matched_price = MODEL_PRICES["gpt-4o-mini"]
        
    cost = (prompt_tokens / 1_000_000 * matched_price["prompt"]) + \
           (completion_tokens / 1_000_000 * matched_price["completion"])
    return round(cost, 6)

