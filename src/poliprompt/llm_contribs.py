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

logger = logging.getLogger(__name__)

# --- 1. 初始化官方 SDK Key (专门供 Embedding 阶段使用) ---
# 这样 Qwen SDK 在调用 MultiModalEmbedding 时会自动使用这个官方 Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 2. 工业级重试配置 ---
llm_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)

# --- 3. [推理专供] 端点分配逻辑 ---
def get_llm_provider_config():
    """
    Annotate 推理阶段专供：
    强制所有模型（GPT, Qwen等）都走中转站 URL。
    """
    return os.getenv("OPENAI_BASE_URL"), os.getenv("OPENAI_API_KEY")

# --- 4. 统一创建 LLM 实例 (推理用) ---
def create_llm(llm_name: str, model_config: Dict) -> ChatOpenAI:
    base_url, api_key = get_llm_provider_config()
    return ChatOpenAI(
        model=llm_name,
        base_url=base_url,
        api_key=api_key,
        temperature=model_config.get("temperature", 0.0),
        max_tokens=model_config.get("max_tokens", 1024),
        max_retries=5,
        timeout=120  # 给中转站足够的响应时间
    )

# --- 5. 具体的底层 Embedding 调用 (带重试) ---

@llm_retry
def _fetch_openai_embedding(client: openai.OpenAI, text: str, model: str) -> List[float]:
    """通过中转站获取 OpenAI 文本向量"""
    return client.embeddings.create(input=[text], model=model).data[0].embedding

@llm_retry
def _fetch_qwen_embedding(text: str, image_path: Optional[str], model: str) -> List[float]:
    """
    通过阿里官方 SDK 获取 Qwen 多模态向量。
    已移除固定 1024 维度的限制，适配模型默认维度。
    """
    input_data = [{"text": text}]
    if image_path and str(image_path).strip() != "" and os.path.exists(image_path):
        clean_path = os.path.abspath(image_path).replace("\\", "/")
        input_data.append({"image": clean_path})
    
    # 🚀 注意：此处不再传 parameters={"dimension": 1024}
    resp = MultiModalEmbedding.call(
        model=model,
        input=input_data
    )
    if resp.status_code == 200:
        return resp.output['embeddings'][0]['embedding']
    raise Exception(f"Qwen Embedding Error (Official SDK): {resp.message}")

# --- 6. 统一的多线程并发引擎 (嵌入用) ---

def get_universal_embeddings(
    docs: List[Dict], 
    model_name: str, 
    max_workers: int = 8
) -> np.ndarray:
    """
    嵌入分发引擎：
    - 如果包含 qwen：调用阿里官方 SDK。
    - 否则：调用中转站 OpenAI 兼容端点。
    """
    total = len(docs)
    results = [None] * total
    
    # 判断当前是哪个 Service
    service = "qwen" if "qwen" in model_name.lower() else "openai"
    
    # 预准备中转站客户端 (仅在需要时)
    client = None
    if service == "openai":
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    print(f"--- 🚀 Dispatching {total} Embedding tasks ({service} | {model_name}) ---")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, doc in enumerate(docs):
            if service == "openai":
                f = executor.submit(_fetch_openai_embedding, client, doc['text'], model_name)
            else:
                f = executor.submit(_fetch_qwen_embedding, doc['text'], doc.get('image_path'), model_name)
            future_to_idx[f] = i

        sample_emb = None
        for f in tqdm(as_completed(future_to_idx), total=total, desc="Embedding Progress"):
            idx = future_to_idx[f]
            try:
                emb = f.result()
                results[idx] = emb
                if sample_emb is None: 
                    sample_emb = emb # 记录第一个成功返回的向量以推断维度
            except Exception as e:
                logger.error(f" Index {idx} failed: {e}")

        # 🚀 维度动态推断逻辑
        dim = len(sample_emb) if sample_emb is not None else None
        
        for i in range(len(results)):
            if results[i] is None:
                if dim is None:
                    raise RuntimeError(" FATAL: All embedding tasks failed! Check API Keys and Network.")
                results[i] = np.zeros(dim).tolist()

    return np.array(results).astype('float32')


# import numpy as np
# from pathlib import Path
# import openai

# from langchain_openai import ChatOpenAI
# from langchain_core.language_models.chat_models import BaseChatModel

# from ratelimit import limits, sleep_and_retry
# from typing import List, Dict
# import os
# import dashscope
# from dashscope import MultiModalEmbedding
# import base64
# from langchain_core.messages import HumanMessage, SystemMessage


# import logging
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dotenv import load_dotenv
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# # from .utils import get_base64_image





# def create_llm(llm_name, model_config: Dict) -> BaseChatModel:
#     # 逻辑：如果是 qwen 模型，使用通义千问的兼容端点；否则使用中转站端点
#     if "qwen" in llm_name.lower():
#         base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#         api_key = os.getenv("DASHSCOPE_API_KEY")
#     else:
#         base_url = "https://api2.aigcbest.top/v1"
#         api_key = os.getenv("OPENAI_API_KEY")
        
#     return ChatOpenAI(
#         model=llm_name, 
#         # max_tokens=model_config.get("max_tokens", 1024),
#         # temperature=model_config.get("temperature", 0.7),
#         base_url=base_url,
#         api_key=api_key
#     )



# def call_llm_wrapper(calls, period):
#     @sleep_and_retry
#     @limits(calls=calls, period=period)
#     def call_llm(func, *args, **kwargs):
#         """
#         万能限速包装器：不再限制参数格式，直接执行传入的函数。
#         """
#         # func 就是你传进来的 model.invoke 或者 chain.invoke
#         # *args 和 **kwargs 会原封不动地传给那个函数
#         response = func(*args, **kwargs)
#         return response

#     return call_llm


# # def get_llm_embeddings(service, docs, batch_size, model,**kwargs):
# #     """
# #     Fetches embeddings for a batch of texts using the voyageai model.

# #     Parameters:
# #         - docs (list of str): A list of docs to embed.
# #         - batch_size (int): The number of texts to process in each batch.
# #         - model: The embedding model to use.

# #     Returns:
# #         - numpy array of vectors of float: The embeddings for the input docs.
# #     """
# #     if service == "openai":
# #         embeddings = get_openai_embeddings(docs, batch_size, model)
# #     elif service == "qwen":
# #         embeddings = get_qwen_embeddings(docs, batch_size, model,**kwargs)
# #     # elif service == "voyageai":
# #     #     embeddings = get_voyageai_embeddings(docs, batch_size, model)
# #     else:
# #         raise ValueError(f"Unsupported service: {service}. Choose 'voyageai' or 'openai'.")
# #     return embeddings




# def get_openai_embeddings(docs, batch_size, model="text-embedding-3-small"):
#     """
#     Fetches embeddings for a batch of texts using the openai model.

#     Parameters:
#         - docs (list of str): A list of docs to embed.
#         - batch_size (int): The number of texts to process in each batch.
#         - model: The embedding model to use.

#     Returns:
#         - numpy array of vectors of float: The embeddings for the input docs.
#     """

#     client = openai.OpenAI(base_url="https://api2.aigcbest.top/v1",api_key=os.getenv("OPENAI_API_KEY"))
#     embeddings = []

#     for i in range(0, len(docs), batch_size):
#         batch_docs = docs[i : i + batch_size]
#         batch_ebs = client.embeddings.create(input=batch_docs, model=model).data
#         # 返回一个列表里面每一个元素是一个对象，包含embedding对应的向量
#         embeddings.extend([eb.embedding for eb in batch_ebs])
#         # 对于刚才拿到的对象（eb），把embedding那一串数字向量找出来，得到一个组干净的向量列表
#         # extend作用是把新列表里的每一个向量拆开，一个个加紧原本的embeddings大列表里，extend是全部展开还是原来的维度，但是append会增加维度，但是后面的降维等函数只认二维数组
#     return np.array(embeddings)
#     # 转换为Numpy数组


# # 1. 初始化日志 (确保能看到每个线程的进度)
# logger = logging.getLogger(__name__)

# # 2. 从 .env 文件加载 API 密钥
# load_dotenv()
# dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# # 3. 定义安全的 API 调用 (带指数退避重试)
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type(Exception)
# )
# def call_qwen_api_safe(input_data: List[Dict], model: str) -> List[np.ndarray]:
#     """
#     线程安全地调用 Qwen API，带指数退避和重试。
#     """
#     try:
#         resp = MultiModalEmbedding.call(
#             model=model,
#             input=input_data,
#             parameters={"dimension": 1024}
#         )
#         if resp.status_code == 200:
#             embeddings = [np.array(eb['embedding']) for eb in resp.output['embeddings']]
#             return embeddings
#         else:
#             raise Exception(f"API Error: {resp.code} - {resp.message}")
#     except Exception as e:
#         logger.error(f"API 调用失败: {e}")
#         raise


# def get_qwen_embeddings(
#     docs: List[Dict],
#     batch_size: int = 5,
#     model="qwen3-vl-embedding",
#     max_workers: int = 8 
# ) -> np.ndarray:
    
#     total_docs = len(docs)
#     all_embeddings = [None] * total_docs

#     def process_batch(start_idx: int) -> None:
#         end_idx = min(start_idx + batch_size, total_docs)
#         batch_docs = docs[start_idx:end_idx]
        
#         input_data = []
#         for item in batch_docs:
#             raw_path = item["image_path"]
#             # 🚀 铁血逻辑：只要绝对路径，且把反斜杠换成正斜杠
#             # 阿里 SDK 在识别 D:/xxx 这种格式时最稳
#             clean_abs_path = os.path.abspath(raw_path).replace("\\", "/")
            
#             # 🚨 物理检查：如果这时候硬盘上都找不到，直接报错拦截，不要发给 API
#             if not os.path.exists(clean_abs_path):
#                 raise FileNotFoundError(f"图片物理路径不存在: {clean_abs_path}")
                
#             input_data.append({"text": item["text"], "image": clean_abs_path})

#         try:
#             embeddings = call_qwen_api_safe(input_data, model)
#             for i, emb in enumerate(embeddings):
#                 all_embeddings[start_idx + i] = emb
#             logger.info(f"线程 {os.getpid()}：已完成 {start_idx + 1} - {end_idx}")
#         except Exception as e:
#             logger.error(f"❌ 批次 {start_idx} 失败: {e}")
#             raise 

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(process_batch, i) for i in range(0, total_docs, batch_size)]
#         for future in as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 raise e 

#     return np.array(all_embeddings)


# # # 4. 定义核心的 Embedding 函数 (多线程并发)
# # def get_qwen_embeddings(
# #     docs: List[Dict],
# #     batch_size: int = 5,
# #     model="qwen3-vl-embedding",
# #     max_workers: int = 8  # 调整并发线程数
# # ) -> np.ndarray:
    
# #     total_docs = len(docs)
# #     all_embeddings = [None] * total_docs  # 预分配空间，保证顺序

# #     # 内部函数：处理单个批次
# #     def process_batch(start_idx: int) -> None:
# #         end_idx = min(start_idx + batch_size, total_docs)
# #         batch_docs = docs[start_idx:end_idx]
# #         input_data = [{"text": item["text"], "image": item["image_path"]} for item in batch_docs]

# #         try:
# #             embeddings = call_qwen_api_safe(input_data, model)   # 传入 model
# #             for i, emb in enumerate(embeddings):
# #                 all_embeddings[start_idx + i] = emb
# #             logger.info(f"线程 {os.getpid()}：已完成 {start_idx + 1} - {end_idx} / {total_docs} 的 Embedding")
# #         except Exception as e:
# #             logger.error(f"线程 {os.getpid()}：批次 {start_idx} 发生错误: {e}")
# #             raise   # 重新抛出，让 future.result() 能捕获

# #     # 创建线程池
# #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
# #         # 提交所有批次任务
# #         futures = [executor.submit(process_batch, i) for i in range(0, total_docs, batch_size)]
        
# #         # 等待所有任务完成 (as_completed 保证了即使有异常也能正常退出)
# #         for future in as_completed(futures):
# #             try:
# #                 future.result()  # 如果有异常，在这里会被抛出
# #             except Exception as e:
# #                 logger.error(f"线程池中发生未处理的异常: {e}")
# #                 raise 

# #     # 转换为 NumPy 数组
# #     return np.array(all_embeddings)



# # def _prepare_multimodal_message(system_prompt, text_content, image_path):
# #     # 使用带缓存的读取
# #     b64_image = get_base64_image(image_path)
    
# #     messages = [
# #         SystemMessage(content=system_prompt),
# #         HumanMessage(content=[
# #             {"type": "text", "text": text_content},
# #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "low"}}
# #         ])
# #     ]
# #     return messages



