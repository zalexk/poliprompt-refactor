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

# import logging
# logger = logging.getLogger(__name__)

def create_llm(llm_name, model_config: Dict) -> BaseChatModel:
    if "gpt" in llm_name:
        model = ChatOpenAI(
            model_name=llm_name, max_tokens=model_config["max_tokens"], temperature=model_config["temperature"],base_url="https://api2.aigcbest.top/v1"
        )
    # elif "claude" in llm_name:
    #     model = ChatAnthropic(
    #         model_name=llm_name, max_tokens=model_config["max_tokens"], temperature=model_config["temperature"],base_url="https://api2.aigcbest.top/v1"
    #     )
    # elif "mistral" in llm_name:
    #     model = ChatMistralAI(
    #         model=llm_name, max_tokens=model_config["max_tokens"], temperature=model_config["temperature"],base_url="https://api2.aigcbest.top/v1"
    #     )
    else:
        raise ValueError(f"The specified model {llm_name} is not supported.")
    return model


# LLM API direct with exponential backoff
# def call_llm_wrapper(calls, period):
#     @sleep_and_retry
#     @limits(calls=calls, period=period)
#     def call_llm(chain, input_pairs: Dict):
#         response = chain.invoke(input_pairs)
#         return response

#     return call_llm

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


def get_llm_embeddings(service, docs, batch_size, model):
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
        embeddings = get_qwen_embeddings(docs, batch_size, model)
    # elif service == "voyageai":
    #     embeddings = get_voyageai_embeddings(docs, batch_size, model)
    else:
        raise ValueError(f"Unsupported service: {service}. Choose 'voyageai' or 'openai'.")
    return embeddings


# def get_voyageai_embeddings(docs, batch_size, model="voyage-2"):
#     """
#     Fetches embeddings for a batch of texts using the voyageai model.

#     Parameters:
#         - docs (list of str): A list of docs to embed.
#         - batch_size (int): The number of texts to process in each batch.
#         - model: The embedding model to use.

#     Returns:
#         - numpy array of vectors of float: The embeddings for the input docs.
#     """

#     client = voyageai.Client(base_url="https://api2.aigcbest.top/v1")
#     embeddings = []

#     # Process the docs in batches
#     for i in range(0, len(docs), batch_size):
#         batch_docs = docs[i : i + batch_size]
#         batch_ebs = client.embed(batch_docs, model=model)
#         embeddings.extend(batch_ebs.embeddings)
#     return np.array(embeddings)


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

    client = openai.OpenAI(base_url="https://api2.aigcbest.top/v1")
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

def get_qwen_embeddings(docs, batch_size, model="qwen3-vl-embedding"):
    """

    """
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    all_embeddings = []
    for item in docs:
        input_data = [
            {
                "text" : item['text'],
                "image" : item['image_path']
            }
        ]
        resp = MultiModalEmbedding.call(
            model = model,
            input = input_data, # type: ignore
            parameters={"dimension":1024}
        )
        if resp.status_code == 200:
            all_embeddings.append(resp.output['embeddings'][0]['embedding'])
        else:
            print(f"Error calling DashScope:{resp.message}")
    return np.array(all_embeddings)


def _prepare_multimodal_message(system_prompt,text_content,image_path):
    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {
                "type": "text",
                "text": text_content
            },
            {
                "type": "image_url",
                "image_url": {"url":f"data:image/jpeg;base64,{b64_image}"}
            }
        ])
    ]
    return messages
