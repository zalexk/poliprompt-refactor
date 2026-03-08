from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import faiss
import logging


logger = logging.getLogger(__name__)

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    pool_embeddings: np.ndarray,
    initial_indices: list,
    k: int,
    lambda_param: float
) -> list:
    """
    PoliPrompt v3.1 增强版 MMR 逻辑
    参数:
        query_embedding: 当前 Query 的向量 (1, dim)
        pool_embeddings: 候选池的向量 (N, dim)
        lambda_param: 相关性与多样性的平衡系数 [0, 1]
        k: 最终需要选出的总样本数
        initial_indices: 初始已选中的索引 (例如通过类别均衡预筛选出的)
    返回:
        包含初始索引在内的、经过 MMR 排序后的索引列表
    """
    # 0. 鲁棒性检查：如果已经选够了，直接返回
    if len(initial_indices) >= k:
        return initial_indices[:k]

    # 1. 确保 query_embedding 是二维的
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # 2. 计算相关性得分 (Cosine Similarity: 越大越好)
    # scipy 的 cdist(metric='cosine') 返回的是余弦距离 = 1 - cos_sim
    # 所以 1 - cos_dist = cos_sim，范围在 [-1, 1]
    relevance_scores = 1.0 - cdist(query_embedding, pool_embeddings, metric="cosine").flatten()

    # 3. 初始化选择列表
    selected_indices = list(initial_indices)

    # 🚨 核心修复 1: 屏蔽掉初始索引，确保它们不会在后续循环中被再次选中
    # 将其分数设为负无穷
    relevance_scores[selected_indices] = -np.inf

    # 4. 迭代选择剩余的样本
    num_to_select = k - len(selected_indices)
    
    for _ in range(num_to_select):
        # 如果当前已经没有可选的了（逻辑兜底）
        if np.all(relevance_scores == -np.inf):
            break
        if not selected_indices:
            # 此时没有已选样本，无法计算多样性，直接在剩余候选中选最相关的
            best_idx = np.argmax(relevance_scores)
        else:
            # 核心修复 2: 计算多样性得分 (Diversity Score)
            # 目标：寻找一个候选点，使其与“已选集合”中最像的那个点，尽可能不像。
            # 即：求 candidate 与 selected_set 的最大相似度，并在 MMR 公式中减去它。
            
            # 计算 pool 与已选集合之间的余弦距离矩阵 (N_pool, N_selected)
            dist_matrix = cdist(pool_embeddings, pool_embeddings[selected_indices], metric="cosine")
            # 转换为相似度 (N_pool, N_selected)
            sim_matrix = 1.0 - dist_matrix
            
            # 对于 pool 中的每个点，找到它与已选集合中最相似的那个点的相似度值
            diversity_scores = np.max(sim_matrix, axis=1)

            # 🚨 核心修复 3: MMR 公式
            # 公式：Score = λ * Relevance - (1 - λ) * Max_Similarity_to_Selected
            mmr_scores = lambda_param * relevance_scores - (1.0 - lambda_param) * diversity_scores

            # 选取 MMR 得分最高的索引
            best_idx = np.argmax(mmr_scores)
            
        # 记录并更新屏蔽
        selected_indices.append(int(best_idx))
        relevance_scores[best_idx] = -np.inf

    # 5. 重复性最终检查
    if len(selected_indices) != len(set(selected_indices)):
        logger.warning(f"MMR 警告: 选中的索引存在重复! Indices: {selected_indices}")

    return selected_indices



def get_k_nearest_distinct_classes(query_embedding, pool_embeddings, pool_labels, kshots, options, metric="cosine"):
    """
    Get the indices of k embeddings from distinct classes that are most similar to the query_embedding.

    Parameters:
        - query_embedding (np.ndarray): The embedding of the query item.
        - pool_embeddings (np.ndarray): The embeddings of the items in the pool.
        - pool_labels (list or np.ndarray): The class labels corresponding to the items in the pool.
        - kshots (int): The number of distinct classes to select.
        - options (list of str): The classes to select from.
        - metric (str): The distance metric to use (default is 'euclidean').

    Returns:
        - selected_indices (list): The indices of the selected items in the pool.
    """

    # Calculate the distances between the query and the pool embeddings
    distances = cdist(query_embedding, pool_embeddings, metric=metric).flatten()

    # Get the sorted indices based on distance (smallest to largest)
    sorted_indices = np.argsort(distances)

    # Initialize a list to store selected indices and a set for the selected classes
    selected_indices = []
    un_selected_classes = set(options)
    k_selected = 0

    # Iterate through the sorted indices and select distinct classes
    for idx in sorted_indices:
        if pool_labels[idx] in un_selected_classes:
            selected_indices.append(int(idx))
            un_selected_classes.remove(pool_labels[idx])
            k_selected += 1

        # Stop when we've selected k distinct classes
        if len(un_selected_classes) == 0 or k_selected == kshots:
            break

    return selected_indices


def select_kshots(
    ds: pd.DataFrame,
    feature_col: str,
    image_col: str,
    answer_col: str,
    kshots: int,
    idx: int,
    indices: List[int],
    index: faiss.Index,
    lambda_param: float = 1.0,
    options=None,
    rules_dict: dict = None
) ->  Tuple[List[Dict[str, str]], List[float], List[str], List[int]]:
    """
    
    
    """
    # Extract the embedding for the query example
    # index.reconstruct(idx)让faiss根据编号，去它内部压缩、优化过的数据库里，把原始向量重新“组装”拿出来
    # .reshape(1,-1) 
    query_embedding = np.array(index.reconstruct(int(idx))).reshape(1,-1).astype('float32') # type:ignore

    # Extract the embeddings for the pool of examples
    # faiss不擅长随机切片，根据indices去把n个向量取出来，拼成一个numpy矩阵
    pool_embeddings = np.array([index.reconstruct(int(i)) for i in indices]).astype('float32') # type:ignore
    if pool_embeddings.ndim == 1:
        pool_embeddings = pool_embeddings.reshape(1, -1)
    
    print(f"DEBUG: query_embedding shape: {query_embedding.shape}")
    print(f"DEBUG: pool_embeddings shape: {pool_embeddings.shape}")

    # pool_labels = ds.iloc[indices][answer_col].tolist()
    pool_labels = ds.iloc[indices][answer_col].astype(str).tolist()

    # Initialize the list for selected indices
    selected_indices = []
    if options:
        selected_indices = get_k_nearest_distinct_classes(
            query_embedding, pool_embeddings, pool_labels, kshots, options
        )
        print(f"DEBUG: Query 找到了初始索引: {selected_indices}")

    # Perform Maximal Marginal Relevance (MMR) selection
    selected_indices = maximal_marginal_relevance(
        query_embedding, pool_embeddings, selected_indices, kshots, lambda_param=lambda_param
    )
    print(f"DEBUG: MMR 最终选出的索引: {selected_indices}")
    
       # 4. 【核心采集】提取这几个被选中样本的所有信息，包括图片路径！
    kshots_indices = [indices[i] for i in selected_indices]
    
    kshot_distances = []
    rag_labels = []
    selected_data = []

    for i_in_pool, real_idx in zip(selected_indices, kshots_indices):
        # 记录距离
        dist = np.linalg.norm(query_embedding - pool_embeddings[i_in_pool])
        kshot_distances.append(float(dist))
        
        # 记录标签
        label = ds.loc[real_idx, answer_col]
        rag_labels.append(str(label))
        
        # --- 【关键修正点】 ---
        # 以前只拿了 text，现在我们要把 text 和 image 路径都打包
        feature_text = ds.loc[real_idx, feature_col]
        image_path = ds.loc[real_idx, image_col] # <--- 拿到图片路径
        explanation = rules_dict.get(str(real_idx), "No explanation available.") if rules_dict else "No explanation available."
        
        selected_data.append({
            "content": feature_text, 
            "image_path": image_path, # <--- 存入字典，给 Prompt 喂图
            "answer": label,
            "explanation": explanation
        })

    return selected_data, kshot_distances, rag_labels, kshots_indices
