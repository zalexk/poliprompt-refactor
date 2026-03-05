from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import faiss


def maximal_marginal_relevance(
    query_embedding, pool_embeddings, selected_indices, k, lambda_param=1.0, metric="euclidean"
):
    """
    Apply the Maximal Marginal Relevance (MMR) algorithm to select k items.

    Parameters:
        - query_embedding (np.ndarray): The embedding of the query item.
        - pool_embeddings (np.ndarray): The embeddings of the items in the pool.
        - selected_indices (list of str): The preselected indices.
        - k (int): The number of items to select.
        - lambda_param (float): The trade-off parameter between relevance and diversity (0 <= lambda_param <= 1).

    Returns:
        - selected_indices (list): The indices of the selected items in the pool.
    """

    # Calculate the distances between the query and the pool embeddings
    relevance_scores = cdist(query_embedding, pool_embeddings, metric=metric).flatten()

    # List to store the indices of selected items
    if not selected_indices:
        selected_indices = []

    # While we have not yet selected k items
    for _ in range(k):
        if not selected_indices:
            # Select the item with the highest relevance (smallest distance)
            selected_idx = int(np.argmin(relevance_scores))
        else:
            # Calculate diversity for each unselected item
            diversity_scores = np.min(cdist(pool_embeddings[selected_indices], pool_embeddings, metric=metric), axis=0)
            # Combine relevance and diversity to score items
            mmr_scores = lambda_param * (-relevance_scores) + (1 - lambda_param) * diversity_scores
            # mmr_scores = lambda_param * relevance_scores - (1 - lambda_param) * diversity_scores
            # Select the item with the highest MMR score
            selected_idx = int(np.argmax(mmr_scores))

        # Append the selected index to the list of selected indices
        selected_indices.append(selected_idx)

        # Mark the selected item as used by setting its score to -infinity
        relevance_scores[selected_idx] = np.inf
        # relevance_scores[selected_idx] = -np.inf

    return selected_indices



def get_k_nearest_distinct_classes(query_embedding, pool_embeddings, pool_labels, kshots, options, metric="euclidean"):
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

    pool_labels = ds.iloc[indices][answer_col].tolist()

    # Initialize the list for selected indices
    selected_indices = []
    if options:
        selected_indices = get_k_nearest_distinct_classes(
            query_embedding, pool_embeddings, pool_labels, kshots, options
        )

    # Perform Maximal Marginal Relevance (MMR) selection
    selected_indices = maximal_marginal_relevance(
        query_embedding, pool_embeddings, selected_indices, kshots - len(selected_indices), lambda_param=lambda_param
    )
    
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
        
        selected_data.append({
            "content": feature_text, 
            "image_path": image_path, # <--- 存入字典，给 Prompt 喂图
            "answer": label
        })

    return selected_data, kshot_distances, rag_labels, kshots_indices
