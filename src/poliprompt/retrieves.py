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
    lambda_param: float,
    metric: str = "cosine"
) -> list:
    """
    Select k examples from the candidate pool using Maximal Marginal Relevance.

    Args:
        query_embedding: Embedding of the current query (1, dim).
        pool_embeddings: Embeddings of all candidates (N, dim).
        initial_indices: Pre-selected indices (e.g. from class-balance pre-filter).
        k: Total number of examples to return (including initial_indices).
        lambda_param: Balance between relevance (1.0) and diversity (0.0).
        metric: Distance metric — 'cosine' or 'euclidean'.

    Returns:
        List of selected indices of length <= k.
    """
    # Validate initial_indices for duplicates before any computation
    if len(initial_indices) != len(set(initial_indices)):
        logger.warning(f"MMR: duplicate indices in initial_indices {initial_indices}; deduplicating.")
        initial_indices = list(dict.fromkeys(initial_indices))  # deduplicate, preserve order

    # If the pre-selected set already meets k, return early
    if len(initial_indices) >= k:
        return initial_indices[:k]

    # Ensure query_embedding is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Compute relevance scores against the query
    # cdist with metric='cosine' returns cosine distance = 1 - cos_sim
    if metric == "cosine":
        relevance_scores = 1.0 - cdist(query_embedding, pool_embeddings, metric="cosine").flatten()
    else:
        # Negate Euclidean distance so that closer = higher score
        relevance_scores = -cdist(query_embedding, pool_embeddings, metric="euclidean").flatten()

    selected_indices = list(initial_indices)

    # Mask out already-selected indices so they are not chosen again
    relevance_scores[selected_indices] = -np.inf

    num_to_select = k - len(selected_indices)

    for _ in range(num_to_select):
        if np.all(relevance_scores == -np.inf):
            break
        if not selected_indices:
            # No selected examples yet — pick the most relevant candidate
            best_idx = np.argmax(relevance_scores)
        else:
            if metric == "cosine":
                dist_matrix = cdist(pool_embeddings, pool_embeddings[selected_indices], metric="cosine")
                diversity_scores = np.max(1.0 - dist_matrix, axis=1)
            else:
                dist_matrix = cdist(pool_embeddings, pool_embeddings[selected_indices], metric="euclidean")
                diversity_scores = -np.min(dist_matrix, axis=1)

            # MMR score: λ * relevance - (1 - λ) * max_similarity_to_selected
            mmr_scores = lambda_param * relevance_scores - (1.0 - lambda_param) * diversity_scores
            best_idx = np.argmax(mmr_scores)

        selected_indices.append(int(best_idx))
        relevance_scores[best_idx] = -np.inf

    return selected_indices


def get_k_nearest_distinct_classes(
    query_embedding, pool_embeddings, pool_labels, kshots, options, metric="cosine"
) -> list:
    """
    Select one nearest example per class from the pool, up to kshots total.

    Ensures at least one representative per class before MMR re-ranking.
    """
    distances = cdist(query_embedding, pool_embeddings, metric=metric).flatten()
    sorted_indices = np.argsort(distances)

    selected_indices = []
    un_selected_classes = set(options)
    k_selected = 0

    for idx in sorted_indices:
        if str(pool_labels[idx]) in un_selected_classes:
            selected_indices.append(int(idx))
            un_selected_classes.remove(str(pool_labels[idx]))
            k_selected += 1

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
    pool_embeddings: np.ndarray,
    lambda_param: float = 1.0,
    options=None,
    metric: str = "cosine",
    rules_dict: dict = None
) -> Tuple[List[Dict[str, str]], List[float], List[str], List[int]]:
    """
    Retrieve k few-shot examples for a given query row using class-balanced MMR.

    Args:
        ds: Full dataset DataFrame.
        feature_col: Text column name.
        image_col: Image column name (None for text-only tasks).
        answer_col: Label column name.
        kshots: Number of examples to retrieve.
        idx: Row index of the current query.
        indices: Exemplar pool indices.
        index: FAISS index containing all embeddings.
        pool_embeddings: Pre-cached embeddings for the exemplar pool.
        lambda_param: MMR balance parameter.
        options: Valid class labels for class-balance pre-filtering.
        metric: Distance metric.
        rules_dict: Per-exemplar reasoning strings from the Map-Reduce phase.

    Returns:
        Tuple of (selected_data, distances, labels, global_indices).
    """
    # Retrieve the query embedding directly from the FAISS index
    query_embedding = np.array(index.reconstruct(int(idx))).reshape(1, -1).astype('float32')  # type: ignore

    pool_labels = ds.iloc[indices][answer_col].astype(str).tolist()

    selected_indices = []
    if options:
        selected_indices = get_k_nearest_distinct_classes(
            query_embedding, pool_embeddings, pool_labels, kshots, options,
            metric=metric
        )

    selected_indices = maximal_marginal_relevance(
        query_embedding, pool_embeddings, selected_indices, kshots,
        lambda_param=lambda_param, metric=metric
    )

    # Map pool-relative indices back to global DataFrame indices
    kshots_indices = [indices[i] for i in selected_indices]

    kshot_distances = []
    rag_labels = []
    selected_data = []

    for i_in_pool, real_idx in zip(selected_indices, kshots_indices):
        dist = np.linalg.norm(query_embedding - pool_embeddings[i_in_pool])
        kshot_distances.append(float(dist))

        label = ds.loc[real_idx, answer_col]
        rag_labels.append(str(label))

        feature_text = ds.loc[real_idx, feature_col]
        image_path = ds.loc[real_idx, image_col] if image_col else None
        explanation = (
            rules_dict.get(str(real_idx), "No explanation available.")
            if rules_dict else "No explanation available."
        )

        selected_data.append({
            "content": feature_text,
            "image_path": image_path,
            "answer": label,
            "explanation": explanation
        })

    return selected_data, kshot_distances, rag_labels, kshots_indices
