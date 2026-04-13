from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import KMeans


class ExemplarSelector(ABC):
    """Abstract base class for selecting exemplars from a set of embeddings."""

    @abstractmethod
    def select_exemplars(self, embeddings: np.ndarray, n_exemplars: int = 10, **kwargs) -> list:
        """
        Select exemplar indices from the given embeddings.

        Parameters:
            - embeddings (numpy.ndarray): Embeddings to select from.
            - n_exemplars (int): Number of exemplars to select.

        Returns:
            - list: Indices of the selected exemplars.
        """
        pass


class RandomExemplarSelector(ExemplarSelector):
    """Selects exemplars uniformly at random."""

    def select_exemplars(self, embeddings: np.ndarray, n_exemplars: int = 10, **kwargs) -> list:
        """
        Randomly select n_exemplars indices without replacement.

        Parameters:
            - embeddings (np.ndarray): Embeddings to select from.
            - n_exemplars (int): Number of exemplars to select.

        Returns:
            - list: Randomly selected indices.
        """
        return np.random.choice(len(embeddings), n_exemplars, replace=False).tolist()


class KMeansExemplarSelector(ExemplarSelector):
    """Selects exemplars as the closest point to each KMeans cluster center."""

    def select_exemplars(self, embeddings: np.ndarray, n_exemplars: int = 10, **kwargs) -> list:
        """
        Select exemplars using KMeans clustering.

        One exemplar is chosen per cluster: the point closest to that cluster's centroid.

        Parameters:
            - embeddings (numpy.ndarray): Embeddings to select from.
            - n_exemplars (int): Number of clusters (and exemplars) to create.
            - **kwargs: Additional keyword arguments passed to KMeans.

        Returns:
            - list: Indices of the selected exemplars.

        Raises:
            - RuntimeError: If KMeans produces an empty cluster, which indicates
              n_exemplars exceeds the number of distinct data points.
        """
        kmeans = KMeans(n_clusters=n_exemplars, **kwargs)
        kmeans.fit(embeddings)

        exemplars_indices = []
        for i in range(n_exemplars):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) == 0:
                raise RuntimeError(
                    f"KMeans produced an empty cluster (cluster {i}). "
                    f"n_exemplars ({n_exemplars}) likely exceeds the number of distinct points "
                    f"in the dataset ({len(embeddings)})."
                )
            cluster_center = kmeans.cluster_centers_[i]
            closest_index = cluster_indices[
                np.argmin(np.linalg.norm(embeddings[cluster_indices] - cluster_center, axis=1))
            ]
            exemplars_indices.append(closest_index)

        return exemplars_indices


def create_selector(method: str) -> ExemplarSelector:
    """
    Factory function that returns an ExemplarSelector for the given method name.

    Parameters:
        - method (str): Selection strategy — 'kmeans' or 'random'.

    Returns:
        - ExemplarSelector: An instance of the requested selector.
    """
    selectors = {
        "kmeans": KMeansExemplarSelector,
        "random": RandomExemplarSelector,
    }

    if method.lower() not in selectors:
        raise ValueError(
            f"Unsupported method: '{method}'. "
            f"Choose from {list(selectors.keys())} (case-insensitive)."
        )

    return selectors[method.lower()]()
