"""Unit tests for poliprompt.selectors."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from poliprompt.selectors import (
    KMeansExemplarSelector,
    RandomExemplarSelector,
    create_selector,
)

N = 30  # number of embeddings in test data


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    return rng.standard_normal((N, 16)).astype("float32")


# ---------------------------------------------------------------------------
# KMeansExemplarSelector
# ---------------------------------------------------------------------------

def test_kmeans_returns_correct_count(embeddings):
    indices = KMeansExemplarSelector().select_exemplars(embeddings, n_exemplars=5, random_state=0)
    assert len(indices) == 5


def test_kmeans_indices_in_valid_range(embeddings):
    indices = KMeansExemplarSelector().select_exemplars(embeddings, n_exemplars=5, random_state=0)
    assert all(0 <= i < N for i in indices)


def test_kmeans_no_duplicate_indices(embeddings):
    indices = KMeansExemplarSelector().select_exemplars(embeddings, n_exemplars=5, random_state=0)
    assert len(indices) == len(set(indices))


def test_kmeans_deterministic_with_same_seed(embeddings):
    s1 = KMeansExemplarSelector().select_exemplars(embeddings, n_exemplars=5, random_state=7)
    s2 = KMeansExemplarSelector().select_exemplars(embeddings, n_exemplars=5, random_state=7)
    assert sorted(s1) == sorted(s2)


def test_kmeans_single_exemplar(embeddings):
    indices = KMeansExemplarSelector().select_exemplars(embeddings, n_exemplars=1, random_state=0)
    assert len(indices) == 1


def test_kmeans_n_exemplars_exceeds_points_raises():
    """Requesting more clusters than data points must raise an error.

    sklearn raises ValueError at fit() time (n_samples < n_clusters); our
    RuntimeError guard covers any degenerate empty-cluster case that survives
    that check.  Either way the call must not silently succeed.
    """
    tiny = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    with pytest.raises((ValueError, RuntimeError)):
        KMeansExemplarSelector().select_exemplars(tiny, n_exemplars=10, random_state=0)


# ---------------------------------------------------------------------------
# RandomExemplarSelector
# ---------------------------------------------------------------------------

def test_random_returns_correct_count(embeddings):
    indices = RandomExemplarSelector().select_exemplars(embeddings, n_exemplars=8)
    assert len(indices) == 8


def test_random_no_duplicates(embeddings):
    indices = RandomExemplarSelector().select_exemplars(embeddings, n_exemplars=8)
    assert len(set(indices)) == 8


def test_random_indices_in_valid_range(embeddings):
    indices = RandomExemplarSelector().select_exemplars(embeddings, n_exemplars=10)
    assert all(0 <= i < N for i in indices)


# ---------------------------------------------------------------------------
# create_selector factory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,expected_type", [
    ("kmeans", KMeansExemplarSelector),
    ("random", RandomExemplarSelector),
    ("KMeans", KMeansExemplarSelector),   # case-insensitive
    ("RANDOM", RandomExemplarSelector),
])
def test_create_selector_returns_correct_type(method, expected_type):
    assert isinstance(create_selector(method), expected_type)


def test_create_selector_invalid_method_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        create_selector("pca")
