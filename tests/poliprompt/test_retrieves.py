"""Unit tests for poliprompt.retrieves."""

import numpy as np
import pandas as pd
import faiss
import pytest
from hypothesis import given, settings, strategies as st

from poliprompt.retrieves import (
    maximal_marginal_relevance,
    get_k_nearest_distinct_classes,
    select_kshots,
)

OPTIONS = ["0", "1"]


# ---------------------------------------------------------------------------
# maximal_marginal_relevance
# ---------------------------------------------------------------------------

def test_mmr_returns_k_items(embeddings):
    result = maximal_marginal_relevance(embeddings[0:1], embeddings, [], k=4, lambda_param=0.5)
    assert len(result) == 4


def test_mmr_no_duplicate_indices(embeddings):
    result = maximal_marginal_relevance(embeddings[0:1], embeddings, [], k=6, lambda_param=0.5)
    assert len(result) == len(set(result))


def test_mmr_preserves_initial_indices(embeddings):
    initial = [2, 5]
    result = maximal_marginal_relevance(embeddings[0:1], embeddings, initial, k=5, lambda_param=0.5)
    assert 2 in result
    assert 5 in result


def test_mmr_early_return_when_initial_meets_k(embeddings):
    initial = [0, 1, 2, 3]
    result = maximal_marginal_relevance(embeddings[0:1], embeddings, initial, k=4, lambda_param=0.5)
    assert result == initial


def test_mmr_lambda_1_top_result_is_self(embeddings):
    """With λ=1 (pure relevance) the top pick must be the query itself."""
    query_idx = 3
    result = maximal_marginal_relevance(
        embeddings[query_idx:query_idx + 1], embeddings, [], k=1, lambda_param=1.0
    )
    assert result[0] == query_idx


def test_mmr_euclidean_metric(embeddings):
    result = maximal_marginal_relevance(
        embeddings[0:1], embeddings, [], k=3, lambda_param=0.5, metric="euclidean"
    )
    assert len(result) == 3
    assert len(set(result)) == 3


def test_mmr_1d_query_is_accepted(embeddings):
    """A 1-D query embedding should be reshaped internally without raising."""
    result = maximal_marginal_relevance(embeddings[0], embeddings, [], k=2, lambda_param=0.5)
    assert len(result) == 2


def test_mmr_duplicate_initial_indices_deduped(embeddings):
    """Duplicate indices in initial_indices must be silently deduplicated."""
    result = maximal_marginal_relevance(
        embeddings[0:1], embeddings, [2, 2, 5], k=4, lambda_param=0.5
    )
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# get_k_nearest_distinct_classes
# ---------------------------------------------------------------------------

def test_distinct_classes_covers_all_options(embeddings):
    labels = [str(i % 2) for i in range(len(embeddings))]
    result = get_k_nearest_distinct_classes(
        embeddings[0:1], embeddings, labels, kshots=2, options=OPTIONS
    )
    returned_labels = {labels[i] for i in result}
    assert returned_labels == {"0", "1"}


def test_distinct_classes_count_does_not_exceed_kshots(embeddings):
    labels = [str(i % 2) for i in range(len(embeddings))]
    result = get_k_nearest_distinct_classes(
        embeddings[0:1], embeddings, labels, kshots=2, options=OPTIONS
    )
    assert len(result) <= 2


def test_distinct_classes_returns_valid_indices(embeddings):
    labels = [str(i % 2) for i in range(len(embeddings))]
    result = get_k_nearest_distinct_classes(
        embeddings[0:1], embeddings, labels, kshots=2, options=OPTIONS
    )
    assert all(0 <= i < len(embeddings) for i in result)


# ---------------------------------------------------------------------------
# select_kshots
# ---------------------------------------------------------------------------

@pytest.fixture
def kshot_fixtures(embeddings, faiss_index, sample_df):
    return dict(
        ds=sample_df,
        feature_col="text",
        image_col=None,
        answer_col="label",
        kshots=3,
        idx=0,
        indices=list(range(len(embeddings))),
        index=faiss_index,
        pool_embeddings=embeddings,
        lambda_param=0.5,
        options=OPTIONS,
    )


def test_select_kshots_count(kshot_fixtures):
    selected, dists, labels, indices = select_kshots(**kshot_fixtures)
    assert len(selected) == 3
    assert len(dists) == 3
    assert len(labels) == 3
    assert len(indices) == 3


def test_select_kshots_labels_are_valid_options(kshot_fixtures):
    _, _, labels, _ = select_kshots(**kshot_fixtures)
    assert all(l in OPTIONS for l in labels)


def test_select_kshots_selected_data_keys(kshot_fixtures):
    selected, _, _, _ = select_kshots(**kshot_fixtures)
    for ex in selected:
        assert "content" in ex
        assert "answer" in ex
        assert "explanation" in ex


def test_select_kshots_uses_rules_dict(kshot_fixtures):
    rules = {str(i): f"explanation_{i}" for i in range(len(kshot_fixtures["indices"]))}
    selected, _, _, _ = select_kshots(**kshot_fixtures, rules_dict=rules)
    assert all(ex["explanation"].startswith("explanation_") for ex in selected)


def test_select_kshots_no_rules_dict_fallback(kshot_fixtures):
    selected, _, _, _ = select_kshots(**kshot_fixtures, rules_dict=None)
    assert all(ex["explanation"] == "No explanation available." for ex in selected)


def test_select_kshots_distances_are_non_negative(kshot_fixtures):
    _, dists, _, _ = select_kshots(**kshot_fixtures)
    assert all(d >= 0.0 for d in dists)
