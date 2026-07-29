"""Unit tests for poliprompt.retrieves."""

import numpy as np
import pandas as pd
import faiss
import pytest
from hypothesis import given, settings, strategies as st

from collections import Counter

from poliprompt.retrieves import (
    maximal_marginal_relevance,
    get_k_nearest_distinct_classes,
    select_kshots,
    allocate_proportional_quotas,
    select_kshots_proportional,
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


# ---------------------------------------------------------------------------
# allocate_proportional_quotas
# ---------------------------------------------------------------------------

def test_allocate_quotas_exact_proportions():
    assert allocate_proportional_quotas({"A": 30, "B": 15, "C": 5}, 20) == {"A": 12, "B": 6, "C": 2}


def test_allocate_quotas_tie_breaks_by_label():
    """Equal fractional remainders are resolved deterministically by label."""
    assert allocate_proportional_quotas({"A": 1, "B": 1, "C": 1}, 2) == {"A": 1, "B": 1, "C": 0}


def test_allocate_quotas_leftover_to_largest_remainder():
    assert allocate_proportional_quotas({"A": 10, "B": 10}, 3) == {"A": 2, "B": 1}


def test_allocate_quotas_degenerate_inputs():
    assert allocate_proportional_quotas({}, 5) == {}
    assert allocate_proportional_quotas({"A": 3}, 0) == {}


@given(
    counts=st.dictionaries(
        st.text(min_size=1, max_size=3),
        st.integers(min_value=1, max_value=50),
        min_size=1, max_size=6,
    ),
    total=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=50)
def test_allocate_quotas_always_sum_to_total(counts, total):
    quotas = allocate_proportional_quotas(counts, total)
    assert set(quotas) == set(counts)
    assert sum(quotas.values()) == total
    assert all(v >= 0 for v in quotas.values())


# ---------------------------------------------------------------------------
# select_kshots_proportional
# ---------------------------------------------------------------------------

def _make_pool(cluster_sizes, seed):
    """Tight clusters along orthogonal axes: cluster i sits near axis i."""
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for i, (label, size) in enumerate(cluster_sizes):
        loc = [0.0] * 8
        loc[i] = 2.0
        blocks.append(rng.normal(loc=loc, scale=0.01, size=(size, 8)))
        labels.extend([label] * size)
    embs = np.vstack(blocks).astype("float32")
    df = pd.DataFrame({"text": [f"row{i}" for i in range(len(labels))], "label": labels})
    index = faiss.IndexFlatIP(8)
    index.add(embs)
    return df, embs, index


def _prop_kwargs(df, embs, index, **overrides):
    kwargs = dict(
        ds=df, feature_col="text", image_col=None, answer_col="label",
        kshot_ratio=0.5, kshot_neighbors=10,
        idx=10,  # a 'B' row — its neighborhood is the whole B cluster
        indices=list(range(len(df))), index=index, pool_embeddings=embs,
        lambda_param=0.5,
    )
    kwargs.update(overrides)
    return kwargs


def test_proportional_kshots_follow_neighborhood_not_pool_mix():
    """Pool is 50/50 but the query sits in cluster B: all shots must come from B."""
    df, embs, index = _make_pool([("A", 10), ("B", 10)], seed=0)
    _, _, labels, _ = select_kshots_proportional(**_prop_kwargs(df, embs, index))
    assert len(labels) == 10  # 0.5 × 20
    assert set(labels) == {"B"}


def test_proportional_kshots_total_scales_with_ratio():
    df, embs, index = _make_pool([("A", 10), ("B", 10)], seed=0)
    _, _, labels5, _ = select_kshots_proportional(**_prop_kwargs(df, embs, index, kshot_ratio=0.25))
    assert len(labels5) == 5  # 0.25 × 20
    _, _, labels1, _ = select_kshots_proportional(**_prop_kwargs(df, embs, index, kshot_ratio=0.01))
    assert len(labels1) == 1  # max(1, round(0.2)) floor


def test_proportional_kshots_backfills_exhausted_class():
    """Class C's quota exceeds its 2 pool members; the deficit is backfilled from D."""
    df, embs, index = _make_pool([("C", 2), ("D", 10)], seed=1)
    _, _, labels, _ = select_kshots_proportional(
        **_prop_kwargs(df, embs, index, idx=0, kshot_neighbors=4)
    )
    # window = {C:2, D:2} → quotas {C:3, D:3}; C caps at 2, deficit 1 → D
    assert len(labels) == 6  # 0.5 × 12
    assert Counter(labels) == {"C": 2, "D": 4}


def test_proportional_kshots_window_larger_than_pool_uses_whole_pool():
    df, embs, index = _make_pool([("C", 2), ("D", 10)], seed=1)
    _, _, labels, _ = select_kshots_proportional(
        **_prop_kwargs(df, embs, index, idx=0, kshot_neighbors=999)
    )
    # window = whole pool {C:2, D:10} → quotas {C:1, D:5}
    assert Counter(labels) == {"C": 1, "D": 5}


def test_proportional_kshots_result_plumbing():
    df, embs, index = _make_pool([("A", 10), ("B", 10)], seed=0)
    rules = {str(i): f"reason_{i}" for i in range(20)}
    selected, dists, labels, global_idx = select_kshots_proportional(
        **_prop_kwargs(df, embs, index, rules_dict=rules)
    )
    assert len(selected) == len(dists) == len(labels) == len(global_idx) == 10
    assert len(set(global_idx)) == 10               # no duplicates
    assert all(10 <= i < 20 for i in global_idx)    # all from B rows
    # nearest-first by the retrieval metric (cosine), not by the reported L2 dists
    from scipy.spatial.distance import cdist
    cos_d = cdist(embs[10:11], embs[global_idx], metric="cosine").flatten()
    assert list(cos_d) == sorted(cos_d)
    assert all(d >= 0 for d in dists)
    assert all(ex["answer"] == "B" for ex in selected)
    assert all(ex["explanation"].startswith("reason_") for ex in selected)


def test_proportional_kshots_rules_dict_fallback():
    df, embs, index = _make_pool([("A", 10), ("B", 10)], seed=0)
    selected, _, _, _ = select_kshots_proportional(**_prop_kwargs(df, embs, index))
    assert all(ex["explanation"] == "No explanation available." for ex in selected)


def test_proportional_kshots_deterministic():
    df, embs, index = _make_pool([("A", 10), ("B", 10)], seed=0)
    r1 = select_kshots_proportional(**_prop_kwargs(df, embs, index))
    r2 = select_kshots_proportional(**_prop_kwargs(df, embs, index))
    assert r1[3] == r2[3]  # same global indices
    assert r1[1] == r2[1]  # same distances
