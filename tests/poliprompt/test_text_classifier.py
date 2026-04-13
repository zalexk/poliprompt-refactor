"""Unit tests for TextClassifier."""

import numpy as np
import pandas as pd
import faiss
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.messages import SystemMessage, HumanMessage

from poliprompt.text_classifier import TextClassifier


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def clf(text_project_dir, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    with patch("poliprompt.base_classifier.create_llm", return_value=MagicMock()):
        return TextClassifier(
            config_path=text_project_dir / "config.yaml",
            prompt_path=text_project_dir / "prompt.txt",
        )


def _attach_rag(clf, n=8, dim=16):
    """Attach a minimal in-memory FAISS index and exemplar pool to clf."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, dim)).astype("float32")
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(dim)
    idx.add(emb)

    clf.faiss_index = idx
    clf.indices = list(range(n))
    clf.pool_embeddings_cache = emb
    clf.rules_dict = {}
    clf.enhanced_rules = ""
    clf.df = pd.DataFrame({
        "text":  [f"doc {i}" for i in range(n)],
        "label": [str(i % 2) for i in range(n)],
    })
    return clf


# ---------------------------------------------------------------------------
# _convert_df_to_docs
# ---------------------------------------------------------------------------

def test_convert_df_to_docs_length(clf):
    df = pd.DataFrame({"text": ["a", "b", "c"], "label": ["0", "1", "0"]})
    docs = clf._convert_df_to_docs(df)
    assert len(docs) == 3


def test_convert_df_to_docs_text_values(clf):
    df = pd.DataFrame({"text": ["hello", "world"], "label": ["0", "1"]})
    docs = clf._convert_df_to_docs(df)
    assert docs[0]["text"] == "hello"
    assert docs[1]["text"] == "world"


def test_convert_df_to_docs_no_image_path_key(clf):
    """Text classifier docs must not carry an image_path key."""
    df = pd.DataFrame({"text": ["test"], "label": ["0"]})
    docs = clf._convert_df_to_docs(df)
    assert "image_path" not in docs[0]


# ---------------------------------------------------------------------------
# _prepare_agent_messages
# ---------------------------------------------------------------------------

def test_prepare_messages_returns_two_messages(clf):
    clf = _attach_rag(clf)
    messages, _, _, _ = clf._prepare_agent_messages(idx=0, item={"text": "doc 0"})
    assert len(messages) == 2


def test_prepare_messages_first_is_system(clf):
    clf = _attach_rag(clf)
    messages, _, _, _ = clf._prepare_agent_messages(idx=0, item={"text": "doc 0"})
    assert isinstance(messages[0], SystemMessage)


def test_prepare_messages_second_is_human(clf):
    clf = _attach_rag(clf)
    messages, _, _, _ = clf._prepare_agent_messages(idx=0, item={"text": "doc 0"})
    assert isinstance(messages[1], HumanMessage)


def test_prepare_messages_user_content_has_reference_examples(clf):
    clf = _attach_rag(clf)
    messages, _, _, _ = clf._prepare_agent_messages(idx=0, item={"text": "doc 0"})
    assert "REFERENCE EXAMPLES" in messages[1].content


def test_prepare_messages_user_content_has_current_task(clf):
    clf = _attach_rag(clf)
    messages, _, _, _ = clf._prepare_agent_messages(idx=0, item={"text": "doc 0"})
    assert "CURRENT TASK" in messages[1].content


def test_prepare_messages_rules_injected_when_present(clf):
    clf = _attach_rag(clf)
    clf.enhanced_rules = "Rule A: label 0 means X."
    messages, _, _, _ = clf._prepare_agent_messages(idx=0, item={"text": "doc 0"})
    assert "GLOBAL REASONING RULES" in messages[0].content
    assert "Rule A" in messages[0].content


def test_prepare_messages_returns_rag_metadata(clf):
    clf = _attach_rag(clf)
    _, dists, labels, indices = clf._prepare_agent_messages(idx=0, item={"text": "doc 0"})
    assert len(dists) == clf.k_shots
    assert len(labels) == clf.k_shots
    assert len(indices) == clf.k_shots


# ---------------------------------------------------------------------------
# _display_item_for_hitl
# ---------------------------------------------------------------------------

def test_display_item_for_hitl_prints_text(clf, capsys):
    clf._display_item_for_hitl({"text": "review this content please"})
    output = capsys.readouterr().out
    assert "review this content please" in output
