"""Unit tests for MultiModalClassifier."""

import numpy as np
import pandas as pd
import faiss
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.messages import SystemMessage, HumanMessage

from poliprompt.multimodal_classifier import MultiModalClassifier


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def clf(multimodal_project_dir, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    with patch("poliprompt.base_classifier.create_llm", return_value=MagicMock()):
        instance = MultiModalClassifier(
            config_path=multimodal_project_dir / "config.yaml",
            prompt_path=multimodal_project_dir / "prompt.txt",
        )
    return instance


def _attach_rag(clf, n=4, dim=16):
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
        "img":   ["001.jpg"] * n,
        "label": [str(i % 2) for i in range(n)],
    })
    return clf


# ---------------------------------------------------------------------------
# _convert_df_to_docs
# ---------------------------------------------------------------------------

def test_convert_df_to_docs_has_text(clf):
    df = pd.DataFrame({"text": ["hello"], "img": ["001.jpg"], "label": ["0"]})
    docs = clf._convert_df_to_docs(df)
    assert docs[0]["text"] == "hello"


def test_convert_df_to_docs_has_image_path(clf, multimodal_project_dir):
    df = pd.DataFrame({"text": ["hello"], "img": ["001.jpg"], "label": ["0"]})
    docs = clf._convert_df_to_docs(df)
    assert "image_path" in docs[0]
    assert "001.jpg" in docs[0]["image_path"]


def test_convert_df_to_docs_length(clf):
    df = pd.DataFrame({"text": ["a", "b"], "img": ["x.jpg", "y.jpg"], "label": ["0", "1"]})
    docs = clf._convert_df_to_docs(df)
    assert len(docs) == 2


# ---------------------------------------------------------------------------
# _abs_image_path
# ---------------------------------------------------------------------------

def test_abs_image_path_resolves_relative(clf, multimodal_project_dir):
    result = clf._abs_image_path("001.jpg")
    assert result is not None
    assert Path(result).is_absolute()
    assert Path(result).exists()


def test_abs_image_path_absolute_existing(clf, multimodal_project_dir):
    abs_path = str(multimodal_project_dir / "images" / "001.jpg")
    assert clf._abs_image_path(abs_path) == abs_path


def test_abs_image_path_missing_file_returns_none(clf):
    assert clf._abs_image_path("does_not_exist.jpg") is None


def test_abs_image_path_empty_string_returns_none(clf):
    assert clf._abs_image_path("") is None


def test_abs_image_path_none_returns_none(clf):
    assert clf._abs_image_path(None) is None


@pytest.mark.parametrize("sentinel", ["None", "nan"])
def test_abs_image_path_sentinel_strings_return_none(clf, sentinel):
    assert clf._abs_image_path(sentinel) is None


# ---------------------------------------------------------------------------
# _prepare_agent_messages
# ---------------------------------------------------------------------------

def test_prepare_messages_structure(clf, multimodal_project_dir):
    clf = _attach_rag(clf)
    item = {clf.text_col: "doc 0", "image_path": None}
    messages, dists, labels, indices = clf._prepare_agent_messages(idx=0, item=item)

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[1].content, list)  # multimodal payload is a list


def test_prepare_messages_payload_has_text_blocks(clf):
    clf = _attach_rag(clf)
    item = {clf.text_col: "doc 0", "image_path": None}
    messages, _, _, _ = clf._prepare_agent_messages(idx=0, item=item)
    types = [b["type"] for b in messages[1].content]
    assert "text" in types


def test_prepare_messages_rules_in_system_content(clf):
    clf = _attach_rag(clf)
    clf.enhanced_rules = "Critical rule: X implies 1."
    item = {clf.text_col: "doc 0", "image_path": None}
    messages, _, _, _ = clf._prepare_agent_messages(idx=0, item=item)
    assert "Critical rule" in messages[0].content


# ---------------------------------------------------------------------------
# _display_item_for_hitl
# ---------------------------------------------------------------------------

def test_display_item_prints_text(clf, capsys):
    clf._display_item_for_hitl({"text": "sample review content", "image_path": None})
    output = capsys.readouterr().out
    assert "sample review content" in output


def test_display_item_missing_image_prints_message(clf, capsys):
    clf._display_item_for_hitl({"text": "text", "image_path": "/nonexistent/path.jpg"})
    output = capsys.readouterr().out
    assert "not found" in output.lower() or "/nonexistent" in output
