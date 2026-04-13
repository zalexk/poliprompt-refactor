"""
Unit tests for BaseClassifier via TextClassifier (concrete stand-in).

Covers: config parsing, message formatting, item standardisation,
        RAG resource loading, and evaluate().
"""

import json
import numpy as np
import pandas as pd
import faiss
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.messages import SystemMessage, HumanMessage

from poliprompt.text_classifier import TextClassifier


# ---------------------------------------------------------------------------
# Fixture: a fully constructed TextClassifier with mocked LLMs
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


# ---------------------------------------------------------------------------
# _parse_config_to_self — validation
# ---------------------------------------------------------------------------

def _make_bad_config(tmp_path, overrides):
    """Write a minimal project dir whose config.yaml has the given overrides applied."""
    import yaml
    base = {
        "project": {"name": "T", "modality": "text", "work_station": str(tmp_path), "data_path": "d.csv", "outfiles_dir": "out"},
        "column_mapping": {"text_col": "text", "answer_col": "label"},
        "user_settings": {"options": ["0", "1"], "k_shots": 2, "lambda_param": 0.5},
        "models": {"primary_llm": "gpt-4o-mini", "secondary_llm": "gpt-4o-mini", "expert_llm": "gpt-4o-mini"},
        "retrieval": {}, "parallel": {}, "observability": {"enabled": False},
    }
    for key_path, val in overrides.items():
        section, key = key_path.split(".")
        base[section][key] = val
    (tmp_path / "out").mkdir(exist_ok=True)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump(base))
    (tmp_path / "prompt.txt").write_text("Classify.", encoding="utf-8")
    return tmp_path


def test_empty_options_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    proj = _make_bad_config(tmp_path, {"user_settings.options": []})
    with patch("poliprompt.base_classifier.create_llm", return_value=MagicMock()):
        with pytest.raises(ValueError, match="options"):
            TextClassifier(config_path=proj / "config.yaml", prompt_path=proj / "prompt.txt")


def test_zero_k_shots_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    proj = _make_bad_config(tmp_path, {"user_settings.k_shots": 0})
    with patch("poliprompt.base_classifier.create_llm", return_value=MagicMock()):
        with pytest.raises(ValueError, match="k_shots"):
            TextClassifier(config_path=proj / "config.yaml", prompt_path=proj / "prompt.txt")


# ---------------------------------------------------------------------------
# _parse_config_to_self
# ---------------------------------------------------------------------------

def test_project_name_parsed(clf, text_project_dir):
    assert clf.project_name == "TestProject"


def test_modality_parsed(clf):
    assert clf.modality == "text"


def test_options_are_strings(clf):
    assert clf.options == ["0", "1"]


def test_k_shots_parsed(clf):
    assert clf.k_shots == 2


def test_lambda_param_parsed(clf):
    assert clf.lambda_param == 0.5


def test_outfiles_dir_is_absolute_path(clf, text_project_dir):
    assert clf.outfiles_dir == text_project_dir / "outfiles"
    assert clf.outfiles_dir.is_absolute()


def test_image_col_none_string_normalised_to_python_none(clf):
    assert clf.image_col is None


# ---------------------------------------------------------------------------
# _format_messages_for_llm
# ---------------------------------------------------------------------------

def test_format_messages_system_role(clf):
    msgs = [SystemMessage(content="system text"), HumanMessage(content="user text")]
    raw = clf._format_messages_for_llm(msgs)
    assert raw[0]["role"] == "system"
    assert raw[0]["content"] == "system text"


def test_format_messages_user_role(clf):
    msgs = [SystemMessage(content="sys"), HumanMessage(content="human")]
    raw = clf._format_messages_for_llm(msgs)
    assert raw[1]["role"] == "user"
    assert raw[1]["content"] == "human"


def test_format_messages_preserves_order(clf):
    msgs = [SystemMessage(content="A"), HumanMessage(content="B")]
    raw = clf._format_messages_for_llm(msgs)
    assert [m["role"] for m in raw] == ["system", "user"]


# ---------------------------------------------------------------------------
# _get_item_standardized
# ---------------------------------------------------------------------------

def test_get_item_has_text_key(clf):
    clf.df = pd.DataFrame({"text": ["hello world"], "label": ["0"]})
    item = clf._get_item_standardized(0)
    assert item["text"] == "hello world"


def test_get_item_text_mode_image_path_is_none(clf):
    clf.df = pd.DataFrame({"text": ["test"], "label": ["0"]})
    item = clf._get_item_standardized(0)
    assert item["image_path"] is None


# ---------------------------------------------------------------------------
# _setup_rag_resources
# ---------------------------------------------------------------------------

def test_setup_rag_mandatory_missing_index_raises(clf):
    with pytest.raises(FileNotFoundError, match="create_few_shot_pool"):
        clf._setup_rag_resources(mandatory=True)


def test_setup_rag_non_mandatory_missing_files_silent(clf):
    clf._setup_rag_resources(mandatory=False)  # should not raise


def test_setup_rag_loads_existing_files(clf, text_project_dir):
    outfiles = text_project_dir / "outfiles"

    # Write exemplar indices
    (outfiles / "exemplar_indices.json").write_text(json.dumps([0, 1, 2, 3]))

    # Build a tiny FAISS index
    dim, n = 8, 8
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((n, dim)).astype("float32")
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(dim)
    idx.add(emb)
    faiss.write_index(idx, str(outfiles / "embeddings.index"))

    clf._setup_rag_resources(mandatory=True)

    assert clf.indices == [0, 1, 2, 3]
    assert clf.faiss_index is not None
    assert clf.pool_embeddings_cache.shape == (4, dim)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_evaluate_correct_accuracy(clf, text_project_dir):
    backup = text_project_dir / "outfiles" / "TestProject_backup.csv"
    pd.DataFrame({
        "text":            ["a", "b", "c", "d"],
        "label":           ["0", "1", "0", "1"],
        "predicted_label": ["0", "1", "1", "1"],   # 3 / 4 correct
    }).to_csv(backup, encoding="utf-8-sig", index=False)

    metrics = clf.evaluate()
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["n_evaluated"] == 4


def test_evaluate_excludes_rows_without_ground_truth(clf, text_project_dir):
    backup = text_project_dir / "outfiles" / "TestProject_backup.csv"
    pd.DataFrame({
        "text":            ["a", "b", "c"],
        "label":           ["0", "",  "1"],   # row 1 has no ground truth
        "predicted_label": ["0", "0", "1"],
    }).to_csv(backup, encoding="utf-8-sig", index=False)

    metrics = clf.evaluate()
    assert metrics["n_evaluated"] == 2
    assert metrics["n_no_gt"] == 1


def test_evaluate_excludes_failed_inference(clf, text_project_dir):
    backup = text_project_dir / "outfiles" / "TestProject_backup.csv"
    pd.DataFrame({
        "text":            ["a", "b", "c"],
        "label":           ["0", "1", "0"],
        "predicted_label": ["0", None, "0"],   # row 1 inference failed
    }).to_csv(backup, encoding="utf-8-sig", index=False)

    metrics = clf.evaluate()
    assert metrics["n_infer_fail"] == 1
    assert metrics["n_evaluated"] == 2


def test_evaluate_no_backup_raises(clf):
    with pytest.raises(FileNotFoundError, match="annotate"):
        clf.evaluate()


def test_evaluate_returns_confusion_matrix(clf, text_project_dir):
    backup = text_project_dir / "outfiles" / "TestProject_backup.csv"
    pd.DataFrame({
        "text":            ["a", "b", "c", "d"],
        "label":           ["0", "1", "0", "1"],
        "predicted_label": ["0", "1", "0", "1"],
    }).to_csv(backup, encoding="utf-8-sig", index=False)

    metrics = clf.evaluate()
    cm = metrics["confusion_matrix"]
    assert isinstance(cm, pd.DataFrame)
    assert cm.shape == (2, 2)
