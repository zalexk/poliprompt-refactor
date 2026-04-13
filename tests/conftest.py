"""
Shared fixtures for the PoliPrompt test suite.

Fixtures here are available to all tests without explicit import.
File-specific fixtures live in the individual test_*.py files.
"""

import json
import yaml
import numpy as np
import pandas as pd
import faiss
import pytest


# ---------------------------------------------------------------------------
# Embedding / FAISS primitives
# ---------------------------------------------------------------------------

DIM = 16  # embedding dimension used across tests


@pytest.fixture
def rng():
    """Seeded NumPy default_rng for reproducible random data."""
    return np.random.default_rng(42)


@pytest.fixture
def raw_embeddings(rng):
    """10 × DIM float32 embeddings, NOT normalised."""
    return rng.standard_normal((10, DIM)).astype("float32")


@pytest.fixture
def embeddings(raw_embeddings):
    """10 × DIM float32 embeddings, L2-normalised (cosine-ready)."""
    emb = raw_embeddings.copy()
    faiss.normalize_L2(emb)
    return emb


@pytest.fixture
def faiss_index(embeddings):
    """IndexFlatIP populated with the shared embeddings fixture."""
    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# DataFrame primitives
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """10-row text-only DataFrame with binary labels."""
    return pd.DataFrame({
        "text":  [f"document {i}" for i in range(10)],
        "label": [str(i % 2) for i in range(10)],
    })


@pytest.fixture
def sample_df_multimodal():
    """10-row multimodal DataFrame with image filenames."""
    return pd.DataFrame({
        "text":  [f"document {i}" for i in range(10)],
        "img":   [f"img/{i:03d}.jpg" for i in range(10)],
        "label": [str(i % 2) for i in range(10)],
    })


# ---------------------------------------------------------------------------
# Project directory (used by classifier-level tests)
# ---------------------------------------------------------------------------

def _write_project(tmp_path, *, modality="text", data_filename="data.csv"):
    """
    Write the minimal files a classifier constructor needs to start up:
      .env, config.yaml, prompt.txt, data file, outfiles/
    Returns the project directory Path.
    """
    config = {
        "project": {
            "name":        "TestProject",
            "modality":    modality,
            "work_station": str(tmp_path),
            "data_path":   data_filename,
            "outfiles_dir": "outfiles",
            **( {"image_dir": "images"} if modality == "multimodal" else {} ),
        },
        "column_mapping": {
            "text_col":   "text",
            "image_col":  "img" if modality == "multimodal" else "None",
            "answer_col": "label",
        },
        "user_settings": {
            "options":      ["0", "1"],
            "k_shots":      2,
            "lambda_param": 0.5,
            "testing":      False,
        },
        "models": {
            "embedding_llm":  "qwen-vl-max" if modality == "multimodal" else "text-embedding-3-small",
            "primary_llm":    "gpt-4o-mini",
            "secondary_llm":  "qwen-vl-max",
            "expert_llm":     "gpt-4o",
        },
        "retrieval": {"n_exemplars_pool": 6},
        "parallel":  {"embedding_workers": 1, "inference_workers": 1, "embedding_batch_size": 2},
        "observability": {"enabled": False},
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")
    (tmp_path / "prompt.txt").write_text("Classify as 0 or 1. Reply with JSON.", encoding="utf-8")
    (tmp_path / "outfiles").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture
def text_project_dir(tmp_path):
    """Project directory pre-populated for a text-only classifier."""
    _write_project(tmp_path, modality="text")
    df = pd.DataFrame({
        "text":  [f"doc {i}" for i in range(8)],
        "label": [str(i % 2) for i in range(8)],
    })
    df.to_csv(tmp_path / "data.csv", index=False)
    return tmp_path


@pytest.fixture
def multimodal_project_dir(tmp_path):
    """Project directory pre-populated for a multimodal classifier."""
    _write_project(tmp_path, modality="multimodal", data_filename="data.jsonl")
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # Minimal JPEG stub for image path resolution tests
    (img_dir / "001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)

    rows = [{"text": f"doc {i}", "img": "001.jpg", "label": str(i % 2)} for i in range(4)]
    with open(tmp_path / "data.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return tmp_path
