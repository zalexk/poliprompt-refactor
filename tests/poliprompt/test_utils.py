"""Unit tests for poliprompt.utils."""

import json
import yaml
import pandas as pd
import pytest
from pathlib import Path
from hypothesis import given, settings, strategies as st

from poliprompt.utils import (
    load_yaml_config,
    load_json_config,
    load_and_validate_data,
    read_docs_from_dataframe,
    read_multimodal_docs_from_dataframe,
    load_config,
    parse_llm_response_generic,
    get_base64_image,
)


# ---------------------------------------------------------------------------
# load_yaml_config
# ---------------------------------------------------------------------------

def test_load_yaml_config_returns_dict(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("key: value\nnested:\n  a: 1\n")
    assert load_yaml_config(p) == {"key": "value", "nested": {"a": 1}}


def test_load_yaml_config_empty_file_returns_empty_dict(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_yaml_config(p) == {}


def test_load_yaml_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# load_json_config
# ---------------------------------------------------------------------------

def test_load_json_config_returns_dict(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"model": "gpt-4o", "temperature": 0.0}')
    assert load_json_config(p) == {"model": "gpt-4o", "temperature": 0.0}


def test_load_json_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_json_config(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# load_and_validate_data
# ---------------------------------------------------------------------------

def test_load_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("text,label\nhello,0\nworld,1\n")
    df = load_and_validate_data(p, text_col="text", answer_col="label")
    assert list(df["text"]) == ["hello", "world"]
    assert list(df["label"].astype(str)) == ["0", "1"]


def test_load_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"text":"a","label":"0"}\n{"text":"b","label":"1"}\n')
    df = load_and_validate_data(p, text_col="text", answer_col="label")
    assert len(df) == 2


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_validate_data(tmp_path / "none.csv", text_col="text")


def test_load_unsupported_format_raises(tmp_path):
    p = tmp_path / "data.xml"
    p.write_text("<data/>")
    with pytest.raises(ValueError, match="Unsupported"):
        load_and_validate_data(p, text_col="text")


def test_load_missing_answer_col_raises(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("text\nhello\n")
    with pytest.raises(ValueError):
        load_and_validate_data(p, text_col="text", answer_col="ghost_col")


def test_load_missing_text_col_raises(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Required columns"):
        load_and_validate_data(p, text_col="text")


def test_load_missing_image_col_raises(tmp_path):
    """Even when text_col is present, a missing image_col must raise."""
    p = tmp_path / "data.csv"
    p.write_text("text\nhello\n")
    with pytest.raises(ValueError, match="Required columns"):
        load_and_validate_data(p, text_col="text", image_col="img")


# ---------------------------------------------------------------------------
# read_docs_from_dataframe
# ---------------------------------------------------------------------------

def test_read_docs_returns_list():
    df = pd.DataFrame({"text": ["alpha", "beta", "gamma"]})
    assert read_docs_from_dataframe(df, "text") == ["alpha", "beta", "gamma"]


def test_read_docs_missing_column_raises():
    df = pd.DataFrame({"text": ["a"]})
    with pytest.raises(ValueError, match="not found"):
        read_docs_from_dataframe(df, "nonexistent")


def test_read_docs_wrong_type_raises():
    with pytest.raises(TypeError):
        read_docs_from_dataframe(["not", "a", "dataframe"], "text")


# ---------------------------------------------------------------------------
# read_multimodal_docs_from_dataframe
# ---------------------------------------------------------------------------

def test_read_multimodal_docs_structure(tmp_path):
    df = pd.DataFrame({"text": ["hello", "world"], "img": ["a.jpg", "b.jpg"]})
    result = read_multimodal_docs_from_dataframe(df, "text", "img", tmp_path)
    assert len(result) == 2
    assert result[0]["text"] == "hello"
    assert result[0]["image_path"] == str(tmp_path / "a.jpg")


def test_read_multimodal_docs_no_img_root():
    df = pd.DataFrame({"text": ["hi"], "img": ["pic.jpg"]})
    result = read_multimodal_docs_from_dataframe(df, "text", "img", img_root=None)
    assert result[0]["image_path"] == "pic.jpg"


def test_read_multimodal_docs_missing_columns_raises():
    df = pd.DataFrame({"text": ["a"]})
    with pytest.raises(ValueError):
        read_multimodal_docs_from_dataframe(df, "text", "img", img_root=None)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_returns_correct_entry(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"gpt-4o": {"temperature": 0.0}, "gpt-4o-mini": {"temperature": 0.2}}')
    assert load_config(p, "gpt-4o") == {"temperature": 0.0}


def test_load_config_missing_key_raises(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"gpt-4o": {}}')
    with pytest.raises(ValueError, match="No configuration"):
        load_config(p, "unknown-model")


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json", "any")


# ---------------------------------------------------------------------------
# parse_llm_response_generic
# ---------------------------------------------------------------------------

def test_parse_clean_json():
    content = '{"label": "1", "reason": "Hateful content detected."}'
    result = parse_llm_response_generic(content, options=["0", "1"])
    assert result["label"] == "1"
    assert "Hateful" in result["reason"]


def test_parse_json_embedded_in_prose():
    content = 'Based on analysis: {"label": "0", "reason": "Benign."} End of response.'
    result = parse_llm_response_generic(content, options=["0", "1"])
    assert result["label"] == "0"


def test_parse_regex_fallback():
    content = "label: 1\nreason: Clearly problematic."
    result = parse_llm_response_generic(content, options=["0", "1"])
    assert result["label"] == "1"


def test_parse_options_alignment_angle_bracket():
    content = "My answer is <positive> based on the tone."
    result = parse_llm_response_generic(content, options=["positive", "negative"])
    assert result["label"] == "positive"


def test_parse_options_alignment_quoted():
    content = 'The label should be "negative" here.'
    result = parse_llm_response_generic(content, options=["positive", "negative"])
    assert result["label"] == "negative"


def test_parse_unrecognised_returns_unknown():
    result = parse_llm_response_generic("No structured output.", options=["0", "1"])
    assert result["label"] == "unknown"


def test_parse_no_options_does_not_raise():
    result = parse_llm_response_generic('{"label": "xyz", "reason": "test"}')
    assert result["label"] == "xyz"


@given(st.text())
@settings(max_examples=200)
def test_parse_never_raises_on_arbitrary_input(content):
    """parse_llm_response_generic must tolerate any string without raising."""
    result = parse_llm_response_generic(content, options=["0", "1"])
    assert "label" in result
    assert "reason" in result


# ---------------------------------------------------------------------------
# get_base64_image
# ---------------------------------------------------------------------------

def test_get_base64_image_returns_string(tmp_path):
    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    result = get_base64_image(str(img))
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_base64_image_is_cached(tmp_path):
    img = tmp_path / "cached.jpg"
    img.write_bytes(b"\x00" * 32)
    r1 = get_base64_image(str(img))
    r2 = get_base64_image(str(img))
    assert r1 is r2  # same object from LRU cache
