"""Unit tests for poliprompt.llm_contribs."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from poliprompt.llm_contribs import create_llm, get_universal_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8


def _fake_embeddings(n):
    """Return n fake embedding vectors of dimension DIM."""
    return [[float(j) for j in range(DIM)] for _ in range(n)]


# ---------------------------------------------------------------------------
# create_llm — vendor detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name,url_fragment", [
    ("gpt-4o",         "openai.com"),
    ("gpt-4o-mini",    "openai.com"),
    ("o3-mini",        "openai.com"),
    ("o4-mini",        "openai.com"),
    ("gemini-pro",     "generativelanguage.googleapis.com"),
    ("qwen-vl-max",    "dashscope.aliyuncs.com"),
    ("qwen-max",       "dashscope.aliyuncs.com"),
])
def test_create_llm_vendor_url(model_name, url_fragment, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY",    "sk-test")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-test")
    monkeypatch.setenv("GOOGLE_API_KEY",    "ggl-test")

    with patch("poliprompt.llm_contribs.ChatOpenAI") as MockChat:
        MockChat.return_value = MagicMock()
        create_llm(model_name)
        _, kwargs = MockChat.call_args
        assert url_fragment in kwargs["base_url"]


# ---------------------------------------------------------------------------
# create_llm — Anthropic (Claude)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", [
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-5",
    "claude-haiku-4-5",
])
def test_create_llm_claude_uses_chat_anthropic(model_name, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
    with patch("poliprompt.llm_contribs.ChatAnthropic") as MockAnthropic:
        MockAnthropic.return_value = MagicMock()
        create_llm(model_name)
        MockAnthropic.assert_called_once()
        _, kwargs = MockAnthropic.call_args
        assert kwargs["model"] == model_name
        assert kwargs["api_key"] == "ant-test"


def test_create_llm_claude_dict_inline_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-ant-key")
    model_cfg = {"model": "claude-3-5-sonnet-20241022", "api_key": "inline-ant-key"}
    with patch("poliprompt.llm_contribs.ChatAnthropic") as MockAnthropic:
        MockAnthropic.return_value = MagicMock()
        create_llm(model_cfg)
        _, kwargs = MockAnthropic.call_args
        assert kwargs["api_key"] == "inline-ant-key"


def test_create_llm_claude_dict_overrides_temp_and_tokens(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
    model_cfg = {"model": "claude-opus-4-5", "temperature": 0.7, "max_tokens": 2048}
    with patch("poliprompt.llm_contribs.ChatAnthropic") as MockAnthropic:
        MockAnthropic.return_value = MagicMock()
        create_llm(model_cfg)
        _, kwargs = MockAnthropic.call_args
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 2048


def test_create_llm_claude_missing_package_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
    with patch("poliprompt.llm_contribs.ChatAnthropic", None):
        with pytest.raises(ImportError, match="langchain-anthropic"):
            create_llm("claude-3-5-sonnet-20241022")


def test_create_llm_unknown_vendor_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("poliprompt.llm_contribs.ChatOpenAI") as MockChat:
        MockChat.return_value = MagicMock()
        create_llm("some-unknown-model")
        _, kwargs = MockChat.call_args
        assert "openai.com" in kwargs["base_url"]


def test_create_llm_string_input_uses_config_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    default_cfg = {"temperature": 0.2, "max_tokens": 512}
    with patch("poliprompt.llm_contribs.ChatOpenAI") as MockChat:
        MockChat.return_value = MagicMock()
        create_llm("gpt-4o-mini", default_config=default_cfg)
        _, kwargs = MockChat.call_args
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512


def test_create_llm_dict_input_overrides_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model_cfg = {"model": "gpt-4o-mini", "temperature": 0.9, "max_tokens": 256}
    with patch("poliprompt.llm_contribs.ChatOpenAI") as MockChat:
        MockChat.return_value = MagicMock()
        create_llm(model_cfg)
        _, kwargs = MockChat.call_args
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 256


def test_create_llm_dict_custom_api_key_used(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    model_cfg = {"model": "gpt-4o", "api_key": "inline-key"}
    with patch("poliprompt.llm_contribs.ChatOpenAI") as MockChat:
        MockChat.return_value = MagicMock()
        create_llm(model_cfg)
        _, kwargs = MockChat.call_args
        assert kwargs["api_key"] == "inline-key"


# ---------------------------------------------------------------------------
# get_universal_embeddings — OpenAI path
# ---------------------------------------------------------------------------

def test_embeddings_openai_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    docs = [{"text": f"doc {i}"} for i in range(6)]

    with patch("poliprompt.llm_contribs._fetch_openai_batch", return_value=_fake_embeddings(3)):
        result = get_universal_embeddings(docs, "text-embedding-3-small", max_workers=1, batch_size=3)

    assert result.shape == (6, DIM)
    assert result.dtype == np.float32


def test_embeddings_openai_failed_batch_zero_filled(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    docs = [{"text": f"doc {i}"} for i in range(4)]

    call_count = {"n": 0}

    def _side_effect(client, texts, model):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient API error")
        return _fake_embeddings(len(texts))

    with patch("poliprompt.llm_contribs._fetch_openai_batch", side_effect=_side_effect):
        result = get_universal_embeddings(docs, "text-embedding-3-small", max_workers=1, batch_size=2)

    # First batch (indices 0-1) failed → zero-filled
    assert np.all(result[0] == 0.0)
    assert np.all(result[1] == 0.0)
    # Second batch (indices 2-3) succeeded → non-zero
    assert not np.all(result[2] == 0.0)


def test_embeddings_all_batches_fail_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    docs = [{"text": "only doc"}]

    with patch("poliprompt.llm_contribs._fetch_openai_batch", side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError, match="All embedding batches failed"):
            get_universal_embeddings(docs, "text-embedding-3-small", max_workers=1, batch_size=1)


# ---------------------------------------------------------------------------
# get_universal_embeddings — Qwen path
# ---------------------------------------------------------------------------

def test_embeddings_qwen_dispatched_to_qwen_backend(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-test")
    docs = [{"text": "hello", "image_path": None}]

    with patch("poliprompt.llm_contribs._fetch_qwen_batch", return_value=_fake_embeddings(1)) as mock_qwen, \
         patch("poliprompt.llm_contribs._fetch_openai_batch") as mock_openai:
        get_universal_embeddings(docs, "qwen-vl-max", max_workers=1, batch_size=1)
        mock_qwen.assert_called_once()
        mock_openai.assert_not_called()
