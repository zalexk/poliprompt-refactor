import json
from typing import List, Dict
import pandas as pd
from pathlib import Path
import base64
from functools import lru_cache
import re
import yaml


def load_yaml_config(path: Path) -> dict:
    """Load a YAML config file and return it as a dict. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_json_config(path: Path) -> dict:
    """Load a JSON config file and return it as a dict. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"JSON config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_and_validate_data(
        file_path: Path | str,
        text_col: str,
        answer_col: str = None,
        image_col: str = None
) -> pd.DataFrame:
    """
    Load a dataset from CSV, JSON, or JSONL and validate that required columns are present.

    Raises FileNotFoundError if the file does not exist, ValueError on format or column errors.
    """
    file_path = Path(file_path).expanduser()

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    suffix = file_path.suffix.lower()
    try:
        if suffix == '.csv':
            df = pd.read_csv(file_path)
        elif suffix == '.json':
            df = pd.read_json(file_path)
        elif suffix == '.jsonl':
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    except Exception as e:
        raise ValueError(f"Error reading file '{file_path}': {e}") from e

    missing_cols = []
    if text_col not in df.columns:
        missing_cols.append(text_col)
    if image_col and image_col not in df.columns:
        missing_cols.append(image_col)
    if answer_col and answer_col not in df.columns:
        missing_cols.append(answer_col)

    if missing_cols:
        raise ValueError(f"Required columns not found in the data: {missing_cols}")

    return df


def read_docs_from_dataframe(df: pd.DataFrame, column_name: str = "text") -> List:
    """
    Extract a list of values from a specified DataFrame column.

    Parameters:
        - df: Source DataFrame.
        - column_name: Column to extract.

    Returns:
        - list: Values from the specified column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got {type(df)} instead.")

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    return df[column_name].tolist()


def read_multimodal_docs_from_dataframe(df, text_col, img_col, img_root):
    """
    Extract text and construct absolute image paths from a DataFrame.

    Args:
        df (pd.DataFrame): Source data.
        text_col (str): Column name for text content.
        img_col (str): Column name for image filenames or relative paths.
        img_root (str | Path): Root directory where images are stored.

    Returns:
        list[dict]: A list of dicts with 'text' and 'image_path' keys.

    Raises:
        ValueError: If specified columns are missing from the DataFrame.
    """
    if text_col not in df.columns or img_col not in df.columns:
        raise ValueError(f"Columns '{text_col}' or '{img_col}' not found in the DataFrame.")

    multimodal_questions = []
    # The leading _ discards the row index returned by df.iterrows()
    for _, row in df.iterrows():
        text_val = str(row[text_col]) if not pd.isna(row.get(text_col)) else ""
        img_val = row[img_col] if not pd.isna(row.get(img_col)) else None
        if img_val and img_root:
            full_path = str(Path(img_root) / img_val)
        else:
            full_path = str(img_val) if img_val else None

        multimodal_questions.append(
            {"text": text_val,
             "image_path": full_path})

    return multimodal_questions


def load_config(configs_path: str | Path, name: str) -> Dict:
    """
    Load a named entry from a JSON config file.

    Parameters:
    - configs_path (str | Path): Path to the configuration JSON file.
    - name (str): Name of the config entry to retrieve.

    Returns:
    - dict: The configuration for the specified name.
    """
    configs = load_json_config(Path(configs_path))
    config = configs.get(name)
    if config is None:
        raise ValueError(f"No configuration found for: {name}")
    return config


def parse_llm_response_generic(content: str, options: list = None) -> dict:
    """
    Parse a raw LLM response string into a structured result dict.

    Strategy A: Try standard JSON extraction.
    Strategy B: Fall back to regex matching for 'label' and 'reason' fields.
    """
    result = {"label": "unknown", "reason": content}

    # Strategy A: JSON extraction
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "label" in data:
                result["label"] = str(data["label"])
            if "reason" in data:
                result["reason"] = str(data["reason"])
            return result
    except Exception:
        pass

    # Strategy B: Regex matching (supports multi-line responses)
    label_pattern = r'["\']?label["\']?\s*:\s*["\']?([^"\'\s,}]+)["\']?'
    reason_pattern = r'["\']?reason["\']?\s*:\s*["\']?(.*?)(?=\s*[,}]\s*["\']?\w+["\']?\s*:|$)'

    label_match = re.search(label_pattern, content, re.IGNORECASE | re.DOTALL)
    if label_match:
        result["label"] = label_match.group(1).strip()

    reason_match = re.search(reason_pattern, content, re.IGNORECASE | re.DOTALL)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()

    # Align against provided options if the parsed label is not recognized
    if options and result["label"] not in [str(o) for o in options]:
        for opt in options:
            if f"<{opt}>" in content or f'"{opt}"' in content or f"'{opt}'" in content:
                result["label"] = str(opt)
                break

    return result


@lru_cache(maxsize=500)
def get_base64_image(image_path: str) -> str:
    """Read an image file and return its base64-encoded string. Results are LRU-cached (up to 500 images)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
