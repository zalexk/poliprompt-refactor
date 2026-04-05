import os
import csv
import json
from typing import List, Dict
import pandas as pd
from datetime import datetime as dt
from pathlib import Path
import base64
from functools import lru_cache
import re
import yaml


def load_yaml_config(path: Path) -> dict:
    """加载 YAML 配置文件，返回字典。文件必须存在。"""
    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_json_config(path: Path) -> dict:
    """加载 JSON 配置文件，返回字典。文件必须存在。"""
    if not path.exists():
        raise FileNotFoundError(f"JSON config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def load_and_validate_data(
        file_path : Path | str,
        text_col : str,
        answer_col : str=None,
        image_col : str=None
)-> pd.DataFrame:
    """
 
    """
    file_path = Path(os.path.expanduser(file_path))

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
        raise ValueError(f"Error reading file '{file_path}':{e}")
    
    missing_cols = []
    
    if (text_col not in df.columns) and (image_col not in df.columns):
        raise ValueError("The dataset must contain at least one valid feature column ('text' or 'image').")
    
    if answer_col and answer_col not in df.columns:
        missing_cols.append(answer_col)
    
    if missing_cols:
        raise ValueError(f"Specified columns {missing_cols} not found in the file.")
    return df


def read_docs_from_dataframe(df: pd.DataFrame, column_name: str="text"):
    """
    Reads texts from a specified column in a CSV file.

    Parameters:
        - file_path (str): The path to the CSV file.
        - column_name (str): The name of the column containing the texts.

    Returns:
        - list: A list of docs from the specified column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got {type(df)} instead.")

    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Extract the texts as a list
    docs = df[column_name].tolist()

    return docs


def read_multimodal_docs_from_dataframe(df,text_col,img_col,img_root):
    """
    Extracts text and constructs absolute image paths from a DataFrame.

    Args:
        df (pd.DataFrame): Source data.
        text_col (str): Column name for text content.
        img_col (str): Column name for image filenames/relative paths.
        img_root (str | Path): Root directory where images are stored.

    Returns:
        list[dict]: A list of dictionaries, e.g., [{'text': '...', 'image_path': '...'}]
    
    Raises:
        ValueError: If specified columns are missing from the DataFrame.
    """
    # Check if the column exists in the DataFrame
    if text_col not in df.columns or img_col not in df.columns:
        raise ValueError(f"Columns '{text_col}'or'{img_col}' not found in the DataFrame.")

    multimodal_questions = []
    # for循环中前面的_代表丢弃这个索引的意思，df.iterrows()会返回索引和这一行的内容
    for _,row in df.iterrows():
        text_val = str(row[text_col]) if not pd.isna(row.get(text_col)) else""
        img_val = row[img_col] if not pd.isna(row.get(img_col)) else None
        if img_val and img_root:
            full_path = str(Path(img_root)/img_val)
        else:
            full_path = str(img_val) if img_val else None
        
        
        multimodal_questions.append(
            {"text" : text_val,
             "image_path" : full_path})
    
    return multimodal_questions


def load_config(configs_path: str | Path, name: str) -> Dict:
    """
    Load the configuration from a specific json file.

    Parameters:
    - configs_path (str): Path to the configuration JSON file.
    - name (str): Name of the config to use.

    Returns:
    - dict: The configuration for the specified LLM.
    """
    try:
        with open(configs_path, "r") as file:
            configs = json.load(file)
            config = configs.get(name)
            if config is None:
                raise ValueError(f"No configuration found for: {name}")
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {configs_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from file: {configs_path}")



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def parse_llm_response_generic(content: str, options: list = None) -> dict:
    """
    产品化解析器：
    1. 优先尝试标准的 JSON 解析。
    2. 如果失败，通过正则提取 'label' 和 'reason' 字段后的值。
    """
    result = {"label": "unknown", "reason": content}
    
    # --- 策略 A: 尝试 JSON 提取 ---
    try:
        # 寻找最近的 JSON 块
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

    # --- 策略 B: 正则匹配 (支持跨行) ---
    # 匹配 label: value (value 可能带引号)
    label_pattern = r'["\']?label["\']?\s*:\s*["\']?([^"\'\s,}]+)["\']?'
    # 匹配 reason: 后面的内容直到下一个字段或结尾（跨行）
    reason_pattern = r'["\']?reason["\']?\s*:\s*["\']?(.*?)(?=\s*[,}]\s*["\']?\w+["\']?\s*:|$)'
    
    label_match = re.search(label_pattern, content, re.IGNORECASE | re.DOTALL)
    if label_match:
        result["label"] = label_match.group(1).strip()
    
    reason_match = re.search(reason_pattern, content, re.IGNORECASE | re.DOTALL)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()
    
    # 如果用户提供了 options，进行一次对齐
    if options and result["label"] not in [str(o) for o in options]:
        for opt in options:
            # 检查 content 中是否有明确的选项标志（如 <opt> 或 "opt"）
            if f"<{opt}>" in content or f'"{opt}"' in content or f"'{opt}'" in content:
                result["label"] = str(opt)
                break
                
    return result


@lru_cache(maxsize=500)  # 缓存最近 500 张图片
def get_base64_image(image_path: str) -> str:
    """读取图片并返回 base64 编码字符串，使用 LRU 缓存"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

