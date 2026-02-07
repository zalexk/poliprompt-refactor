import os
import csv
import json
from typing import List, Dict
import pandas as pd
from datetime import datetime as dt
from pathlib import Path
import base64

def ensure_workstation_directories(work_station: Path):
    """
    Ensures that the WORK_STATION directory and its subdirectories (infiles/ and outfiles/) exist.
    """
    if not work_station.parent.exists():
        raise FileNotFoundError(f"No valid parent directories exist for the path '{work_station}'.")
    # parent是path的一个属性，返回父目录；raise代表主动报错
    # Paths to infiles/ and outfiles/ directories
    infiles_dir = work_station / "infiles"
    configs_dir = infiles_dir / "configs"
    prompts_dir = infiles_dir / "prompts"
    outfiles_dir = work_station / "outfiles"
    logs_dir = outfiles_dir / "logs"
    images_out_dir = outfiles_dir / "images"
    # 这里是存了一个地址字符串，但还没有这几个文件夹

    # Create directories if they do not exist
    # 开始创建这几个文件夹，parents代表如果要创建C:/A/B/C,但现在连C:/A都没有，就先递归的先建前面的
    # exist_ok代表如果文件已经存在就跳过，如果是false第二次运行时存在就会报错
    if not work_station.exists():
        work_station.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {work_station}")

    if not infiles_dir.exists():
        infiles_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {infiles_dir}")

    if not configs_dir.exists():
        configs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {configs_dir}")

    if not prompts_dir.exists():
        prompts_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {prompts_dir}")

    if not outfiles_dir.exists():
        outfiles_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {outfiles_dir}")

    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {logs_dir}")

    if not images_out_dir.exists():
        images_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {images_out_dir}")

# def load_data_file(file_path: Path) -> pd.DataFrame:
#     """
#     Unified Interface: Automatically selects CSV or JSONL reading based on file extension.
#     """
#     if not file_path.exists():
#         raise FileNotFoundError(f"Data file not found: {file_path}")
        
#     if file_path.suffix == '.jsonl':
#         return pd.read_json(file_path, lines=True)
#     elif file_path.suffix == '.csv':
#         return pd.read_csv(file_path, index_col=False)
#     else:
#         raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .csv or .jsonl")


# def validate_csv_file(file_path: str, feature_col: str, answer_col: str):
#     # Check if the file exists
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File '{file_path}' does not exist.")

#     # Check if the file is a CSV
#     if not file_path.endswith(".csv"):
#         raise ValueError(f"File '{file_path}' is not a CSV file.")

#     # Load the CSV file using pandas
#     try:
#         df = pd.read_csv(file_path)
#     except Exception as e:
#         raise ValueError(f"Failed to read the CSV file '{file_path}': {e}")

#     # Check if feature_col exists in the DataFrame
#     if feature_col not in df.columns:
#         raise ValueError(f"Feature column '{feature_col}' does not exist in the CSV file.")

#     # Check if answer_col exists in the DataFrame
#     if answer_col not in df.columns:
#         raise ValueError(f"Answer column '{answer_col}' does not exist in the CSV file.")

def load_and_validate_data(
        file_path : Path | str,
        feature_col : str,
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
    
    if (feature_col not in df.columns) and (image_col not in df.columns):
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


def track_computation_time(elapsed_time: float) -> None:
    """
    Tracks the computation time and prints it in hours, minutes, and seconds.

    Parameters:
        - elapsed_time (float): The elapsed time using time.time().
    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds
    print(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")


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


def get_prompt(prompt_file: str | Path) -> str:
    if isinstance(prompt_file, str):
        prompt_file = Path(prompt_file)
    return prompt_file.read_text()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

