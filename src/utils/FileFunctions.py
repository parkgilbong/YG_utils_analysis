import os
from pathlib import Path
from typing import List, Optional, NamedTuple
import pandas as pd
import numpy as np
import yaml
from contextlib import contextmanager

def set_working_directory(base_folder: str, *args: str) -> str:
    """
    Creates and sets the working directory using the base path and additional subfolders.

    Parameters:
    - base_folder (str): The base directory.
    - *args (str): Additional subfolders to create within the base directory.

    Returns:
    - str: The full path of the working directory.

    Example:
        set_working_directory("C:/Users/Example", "Project", "Data")
    """
    full_path = Path(base_folder).joinpath(*args)
    full_path.mkdir(parents=True, exist_ok=True)
    os.chdir(full_path)
    print(f"The working directory is set to {full_path}.")
    return str(full_path)

def grab_files(folder_path: str, ext: str = "", recursive: bool = False) -> List[str]:
    """
    Retrieves file paths with a specific extension from a folder, optionally including subfolders.

    Parameters:
    - folder_path (str): Directory to search.
    - ext (str): File extension filter (e.g., ".txt", ".csv").
    - recursive (bool): Whether to include subdirectories.

    Returns:
    - List[str]: List of matching file paths.

    Example:
        grab_files("./data", ext=".csv", recursive=True)
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"The provided path {folder_path} is not a valid directory.")
    if recursive:
        return [str(Path(root) / file)
                for root, _, files in os.walk(folder_path)
                for file in files if file.endswith(ext)]
    return [str(file) for file in folder.glob(f'*{ext}') if file.is_file()]

def grab_folders(folder_path: str, recursive: bool = False, names_only: bool = False) -> List[str]:
    """
    Retrieves folder paths or names from a directory, optionally including subfolders.

    Parameters:
    - folder_path (str): Directory to search.
    - recursive (bool): Whether to include subdirectories.
    - names_only (bool): If True, return only folder names instead of full paths.

    Returns:
    - List[str]: List of folder paths or names.

    Example:
        grab_folders("./projects", recursive=False, names_only=True)
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"The provided path {folder_path} is not a valid directory.")
    if recursive:
        return [name if names_only else os.path.join(root, name)
                for root, dirs, _ in os.walk(folder_path) for name in dirs]
    folders = [f for f in folder.glob('*') if f.is_dir()]
    return [f.name if names_only else str(f) for f in folders]

class PathInfo(NamedTuple):
    parent: str
    stem: str

def get_dirname_and_basename(path: str) -> PathInfo:
    """
    Extracts the parent folder name and base file name (without extension).

    Parameters:
    - path (str): File path.

    Returns:
    - PathInfo: Named tuple with parent folder and base name.

    Example:
        get_dirname_and_basename("/Users/name/data/file.csv")
    """
    p = Path(path)
    return PathInfo(parent=str(p.parent.parent.name), stem=p.stem)

def load_dataframes(file_list: List[str], file_type: str = 'pickle', trace_start_idx: int = 1):
    """
    Loads multiple data files and extracts traces and optional time columns.

    Parameters:
    - file_list (List[str]): List of file paths.
    - file_type (str): File type ('pickle' or 'csv').
    - trace_start_idx (int): Index of first trace column (0 means no time column).

    Returns:
    - np.ndarray: All traces (combined and transposed).
    - Optional[np.ndarray]: Time vector (or None).
    - List[str]: Labels for each trace based on file index.

    Example:
        load_dataframes(['file1.pkl', 'file2.pkl'], file_type='pickle', trace_start_idx=1)
    """
    all_traces = []
    time_vector = None
    trace_labels = []

    for file_idx, file_path in enumerate(file_list):
        if file_type == 'pickle':
            df = pd.read_pickle(file_path)
        elif file_type == 'csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Supported file types: 'pickle' or 'csv'")

        if trace_start_idx > 0:
            time_data = df.iloc[:, :trace_start_idx].values.squeeze()
        else:
            time_data = None

        traces = df.iloc[:, trace_start_idx:].values.T

        if time_data is not None:
            if time_vector is None:
                time_vector = time_data
            elif not np.allclose(time_vector, time_data, equal_nan=True):
                raise ValueError(f"Time mismatch in {file_path}")

        all_traces.append(traces)
        trace_labels.extend([f"file_{file_idx}"] * traces.shape[0])

    return np.vstack(all_traces), time_vector, trace_labels

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    Parameters:
    - config_path (str): Path to the YAML configuration file.
    Returns:
    - dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config_copy(cfg: dict, output_dir: Path) -> None:
    """
    Save a copy of the config used for analysis into the output directory.
    Parameters:
    - cfg (dict): Configuration dictionary.
    - output_dir (Path): Directory where the config copy will be saved.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    config_copy_path = output_dir / "config_used.yaml"
    with open(config_copy_path, "w") as f:
        yaml.safe_dump(cfg, f)

@contextmanager
def temp_chdir(path):
    """
    A context manager for temporarily changing the current working directory.
    Args:
        path (str): The target directory to change into. If the directory does not exist, it will be created.

    Yields:
        None

    Usage:
        with temp_chdir('/path/to/dir'):
            # Code executed inside the specified directory

    Upon exiting the context, the working directory is restored to its previous value.
    """
    prev_dir = os.getcwd()
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[temp_chdir] Created directory: {path}")
    else:
        print(f"[temp_chdir] Entering directory: {path}")
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_dir)
        print(f"[temp_chdir] Returned to directory: {prev_dir}")