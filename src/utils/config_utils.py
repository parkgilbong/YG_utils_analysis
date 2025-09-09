import os
from typing import Any, Dict, List,  Optional
from utils.FileFunctions import load_yaml

def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries.

    For each key in the `update` dictionary:
    - If the key exists in `base` and both values are dictionaries, merges them recursively.
    - Otherwise, the value from `update` overwrites the value in `base`.

    Args:
        base (Dict[str, Any]): The base dictionary to merge into.
        update (Dict[str, Any]): The dictionary with updates to apply.

    Returns:
        Dict[str, Any]: A new dictionary with the merged contents.
    """
    out = dict(base)
    for k, v in (update or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_cfg(config_path: Optional[str]) -> Dict[str, Any]:
    return load_yaml(config_path) if config_path else {}


def pick(cli_val, cfg: Dict[str, Any], key: str, default=None):
    """CLI 우선, 없으면 cfg[key], 둘 다 없으면 default."""
    return cli_val if cli_val is not None else cfg.get(key, default)

def resolve_path(
    cli_path: Optional[List[str]],
    cfg: Dict[str, Any],
    config_key: str,
    config_section: Optional[str] = None,
) -> List[str]:
    """
    Resolves file paths, prioritizing CLI arguments over config file entries.
    If using the config, it constructs the path from 'ROOT_DIR' and a filename key.

    Args:
        cli_path: The path(s) from command-line arguments (e.g., args.excels).
        cfg: The main configuration dictionary.
        config_key: The key for the filename in the config section (e.g., "excel_file").
        config_section: The section in the config to look into. If None, looks in root of cfg.

    Returns:
        A list of full file paths.

    Raises:
        ValueError: If required configuration keys ('ROOT_DIR', `config_key`) are missing
                    when falling back to the config file.
    """
    # 1. Prioritize command-line argument
    if cli_path:
        return cli_path

    # 2. Construct path from config
    root_dir = cfg.get("ROOT_DIR")
    if not root_dir:
        raise ValueError("Configuration error: 'ROOT_DIR' must be specified in the config file.")

    section_cfg = cfg.get(config_section, {}) if config_section else cfg
    filename = section_cfg.get(config_key)
    if not filename:
        raise ValueError(
            f"Configuration error: '{config_key}' must be specified under '{config_section}' "
            "if no CLI path is provided."
        )

    # Handle both single and multiple filenames from config
    if isinstance(filename, list):
        return [os.path.join(root_dir, f) for f in filename]

    return [os.path.join(root_dir, str(filename))]