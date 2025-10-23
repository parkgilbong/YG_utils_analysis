import os
from typing import Any, Dict, List,  Optional
from utils.FileFunctions import load_yaml
from pathlib import Path

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
    cfg: Dict,
    config_key: str,
    config_section: Optional[str] = None,
    is_input: bool = False,
) -> List[str]:
    """
    Resolves file paths from CLI arguments or a configuration dictionary,
    making them absolute based on the project root.

    The function prioritizes paths from the CLI. If not provided, it falls
    back to the configuration file.

    Args:
        cli_path: Path(s) from argparse.
        cfg: The full configuration dictionary.
        config_key: The key for the path in the config section.
        config_section: The name of the section in the config (e.g., "filtering").
        is_input: If True, resolves the path relative to the project root.
                  If False (default), resolves it relative to the `ROOT_DIR`
                  specified in the config, which is typically for outputs.

    Returns:
        A list of absolute, resolved file paths. Returns an empty list if
        no path is found.
    """
    # --- Determine the base directory for resolving paths ---
    # This utility might be in a different project (`YG_utils_analysis`) than the
    # calling script (`RNA-Seq`). We need the project root of the *calling* script.
    # We can get this by inspecting the call stack.
    import inspect
    main_script_path = inspect.stack()[-1].filename
    project_root = Path(main_script_path).resolve().parents[2]
    
    # For outputs, the base is the ROOT_DIR from the config.
    # For inputs, the base is the project root itself.
    if is_input:
        base_dir = project_root
    else:
        # Use the sample-specific ROOT_DIR for outputs
        root_dir_str = cfg.get("ROOT_DIR", ".")
        base_dir = project_root / root_dir_str

    # Get the relative path from CLI or config
    path_val = None
    if cli_path:
        path_val = cli_path
    else:
        section = cfg.get(config_section, cfg) if config_section else cfg
        path_val = section.get(config_key)

    if not path_val:
        return []

    # Ensure we're working with a list
    paths = path_val if isinstance(path_val, list) else [path_val]

    # Resolve each path relative to the appropriate base directory
    resolved_paths = []
    for p in paths:
        if Path(p).is_absolute():
            resolved_paths.append(str(p)) # If path is already absolute, use it as is.
        else:
            resolved_paths.append(str(base_dir / p))
    
    return resolved_paths