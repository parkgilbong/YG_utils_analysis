# src/analysis/rename_files.py
"""
Batch rename TIFF files based on a YAML configuration.

This script reads settings from a YAML file (default: config.yaml) and for each specified subfolder:
  - Ensures the folder exists (creates if missing)
  - Scans for files with the target extension
  - Strips a fixed number of leading characters from each filename
  - Renames files whose suffix is purely numeric using a zero-padded prefix
  - Logs all operations to console and a log file

Usage:
    python rename_files.py [--config CONFIG]

Example:
    python rename_files.py -c /path/to/config.yaml
"""
import os
import argparse
import logging
from pathlib import Path
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    """
    Load YAML configuration from the given file path.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary of configuration parameters.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logger(log_path: str) -> None:
    """
    Configure the root logger to write INFO-level messages to both console and file.

    Args:
        log_path: File path for the log output.
    """
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a', encoding='utf-8'),
            # logging.StreamHandler() # Remove or comment out this line to disable console logging
        ]
    )


def rename_files(config: dict) -> None:
    """
    Rename files in each configured subfolder according to numeric suffix.

    Args:
        config: Configuration dictionary loaded from YAML.
    """
    base = Path(config['base'])
    ext = config.get('file_extension', '')
    strip_chars = config.get('strip_chars', 0)
    prefix = config.get('prefix', 'img')
    padding = config.get('padding', 5)

    for sub in tqdm(config['sub_folders'], desc='Processing folders'):
        target_dir = base / sub
        if not target_dir.exists():
            logging.info(f"Creating directory: {target_dir}")
            target_dir.mkdir(parents=True)

        logging.info(f"Scanning directory: {target_dir}")
        files = list(target_dir.glob(f"*{ext}"))

        for file_path in files:
            stem = file_path.stem
            if len(stem) > strip_chars:
                suffix = stem[strip_chars:]
                if suffix.isdigit():
                    num = int(suffix)
                    new_name = f"{prefix}{num:0{padding}d}{ext}"
                    new_path = target_dir / new_name
                    file_path.rename(new_path)
                    logging.info(f"Renamed: {file_path.name} -> {new_name}")
                else:
                    logging.warning(f"Filename suffix not numeric: {file_path.name}")
            else:
                logging.warning(f"Filename too short to strip: {file_path.name}")


def main():
    """
    Entry point for the script.

    Parses command-line arguments and initiates batch renaming.
    """
    parser = argparse.ArgumentParser(
        description='Batch rename files based on YAML config.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python rename_files.py -c /path/to/config.yaml'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to the YAML configuration file'
    )
    args = parser.parse_args()

    conf = load_config(args.config)
    setup_logger(conf.get('log_file', 'rename_files.log'))
    rename_files(conf)


if __name__ == '__main__':
    main()