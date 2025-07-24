#src/analysis/flip_videos.py
"""
Script to vertically flip multiple AVI videos as defined in a YAML config,
using a global output directory and creating directories as needed.
Usage:
    python -m src.analysis.flip_videos --config path/to/config.yaml

    

YAML config example:
    output_dir: /path/to/flipped_videos
    files:
      - input: /path/to/video1.avi
        output: video1_flipped.avi
      - input: /path/to/video2.avi
        output: subdir/video2_flipped.avi
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import yaml
from tqdm import tqdm

from src.utils.VideoFunctions import flip_video


def setup_logging() -> None:
    """Configure basic logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: Path) -> Tuple[Path, List[Dict[str, str]]]:
    """Load YAML config and return global output_dir and list of file entries."""
    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    output_dir = data.get("output_dir")
    if not output_dir:
        raise ValueError(f"'output_dir' must be defined in {config_path}")
    output_dir_path = Path(output_dir)

    files = data.get("files")
    if not isinstance(files, list):
        raise ValueError(f"'files' must be a list in {config_path}")

    return output_dir_path, files


def process_files(output_dir: Path, files: List[Dict[str, str]]) -> None:
    """Iterate through config entries, ensure output dirs, and flip each video."""
    for entry in tqdm(files, desc="Processing videos"):
        input_path = entry.get("input") or entry.get("input_path")
        output_name = entry.get("output") or entry.get("output_name")

        if not input_path or not output_name:
            logging.error("Invalid entry (missing 'input' or 'output'): %s", entry)
            continue

        input_path = Path(input_path)
        output_path = output_dir / output_name

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.exception("Failed to create directory for %s: %s", output_path, e)
            continue

        try:
            logging.info("Flipping video: %s -> %s", input_path, output_path)
            flip_video(str(input_path), str(output_path))  # type: ignore
            logging.info("Successfully processed %s", input_path)
        except Exception as e:
            logging.exception("Failed to process %s: %s", input_path, e)


def main() -> None:
    """Parse arguments, load config, and initiate processing."""
    parser = argparse.ArgumentParser(
        description="Flip videos as specified in a YAML config using a global output directory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/flip_video.yaml"),
        help="Path to YAML config file listing videos to process",
    )
    args = parser.parse_args()

    setup_logging()
    logging.info("Loading config from %s", args.config)

    try:
        output_dir, files = load_config(args.config)
    except Exception as e:
        logging.exception("Failed to load config: %s", e)
        return

    process_files(output_dir, files)


if __name__ == "__main__":
    main()
