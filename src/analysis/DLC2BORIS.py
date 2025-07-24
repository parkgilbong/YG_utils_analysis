# File: src/analysis/DLC2BORIS.py
"""
DLC2BORIS.py

This module converts DeepLabCut (DLC) CSV pose estimation output into BORIS-compatible CSV event format.

Usage:
------
▶ Run from command line:
    $ python src/analysis/DLC2BORIS.py --config path/to/config.yaml

▶ Import from another Python script or Jupyter notebook:
    from src.analysis.DLC2BORIS import process_animal, load_config
    
    cfg = load_config("../configs/config.yaml")
    process_animal("test_rat", cfg, Path(cfg["OUTPUT_DIR"]))

"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from src.utils.FileFunctions_new import grab_files, save_config_copy, load_config
from src.utils.DLCFunctions import df_to_dic_multi, annotate_body_part_proximity
from src.utils.boris_export import extract_bouts_from_annotation, prepare_boris_export



def process_animal(animal_id: str, cfg: dict, output_base: Path):
    print(f"Processing animal: {animal_id}")

    animal_data_dir = Path(cfg["DATA_DIR"]) / animal_id
    if not animal_data_dir.exists():
        print(f"Data directory for {animal_id} does not exist.")
        return

    # Load DLC CSV (multi-index)
    csv_file = grab_files(animal_data_dir, ".csv", recursive=False)[0]
    df = pd.read_csv(csv_file, header=[2, 3])

    subject_bodyparts, partner_bodyparts = df_to_dic_multi(df)

    body_parts_pair = cfg.get("body_parts_pair", [
        ('Nose', 'Nose'),
        ('Nose', 'Left_ear'),
        ('Nose', 'Right_ear'),
        ('Nose', 'Left_fhip'),
        ('Nose', 'Right_fhip'),
        ('Nose', 'Tail_base'),
        ('Left_ear', 'Nose'),
        ('Right_ear', 'Nose'),
        ('Left_fhip', 'Nose'),
        ('Right_fhip', 'Nose'),
        ('Tail_base', 'Nose'),
        ('Tail_base', 'Tail_base')
    ])

    all_events = []
    for subject_bp, partner_bp in body_parts_pair:
        events_df = annotate_body_part_proximity(
            subject_bodyparts, partner_bodyparts,
            subject_bp, partner_bp,
            cfg['d_threshold'], cfg['d_threshold2'], cfg['extra_threshold'],
            cfg['pcutoff']
        )

        events_df = extract_bouts_from_annotation(events_df)
        events_df["behavior"] = cfg["boris_behavior_map"].get(f"{subject_bp}_{partner_bp}", "unknown")
        all_events.append(events_df)

    if all_events:
        combined_events = pd.concat(all_events)
        boris_ready_df = prepare_boris_export(combined_events, cfg["event_columns"], cfg["FPS"])

        boris_file = output_base / f"{animal_id}_boris.csv"
        boris_ready_df.to_csv(boris_file, index=False)
        print(f"BORIS CSV saved: {boris_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert DLC pose data to BORIS format")
    parser.add_argument("--config", default="../configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(cfg["OUTPUT_DIR"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    save_config_copy(cfg, output_dir)

    for animal_id in cfg["ANIMAL_ID_LIST"]:
        process_animal(animal_id, cfg, output_dir)


if __name__ == "__main__":
    main()