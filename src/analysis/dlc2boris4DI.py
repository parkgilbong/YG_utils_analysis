"""
dlc2boris.py

This module processes DeepLabCut (DLC) tracking data and converts proximity-based interaction events
into BORIS-compatible behavioral annotations.

Functions:
    - load_config(config_path): Loads a YAML configuration file.
    - run_dlc2boris_pipeline(cfg): Processes all animals specified in the configuration file.
    - process_animal(animal_id, animal_data_dir, output_dir, cfg): Processes a single animal's tracking data,
      detects interactions, extracts behavioral events, and exports results.

Expected YAML config keys:
    DATA_DIR: Directory containing raw tracking data (CSV)
    OUTPUT_DIR: Directory to store results
    SESSION_ID: Session name or ID
    GROUP_ID: Group name or ID
    ANIMAL_ID_LIST: List of animal ID folder names
    d_threshold, d_threshold2, extra_threshold: Distance thresholds for proximity annotation
    pcutoff: Confidence cutoff for pose estimation
    FPS: Frames per second of video
    boris_behavior_map: Mapping of event to BORIS behavior string
    event_columns: List of event columns to extract

Usage:
    from src.analysis import dlc_2_boris
    cfg = dlc_2_boris.load_config("../configs/dlc2boris.yaml")
    dlc2boris.run_dlc2boris_pipeline(cfg)

Command-line usage:
    python -m src.analysis.dlc2boris ../configs/config.yaml
"""

import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys
import yaml

from src.utils.FileFunctions import grab_files, save_config_copy, load_config
from src.utils.DLCFunctions import df_to_dic_multi, annotate_body_part_proximity
from src.utils.boris_export import extract_bouts_from_annotation, prepare_boris_export

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_dlc2boris_pipeline(cfg: dict):
    data_dir = Path(cfg["DATA_DIR"])
    output_dir = Path(cfg["OUTPUT_DIR"])
    session_id = cfg["SESSION_ID"]
    animal_ids = cfg["ANIMAL_ID_LIST"]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_output_dir = output_dir / timestamp
    session_output_dir.mkdir(parents=True, exist_ok=True)

    for animal_id in animal_ids:
        print(f"Processing animal: {animal_id}")
        animal_data_dir = data_dir / animal_id

        if not animal_data_dir.exists():
            print(f"Data directory for {animal_id} does not exist.")
            continue

        try:
            process_animal(animal_id, animal_data_dir, session_output_dir, cfg)
        except Exception as e:
            print(f"Error processing {animal_id}: {e}")

    save_config_copy(cfg, session_output_dir)

def process_animal(animal_id: str, animal_data_dir: Path, output_dir: Path, cfg: dict):
    file = grab_files(animal_data_dir, ".csv", recursive=False)[0]
    df = pd.read_csv(file, header=[2,3])

    subject_bodyparts, partner_bodyparts = df_to_dic_multi(df)
    points_df = pd.DataFrame()

    body_parts_pair = [
        ('Nose', 'Nose'), ('Nose', 'Left_ear'), ('Nose', 'Right_ear'),
        ('Nose', 'Left_fhip'), ('Nose', 'Right_fhip'), ('Nose', 'Tail_base'),
        ('Left_ear', 'Nose'), ('Right_ear', 'Nose'), ('Left_fhip', 'Nose'),
        ('Right_fhip', 'Nose'), ('Tail_base', 'Nose'), ('Tail_base', 'Tail_base')
    ]

    for part1, part2 in body_parts_pair:
        annotate_body_part_proximity(
            subject_bodyparts, partner_bodyparts, points_df,
            body_part_name1=part1, body_part_name2=part2,
            pcutoff=cfg["pcutoff"],
            d_threshold=cfg["d_threshold"],
            d_threshold2=cfg["d_threshold2"]
        )

    out_csv = output_dir / f"DLC_Analysis_{animal_id}.csv"
    points_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    evt_dict = {
        evt_name: extract_bouts_from_annotation(points_df[evt_name], cfg["FPS"])
        for evt_name in cfg["event_columns"]
    }

    final_df = prepare_boris_export(evt_dict, cfg["FPS"], animal_id, cfg["boris_behavior_map"])
    out_excel = output_dir / f"DLC_Analysis2BORIS_{animal_id}.xlsx"
    final_df.to_excel(out_excel, index=False)
    print(f"Exported: {out_excel}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.analysis.dlc_2_boris <path_to_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = load_config(config_path)
    run_dlc2boris_pipeline(cfg)
