#src/analysis/dlc2boris43CT.py
"""
dlc2boris43CT.py

A modular pipeline to process DeepLabCut (DLC) outputs for multiple animals and sessions,
perform movement and proximity analyses, ROI entry detection, and export event data in
BORIS-compatible format.

This updated version loads configuration from a YAML file with the following structure:

1) Base paths
   root_folder: str            # Top-level data directory
   batch_folder: str           # Subfolder for this batch of analyses
   dlc_analysis_subfolder: str # Name of DLC output subfolder (default: DLC_Analysis)
   output_dir: str             # Directory to save BORIS export files

2) Group and animal lists
   group_id: str               # Prefix for DLC file names (e.g., "CaMKII")
   animal_id_list:            # List of animal IDs (e.g., ["G14_001", "G14_002"])
     - "G14_001"
     - "G14_002"
   session:                    # List of session identifiers (e.g., ["EE", "SE"])
     - "EE"
     - "SE"

3) DLC analysis parameters
   dlc_scorer: str             # Suffix for DLC CSV file names (e.g., DLCscorer_v2)
   bodypart4velocity: str      # Body part to compute velocity
   bodyparts4distance:        # Two body parts for proximity analysis
     - "Nose"
     - "snout1"
   bodypart4roientry: str      # Body part for ROI entry detection
   fps: float                  # Frames per second of the video
   pcutoff: float              # Probability cutoff for body-point detection
   distance_thres: float       # Pixel threshold for proximity detection

4) ROI definitions
   roi_folder: str             # Path to folder containing ROI CSV files
   roi_list:                   # Filenames (without extension) for ROIs
     - "S_sniffing_zone"
     - "S1"
     - "S2"
     - "E_sniffing_zone"
     - "E1"
     - "E2"

5) Output options
   save_data: bool             # Save per-frame analysis CSV (default: false)
   export2boris: bool          # Export BORIS Excel file (default: false)
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

from utils.DLCFunctions import (
    df_to_dic_single,
    get_velocity,
    get_bodypoints_distance,
    roi_entry_analysis,
)
from utils.boris_export import extract_bouts_from_annotation, prepare_boris_export
from utils.FileFunctions import save_config_copy


# def log_status(message: str) -> None:
#     """
#     Print a timestamped status message.
#     """
#     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def setup_logging():
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/dlc2boris43CT.log', mode='a'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: Path) -> dict:
    """
    Load processing parameters from a YAML configuration file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    return config


def process_animal(animal: str, session: str, config: dict, output_dir: Path) -> None:
    """
    Process DLC outputs and extract movement and event metrics for one animal-session.
    Results are saved into the provided output_dir.
    """
    setup_logging()

    alias = f"{config['group_id']}_{animal}"
    logging.info(f"Loading DLC data for {alias} session {session}")

    # Build paths
    base = Path(config['root_folder']) / config['batch_folder'] / animal / session
    dlc_folder = base / config.get('dlc_analysis_subfolder', 'DLC_Analysis')
    dlc_folder.mkdir(parents=True, exist_ok=True)

    # Read DLC CSV
    csv_file = f"{alias}_{session}{config['dlc_scorer']}.csv"
    df = pd.read_csv(dlc_folder / csv_file, header=[1, 2])
    body_data = df_to_dic_single(df=df)
    logging.info("DLC data loaded and parsed into dictionary")

    result = {}
    events = {}

    # 1) Compute velocity
    logging.info("Computing velocity")
    time, velocity = get_velocity(
        DLCresult=body_data,
        bpt=config['bodypart4velocity'],
        FPS=config['fps'],
        pcutoff=config['pcutoff'],
    )
    result['Time'] = time
    result['Velocity'] = velocity
    logging.info("Velocity computed")

    # 2) Compute proximity between two body parts
    logging.info("Computing proximity distance and epochs")
    bpt1, bpt2 = config['bodyparts4distance']
    dist, epochs = get_bodypoints_distance(
        DLCresult=body_data,
        bpt=bpt1,
        bpt2=bpt2,
        pcutoff=config['pcutoff'],
        distance_thres=config['distance_thres'],
    )
    result[f"{bpt1}_{bpt2}_distance"] = dist
    result[f"{bpt1}_{bpt2}_epoch"] = epochs
    logging.info("Proximity analysis completed")

    # 3) ROI entry detection
    logging.info("Starting ROI entry detection")
    for roi in config['roi_list']:
        roi_path = Path(config['roi_folder']) / f"{roi}.csv"
        if not roi_path.exists():
            logging.warning(f"ROI file missing: {roi_path}")
            continue
        roi_df = pd.read_csv(roi_path)
        coords = list(zip(roi_df['X'], roi_df['Y']))
        result[roi] = roi_entry_analysis(
            DLCresult=body_data,
            bpt=config['bodypart4roientry'],
            pcutoff=config['pcutoff'],
            ROI=coords)
    logging.info("ROI entry detection completed")

    # 4) Construct DataFrame for analysis
    analysis_df = pd.DataFrame(result)

    # 5) Save per-frame analysis if requested
    if config.get('save_data', False):
        filename = f"{alias}_{session}_frame_analysis.csv"
        path = output_dir / filename
        analysis_df.to_csv(path, index=False)
        logging.info(f"Per-frame analysis saved: {path}")

    # 6) Extract event bouts for boolean event columns
    logging.info("Extracting event bouts from boolean columns")
    evt_dict = {}
    for evt in config['event_columns']:
        series = analysis_df[evt]
        # boolean → 'on'/'off' convert
        for evt in config['event_columns']:
            series = analysis_df[evt]
            # Convert to 'on'/'off' robustly, handling blanks and NaN
            series = series.apply(lambda x: 'on' if bool(x) and str(x).strip().lower() not in ['off', 'nan', ''] else 'off')
            evt_dict[evt] = extract_bouts_from_annotation(series, config['fps'])
        print(evt_dict[evt])
    logging.info("Event bouts extracted")

    # 7) Export BORIS file if configured
    boris_df = prepare_boris_export(evt_dict, config['fps'], animal, config["boris_behavior_map"])
    path = output_dir / f"{alias}_{session}_BORIS.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    boris_df.to_excel(path, index=False)
    logging.info(f"BORIS export saved: {path}")


def run_dlc2boris_pipeline(config: dict) -> None:
    """
    Execute the DLC-to-BORIS pipeline over all specified animals and sessions.
    Creates a timestamped session directory under the configured output_dir.
    """
    setup_logging()
    
    base_out = Path(config['output_dir'])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_output_dir = base_out / timestamp
    session_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Session output directory created: {session_output_dir}")

    for animal in config['animal_id_list']:
        for session in config['session']:
            logging.info(f"Starting: {animal} - {session}")
            try:
                process_animal(animal, session, config, session_output_dir)
            except Exception as e:
                logging.error(f"Error processing {animal} session {session}: {e}")
            logging.info(f"Finished: {animal} - {session}")
    save_config_copy(config, session_output_dir)
    logging.info(f"Configuration saved to {session_output_dir / 'config_used.yaml'}")

def main() -> None:
    """
    Command-line entry point.
    """
    parser = argparse.ArgumentParser(
        description="Process DeepLabCut outputs (CT version) and export BORIS event files."
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('dlc2boris43CT.yaml'),
        help='YAML configuration file path.'
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    cfg = load_config(args.config)
    run_dlc2boris_pipeline(cfg)


if __name__ == '__main__':
    main()
