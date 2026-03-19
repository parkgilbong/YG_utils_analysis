# src/analysis/peak_analysis.py
"""
Peak Analysis Pipeline for Fiber Photometry signals

This script loads preprocessing results (e.g., raw trace pickle files) for each animal/session,
performs peak detection, and logs the summary.

Usage (Command Line):
    $ python src/analysis/peak_analysis.py

Usage (Jupyter Notebook):
    from src.analysis import peak_analysis
    config = peak_analysis.load_config("configs/peak_analysis.yaml")
    peak_analysis.run_peak_analysis(config)

Required:
- `Final_table_raw_trace.pkl` should exist in the Preprocessing folder of each session.
- See `configs/peak_analysis.yaml` for all configuration options.
"""

import os
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.utils import FPFunctions, FileFunctions
from src.utils.FileFunctions import temp_chdir

# Ensure working directory is project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
print(f"Working directory set to: {os.getcwd()}")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_peak_analysis(config: dict):
    base = config["base"]
    params = config["peak_params"]
    output_cfg = config["output"]

    base_folder = os.path.abspath(base['base_folder'])

    root_folder = os.path.join(base_folder, base["batch"])
    session_list = base["session_list"]

    datafolder_list = FileFunctions.grab_folders(root_folder, recursive=False, names_only=True)
    animal_id_list = base["animal_id_list"]
    autofilled = False

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(output_cfg["save_folder"]) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    FileFunctions.save_config_copy(config, out_dir)

    if animal_id_list is None:
        animal_id_list = [x[:7] for x in datafolder_list]
        autofilled = True

    summary_records = []

    for animal_id in animal_id_list:
        for session in session_list:
            destfolder = os.path.join(root_folder, animal_id, session)
            destfolder2 = os.path.join(out_dir, animal_id, session)
            # FileFunctions.set_working_directory(destfolder)

            pkl_path = os.path.join(destfolder, "Preprocessing", "Final_table_raw_trace.pkl")
            video_path = os.path.join(destfolder, f'{base["batch"]}_{animal_id}_{session}{output_cfg["video_suffix"]}')

            if not os.path.exists(pkl_path):
                summary_records.append({
                    "AnimalID": animal_id,
                    "Session": session,
                    "PeakCount": "Error: pkl file not found"
                })
                continue

            try:
                
                with temp_chdir(destfolder2):
                    peak_count = FPFunctions.Peak_Analysis(
                        pkl_path=pkl_path,
                        signal2use=params["signal2use"],
                        prominence_thres=params["prominence_thres"],
                        amplitude_thres=params["amplitude_thres"],
                        FPS=params["FPS"],
                        pre_window_len=params["pre_window_len"],
                        post_window_len=params["post_window_len"],
                        SavePlots=output_cfg["save_plots"],
                        SaveData=output_cfg["save_data"],
                        SaveVideos=output_cfg["save_videos"],
                        video_path=video_path
                    )
                summary_records.append({
                    "AnimalID": animal_id,
                    "Session": session,
                    "PeakCount": peak_count
                })
            except Exception as e:
                summary_records.append({
                    "AnimalID": animal_id,
                    "Session": session,
                    "PeakCount": f"Error: {str(e)}"
                })

    # Save summary
    summary_path = os.path.join(out_dir, f"summary_{timestamp}.csv")
    df_summary = pd.DataFrame(summary_records)
    df_summary.attrs["animal_id_list_source"] = "autofilled from folder names" if autofilled else "from config"
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    print(f"AnimalID list source: {df_summary.attrs['animal_id_list_source']}")

def main():
    config = load_config("configs/peak_analysis.yaml")
    run_peak_analysis(config)

if __name__ == "__main__":
    main()
