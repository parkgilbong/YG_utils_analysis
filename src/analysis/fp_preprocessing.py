# src/analysis/fp_preprocessing.py

import os
import yaml
import argparse
import logging
from pathlib import Path

import src.utils.FPFunctions as FPFunctions
import src.utils.FileFunctions as FileFunctions
import src.utils.ReportGeneration as ReportGeneration

def setup_logging():
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/fp_preprocessing.log', mode='a'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_preprocessing(config):
    setup_logging()

    logging.info("Starting preprocessing with provided config dictionary.")

    raw_data_path = config['raw_data_path']
    recorded_date = config['recorded_date']
    base_folder = os.path.abspath(config['base_folder'])
    batch_folder = config['batch_folder']
    sys = config.get('sys', 'tdt')  # default to 'tdt'
    detrending_method = config.get('detrending_method', 'Highpass_filter')
    use_camtick = config.get('use_camtick', True)
    duration_mode = config.get('duration_mode', 'fixed')  # default to fixed
    fps = config.get('fps', 25)
    rec_duration = config.get('rec_duration', 600)
    save_as_csv = config.get('save_as_csv', True)

    datafolder_list = config.get('datafolder_list')
    if datafolder_list is None:
        datafolder_list = FileFunctions.grab_folders(
            os.path.join(raw_data_path, recorded_date), 
            recursive=False, 
            names_only=True
        )
    
    animal_id_list = config.get('animal_id_list')
    if animal_id_list is None:
        animal_id_list = [x[:7] for x in datafolder_list]

    logging.info(f"Animal ID list: {animal_id_list}")

    session_list = config.get('session_list', ['EE', 'SE'])

    # Save config copy once at the batch_folder level
    batch_folder_path = os.path.join(base_folder, batch_folder)
    FileFunctions.save_config_copy(config, Path(batch_folder_path))
    logging.info(f"Saved config_used.yaml to {batch_folder_path}")

    for i, folder in enumerate(datafolder_list):
        try:
            if sys == 'tdt':
                tdt_tanks_path = os.path.join(raw_data_path, recorded_date, folder)
            else:
                tdt_tanks_path = os.path.join(raw_data_path, recorded_date, folder, 'Fluorescence.csv')

            dest_folder = os.path.join(base_folder, batch_folder, animal_id_list[i], session_list[i % len(session_list)], 'Preprocessing')

            logging.info(f"Processing {animal_id_list[i]} - Session: {session_list[i % len(session_list)]}")

            FPFunctions.FP_preprocessing_1ch(
                Tank_path=tdt_tanks_path,
                Dest_folder=dest_folder,
                sys=sys,
                Detrending_method=detrending_method,
                Use_CamTick=use_camtick,
                duration_mode=duration_mode,
                FPS=fps,
                Rec_duration=rec_duration,
                SaveAsCSV=save_as_csv
            )

            image_paths = [
                os.path.join(dest_folder, 'Plot_Raw_signal_465.png'),
                os.path.join(dest_folder, 'Plot_Denoised_signals.png'),
                os.path.join(dest_folder, 'Plot_405_465_correlation.png'),
                os.path.join(dest_folder, 'Plot_465_z-score.png')
            ]

            comments = [
                "Figure1. Raw signals",
                "Figure2. Denoised signals",
                "Figure3. Correlation between 405 and 465",
                "Figure4. 465 zscore"
            ]

            ReportGeneration.FP_preprocessing(
                output_path=os.path.join(base_folder, batch_folder, animal_id_list[i], session_list[i % len(session_list)]),
                title=f'{animal_id_list[i]}_{session_list[i % len(session_list)]}',
                image_paths=image_paths,
                comments=comments
            )

            logging.info(f"Completed {animal_id_list[i]} - Session: {session_list[i % len(session_list)]}")
        except Exception as e:
            logging.error(f"Error processing {animal_id_list[i]} - Session: {session_list[i % len(session_list)]}: {e}", exc_info=True)

    logging.info("All preprocessing completed.")

def main():
    parser = argparse.ArgumentParser(description='Run FP preprocessing pipeline.')
    parser.add_argument('--config', type=str, default='configs/fp_preprocessing.yaml',
                        help='Path to the config YAML file.')
    args = parser.parse_args()

    config = load_config(args.config)
    run_preprocessing(config)

if __name__ == "__main__":
    main()
