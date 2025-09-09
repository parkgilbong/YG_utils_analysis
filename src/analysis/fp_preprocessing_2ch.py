# src/analysis/fp_preprocessing.py

import os
import yaml
import argparse
import logging
from pathlib import Path

import utils.FPFunctions as FPFunctions
import utils.FileFunctions as FileFunctions
import utils.ReportGeneration as ReportGeneration

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

    detrending_method = config.get('detrending_method', 'Highpass_filter')
    use_camtick = config.get('use_camtick', True)
    duration_mode = config.get('duration_mode', 'fixed')  # default to fixed
    fps = config.get('fps', 25)
    rec_duration = config.get('rec_duration', 600)
    save_as_csv = config.get('save_as_csv', True)
    sensor_list = config.get('sensor_list', ['465', '560'])

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
        TDT_Tanks_path = os.path.join(raw_data_path, recorded_date, folder)
        Dest_folder = os.path.join(base_folder, batch_folder, animal_id_list[i], session_list[i % len(session_list)],'Preprocessing')
        Name465 = sensor_list[0]
        Name560 = sensor_list[1]
        FPFunctions.FP_preprocessing_2ch_new(Tank_path=TDT_Tanks_path, 
                        Dest_folder = Dest_folder,
                        Detrending_method=detrending_method,
                        Use_CamTick=use_camtick, 
                        duration_mode=duration_mode,
                        FPS=fps, 
                        Rec_duration=rec_duration,
                        Namefor465=Name465,
                        Namefor560=Name560)
        
        image_paths = [os.path.join(Dest_folder, f'Plot_Raw_signal_{Name465}.png'),
                        os.path.join(Dest_folder, f'Plot_Raw_signal_{Name560}.png'),
                        os.path.join(Dest_folder, f'Plot_Denoised_signals_{Name465}.png'),
                        os.path.join(Dest_folder, f'Plot_Denoised_signals_{Name560}.png'), 
                        os.path.join(Dest_folder, f'Plot_ISOS_{Name465}_correlation.png'),
                        os.path.join(Dest_folder, f'Plot_ISOS_{Name560}_correlation.png'), 
                        os.path.join(Dest_folder, f'Plot_{Name465}_z-score.png'),
                        os.path.join(Dest_folder, f'Plot_{Name560}_z-score.png')]

        comments = [f"Figure1. Raw signals: {Name465}",
                    f"Figure2. Raw signals: {Name560}",
                    f"Figure3. Denoised signals: {Name465}",
                    f"Figure4. Denoised signals: {Name560}",
                    f"Figure5. Correlation between ISOS and {Name465}",
                    f"Figure5. Correlation between ISOS and {Name560}",
                    f"Figure6. {Name465} z-score",
                    f"Figure7. {Name560} z-score"]

        ReportGeneration.FP_preprocessing(output_path=os.path.join(batch_folder_path, animal_id_list[i]), 
                                        title=f'{animal_id_list[i]}_{session_list[i % len(session_list)]}', 
                                        image_paths=image_paths, 
                                        comments=comments)

        logging.info(f"Completed {animal_id_list[i]} - Session: {session_list[i % len(session_list)]}")

    logging.info("All preprocessing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FP preprocessing pipeline.')
    parser.add_argument('--config', type=str, default='configs/fp_preprocessing.yaml',
                        help='Path to the config YAML file.')
    args = parser.parse_args()

    config = load_config(args.config)
    run_preprocessing(config)
