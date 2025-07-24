# src/analysis/preprocess_behavioral_data.py
"""
Preprocess Behavioral Video Data with Segment Support

This script processes raw behavioral video frame data into one or more videos per session,
allowing per-animal segmentation defined in a YAML config.

Usage:
    python src/analysis/preprocess_behavioral_data.py --config config/behavior_preprocessing.yaml
    or
    from src.analysis.preprocess_behavioral_data import preprocess_behavioral_data
    preprocess_behavioral_data("config/behavior_preprocessing.yaml")
"""
import os
import yaml
import logging
from tqdm import tqdm
from src.utils.VideoFunctions import create_video_from_images, resize_video
from src.utils.FileFunctions import grab_folders, save_config_copy

def setup_logging():
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/behavior_preprocessing.log', mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

def create_video_segments_from_images(
    image_folder: str,
    output_basename: str,
    frame_rate: int,
    segments: list,
    codec: str = 'mp4v',
    quality: int = 95
) -> list:
    """
    Create multiple video segments from a folder of images based on index ranges.
    Args:
        image_folder: Path to input images.
        output_basename: Base path/name for output videos (without extension).
        frame_rate: Frames per second for output videos.
        segments: List of dicts with 'start_index', 'end_index', and optional 'postfix'.
        codec: FourCC codec string.
        quality: JPEG quality (placeholder).
    Returns:
        List of generated video file paths.
    """
    import cv2

    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.png', '.tiff'))
    ])
    if not image_files:
        logging.info(f"No images found in {image_folder}")
        return []

    outputs = []
    for seg in segments:
        start = seg.get('start_index', 0)
        end   = seg.get('end_index', len(image_files))
        postfix = seg.get('postfix', f"{start}_{end}")
        subset = image_files[start:end]
        if not subset:
            logging.info(f"No frames in specified range {start}-{end}")
            continue

        first_frame = cv2.imread(subset[0])
        h, w, _ = first_frame.shape
        fourcc    = cv2.VideoWriter_fourcc(*codec)
        ext_map = {
            'mp4v': '.mp4',
            'avc1': '.mp4',
            'XVID': '.avi',
            'MJPG': '.avi',
            'DIVX': '.avi',
            'H264': '.mp4',
            'h264': '.mp4',
        }
        ext = ext_map.get(codec, '.avi')
        out_path  = f"{output_basename}_{postfix}{ext}"
        writer = cv2.VideoWriter(out_path, fourcc, frame_rate, (w, h))

        for img in tqdm(subset, desc=f"Writing segment {postfix}", leave=False):
            frame = cv2.imread(img)
            writer.write(frame)
        writer.release()
        logging.info(f"Segment created: {out_path}")
        outputs.append(out_path)
    return outputs


def preprocess_behavioral_data(config_path: str):
    """
    Read YAML config and generate videos per session, with optional per-animal segments.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    root_folder      = config['root_folder']
    group_list       = config['group_list']
    session_list     = config['session_list']
    fps              = config['fps']
    default_segs     = config.get('segments', [])
    animal_segs_map  = config.get('animal_segments', {})  # e.g. {'GroupA': {'Mouse1': [...], ...}, ...}
    duration         = config.get('duration', 600)
    scale_factor     = config.get('scale_factor', 0.5)
    quality          = config.get('quality', 95)
    codec            = config.get('codec', 'mp4v')
    resize_enabled   = config.get('resize', True)

    # save_config_copy(config, config_path)

    tasks = []
    for group in group_list:
        group_path = os.path.join(root_folder, group)
        for animal in grab_folders(group_path, recursive=False, names_only=True):
            for session in session_list:
                tasks.append((group, animal, session))

    # save_config_copy(config, group_path)

    logging.info(f"Generating videos for {len(tasks)} sessions...")
    generated = []

    with tqdm(total=len(tasks), desc="Processing", ncols=80) as bar:
        for group, animal, session in tasks:
            img_folder = os.path.join(root_folder, group, animal, session)
            base_name  = os.path.join(root_folder, f'B6_{group}_{animal}_{session}')

            # select segments: per-animal override or default
            segs = default_segs
            if group in animal_segs_map and animal in animal_segs_map[group]:
                segs = animal_segs_map[group][animal]

            if segs:
                outputs = create_video_segments_from_images(
                    img_folder, base_name, fps, segs, codec=codec, quality=quality
                )
                generated.extend(outputs)
            else:
                out_file = f"{base_name}.avi"
                create_video_from_images(
                    img_folder, out_file,
                    frame_rate=fps,
                    duration=duration,
                    codec=codec,
                    quality=quality
                )
                generated.append(out_file)

            bar.update(1)

    if resize_enabled and generated:
        logging.info(f"Resizing {len(generated)} videos...")
        for vid in tqdm(generated, desc="Resizing", ncols=80):
            resized_name = vid.replace('.avi', '_resized.avi')
            resize_video(vid, os.path.join(root_folder, resized_name), scale_factor=scale_factor)
    else:
        logging.info("Skipping resize step as configured.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess behavioral video data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    setup_logging()
    preprocess_behavioral_data(args.config)

if __name__ == "__main__":
    main()