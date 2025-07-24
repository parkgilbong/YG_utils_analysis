# src/analysis/video_resizing.py
"""
Separate resizing module: reads config.yaml, finds videos, and calls VideoFunctions.resize_video fileciteturn3file1
"""
import os
import yaml
from tqdm import tqdm
from src.utils.VideoFunctions import resize_video


def video_resizing(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    root = config['root_folder']
    scale = config.get('scale_factor', 0.5)
    resize_enabled = config.get('resize', True)
    exts = config.get('video_extensions', ['.avi'])

    if not resize_enabled:
        print("Resize is disabled in config.")
        return

    videos = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in exts) and not fname.lower().endswith('_resized' + os.path.splitext(fname)[1]):
                videos.append(os.path.join(dirpath, fname))

    if not videos:
        print(f"No videos found for resizing in {root}")
        return

    print(f"Resizing {len(videos)} videos...")
    for vid in tqdm(videos, desc="Resizing"):
        base, ext = os.path.splitext(vid)
        out_path = base + '_resized' + ext
        resize_video(vid, out_path, scale_factor=scale)
    print("Resizing completed.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Resize videos using config YAML")
    parser.add_argument('--config', required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
