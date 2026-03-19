"""CLI commands for Fiber Photometry pipelines."""
import argparse
from fp_behav.fp.preprocessing import run_preprocessing, load_config as load_fp_config
from fp_behav.fp.preprocessing_2ch import run_preprocessing as run_preprocessing_2ch
from fp_behav.fp.epoch import run_epoch_analysis, load_config as load_epoch_config
from fp_behav.fp.peak import run_peak_analysis


def fp_preprocess():
    """Entry point: fp-preprocess"""
    parser = argparse.ArgumentParser(description='Run 1-channel FP preprocessing pipeline.')
    parser.add_argument('--config', type=str, default='configs/fp_1ch.yaml',
                        help='Path to the config YAML file.')
    args = parser.parse_args()
    config = load_fp_config(args.config)
    run_preprocessing(config)


def fp_preprocess_2ch():
    """Entry point: fp-preprocess-2ch"""
    parser = argparse.ArgumentParser(description='Run 2-channel FP preprocessing pipeline.')
    parser.add_argument('--config', type=str, default='configs/fp_2ch.yaml',
                        help='Path to the config YAML file.')
    args = parser.parse_args()
    config = load_fp_config(args.config)
    run_preprocessing_2ch(config)


def fp_epoch():
    """Entry point: fp-epoch"""
    parser = argparse.ArgumentParser(description='Run epoch analysis.')
    parser.add_argument('--config', type=str, default='configs/epoch.yaml')
    args = parser.parse_args()
    config = load_epoch_config(args.config)
    import os
    run_epoch_analysis(config, os.getcwd())


def fp_peak():
    """Entry point: fp-peak"""
    parser = argparse.ArgumentParser(description='Run peak analysis.')
    parser.add_argument('--config', type=str, default='configs/peak.yaml')
    args = parser.parse_args()
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run_peak_analysis(config)
