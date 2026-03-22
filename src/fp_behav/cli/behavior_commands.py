"""CLI commands for Behavior analysis pipelines."""
import argparse


def behav_preprocess():
    """Entry point: behav-preprocess"""
    parser = argparse.ArgumentParser(description='Run behavioral video preprocessing.')
    parser.add_argument('--config', type=str, default='configs/behavior.yaml')
    args = parser.parse_args()
    from fp_behav.behavior.preprocessing import preprocess_behavioral_data
    preprocess_behavioral_data(args.config)


def behav_dlc2boris():
    """Entry point: behav-dlc2boris"""
    parser = argparse.ArgumentParser(description='Convert DLC output to BORIS format.')
    parser.add_argument('--config', type=str, default='configs/behavior.yaml')
    parser.add_argument('--variant', choices=['base', 'ct', 'di'], default='base',
                        help='DLC2BORIS variant to use.')
    args = parser.parse_args()
    if args.variant == 'base':
        from fp_behav.behavior.boris.base import main
    elif args.variant == 'ct':
        from fp_behav.behavior.boris.ct import main
    else:
        from fp_behav.behavior.boris.di import main
    main()
