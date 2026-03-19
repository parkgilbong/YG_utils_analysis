#src/analysis/Epoch_analysis.py
"""
Module: epoch_analysis

This module performs epoch-based analysis on fiber photometry data.
It loads parameters from a YAML config, sets up logging, and processes
signal traces aligned to manual scoring events.

Usage (CLI):
    $ cd <project_root>
    $ python -m src.analysis.epoch_analysis

Usage (Notebook import):
    from src.analysis.epoch_analysis import load_config, setup_logging, run_epoch_analysis
    # Load configuration
    cfg = load_config('configs/epoch_analysis.yaml')
    # Initialize logging (optional)
    setup_logging(cfg.get('logging', {}), '<project_root>')
    # Execute analysis
    run_epoch_analysis(cfg, '<project_root>')
"""
import os
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FPFunctions
import PlotFunctions
import FileFunctions


def load_config(path: str) -> dict:
    """
    Load parameters from a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary of configuration settings.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_conf: dict, project_root: str) -> None:
    """
    Configure file-based logging.

    This will create the log directory if it does not exist.

    Args:
        log_conf: Dictionary with 'log_file' and 'level' keys.
        project_root: Base directory to resolve relative log paths.
    """
    log_file = log_conf.get('log_file')
    if not log_file:
        return
    log_path = os.path.join(project_root, log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=getattr(logging, log_conf.get('level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging initialized.")


def run_epoch_analysis(cfg: dict, project_root: str) -> None:
    """
    Execute epoch-based signal extraction, statistic computation, plotting,
    and saving of results for each animal and session.

    Args:
        cfg: Configuration dictionary from load_config().
        project_root: Base directory for resolving paths.
    """
    # Unpack parameters
    root_folder = cfg['root_folder']
    batch_folder = cfg['batch_folder']
    group_list = cfg['group_list']
    session_list = cfg['session_list']
    signal2use = cfg['Signal2Use']
    behavior2use = cfg['Behavior2Use'] # Now expected to be a list
    fps_bout = cfg['FPS']
    pre_win = cfg['pre_window_length']
    post_win = cfg['post_window_length']
    ylim_bottom = cfg['ylim_bottom']
    ylim_top = cfg['ylim_top']
    save_plots = cfg['SavePlots']
    save_data = cfg['SaveData']
    saveAscsv = cfg['SaveAscsv']
    
    to_clip = cfg['ToClip']
    align_point = cfg['Point2align']

    if group_list:
        for group in group_list:
            animal_list = FileFunctions.Grab_folder_names_in_folder(
                os.path.join(root_folder, batch_folder, group)
            )
            for animal in animal_list:
                for session in session_list:
                    try:
                        base = os.path.join(root_folder, batch_folder, group, animal, session)
                        os.chdir(base)
                        logging.info(f"Processing: {group}/{animal}/{session}")

                        # Load preprocessed signal
                        gcamp_df = pd.read_pickle(os.path.join(base, 'Preprocessing', 'Final_table_raw_trace.pkl'))
                        time_sec = gcamp_df['time'].to_numpy()
                        for signalinuse in signal2use:   
                            signal = gcamp_df[signalinuse].to_numpy()

                            for behavior in behavior2use:
                                # Load manual scoring bouts
                                scoring_file = os.path.join(base, 'Epoch_Analysis', 'Manual_scoring.tsv')
                                bouts = FPFunctions.Import_manual_scoring(
                                    file_path=scoring_file,
                                    Event=behavior,
                                    FPS=fps_bout
                                )
                                if bouts is None or len(bouts) == 0:
                                    logging.warning(f"No {behavior} epochs found for {group}/{animal}/{session}. Skipping.")
                                    continue

                                # Extract epoch-aligned traces
                                traces, trace_time = FPFunctions.extract_traces_with_padding(
                                    signal=signal,
                                    time=time_sec,
                                    time_tuples=bouts,
                                    pre_window_sec=pre_win,
                                    post_window_sec=post_win,
                                    FPS=fps_bout,
                                    align_to=align_point
                                )
                                arr = np.array(traces)

                                # Compute mean and std
                                mean_vals = np.nanmean(arr, axis=0)
                                std_vals = np.nanstd(arr, axis=0)

                                # Ensure mean_vals and std_vals are always 1D arrays
                                if np.isscalar(mean_vals):
                                    mean_vals = np.array([mean_vals])
                                if np.isscalar(std_vals):
                                    std_vals = np.array([std_vals])
                                length = min(len(mean_vals), len(std_vals))
                                x_data = (np.linspace(0, pre_win + post_win, length) - pre_win)
                                df_mean = pd.DataFrame({
                                    'Time': x_data,
                                    'Mean': mean_vals[:length],
                                    'STD': std_vals[:length]
                                })

                                # Collate all traces
                                all_df = pd.DataFrame(arr.T)
                                all_df.insert(0, 'time', trace_time)

                                # Plot traces and heatmap
                                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
                                PlotFunctions.plot_traces_with_mean(
                                    arr, trace_time,
                                    title=f"{animal} - {session} {behavior}",
                                    color='green', ax=ax1,
                                    xlabel='Time (sec)', ylabel=f'{signalinuse}',
                                    mode = 'std'
                                )
                                im = PlotFunctions.plot_trace_heatmap(
                                    arr,
                                    trace_time,
                                    vmin=ylim_bottom,
                                    vmax=ylim_top,
                                    ax=ax2,
                                    xlabel='Time (sec)',
                                    ylabel='Epoch Number',
                                    cmap='viridis'
                                )
                                fig.colorbar(im, ax=ax2, orientation='horizontal')
                                plt.tight_layout()
                                
                                # Save results
                                analysis_dir = os.path.join(base, 'Epoch_Analysis')
                                os.makedirs(analysis_dir, exist_ok=True)
                                if save_plots:
                                    out_png = os.path.join(
                                        analysis_dir,
                                        f"Plot_Epoch_Analysis_{signalinuse}_{behavior}_{align_point}.png"
                                    )
                                    fig.savefig(out_png)
                                if to_clip:
                                    df_mean.to_clipboard(sep='	', index=False, header=True)
                                if save_data:
                                    all_df.to_pickle(
                                        os.path.join(
                                            analysis_dir,
                                            f"Data_peri_event_signal_individual_{signalinuse}_{behavior}_{align_point}.pkl"
                                        )
                                    )
                                    if saveAscsv:
                                        all_df.to_csv(
                                            os.path.join(
                                                analysis_dir,
                                                f"Data_peri_event_signal_individual_{signalinuse}_{behavior}_{align_point}.csv"
                                            )
                                        )
                                    df_mean.to_csv(
                                        os.path.join(analysis_dir, f"Data_peri_event_signal_averaged_{signalinuse}_{behavior}_{align_point}.csv"),
                                        index=False
                                    )
                                plt.show()
                                plt.close(fig)

                    except Exception as e:
                        logging.error(
                            f"Failed {group}/{animal}/{session}: {e}",
                            exc_info=True
                        )
    else:
        # If group_list is empty, skip group loop and use batch_folder directly
        animal_list = FileFunctions.Grab_folder_names_in_folder(
            os.path.join(root_folder, batch_folder)
        )
        for animal in animal_list:
            for session in session_list:
                try:
                    base = os.path.join(root_folder, batch_folder, animal, session)
                    os.chdir(base)
                    logging.info(f"Processing: {animal}/{session}")

                    # Load preprocessed signal
                    gcamp_df = pd.read_pickle(
                        os.path.join(base, 'Preprocessing', 'Final_table_raw_trace.pkl')
                    )
                    time_sec = gcamp_df['time'].to_numpy()
                    for signalinuse in signal2use:
                        signal = gcamp_df[signalinuse].to_numpy()

                        for behavior in behavior2use:
                            # Load manual scoring bouts
                            scoring_file = os.path.join(base, 'Epoch_Analysis', 'Manual_scoring.tsv')
                            bouts = FPFunctions.Import_manual_scoring(
                                file_path=scoring_file,
                                Event=behavior,
                                FPS=fps_bout
                            )
                            if bouts is None or len(bouts) == 0:
                                logging.warning(f"No {behavior} epochs found for {animal}/{session}. Skipping.")
                                continue

                            # Extract epoch-aligned traces
                            traces, trace_time = FPFunctions.extract_traces_with_padding(
                                signal=signal,
                                time=time_sec,
                                time_tuples=bouts,
                                pre_window_sec=pre_win,
                                post_window_sec=post_win,
                                FPS=fps_bout,
                                align_to=align_point
                            )
                            arr = np.array(traces)

                            # Compute mean and std
                            mean_vals = np.nanmean(arr, axis=0)
                            std_vals = np.nanstd(arr, axis=0)

                            # Ensure mean_vals and std_vals are always 1D arrays
                            if np.isscalar(mean_vals):
                                mean_vals = np.array([mean_vals])
                            if np.isscalar(std_vals):
                                std_vals = np.array([std_vals])

                            length = min(len(mean_vals), len(std_vals))
                            x_data = (np.linspace(0, pre_win + post_win, length) - pre_win)
                            df_mean = pd.DataFrame({
                                'Time': x_data,
                                'Mean': mean_vals[:length],
                                'STD': std_vals[:length]
                            })

                            # Collate all traces
                            all_df = pd.DataFrame(arr.T)
                            all_df.insert(0, 'time', trace_time)

                            # Plot traces and heatmap
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
                            PlotFunctions.plot_traces_with_mean(
                                arr, trace_time,
                                title=f"{animal} - {session} {behavior}",
                                color='green', ax=ax1,
                                xlabel='Time (sec)', ylabel=f'{signalinuse}'
                            )
                            im = PlotFunctions.plot_trace_heatmap(
                                arr,
                                trace_time,
                                vmin=ylim_bottom,
                                vmax=ylim_top,
                                ax=ax2,
                                xlabel='Time (sec)',
                                ylabel='Epoch Number',
                                cmap='viridis'
                            )
                            fig.colorbar(im, ax=ax2, orientation='horizontal')
                            plt.tight_layout()
                            
                            # Save results
                            analysis_dir = os.path.join(base, 'Epoch_Analysis')
                            os.makedirs(analysis_dir, exist_ok=True)
                            if save_plots:
                                out_png = os.path.join(
                                    analysis_dir,
                                    f"Plot_Epoch_Analysis_{signalinuse}_{behavior}_{align_point}.png"
                                )
                                fig.savefig(out_png)
                            if to_clip:
                                df_mean.to_clipboard(sep='	', index=False, header=True)
                            if save_data:
                                    all_df.to_pickle(
                                        os.path.join(
                                            analysis_dir,
                                            f"Data_peri_event_signal_individual_{signalinuse}_{behavior}_{align_point}.pkl"
                                        )
                                    )
                                    if saveAscsv:
                                        all_df.to_csv(
                                            os.path.join(
                                                analysis_dir,
                                                f"Data_peri_event_signal_individual_{signalinuse}_{behavior}_{align_point}.csv"
                                            )
                                        )
                                    df_mean.to_csv(
                                        os.path.join(analysis_dir, f"Data_peri_event_signal_averaged_{signalinuse}_{behavior}_{align_point}.csv"),
                                        index=False
                                    )
                            plt.show()
                            plt.close(fig)

                except Exception as e:
                    logging.error(
                        f"Failed {animal}/{session}: {e}",
                        exc_info=True
                    )

if __name__ == '__main__':
    # Determine project paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    config_path = os.path.join(project_root, 'configs', 'epoch_analysis.yaml')

    # Load config and run
    cfg = load_config(config_path)
    setup_logging(cfg.get('logging', {}), project_root)
    run_epoch_analysis(cfg, project_root)