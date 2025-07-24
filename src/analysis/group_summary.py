"""
Module: group_summary.py

Provides a pipeline for loading time-series data, computing group summaries
(mean & SEM) across multiple signals/events defined in a nested YAML structure,
and generating plots with signal-specific colors.

Functions:
  - load_config: Load nested YAML config.
  - gather_files: Find PKL files per signal/event.
  - compute_group_summary: Compute and save CSV summaries.
  - plot_group_trace: Plot & save group averages with custom color.
  - plot_with_individual: Plot individual traces with mean overlay and custom color.
  - process_analysis: Execute one signal/event analysis.
  - main: Iterate nested signals/events from YAML.

Usage:
  CLI:
    ```bash
    python src/analysis/group_summary.py --config config/group_summary.yaml
    ```

  Jupyter Notebook / IPython:
    ```python
    # 1) Notebook용 매직 커맨드 활성화
    %matplotlib inline

    # 2) 모듈 import
    from src.analysis.group_summary import load_config, process_analysis

    # 3) 설정 불러오기
    cfg = load_config('config/group_summary.yaml')

    # 4) 원하는 signal/event 분석 호출: return_fig=True
    #    예: 첫번째 signal의 첫 이벤트
    sig_conf = cfg['signals'][0]
    fig, ax = process_analysis(sig_conf, cfg, return_fig=True)

    # 5) ax, fig를 통해 세부 속성 변경
    ax.set_title(f"Custom Title: {sig_conf['name']} / {sig_conf['events'][0]}")
    ax.grid(False)
    fig.set_size_inches(10,5)

    # 6) 그림 출력
    fig  # Notebook에서 자동 렌더링

    # 7) 반복 처리
    for sig in cfg['signals']:
        for ev in sig['events']:
            sig_conf = {'name': sig['name'], 'events': [ev]}
            fig, ax = process_analysis(sig_conf, cfg, return_fig=True)
            fig.show()
    ```
  CLI:
    ```bash
    python src/analysis/group_summary.py --config config/group_summary.yaml
    ```

  Notebook:
    ```python
    from src.analysis.group_summary import load_config, main
    # Running all nested analyses
    main('config/group_summary.yaml')
    %matplotlib inline
    from src.analysis.group_summary import load_config, process_analysis
    cfg = load_config('config/group_summary.yaml')
    sig_conf = cfg['signals'][0]
    fig, ax = process_analysis(sig_conf, cfg, return_fig=True)
    ax.set_title(f"Custom: {sig_conf['name']} / {sig_conf['events'][0]}")
    fig
    ```
"""
import os
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FileFunctions
import PlotFunctions

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load YAML config with nested signals/events."""
    logger.info("Loading config: %s", config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.debug("Config: %s", cfg)
    return cfg

def gather_files(root: str, batches: list, session: str,
                 signal: str, event: str, align: str) -> tuple:
    """Collect valid PKL paths & labels for one signal/event."""
    pkl_files, labels = [], []
    for batch in batches:
        batch_dir = os.path.join(root, batch)
        logger.info("Batch: %s", batch_dir)
        if not os.path.isdir(batch_dir):
            logger.warning("Batch directory not found, skipping: %s", batch_dir)
            continue
        for animal in filter(lambda d: d.startswith('G'), os.listdir(batch_dir)):
            p = os.path.join(
                batch_dir, animal, session, 'Epoch_Analysis',
                f'Data_peri_event_signal_individual_{signal}_{event}_{align}.pkl'
            )
            if os.path.exists(p):
                pkl_files.append(p)
                labels.append(animal)
            else:
                logger.warning("PKL file not found for animal %s, skipping: %s", animal, p)
    logger.info("Found %d valid PKLs for %s/%s", len(pkl_files), signal, event)
    return pkl_files, labels

def compute_group_summary(pkl_files: list, group: str,
                          signal: str, event: str, align: str,
                          root: str) -> tuple:
    """Compute mean & SEM, save summary CSV."""
    traces, time, _ = FileFunctions.load_dataframes(pkl_files)
    mean = np.nanmean(traces, axis=0)
    sem = np.nanstd(traces, axis=0)/np.sqrt(traces.shape[0])
    df = pd.DataFrame({'Time': time, 'Mean': mean, 'SEM': sem})
    out_csv = os.path.join(root,
        f'{group}_{signal}_{event}_summary_{align}.csv')
    df.to_csv(out_csv, index=False)
    logger.info("Saved CSV: %s", out_csv)
    return mean, sem, time, traces

def plot_group_trace(time, mean, sem, group,
                     signal, event, align, root,
                    color: str = 'green', return_fig=False):
    """Plot & save average with custom color; opt. return fig, ax."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time, mean, color=color, lw=2, label='Mean')
    ax.fill_between(time, mean - sem, mean + sem,
                    color=color, alpha=0.3, label='±SEM')
    ax.axvline(0, color='slategray', ls='--')
    ax.set(xlabel='Time (s)', ylabel='Z-score',
           title=f"{group}: {signal}/{event}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    
    out_fig = os.path.join(
        root, f'{group}_{signal}_{event}_avg_{align}.png'
    )
    fig.savefig(out_fig, dpi=300)
    logger.info("Saved avg plot for %s/%s with color %s", signal, event, color)
    if return_fig:
        return fig, ax
    plt.close(fig)


def plot_with_individual(time, traces, group,
                         signal, event, align, root,
                         color: str = 'green', return_fig=False):
    """Plot individual traces w/ mean overlay in custom color; opt. return fig, ax."""
    fig, ax = plt.subplots(figsize=(8, 6))
    PlotFunctions.plot_traces_with_mean(
        traces, time,
        ax=ax,
        title=f"{group}: {signal}/{event}",
        color=color,
        xlabel='Time (s)', ylabel='Z-score'
    )
    plt.tight_layout()
    
    out_fig = os.path.join(
        root, f'{group}_{signal}_{event}_indiv_{align}.png'
    )
    fig.savefig(out_fig, dpi=300)
    logger.info("Saved indiv plot for %s/%s with color %s", signal, event, color)
    if return_fig:
        return fig, ax
    plt.close(fig)

def process_analysis(sig_conf: dict, global_cfg: dict,
                     return_fig=False):
    """Handle one signal/event analysis with color mapping."""
    signal = sig_conf['name']
    events = sig_conf.get('events', [])
    align = sig_conf.get('align_point', global_cfg['time2align'])
    color = sig_conf['color'] 
    for event in events:
        pkl_files, _ = gather_files(
            global_cfg['root_folder'], global_cfg['batch_folders'],
            global_cfg['session'], signal, event, align
        )
        mean, sem, time, traces = compute_group_summary(
            pkl_files,
            global_cfg['group_id'], signal, event, align,
            global_cfg['root_folder']
        )
        avg = plot_group_trace(
            time, mean, sem,
            global_cfg['group_id'], signal, event, align,
            global_cfg['root_folder'], color=color,
            return_fig=return_fig
        )
        if sig_conf.get('plot_individual', global_cfg.get('plot_individual', False)):
            plot_with_individual(
                time, traces,
                global_cfg['group_id'], signal, event, align,
                global_cfg['root_folder'], color=color
            )
        if return_fig:
            return avg


def main(config_path: str):
    """Entry: load config and run all signals/events."""
    cfg = load_config(config_path)
    for sig in cfg.get('signals', []):
        process_analysis(sig, cfg)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    main(p.parse_args().config)
