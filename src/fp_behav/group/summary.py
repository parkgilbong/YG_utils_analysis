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
from fp_behav.io import files as FileFunctions
from fp_behav.plot import functions as PlotFunctions

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


# ---------------------------------------------------------------------------
# Multi-group plotting (merged from group_summary_2.py)
# ---------------------------------------------------------------------------

def load_multi_group_config(config_path: str) -> dict:
    """
    Load YAML configuration for multi-group plotting.

    Expected YAML structure:
      root_folder: str
      align_point: str
      save: bool
      title: str (optional)
      x_label: str (optional)
      y_label: str (optional)
      legend_loc: str (optional)
      axvline: float (optional)
      axhline: float (optional)
      groups:
        - name: str
          color: str (hex or name)
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_group_summaries(root_folder: str,
                         groups: list,
                         align_point: str = 'onset'):
    """
    Load time and mean/SEM series for each group.

    Returns:
      xy_pairs: list of (Time, Mean) tuples
      sem_pairs: list of SEM series
    """
    import pandas as pd
    xy_pairs = []
    sem_pairs = []
    for grp in groups:
        filepath = os.path.join(
            root_folder,
            f"{grp}_summary_{align_point}.csv"
        )
        df = pd.read_csv(filepath)
        xy_pairs.append((df['Time'], df['Mean']))
        sem_pairs.append(df['SEM'])
    return xy_pairs, sem_pairs


def plot_multi_group(root_folder: str,
                     groups: list,
                     align_point: str = 'onset',
                     colors=None,
                     save: bool = True,
                     **kwargs) -> None:
    """
    Plot multiple group summaries with optional SEM shading.

    Parameters:
      root_folder: base folder containing summary CSVs
      groups: list of group names
      align_point: event alignment key
      colors: list of colors matching groups
      save: whether to save figure
      **kwargs: forwarded to PlotFunctions.plot_multi_line
    """
    xy_pairs, sem_pairs = load_group_summaries(root_folder, groups, align_point)
    y_labels = [f"{grp}" for grp in groups]
    PlotFunctions.plot_multi_line(
        xy_pairs,
        sem_pairs=sem_pairs,
        fig_size=kwargs.get('fig_size', (8, 6)),
        title=kwargs.get('title', f"Ca2++ responses to {align_point}"),
        x_label=kwargs.get('x_label', 'Time (sec)'),
        y_label=kwargs.get('y_label', 'Z-score'),
        y_labels=y_labels,
        x_lim=kwargs.get('x_lim', None),
        y_lim=kwargs.get('y_lim', None),
        colors=colors,
        line_styles=kwargs.get('line_styles', None),
        save=save,
        font_size=kwargs.get('font_size', 12),
        title_size=kwargs.get('title_size', 14),
        legend_size=kwargs.get('legend_size', 10),
        legend_loc=kwargs.get('legend_loc', 'upper right'),
        axvline=kwargs.get('axvline', 0),
        axhline=kwargs.get('axhline', 0)
    )


def main_multi_group():
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-group summary plotting via YAML config"
    )
    parser.add_argument(
        '--config', '-c', required=True,
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()

    cfg = load_multi_group_config(args.config)
    root_folder = cfg['root_folder']
    align_point = cfg.get('align_point', 'onset')
    save = cfg.get('save', True)

    groups = [g['name'] for g in cfg['groups']]
    colors = [g.get('color') for g in cfg['groups']]

    plot_kwargs = {}
    for key in ['title', 'x_label', 'y_label', 'legend_loc', 'axvline', 'axhline']:
        if key in cfg:
            plot_kwargs[key] = cfg[key]

    plot_multi_group(
        root_folder=root_folder,
        groups=groups,
        align_point=align_point,
        colors=colors,
        save=save,
        **plot_kwargs
    )
