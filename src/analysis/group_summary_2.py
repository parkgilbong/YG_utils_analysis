import os
import argparse
import yaml
import pandas as pd
import numpy as np
import PlotFunctions


def load_config(config_path: str) -> dict:
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
                         groups: list[str],
                         align_point: str = 'onset') -> tuple[list[tuple[pd.Series, pd.Series]], list[pd.Series]]:
    """
    Load time and mean/SEM series for each group.

    Returns:
      xy_pairs: list of (Time, Mean) tuples
      sem_pairs: list of SEM series
    """
    xy_pairs: list[tuple[pd.Series, pd.Series]] = []
    sem_pairs: list[pd.Series] = []
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
                     groups: list[str],
                     align_point: str = 'onset',
                     colors: list[str] | None = None,
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


def main():
    parser = argparse.ArgumentParser(
        description="Multi-group summary plotting via YAML config"
    )
    parser.add_argument(
        '--config', '-c', required=True,
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root_folder = cfg['root_folder']
    align_point = cfg.get('align_point', 'onset')
    save = cfg.get('save', True)

    # Extract group names and colors
    groups = [g['name'] for g in cfg['groups']]
    colors = [g.get('color') for g in cfg['groups']]

    # Collect optional plotting kwargs
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

if __name__ == '__main__':
    main()
