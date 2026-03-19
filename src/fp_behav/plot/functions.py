# plot_functions.py

import numpy as np
import matplotlib.pyplot as plt


def plot_single_line(x, y, fig_size, fig_title, x_label, y_label, x_lim, y_lim, color, save=False, ax=None):
    """
    Create a simple single-line plot.

    Parameters:
    - x (array-like): X-axis values
    - y (array-like): Y-axis values
    - fig_size (tuple): Figure size (width, height)
    - fig_title (str): Title of the figure
    - x_label (str): Label for X-axis
    - y_label (str): Label for Y-axis
    - x_lim (tuple): Limits for X-axis (min, max)
    - y_lim (tuple): Limits for Y-axis (min, max)
    - color (str): Line color
    - save (bool): Save figure as PNG file
    - ax (matplotlib.axes.Axes or None): Axis to draw on

    Example:
        plot_single_line(x, y, (10, 5), "Sine", "Time", "Value", (0, 10), (-1, 1), "blue")
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    plot = ax.plot(x, y, color=color, label=y_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(fig_title)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save and fig is not None:
        safe_title = "".join(c if c.isalnum() else "_" for c in fig_title)
        fig.savefig(f"Plot_{safe_title}.png")

    if fig is not None:
        plt.show()


def plot_dual_line(x1, y1, x2, y2, fig_size=(10, 6), fig_title='', x_label='', y1_label='', y2_label='',
                   x_lim=(), y1_lim=(), y2_lim=(), color1='k', color2='k', save=False,
                   font_size=12, title_size=14, legend_size=10, ax=None):
    """
    Plot a dual-line graph with two Y-axes.

    Parameters:
    - x1, y1: Data for primary Y-axis
    - x2, y2: Data for secondary Y-axis
    - fig_size: Figure size
    - fig_title: Title
    - x_label, y1_label, y2_label: Axis labels
    - x_lim, y1_lim, y2_lim: Axis limits
    - color1, color2: Colors for each line
    - save: Save image
    - font_size, title_size, legend_size: Font sizes
    - ax: Optional axis

    Example:
        plot_dual_line(x, sinx, x, cosx, fig_title="Dual", y1_label="sin", y2_label="cos")
    """
    fig = None
    if ax is None:
        fig, ax1 = plt.subplots(figsize=fig_size)
    else:
        ax1 = ax

    plot1 = ax1.plot(x1, y1, color=color1, label=y1_label)
    ax2 = ax1.twinx()
    plot2 = ax2.plot(x2, y2, color=color2, label=y2_label)

    ax1.set_xlim(x_lim)
    ax1.set_ylim(y1_lim)
    ax2.set_ylim(y2_lim)

    ax1.set_xlabel(x_label, fontsize=font_size)
    ax1.set_ylabel(y1_label, color=color1, fontsize=font_size)
    ax2.set_ylabel(y2_label, color=color2, fontsize=font_size)
    ax1.set_title(fig_title, fontsize=title_size)

    lines = plot1 + plot2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=legend_size)

    if save and fig is not None:
        safe_title = "".join(c if c.isalnum() else "_" for c in fig_title)
        fig.savefig(f"Plot_{safe_title}.png")

    if fig is not None:
        plt.show()


def plot_traces_with_mean(trace_array, trace_time, ax=None, color='green', title='', xlabel='', ylabel='',
                          mode='std'):
    """
    Plot traces with individual trials and overlay the mean ± standard error or standard deviation.

    Parameters:
    - trace_array (ndarray): shape (n_trials, n_samples), may include NaNs
    - trace_time (ndarray): shape (n_samples,), time axis
    - ax: Axis to plot on (optional)
    - color: Mean line color
    - title, xlabel, ylabel: Labels
    - mode (str): One of 'mean_only', 'std', 'sem'

    Example:
        plot_traces_with_mean(data, time, color='red', mode='mean_only')
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    if mode != 'mean_only':
        for i, trace in enumerate(trace_array):
            if np.isnan(trace).any():
                nan_indices = np.where(np.isnan(trace))[0]
                print(f"Trace {i} contains NaN at indices: {nan_indices}")
            ax.plot(trace_time, trace, color='gray', alpha=0.1, linewidth=1.5)

    mean_trace = np.nanmean(trace_array, axis=0)
    std_trace = np.nanstd(trace_array, axis=0)
    sem_trace = std_trace / np.sqrt(trace_array.shape[0])

    ax.plot(trace_time, mean_trace, color=color, linewidth=2.5, label='Mean')

    if mode == 'std':
        ax.fill_between(trace_time, mean_trace - std_trace, mean_trace + std_trace,
                        facecolor=color, alpha=0.2, label='± SD')
    elif mode == 'sem':
        ax.fill_between(trace_time, mean_trace - sem_trace, mean_trace + sem_trace,
                        facecolor=color, alpha=0.2, label='± SEM')

    ax.axvline(0, color='slategray', linestyle='--', linewidth=1.5, label='Onset')

    xticks = np.arange(np.min(trace_time), np.max(trace_time), 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.astype(int))
    ax.set_xlim(np.min(trace_time), np.max(trace_time))
    ax.set_ylim(np.nanmin(trace_array), np.nanmax(trace_array))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)

    if fig is not None:
        plt.show()



def plot_trace_heatmap(traces, trace_time, vmin=None, vmax=None, ax=None, title='', xlabel='', ylabel='', cmap='viridis'):
    """
    Plot a heatmap of aligned traces with NaN support.

    Parameters:
    - traces (ndarray): shape (n_trials, n_samples)
    - trace_time (ndarray): X-axis time vector
    - vmin, vmax: color scale limits
    - ax: Optional subplot axis
    - title, xlabel, ylabel: Labels
    - cmap: Color map name

    Returns:
    - im: Heatmap image object

    Example:
        plot_trace_heatmap(data, time, cmap='hot')
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    masked_array = np.ma.masked_invalid(traces)
    im = ax.imshow(
        masked_array,
        cmap=cmap,
        aspect='auto',
        interpolation='none',
        extent=[trace_time[0], trace_time[-1], len(traces), 0],
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if fig is not None:
        plt.show()

    return im


def plot_multi_line(xy_pairs, sem_pairs=None, fig_size=(10, 6), title='', x_label='',
                    y_label=None, y_labels=None, x_lim=None, y_lim=None,
                    colors=None, line_styles=None, save=False, font_size=12,
                    title_size=14, legend_size=10, legend_loc='upper left',
                    axvline=None, axhline=None, ax=None):
    """
    Plot multiple line graphs with optional SEM shading.

    Parameters:
    - xy_pairs: List of (x, y) tuples
    - sem_pairs: List of SEM arrays (optional)
    - fig_size: (width, height)
    - title, x_label, y_label: Labels
    - y_labels: List of line labels
    - x_lim, y_lim: Axis limits
    - colors: List of line colors
    - line_styles: List of line styles
    - save: Whether to save to file
    - font_size, title_size, legend_size: Font sizes
    - legend_loc: Legend location
    - axvline, axhline: Optional lines
    - ax: Optional plot axis

    Example:
        plot_multi_line([(x, y1), (x, y2)], y_labels=['Line A', 'Line B'])
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    n = len(xy_pairs)
    if n == 0:
        raise ValueError("xy_pairs must contain at least one (x,y) pair")

    y_labels = y_labels or [f'Line {i + 1}' for i in range(n)]
    colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color'][:n]
    line_styles = line_styles or ['-'] * n

    lines = []

    for i, (x, y) in enumerate(xy_pairs):
        if sem_pairs is not None and i < len(sem_pairs) and sem_pairs[i] is not None:
            sem = sem_pairs[i]
            ax.fill_between(x, y - sem, y + sem, color=colors[i], alpha=0.1)
        line = ax.plot(x, y, color=colors[i], linestyle=line_styles[i], label=y_labels[i])
        lines += line

    ax.set_xlabel(x_label, fontsize=font_size)
    if y_label:
        ax.set_ylabel(y_label, fontsize=font_size)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if axvline is not None:
        ax.axvline(x=axvline, color='gray', linestyle='--', linewidth=1)
    if axhline is not None:
        ax.axhline(y=axhline, color='gray', linestyle='--', linewidth=1)

    ax.set_title(title, fontsize=title_size)
    ax.legend(fontsize=legend_size, loc=legend_loc)

    if save and fig is not None:
        safe_title = "".join(c if c.isalnum() else "_" for c in title)
        fig.savefig(f"Plot_{safe_title}.png", bbox_inches='tight')

    if fig is not None:
        plt.show()
