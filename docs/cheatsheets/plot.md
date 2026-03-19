# PlotFunctions Cheatsheet

Quick reference for common plotting and visualization tasks.

## 🎨 Top 5 Most Common Use Cases

### 1. Plot Single Time Series

```python
from utils.PlotFunctions import plot_single_line
import numpy as np

# Simple line plot
time = np.linspace(0, 10, 1000)
signal = np.sin(time)

plot_single_line(
    x=time,
    y=signal,
    fig_size=(12, 4),
    fig_title='Calcium Signal',
    x_label='Time (s)',
    y_label='dF/F (z-score)',
    x_lim=(0, 10),
    y_lim=(-3, 3),
    color='green',
    save=True
)
```

**What it does:** Creates a clean, publication-ready single line plot with customizable axes and labels.

---

### 2. Plot Dual Signals with Two Y-Axes

```python
from utils.PlotFunctions import plot_dual_line

# Overlay two signals with different scales
plot_dual_line(
    x1=time, y1=calcium_signal,
    x2=time, y2=behavior_score,
    fig_size=(14, 6),
    fig_title='FP Signal with Behavior',
    x_label='Time (s)',
    y1_label='dF/F',
    y2_label='Social Index',
    x_lim=(0, 600),
    y1_lim=(-2, 4),
    y2_lim=(0, 1),
    color1='green',
    color2='orange',
    save=True
)
```

**What it does:** Overlays two time series with independent y-axes, perfect for FP + behavior.

---

### 3. Plot Multiple Traces with Mean ± SEM

```python
from utils.PlotFunctions import plot_traces_with_mean
import matplotlib.pyplot as plt

# Plot event-aligned traces from multiple trials
fig, ax = plt.subplots(figsize=(10, 6))

plot_traces_with_mean(
    trace_array=epoch_traces,  # Shape: (n_trials, n_timepoints)
    trace_time=time_vector,
    ax=ax,
    color='blue',
    title='Peri-Event Traces',
    xlabel='Time from Event (s)',
    ylabel='dF/F (z-score)',
    mode='sem'  # Use 'std' for standard deviation
)

plt.axvline(0, color='red', linestyle='--', label='Event Onset')
plt.legend()
plt.tight_layout()
plt.savefig('epoch_analysis.png', dpi=300)
plt.show()
```

**What it does:** Visualizes multiple trials with mean and shaded error region (SEM or SD).

---

### 4. Create Heatmap of Trial-by-Trial Activity

```python
from utils.PlotFunctions import plot_trace_heatmap
import matplotlib.pyplot as plt

# Heatmap showing all trials
fig, ax = plt.subplots(figsize=(10, 8))

plot_trace_heatmap(
    traces=epoch_traces,  # Shape: (n_trials, n_timepoints)
    trace_time=time_vector,
    vmin=-2,
    vmax=3,
    ax=ax,
    title='Trial-by-Trial Calcium Activity',
    xlabel='Time from Event (s)',
    ylabel='Trial #',
    cmap='RdYlBu_r'
)

plt.axvline(0, color='white', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300)
plt.show()
```

**What it does:** Creates a heatmap showing individual trial responses over time.

---

### 5. Plot Multiple Groups for Comparison

```python
from utils.PlotFunctions import plot_multi_line

# Compare different experimental groups
xy_pairs = [
    (time, control_mean, 'Control'),
    (time, drug_mean, 'Drug'),
    (time, stress_mean, 'Stress')
]

sem_pairs = [
    (time, control_sem),
    (time, drug_sem),
    (time, stress_sem)
]

plot_multi_line(
    xy_pairs=xy_pairs,
    sem_pairs=sem_pairs,
    fig_size=(12, 6),
    title='Group Comparison',
    x_label='Time (s)',
    y_label='dF/F',
    colors=['gray', 'red', 'blue'],
    save=True
)
```

**What it does:** Overlays multiple group averages with error bands for comparison.

---

## 🎯 Advanced Plotting Scenarios

### Create Multi-Panel Figure

```python
import matplotlib.pyplot as plt
from utils.PlotFunctions import plot_single_line, plot_traces_with_mean

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Raw trace
plot_single_line(
    x=time, y=raw_trace,
    fig_size=None,  # Don't create new figure
    fig_title='A. Raw Signal',
    x_label='Time (s)', y_label='Fluorescence',
    x_lim=(0, 100), y_lim=None,
    color='black',
    ax=axes[0, 0]
)

# Panel B: Processed trace
plot_single_line(
    x=time, y=dff_trace,
    fig_size=None,
    fig_title='B. dF/F Signal',
    x_label='Time (s)', y_label='dF/F',
    x_lim=(0, 100), y_lim=None,
    color='green',
    ax=axes[0, 1]
)

# Panel C: Epoch averages
plot_traces_with_mean(
    trace_array=epoch_traces,
    trace_time=epoch_time,
    ax=axes[1, 0],
    title='C. Peri-Event Average',
    xlabel='Time (s)', ylabel='dF/F'
)

# Panel D: Heatmap
from utils.PlotFunctions import plot_trace_heatmap
plot_trace_heatmap(
    traces=epoch_traces,
    trace_time=epoch_time,
    ax=axes[1, 1],
    title='D. Trial Heatmap'
)

plt.tight_layout()
plt.savefig('figure_panel.png', dpi=300)
plt.show()
```

---

## 💡 Tips & Best Practices

1. **Figure sizes**: Use (12, 4) for time series, (10, 6) for comparisons, (10, 8) for heatmaps.

2. **Color choices**:
   - Green/teal: Calcium signals (GCaMP)
   - Blue: Control or isosbestic
   - Red/orange: Events or secondary signals
   - Gray: Control groups

3. **DPI settings**: Use 300 DPI for publications, 150 for presentations, 72 for quick checks.

4. **Axis limits**: Let matplotlib auto-scale first, then adjust if needed for clarity.

5. **Event markers**: Always mark important events (t=0, stimulation) with vertical lines.

6. **Error bands**: Use SEM for group comparisons, STD for individual variability.

---

## 🔗 Related

- See [FPFunctions Cheatsheet](cheatsheet_FPFunctions.md) for data processing
- See [VideoFunctions Cheatsheet](cheatsheet_VideoFunctions.md) for video visualization
