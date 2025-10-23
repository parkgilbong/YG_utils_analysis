# FPFunctions Cheatsheet

Quick reference for common Fiber Photometry data processing tasks.

## 🔥 Top 5 Most Common Use Cases

### 1. Preprocess Single-Channel FP Data (TDT System)

```python
from utils.FPFunctions import FP_preprocessing_1ch

# Basic preprocessing with exponential fit detrending
FP_preprocessing_1ch(
    Tank_path='/path/to/TDT/tank',
    Dest_folder='/path/to/output',
    sys='tdt',
    Detrending_method='Exp_fit',
    Use_CamTick=True,
    FPS=25,
    Rec_duration=600
)
```

**What it does:** Loads raw FP data, detrends using exponential fit, normalizes (dF/F), and saves processed traces.

---

### 2. Preprocess Two-Channel FP Data (465nm & 560nm)

```python
from utils.FPFunctions import FP_preprocessing_2ch_new

# Dual-channel preprocessing
FP_preprocessing_2ch_new(
    Tank_path='/path/to/TDT/tank',
    Dest_folder='/path/to/output',
    Detrending_method='Exp_fit',
    Use_CamTick=True,
    FPS=25,
    Rec_duration=600,
    Namefor465='465',
    Namefor560='560',
    SaveAsCSV=True
)
```

**What it does:** Processes both GCaMP (465nm) and control (560nm) channels simultaneously.

---

### 3. Event-Centered Epoch Analysis

```python
from utils.FPFunctions import Epoch_Analysis_3EVT

# Extract signal around behavioral events
Epoch_Analysis_3EVT(
    pkl_path='Final_table_raw_trace.pkl',
    destfolder='/path/to/output',
    REF_EPOC='Approach',  # Event name from manual scoring
    Pre_samp=5.0,         # Seconds before event
    Post_samp=10.0,       # Seconds after event
    FPS=25,
    baseline_start=-5.0,
    baseline_end=-1.0,
    SaveData=True
)
```

**What it does:** Extracts signal traces aligned to specific behavioral events with baseline correction.

---

### 4. Peak Detection in FP Signals

```python
from utils.FPFunctions import Peak_Analysis

# Detect calcium transients
Peak_Analysis(
    pkl_path='Final_table_raw_trace.pkl',
    destfolder='/path/to/output',
    FPS=25,
    height=1.3,           # Z-score threshold
    min_interval=1.0,     # Minimum time between peaks (sec)
    min_peak_width=0.2,   # Minimum peak width (sec)
    SaveData=True
)
```

**What it does:** Identifies significant calcium transients based on height and temporal constraints.

---

### 5. Import Manual Behavioral Scoring

```python
from utils.FPFunctions import Import_manual_scoring

# Load BORIS or similar manual scoring
events = Import_manual_scoring(
    file_path='/path/to/scoring.csv',
    FPS=25,
    Event='Social_Contact',
    UseFilter=True,
    MinDuration=0.5,      # Min event duration (sec)
    MinInterval=2.0       # Min interval between events (sec)
)
```

**What it does:** Imports behavioral annotations, filters by duration/interval, returns event onset/offset times.

---

## 📊 Additional Useful Functions

### Calculate Area Under Curve (AUC)

```python
from utils.FPFunctions import calculate_auc

# Calculate AUC for specific time intervals
auc_values = calculate_auc(
    time=time_array,
    signal=dff_trace,
    intervals=[(10, 20), (30, 40)]  # Time intervals
)
```

### Extract Traces with Padding

```python
from utils.FPFunctions import extract_traces_with_padding

# Extract signal around specific timepoints
traces = extract_traces_with_padding(
    signal=dff_trace,
    time=time_array,
    time_tuples=[(10.5, 15.0), (25.0, 30.0)],
    pre_window_sec=2.0,
    post_window_sec=5.0,
    FPS=25,
    align_to='onset'
)
```

### Detect Slow Calcium Peaks

```python
from utils.FPFunctions import detect_slow_peaks

# Specialized peak detection for slow signals
peaks = detect_slow_peaks(
    signal=dff_trace,
    sampling_rate=25,
    height=1.3,
    min_interval=2.0,
    min_peak_width=0.5
)
```

---

## 💡 Tips & Best Practices

1. **Always check your sampling rate**: FP systems typically record at 1017 Hz, but behavioral alignment uses camera FPS (usually 25 Hz).

2. **Detrending method choice**:
   - `'Exp_fit'`: Best for signals with exponential decay (most common)
   - `'Highpass_filter'`: For signals with linear drift

3. **Baseline correction**: For epoch analysis, use -5 to -1 seconds before event as baseline for stable dF/F calculation.

4. **Save intermediate files**: Use `SaveAsCSV=True` to keep intermediate processing steps for QC.

5. **Event filtering**: When importing manual scoring, filter out very brief events (< 0.5s) to avoid noise.

---

## 🔗 Related

- See [PlotFunctions Cheatsheet](cheatsheet_PlotFunctions.md) for visualization
- See [FileFunctions Cheatsheet](cheatsheet_FileFunctions.md) for file operations
