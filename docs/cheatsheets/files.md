# FileFunctions Cheatsheet

Quick reference for file operations, configuration management, and directory utilities.

## 📁 Top 5 Most Common Use Cases

### 1. Set Working Directory with Auto-Creation

```python
from utils.FileFunctions import set_working_directory

# Create nested directory structure and set as working dir
output_path = set_working_directory(
    base_folder='/path/to/project',
    'results',
    'experiment_2024',
    'group_A'
)
# Creates: /path/to/project/results/experiment_2024/group_A
# Sets it as current working directory
print(f"Now working in: {output_path}")
```

**What it does:** Creates nested directories if they don't exist and changes to that directory.

---

### 2. Find All Files with Specific Extension

```python
from utils.FileFunctions import grab_files

# Find all CSV files in a directory
csv_files = grab_files(
    folder_path='/path/to/data',
    ext='.csv',
    recursive=False  # Only current directory
)

# Find all pickle files recursively
pkl_files = grab_files(
    folder_path='/path/to/data',
    ext='.pkl',
    recursive=True  # Include subdirectories
)

print(f"Found {len(csv_files)} CSV files")
for file in csv_files:
    print(file)
```

**What it does:** Retrieves all files with a specific extension, optionally searching subdirectories.

---

### 3. Load Configuration from YAML

```python
from utils.FileFunctions import load_config

# Load analysis parameters from YAML
config = load_config('/path/to/config.yaml')

# Access configuration values
tank_path = config['tank_path']
fps = config['fps']
rec_duration = config['rec_duration']

print(f"FPS: {fps}, Duration: {rec_duration}s")
```

**What it does:** Loads YAML configuration files into Python dictionaries.

---

### 4. Load Multiple Data Files

```python
from utils.FileFunctions import load_dataframes

# Load multiple pickle files at once
file_list = [
    '/path/to/mouse1_data.pkl',
    '/path/to/mouse2_data.pkl',
    '/path/to/mouse3_data.pkl'
]

traces, time_vector, labels = load_dataframes(
    file_list=file_list,
    file_type='pickle',
    trace_start_idx=1  # Column 0 is time, traces start at column 1
)

# traces: combined array of all traces
# time_vector: shared time axis
# labels: labels for each trace
print(f"Loaded {traces.shape[0]} traces")
```

**What it does:** Batch loads multiple data files and combines them for group analysis.

---

### 5. Temporary Directory Change

```python
from utils.FileFunctions import temp_chdir

# Temporarily work in a different directory
with temp_chdir('/path/to/temporary/location'):
    # All file operations here use the temporary directory
    with open('temp_file.txt', 'w') as f:
        f.write('Temporary data')
    
    # Process files in this directory
    process_data()

# Automatically returns to original directory
print("Back to original directory")
```

**What it does:** Context manager for temporarily changing directories safely.

---

## 🔧 Advanced File Operations

### Get Folder List

```python
from utils.FileFunctions import grab_folders

# Get all subdirectories (full paths)
folders = grab_folders(
    folder_path='/path/to/data',
    recursive=False,
    names_only=False
)

# Get only folder names (not full paths)
folder_names = grab_folders(
    folder_path='/path/to/data',
    recursive=False,
    names_only=True
)

# Recursively find all subdirectories
all_folders = grab_folders(
    folder_path='/path/to/data',
    recursive=True,
    names_only=False
)
```

---

### Extract Parent and Filename

```python
from utils.FileFunctions import get_dirname_and_basename

# Parse file path
file_path = '/experiments/group_A/mouse_001/data.pkl'
path_info = get_dirname_and_basename(file_path)

print(f"Parent folder: {path_info.parent}")  # group_A
print(f"File name: {path_info.stem}")  # data
```

**What it does:** Extracts parent directory name and file basename (without extension).

---

### Ensure Directory Exists

```python
from utils.FileFunctions import ensure_dir

# Create directory if it doesn't exist (no error if exists)
output_dir = ensure_dir('/path/to/output/directory')

# Now safe to save files
import pandas as pd
df = pd.DataFrame({'data': [1, 2, 3]})
df.to_csv(f"{output_dir}/results.csv")
```

---

### Save Config Copy with Results

```python
from utils.FileFunctions import save_config_copy, load_config
from pathlib import Path

# Load config
config = load_config('config.yaml')

# Run analysis
# ... your analysis code ...

# Save copy of config used
output_dir = Path('/path/to/results')
save_config_copy(config, output_dir)
# Creates: /path/to/results/config_used.yaml
```

**What it does:** Archives the exact configuration used for reproducibility.

---

### Load YAML Safely

```python
from utils.FileFunctions import load_yaml

# Load YAML with error handling
try:
    data = load_yaml('/path/to/config.yaml')
except FileNotFoundError:
    print("Config file not found")
except Exception as e:
    print(f"Error loading YAML: {e}")
```

---

## 💡 Tips & Best Practices

1. **Directory organization**:
   ```
   project/
   ├── raw_data/
   ├── processed/
   │   ├── preprocessing/
   │   └── analysis/
   ├── figures/
   └── configs/
   ```

2. **Configuration files**:
   - Use YAML for all analysis parameters
   - Version control your configs
   - Save a copy with each analysis output

3. **File naming conventions**:
   ```python
   # Good naming
   mouse001_session01_20240115.pkl
   group_control_summary.csv
   
   # Avoid
   data.pkl
   results_final_final2.csv
   ```

4. **Batch processing**:
   ```python
   # Process all files in a folder
   files = grab_files('/data', ext='.csv', recursive=True)
   for file in files:
       process_file(file)
   ```

5. **Path handling**:
   ```python
   from pathlib import Path
   
   # Use pathlib for cross-platform compatibility
   base = Path('/project/data')
   file_path = base / 'subdir' / 'file.csv'
   ```

6. **Error handling**:
   ```python
   from utils.FileFunctions import grab_files
   
   try:
       files = grab_files('/path/to/data', ext='.pkl')
       if not files:
           print("No files found!")
   except ValueError as e:
       print(f"Invalid path: {e}")
   ```

---

## 📋 Common Workflow Pattern

```python
from utils.FileFunctions import (
    load_config,
    set_working_directory,
    grab_files,
    save_config_copy
)
from pathlib import Path

# 1. Load configuration
config = load_config('config.yaml')

# 2. Set up output directory
output_dir = set_working_directory(
    config['base_path'],
    'results',
    config['experiment_id']
)

# 3. Save config copy
save_config_copy(config, Path(output_dir))

# 4. Find input files
input_files = grab_files(
    config['data_path'],
    ext='.pkl',
    recursive=True
)

# 5. Process files
for file in input_files:
    process_file(file, config)

print(f"Analysis complete. Results in: {output_dir}")
```

---

## 🔗 Related

- See [config_utils](cheatsheet_config_utils.md) for advanced configuration management
- See other cheatsheets for domain-specific file operations
