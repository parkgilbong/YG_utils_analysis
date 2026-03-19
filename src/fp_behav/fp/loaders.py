# load_rwd_fpfile.py
"""
Utility functions for loading FP experimental data acquired from RWD fiber photometry system.

This module provides:

- `parse_settings_line`: Parse a custom semicolon-delimited JSON-like settings line.
- `load_fluorescence`: Read experiment settings and measurement data from a CSV file.

Usage Example:

```python
from src.utils.fluorescence_loader import load_fluorescence

# Path to your fluorescence CSV file
file_path = '/mnt/data/Fluorescence.csv'

# Load settings dict and measurement DataFrame
settings, df = load_fluorescence(file_path)

# Inspect parsed settings
print("--- Experiment Settings ---")
for key, value in settings.items():
    print(f"{key}: {value}")

# Preview measurement data
print("\n--- Measurement Data ---")
print(df.head())
```
"""
import json
import pandas as pd
from typing import Tuple, Dict

def _replace_semis_outside_quotes(s: str) -> str:
    """
    Replace semicolons with commas only when outside of double-quoted substrings.
    """
    result = []
    in_str = False
    prev_char = ''
    for ch in s:
        if ch == '"' and prev_char != '\\':
            in_str = not in_str
        if ch == ';' and not in_str:
            result.append(',')
        else:
            result.append(ch)
        prev_char = ch
    return ''.join(result)


def parse_settings_line(line: str) -> Dict:
    """
    Convert first-line custom-format string into a Python dict.
    """
    json_compatible = _replace_semis_outside_quotes(line)
    return json.loads(json_compatible)


def load_fluorescence(file_path: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Read a fluorescence CSV file where the first line contains experiment settings
    in a custom semicolon-delimited JSON-like format, and the rest is tabular data.

    Returns:
        settings: dict of parsed experimental settings
        df: pandas DataFrame of measurement data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    settings = parse_settings_line(first_line)
    # Read DataFrame skipping the settings line, using second row as header
    df = pd.read_csv(file_path, skiprows=1)
    # Drop any columns with empty or unnamed headers
    df = df.loc[:, ~df.columns.str.match(r'^(Unnamed|\s*$)')]
    return settings, df


if __name__ == '__main__':
    import os
    path = '/mnt/data/Fluorescence.csv'
    if os.path.exists(path):
        settings, df = load_fluorescence(path)
        print("Parsed Settings:", settings)
        print("DataFrame Head:\n", df.head())
    else:
        print(f"File not found: {path}")
