# YG_utils_analysis

A centralized Python package containing common utilities and analysis pipelines for fiber photometry and behavioral data analysis projects. This package is designed to improve code reusability, consistency, and maintainability across multiple research projects.

## Main Features

- **Fiber Photometry Preprocessing**: Noise removal, detrending, and normalization routines for 1-channel and 2-channel FP data.
- **Behavioral Data Processing**: Convert DeepLabCut (DLC) tracking data to BORIS format, video processing (flipping, resizing), and more.
- **Epoch and Peak Analysis**: Configuration-based pipelines for event-centered epoch analysis and calcium signal peak detection.
- **Command Line Interface**: Easy-to-use command line scripts for running standard analysis workflows.

## Installation

You can install this package directly from the GitHub repository.

### Standard Installation

```bash
pip install git+https://github.com/parkgilbong/YG_utils_analysis.git
```

### Development Installation

If you want to modify the package code directly, clone the repository and install it in "editable" mode:

```bash
git clone https://github.com/parkgilbong/YG_utils_analysis.git
cd YG_utils_analysis
pip install -e .
```
With this method, any changes you make to the source code will be applied immediately without needing to reinstall.

## Usage

You can use this package in two main ways: by importing functions in scripts or notebooks, or by using the provided command line tools.

### 1. Importing Functions in Python

You can import any function from the `analysis` or `utils` modules directly in your code.

```python
# Example in a Jupyter Notebook or another script
from analysis.group_summary import process_analysis
from utils.FPFunctions import FP_preprocessing_1ch
from utils.PlotFunctions import plot_group_trace

# Now you can call these functions with your data and configuration.
# config = load_config(...)
# process_analysis(config)
```

### 2. Using Command Line Tools

This package provides several command line scripts for running typical analysis pipelines. These are mainly executed from the terminal.

```bash
# Example: Run the fiber photometry preprocessing pipeline
yg-fp-preprocess --config path/to/your/fp_config.yaml

# Example: Run epoch analysis
yg-epoch-analysis --config path/to/your/epoch_config.yaml

# Example: Convert DLC output to BORIS format
yg-dlc2boris --config path/to/your/dlc_config.yaml
```

#### Available Commands

- `yg-behavior-preprocess`: Preprocess behavioral video data.
- `yg-fp-preprocess`: Run the 1-channel fiber photometry preprocessing pipeline.
- `yg-dlc2boris`: Convert DeepLabCut (DLC) output to BORIS format.
- `yg-peak-analysis`: Detect peaks in FP signals.
- `yg-epoch-analysis`: Run event-centered epoch analysis.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.