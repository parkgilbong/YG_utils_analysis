# YG_utils_analysis - Function Summary

This document provides a comprehensive overview of all functions in the package.

---

## Utils Module

### DLCFunctions.py

#### `df_to_dic_single`

**Signature:** `df_to_dic_single(df, ignore_bodyparts: str = 'PatchCordBase')`

**Description:** Convert single dlc output data to a python dictionary. The input data format should be a pandas DataFrame format.

---

#### `df_to_dic_multi`

**Signature:** `df_to_dic_multi(df, ignore_bodyparts: str = 'PatchCordBase')`

**Description:** Convert multi dlc output data to two python dictionaries. The input data format should be a pandas DataFrame format.

---

#### `PostDLC_3CT_3EVTs`

**Signature:** `PostDLC_3CT_3EVTs(DLCresult: dict, destfolder: str = '', ROI: str = 'new', Nose2Snout_dist: float = 30, Evt1: tuple = (0.5, 2), Evt2: tuple = (2, 2), Evt3: tuple = (2, 2), FPS: int = 25, SaveData: bool = False) -> dict`

**Description:** - Social Preference Index based on time spent in different regions of interest (ROIs)

---

#### `get_velocity`

**Signature:** `get_velocity(DLCresult: dict, bpt: str, FPS: int, pcutoff: float = 0.95)`

**Description:** Calculate the velocity of a specific body part from DLC results.

---

#### `roi_entry_analysis`

**Signature:** `roi_entry_analysis(DLCresult: dict, bpt: str, pcutoff: float, ROI: list)`

**Description:** Indentify the entry of a body part into a region of interest (ROI) based on DLC results.

---

#### `get_bodypoints_distance`

**Signature:** `get_bodypoints_distance(DLCresult: dict, bpt: str, bpt2: str, pcutoff: float = 0.95, distance_thres: float = 30)`

**Description:** Calculate the distance between two body parts in DLC results.

---

#### `annotate_body_part_proximity`

**Signature:** `annotate_body_part_proximity(body_part_data1: Dict[str, Dict[str, Any]], body_part_data2: Dict[str, Dict[str, Any]], points_df: pd.DataFrame, body_part_name1: str, body_part_name2: str, pcutoff: float, d_threshold: float, d_threshold2: float = 110.0) -> pd.DataFrame`

**Description:** Compute distance between two body parts across frames and annotate proximity.

---

#### `check_point_in_regions`

**Signature:** `check_point_in_regions(x, y, regions)`

**Description:** *(No docstring)*

---

### FPFunctions.py

#### `FP_preprocessing_1ch`

**Signature:** `FP_preprocessing_1ch(Tank_path: str, Dest_folder: str, sys: str = 'tdt', Detrending_method: str = 'Exp_fit', Use_CamTick: bool = True, duration_mode = 'fixed', FPS: int = 25, Rec_duration: int = 600, Namefor405: str = '405', Namefor465: str = '465', SaveAsCSV: bool = False)`

**Description:** This function preprocesses the 1 channel(465) FP data from the tank file (raw data) and saves it as a .csv file.

---

#### `FP_preprocessing_2ch`

**Signature:** `FP_preprocessing_2ch(Tank_path: str, Dest_folder: str, FPS: int = 25, Rec_duration: int = 600, Namefor405: str = '405', Namefor465: str = '465', Namefor560: str = '560')`

**Description:** This function preprocesses the 2 channels (465, 560) FP data from the tank file (raw data) and saves it as a .csv file.

---

#### `FP_preprocessing_2ch_new`

**Signature:** `FP_preprocessing_2ch_new(Tank_path: str, Dest_folder: str, Detrending_method: str = 'Exp_fit', Use_CamTick: bool = True, duration_mode: str = 'fixed', FPS: int = 25, Rec_duration: int = 600, Namefor405: str = '405', Namefor465: str = '465', Namefor560: str = '560', SaveAsCSV: bool = False) -> None`

**Description:** Preprocesses two-channel FP data (465 & 560 nm) from a TDT tank file.

---

#### `Peak_Analysis`

**Signature:** `Peak_Analysis(pkl_path: str = 'Final_table_raw_trace.pkl', signal2use: str = 'Zscore', prominence_thres: float = 2, amplitude_thres: float = 4, FPS: int = 25, pre_window_len: int = 3, post_window_len: int = 3, output_folder: str = 'Peak_Analysis', SavePlots: bool = False, SaveData: bool = False, SaveVideos: bool = False, video_path: str = 'video.avi')`

**Description:** This function performs peak analysis on the processed fluorescence data, identifying peaks based on prominence and amplitude thresholds, and extracting relevant information about these peaks.

---

#### `Epoch_Analysis_3EVT`

**Signature:** `Epoch_Analysis_3EVT(pkl_path: str = 'Final_table_raw_trace.pkl', signal2use: str = 'normalized', evt_path: str = 'Data_DLC.csv', PRE_TIME: int = 5, POST_TIME: int = 10, FPS: int = 25, Rec_duration: int = 600, SavePlots: bool = False, SaveData: bool = False, output_folder: str = 'Epoch_Analysis')`

**Description:** This function performs epoch analysis on the processed fluorescence data, aligning the data to behavioral events and extracting relevant information about these epochs.

---

#### `Epoch_Analysis_2EVT`

**Signature:** `Epoch_Analysis_2EVT(pkl_path: str = 'Final_table_raw_trace.pkl', evt_path: str = 'Data_DLC.csv', PRE_TIME: int = 5, POST_TIME: int = 10, FPS: int = 25, Rec_duration: int = 600, SavePlots: bool = False, SaveData: bool = False, output_folder: str = 'Epoch_Analysis')`

**Description:** This function performs epoch analysis on the processed fluorescence data, aligning the data to behavioral events and extracting relevant information about these epochs.

---

#### `Eport_Epoch_Info`

**Signature:** `Eport_Epoch_Info(TDT_Tank_path: str, REF_EPOC: str = 'PC1_', Time4Exclude: int = 2, destfolder: str = '', SaveData: bool = False)`

**Description:** This function extracts the information about the epochs from the synapse Tank data and save it as a CSV file.

---

#### `Import_manual_scoring`

**Signature:** `Import_manual_scoring(file_path: str, FPS: int, Event: str, UseFilter: bool = False, MinDuration: float = 0, MinInterval: float = 1)`

**Description:** Import and process manual scoring data from a TSV file.

---

#### `calculate_auc`

**Signature:** `calculate_auc(time, signal, intervals)`

**Description:** Calculate the Area Under the Curve (AUC) for specified intervals and perform statistical analysis.

---

#### `extract_traces_with_padding`

**Signature:** `extract_traces_with_padding(signal, time, time_tuples, pre_window_sec, post_window_sec, FPS, align_to = 'onset')`

**Description:** Extract time-aligned traces around specified indices (onset or offset), with NaN padding for out-of-bounds values.

---

#### `extract_data_at_timepoint`

**Signature:** `extract_data_at_timepoint(traces, time_point, sampling_rate)`

**Description:** Extracts data points from a 2D numpy array at a specified time point.

---

#### `detect_slow_peaks`

**Signature:** `detect_slow_peaks(signal, sampling_rate, height = 1.3, min_interval = 1.0, min_peak_width = 0.2)`

**Description:** Detects slow peaks in a signal optimized for low-frequency events.

---

#### `double_exponential`

**Signature:** `double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier)`

**Description:** Compute a double exponential function with constant offset.

---

#### `double_exp`

**Signature:** `double_exp(x, const, af, as_, ts, tm)`

**Description:** *(No docstring)*

---

#### `plot_norm`

**Signature:** `plot_norm(y_cor, trend, name, col)`

**Description:** *(No docstring)*

---

#### `_get_auc`

**Signature:** `_get_auc(start, end)`

**Description:** *(No docstring)*

---

#### `_get_adjacent`

**Signature:** `_get_adjacent(start, end)`

**Description:** *(No docstring)*

---

#### `double_exponential`

**Signature:** `double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier)`

**Description:** Compute a double exponential function with constant offset.

---

#### `fit_and_sub`

**Signature:** `fit_and_sub(x, y)`

**Description:** *(No docstring)*

---

#### `find_peak_onset`

**Signature:** `find_peak_onset(signal, peak_index, threshold = 0.0005)`

**Description:** *(No docstring)*

---

### FileFunctions.py

#### `set_working_directory`

**Signature:** `set_working_directory(base_folder: str) -> str`

**Description:** Creates and sets the working directory using the base path and additional subfolders.

---

#### `grab_files`

**Signature:** `grab_files(folder_path: str, ext: str = '', recursive: bool = False) -> List[str]`

**Description:** Retrieves file paths with a specific extension from a folder, optionally including subfolders.

---

#### `grab_folders`

**Signature:** `grab_folders(folder_path: str, recursive: bool = False, names_only: bool = False) -> List[str]`

**Description:** Retrieves folder paths or names from a directory, optionally including subfolders.

---

#### `get_dirname_and_basename`

**Signature:** `get_dirname_and_basename(path: str) -> PathInfo`

**Description:** Extracts the parent folder name and base file name (without extension).

---

#### `load_dataframes`

**Signature:** `load_dataframes(file_list: List[str], file_type: str = 'pickle', trace_start_idx: int = 1)`

**Description:** Loads multiple data files and extracts traces and optional time columns.

---

#### `load_config`

**Signature:** `load_config(config_path: str) -> dict`

**Description:** Load configuration from a YAML file.

---

#### `save_config_copy`

**Signature:** `save_config_copy(cfg: dict, output_dir: Path) -> None`

**Description:** Save a copy of the config used for analysis into the output directory.

---

#### `temp_chdir`

**Signature:** `temp_chdir(path)`

**Description:** A context manager for temporarily changing the current working directory.

---

#### `ensure_dir`

**Signature:** `ensure_dir(path: str) -> str`

**Description:** Ensures that the directory at the specified path exists.

---

#### `load_yaml`

**Signature:** `load_yaml(path: str) -> Dict`

**Description:** Load a YAML file and return its contents as a dictionary.

---

### OmicsFunctions.py

#### `load_genes`

**Signature:** `load_genes(path: str, inline_genes = None)`

**Description:** Load gene names from a file or a provided list, removing duplicates and empty entries.

---

#### `filter_genes_by_thresholds`

**Signature:** `filter_genes_by_thresholds(df, gene_col, padj_col, log2fc_col, adj_p_cutoff, log2fc_cutoff, direction = 'both')`

**Description:** Filter genes based on adjusted p-value and log2 fold change thresholds.

---

#### `convert_gene_ids`

**Signature:** `convert_gene_ids(genes, species: str, id_type: str)`

**Description:** Convert gene identifiers to gene symbols using mygene, if available.

---

#### `load_symbol_assoc_from_gaf`

**Signature:** `load_symbol_assoc_from_gaf(gaf_path: str, taxon = '9606', aspect = 'all')`

**Description:** GAF(2.2) 파일에서 'DB Object Symbol'(유전자 심볼) -> {GO_ID} 매핑을 생성합니다.

---

#### `strip_ensembl_version`

**Signature:** `strip_ensembl_version(x: str) -> str`

**Description:** Removes the version suffix from an Ensembl identifier.

---

#### `convert_ensembl_to_symbol`

**Signature:** `convert_ensembl_to_symbol(genes_ens, mapping_df, ens_col = 'ensembl_gene_id', sym_col = 'symbol')`

**Description:** Converts a list of Ensembl gene IDs to their corresponding gene symbols using a mapping DataFrame.

---

#### `read_gene_list`

**Signature:** `read_gene_list(path)`

**Description:** Read a gene list file (one gene per line) and return as a list of strings.

---

### OmicsPlotFunctions.py

#### `plot_gene_ontology_bar`

**Signature:** `plot_gene_ontology_bar(go_data_path: Path, output_path: Path, top_n: int = 15, ax = None)`

**Description:** Generates a bar plot of the top Gene Ontology (GO) terms based on adjusted p-values.

---

#### `plot_heatmap`

**Signature:** `plot_heatmap(pivot_df, title, output_path, pdf = None, ax = None)`

**Description:** Plots a heatmap from a given DataFrame and saves it to the specified output path.

---

#### `barplot`

**Signature:** `barplot(enr_res2d, out_png, top_terms = 20, title = 'GO Enrichment (Top)', ax = None)`

**Description:** Create a horizontal bar plot for GO enrichment results.

---

#### `dotplot`

**Signature:** `dotplot(enr_res2d, out_png, top_terms = 20, title = 'GO Enrichment (Top)', ax = None)`

**Description:** Create a dot plot for GO enrichment results, showing overlap ratios.

---

#### `volcano_plot`

**Signature:** `volcano_plot(df, log2fc_col, padj_col, out_png, log2fc_cutoff, adj_p_cutoff, ax = None)`

**Description:** Create a volcano plot for differential expression results.

---

### PlotFunctions.py

#### `plot_single_line`

**Signature:** `plot_single_line(x, y, fig_size, fig_title, x_label, y_label, x_lim, y_lim, color, save = False, ax = None)`

**Description:** Create a simple single-line plot.

---

#### `plot_dual_line`

**Signature:** `plot_dual_line(x1, y1, x2, y2, fig_size = (10, 6), fig_title = '', x_label = '', y1_label = '', y2_label = '', x_lim = (), y1_lim = (), y2_lim = (), color1 = 'k', color2 = 'k', save = False, font_size = 12, title_size = 14, legend_size = 10, ax = None)`

**Description:** Plot a dual-line graph with two Y-axes.

---

#### `plot_traces_with_mean`

**Signature:** `plot_traces_with_mean(trace_array, trace_time, ax = None, color = 'green', title = '', xlabel = '', ylabel = '', mode = 'std')`

**Description:** Plot traces with individual trials and overlay the mean ± standard error or standard deviation.

---

#### `plot_trace_heatmap`

**Signature:** `plot_trace_heatmap(traces, trace_time, vmin = None, vmax = None, ax = None, title = '', xlabel = '', ylabel = '', cmap = 'viridis')`

**Description:** Plot a heatmap of aligned traces with NaN support.

---

#### `plot_multi_line`

**Signature:** `plot_multi_line(xy_pairs, sem_pairs = None, fig_size = (10, 6), title = '', x_label = '', y_label = None, y_labels = None, x_lim = None, y_lim = None, colors = None, line_styles = None, save = False, font_size = 12, title_size = 14, legend_size = 10, legend_loc = 'upper left', axvline = None, axhline = None, ax = None)`

**Description:** Plot multiple line graphs with optional SEM shading.

---

### ReportGeneration.py

#### `FP_preprocessing`

**Signature:** `FP_preprocessing(output_path: str, title: str, image_paths: list, comments: list)`

**Description:** Generates a PDF report with the given images, comments, title, date, and time.

---

### VideoFunctions.py

#### `extract_frames`

**Signature:** `extract_frames(video_path: str, frame_indices: list, output_folder: str)`

**Description:** Extract the specified frames from a video file (*.avi or *.mp4) and save them as PNG images in the output folder.

---

#### `extract_video_slices`

**Signature:** `extract_video_slices(video_path: str, slices_df, output_folder: str)`

**Description:** Create individual AVI files for each slice (a pair of the start and end indices) specified in the DataFrame and save them in the output folder.

---

#### `create_animated_chart`

**Signature:** `create_animated_chart(data: list, filename: str, interval: int, offset: float)`

**Description:** Create an animated chart from the given data and save it as an MP4 file.

---

#### `add_inset_chart`

**Signature:** `add_inset_chart(video_slice_path: str, chart_path: str, filename: str, position: tuple = ('right', 'bottom'), chart_width: int = 480)`

**Description:** Add an inset chart to a video file.

---

#### `add_inset_char2`

**Signature:** `add_inset_char2(video_slice_path: str, chart_path: str, filename: str, position: tuple = ('right', 'bottom'), chart_width: int = 480)`

**Description:** Add an inset chart to a video file.

---

#### `VideoChopper`

**Signature:** `VideoChopper(input_file: str, tags: list = [], chunk_duration: int = 60, startingIdx: int = 0)`

**Description:** Splits the input video into chunks of given duration.

---

#### `create_video_from_images`

**Signature:** `create_video_from_images(image_folder: str, output_filename: str, frame_rate: int = 25, duration: int = 600, codec: str = 'mp4v', quality = 95)`

**Description:** Creates a video from a sequence of images in a specified folder.

---

#### `resize_video`

**Signature:** `resize_video(input_path, output_path, scale_factor = 0.5)`

**Description:** Resizes a video by a given scale factor and saves the resized video to the specified output path.

---

#### `Generate_montage`

**Signature:** `Generate_montage(input_folder: str, output_filename: str, rows: int = 3, cols: int = 3, frame_rate: int = 25, duration: int = 600, codec: str = 'mp4v', titles: list = [], popups: list = [], scale_factor: float = 1.0)`

**Description:** Generates a montage movie from movies in a specified folder.

---

#### `draw_polygon_on_image`

**Signature:** `draw_polygon_on_image(image, polygon, color = (0, 255, 0), thickness = 2)`

**Description:** Draws a polygon on an image.

---

#### `extract_first_frame_and_draw_rois`

**Signature:** `extract_first_frame_and_draw_rois(video_path: str, rois: list, output_image_path: str)`

**Description:** Extracts the first frame from an AVI-formatted movie, draws the ROIs on the image, and saves the resulting image.

---

#### `flip_video`

**Signature:** `flip_video(input_path: str, output_path: str) -> None`

**Description:** Flips a video vertically and saves the result to a new file.

---

#### `init`

**Signature:** `init()`

**Description:** *(No docstring)*

---

#### `animate`

**Signature:** `animate(i)`

**Description:** *(No docstring)*

---

#### `draw_polygon_on_image`

**Signature:** `draw_polygon_on_image(image, polygon, color = (0, 255, 0), thickness = 2)`

**Description:** *(No docstring)*

---

### boris_export.py

#### `extract_bouts_from_annotation`

**Signature:** `extract_bouts_from_annotation(series: pd.Series, fps: int) -> list[tuple[int, int]]`

**Description:** Identify 'on' bouts in an annotation series by detecting start and end indices.

---

#### `prepare_boris_export`

**Signature:** `prepare_boris_export(evt_dict: dict, fps: int, animal_id: str, behavior_map: dict) -> pd.DataFrame`

**Description:** Prepare BORIS export dataframe from event bouts.

---

### config_utils.py

#### `deep_merge`

**Signature:** `deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]`

**Description:** Recursively merges two dictionaries.

---

#### `get_cfg`

**Signature:** `get_cfg(config_path: Optional[str]) -> Dict[str, Any]`

**Description:** *(No docstring)*

---

#### `pick`

**Signature:** `pick(cli_val, cfg: Dict[str, Any], key: str, default = None)`

**Description:** CLI 우선, 없으면 cfg[key], 둘 다 없으면 default.

---

#### `resolve_path`

**Signature:** `resolve_path(cli_path: Optional[List[str]], cfg: Dict, config_key: str, config_section: Optional[str] = None, is_input: bool = False) -> List[str]`

**Description:** Resolves file paths from CLI arguments or a configuration dictionary,

---

### logging_utils.py

#### `setup_logging`

**Signature:** `setup_logging(log_dir = 'logs', level = logging.INFO)`

**Description:** 프로젝트 전반에 걸쳐 사용할 로거를 설정합니다.

---

#### `setup_named_logger`

**Signature:** `setup_named_logger(logger_name: str, log_dir: str = 'logs', level = logging.INFO) -> Tuple[logging.Logger, str]`

**Description:** Configures and returns a named logger that writes to a timestamped file.

---

### logging_utils_environ.py

#### `setup_logging`

**Signature:** `setup_logging(log_dir = 'logs', level = logging.INFO) -> Tuple[logging.Logger, str]`

**Description:** Initializes the project-wide logger for file and console output.

---

#### `__init__`

**Signature:** `__init__(self, logger, level = logging.INFO)`

**Description:** *(No docstring)*

---

#### `write`

**Signature:** `write(self, buf)`

**Description:** *(No docstring)*

---

#### `flush`

**Signature:** `flush(self)`

**Description:** *(No docstring)*

---

## Analysis Module

### DLC2BORIS.py

#### `process_animal`

**Signature:** `process_animal(animal_id: str, cfg: dict, output_base: Path)`

**Description:** *(No docstring)*

---

#### `main`

**Signature:** `main()`

**Description:** *(No docstring)*

---

### behavior_preprocessing.py

#### `setup_logging`

**Signature:** `setup_logging()`

**Description:** *(No docstring)*

---

#### `create_video_segments_from_images`

**Signature:** `create_video_segments_from_images(image_folder: str, output_basename: str, frame_rate: int, segments: list, codec: str = 'mp4v', quality: int = 95) -> list`

**Description:** Create multiple video segments from a folder of images based on index ranges.

---

#### `preprocess_behavioral_data`

**Signature:** `preprocess_behavioral_data(config_path: str)`

**Description:** Read YAML config and generate videos per session, with optional per-animal segments.

---

#### `main`

**Signature:** `main()`

**Description:** *(No docstring)*

---

### dlc2boris43CT.py

#### `setup_logging`

**Signature:** `setup_logging()`

**Description:** *(No docstring)*

---

#### `load_config`

**Signature:** `load_config(config_path: Path) -> dict`

**Description:** Load processing parameters from a YAML configuration file.

---

#### `process_animal`

**Signature:** `process_animal(animal: str, session: str, config: dict, output_dir: Path) -> None`

**Description:** Process DLC outputs and extract movement and event metrics for one animal-session.

---

#### `run_dlc2boris_pipeline`

**Signature:** `run_dlc2boris_pipeline(config: dict) -> None`

**Description:** Execute the DLC-to-BORIS pipeline over all specified animals and sessions.

---

#### `main`

**Signature:** `main() -> None`

**Description:** Command-line entry point.

---

### dlc2boris4DI.py

#### `load_config`

**Signature:** `load_config(config_path: str) -> dict`

**Description:** *(No docstring)*

---

#### `run_dlc2boris_pipeline`

**Signature:** `run_dlc2boris_pipeline(cfg: dict)`

**Description:** *(No docstring)*

---

#### `process_animal`

**Signature:** `process_animal(animal_id: str, animal_data_dir: Path, output_dir: Path, cfg: dict)`

**Description:** *(No docstring)*

---

### epoch_analysis.py

#### `load_config`

**Signature:** `load_config(path: str) -> dict`

**Description:** Load parameters from a YAML configuration file.

---

#### `setup_logging`

**Signature:** `setup_logging(log_conf: dict, project_root: str) -> None`

**Description:** Configure file-based logging.

---

#### `run_epoch_analysis`

**Signature:** `run_epoch_analysis(cfg: dict, project_root: str) -> None`

**Description:** Execute epoch-based signal extraction, statistic computation, plotting,

---

### flip_videos.py

#### `setup_logging`

**Signature:** `setup_logging() -> None`

**Description:** Configure basic logging format and level.

---

#### `load_config`

**Signature:** `load_config(config_path: Path) -> Tuple[Path, List[Dict[str, str]]]`

**Description:** Load YAML config and return global output_dir and list of file entries.

---

#### `process_files`

**Signature:** `process_files(output_dir: Path, files: List[Dict[str, str]]) -> None`

**Description:** Iterate through config entries, ensure output dirs, and flip each video.

---

#### `main`

**Signature:** `main() -> None`

**Description:** Parse arguments, load config, and initiate processing.

---

### fp_preprocessing.py

#### `setup_logging`

**Signature:** `setup_logging()`

**Description:** *(No docstring)*

---

#### `load_config`

**Signature:** `load_config(config_path)`

**Description:** *(No docstring)*

---

#### `run_preprocessing`

**Signature:** `run_preprocessing(config)`

**Description:** *(No docstring)*

---

#### `main`

**Signature:** `main()`

**Description:** *(No docstring)*

---

### fp_preprocessing_2ch.py

#### `setup_logging`

**Signature:** `setup_logging()`

**Description:** *(No docstring)*

---

#### `load_config`

**Signature:** `load_config(config_path)`

**Description:** *(No docstring)*

---

#### `run_preprocessing`

**Signature:** `run_preprocessing(config)`

**Description:** *(No docstring)*

---

### group_summary.py

#### `load_config`

**Signature:** `load_config(config_path: str) -> dict`

**Description:** Load YAML config with nested signals/events.

---

#### `gather_files`

**Signature:** `gather_files(root: str, batches: list, session: str, signal: str, event: str, align: str) -> tuple`

**Description:** Collect valid PKL paths & labels for one signal/event.

---

#### `compute_group_summary`

**Signature:** `compute_group_summary(pkl_files: list, group: str, signal: str, event: str, align: str, root: str) -> tuple`

**Description:** Compute mean & SEM, save summary CSV.

---

#### `plot_group_trace`

**Signature:** `plot_group_trace(time, mean, sem, group, signal, event, align, root, color: str = 'green', return_fig = False)`

**Description:** Plot & save average with custom color; opt. return fig, ax.

---

#### `plot_with_individual`

**Signature:** `plot_with_individual(time, traces, group, signal, event, align, root, color: str = 'green', return_fig = False)`

**Description:** Plot individual traces w/ mean overlay in custom color; opt. return fig, ax.

---

#### `process_analysis`

**Signature:** `process_analysis(sig_conf: dict, global_cfg: dict, return_fig = False)`

**Description:** Handle one signal/event analysis with color mapping.

---

#### `main`

**Signature:** `main(config_path: str)`

**Description:** Entry: load config and run all signals/events.

---

### group_summary_2.py

#### `load_config`

**Signature:** `load_config(config_path: str) -> dict`

**Description:** Load YAML configuration for multi-group plotting.

---

#### `load_group_summaries`

**Signature:** `load_group_summaries(root_folder: str, groups: list[str], align_point: str = 'onset') -> tuple[list[tuple[pd.Series, pd.Series]], list[pd.Series]]`

**Description:** Load time and mean/SEM series for each group.

---

#### `plot_multi_group`

**Signature:** `plot_multi_group(root_folder: str, groups: list[str], align_point: str = 'onset', colors: list[str] | None = None, save: bool = True) -> None`

**Description:** Plot multiple group summaries with optional SEM shading.

---

#### `main`

**Signature:** `main()`

**Description:** *(No docstring)*

---

### load_rwd_fpfile.py

#### `_replace_semis_outside_quotes`

**Signature:** `_replace_semis_outside_quotes(s: str) -> str`

**Description:** Replace semicolons with commas only when outside of double-quoted substrings.

---

#### `parse_settings_line`

**Signature:** `parse_settings_line(line: str) -> Dict`

**Description:** Convert first-line custom-format string into a Python dict.

---

#### `load_fluorescence`

**Signature:** `load_fluorescence(file_path: str) -> Tuple[Dict, pd.DataFrame]`

**Description:** Read a fluorescence CSV file where the first line contains experiment settings

---

### peak_analysis.py

#### `load_config`

**Signature:** `load_config(config_path: str) -> dict`

**Description:** *(No docstring)*

---

#### `run_peak_analysis`

**Signature:** `run_peak_analysis(config: dict)`

**Description:** *(No docstring)*

---

#### `main`

**Signature:** `main()`

**Description:** *(No docstring)*

---

### rename_files.py

#### `load_config`

**Signature:** `load_config(path: str) -> dict`

**Description:** Load YAML configuration from the given file path.

---

#### `setup_logger`

**Signature:** `setup_logger(log_path: str) -> None`

**Description:** Configure the root logger to write INFO-level messages to both console and file.

---

#### `rename_files`

**Signature:** `rename_files(config: dict) -> None`

**Description:** Rename files in each configured subfolder according to numeric suffix.

---

#### `main`

**Signature:** `main()`

**Description:** Entry point for the script.

---

### video_resizing.py

#### `video_resizing`

**Signature:** `video_resizing(config_path: str)`

**Description:** *(No docstring)*

---

