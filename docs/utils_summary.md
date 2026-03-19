# fp_behav - Function Summary

패키지 전체 함수 목록 및 시그니처. 패키지 구조: `src/fp_behav/`

---

## fp_behav.fp — Fiber Photometry

### functions.py (핵심 신호처리)

#### `FP_preprocessing_1ch`
**Signature:** `FP_preprocessing_1ch(Tank_path: str, Dest_folder: str, sys: str = 'tdt', Detrending_method: str = 'Exp_fit', Use_CamTick: bool = True, duration_mode = 'fixed', FPS: int = 25, Rec_duration: int = 600, Namefor405: str = '405', Namefor465: str = '465', SaveAsCSV: bool = False)`
**Description:** 1채널(465nm) FP 데이터 전처리. TDT/RWD 탱크 파일을 읽어 디트렌딩·정규화 후 CSV/PNG 저장.

---

#### `FP_preprocessing_2ch`
**Signature:** `FP_preprocessing_2ch(Tank_path: str, Dest_folder: str, FPS: int = 25, Rec_duration: int = 600, Namefor405: str = '405', Namefor465: str = '465', Namefor560: str = '560')`
**Description:** 2채널(465, 560nm) FP 데이터 전처리.

---

#### `FP_preprocessing_2ch_new`
**Signature:** `FP_preprocessing_2ch_new(Tank_path: str, Dest_folder: str, Detrending_method: str = 'Exp_fit', Use_CamTick: bool = True, duration_mode: str = 'fixed', FPS: int = 25, Rec_duration: int = 600, Namefor405: str = '405', Namefor465: str = '465', Namefor560: str = '560', SaveAsCSV: bool = False) -> None`
**Description:** 개선된 2채널 FP 전처리 (유연한 파라미터).

---

#### `Peak_Analysis`
**Signature:** `Peak_Analysis(pkl_path: str = 'Final_table_raw_trace.pkl', signal2use: str = 'Zscore', prominence_thres: float = 2, amplitude_thres: float = 4, FPS: int = 25, pre_window_len: int = 3, post_window_len: int = 3, output_folder: str = 'Peak_Analysis', SavePlots: bool = False, SaveData: bool = False, SaveVideos: bool = False, video_path: str = 'video.avi')`
**Description:** prominence/amplitude 임계값 기반 칼슘 transient peak 감지.

---

#### `Epoch_Analysis_3EVT`
**Signature:** `Epoch_Analysis_3EVT(pkl_path: str = 'Final_table_raw_trace.pkl', signal2use: str = 'normalized', evt_path: str = 'Data_DLC.csv', PRE_TIME: int = 5, POST_TIME: int = 10, FPS: int = 25, Rec_duration: int = 600, SavePlots: bool = False, SaveData: bool = False, output_folder: str = 'Epoch_Analysis')`
**Description:** 3가지 행동 이벤트에 정렬된 epoch 분석.

---

#### `Epoch_Analysis_2EVT`
**Signature:** `Epoch_Analysis_2EVT(pkl_path: str = 'Final_table_raw_trace.pkl', evt_path: str = 'Data_DLC.csv', PRE_TIME: int = 5, POST_TIME: int = 10, FPS: int = 25, Rec_duration: int = 600, SavePlots: bool = False, SaveData: bool = False, output_folder: str = 'Epoch_Analysis')`
**Description:** 2가지 행동 이벤트에 정렬된 epoch 분석.

---

#### `Eport_Epoch_Info`
**Signature:** `Eport_Epoch_Info(TDT_Tank_path: str, REF_EPOC: str = 'PC1_', Time4Exclude: int = 2, destfolder: str = '', SaveData: bool = False)`
**Description:** TDT Synapse 탱크에서 epoch 정보 추출 → CSV 저장.

---

#### `Import_manual_scoring`
**Signature:** `Import_manual_scoring(file_path: str, FPS: int, Event: str, UseFilter: bool = False, MinDuration: float = 0, MinInterval: float = 1)`
**Description:** TSV 수동 스코어링 파일 로드 및 duration/interval 필터링.

---

#### `calculate_auc`
**Signature:** `calculate_auc(time, signal, intervals)`
**Description:** 지정 구간의 AUC 계산 및 통계 비교.

---

#### `extract_traces_with_padding`
**Signature:** `extract_traces_with_padding(signal, time, time_tuples, pre_window_sec, post_window_sec, FPS, align_to = 'onset')`
**Description:** 지정 인덱스(onset/offset) 주변 시간 정렬 트레이스 추출 (NaN 패딩 포함).

---

#### `extract_data_at_timepoint`
**Signature:** `extract_data_at_timepoint(traces, time_point, sampling_rate)`
**Description:** 2D numpy 배열에서 특정 시간 포인트의 데이터 포인트 추출.

---

#### `detect_slow_peaks`
**Signature:** `detect_slow_peaks(signal, sampling_rate, height = 1.3, min_interval = 1.0, min_peak_width = 0.2)`
**Description:** 저주파 이벤트에 최적화된 slow peak 감지.

---

### loaders.py (데이터 로더)

#### `load_fluorescence`
**Signature:** `load_fluorescence(file_path: str) -> Tuple[Dict, pd.DataFrame]`
**Description:** RWD 형광 CSV 파일 로드. 첫 번째 줄의 실험 설정과 측정 데이터 반환.

---

#### `parse_settings_line`
**Signature:** `parse_settings_line(line: str) -> Dict`
**Description:** 첫 줄 커스텀 포맷 문자열을 Python dict으로 변환.

---

### preprocessing.py (1채널 파이프라인)

#### `run_preprocessing`
**Signature:** `run_preprocessing(config: dict)`
**Description:** config dict에 따라 1채널 FP 전처리 파이프라인 실행.

---

### preprocessing_2ch.py (2채널 파이프라인)

#### `run_preprocessing`
**Signature:** `run_preprocessing(config: dict)`
**Description:** config dict에 따라 2채널 FP 전처리 파이프라인 실행.

---

### epoch.py (epoch 분석 파이프라인)

#### `run_epoch_analysis`
**Signature:** `run_epoch_analysis(cfg: dict, project_root: str) -> None`
**Description:** YAML config 기반 epoch 분석 실행 (신호 추출, 통계, 플롯, 저장).

---

### peak.py (peak 분석 파이프라인)

#### `run_peak_analysis`
**Signature:** `run_peak_analysis(config: dict)`
**Description:** 여러 동물/세션의 FP 신호에서 peak 감지 파이프라인 실행.

---

---

## fp_behav.behavior — 행동 분석

### dlc.py

#### `df_to_dic_single`
**Signature:** `df_to_dic_single(df, ignore_bodyparts: str = 'PatchCordBase')`
**Description:** 단일 동물 DLC DataFrame → 딕셔너리 변환.

---

#### `df_to_dic_multi`
**Signature:** `df_to_dic_multi(df, ignore_bodyparts: str = 'PatchCordBase')`
**Description:** 멀티 동물 DLC DataFrame → 두 개 딕셔너리로 변환.

---

#### `PostDLC_3CT_3EVTs`
**Signature:** `PostDLC_3CT_3EVTs(DLCresult: dict, destfolder: str = '', ROI: str = 'new', Nose2Snout_dist: float = 30, Evt1: tuple = (0.5, 2), Evt2: tuple = (2, 2), Evt3: tuple = (2, 2), FPS: int = 25, SaveData: bool = False) -> dict`
**Description:** 3-Chamber Test 완전 분석 (Social Preference Index, ROI 진입, nose-poke).

---

#### `get_velocity`
**Signature:** `get_velocity(DLCresult: dict, bpt: str, FPS: int, pcutoff: float = 0.95)`
**Description:** DLC 결과에서 특정 body part 속도 계산.

---

#### `roi_entry_analysis`
**Signature:** `roi_entry_analysis(DLCresult: dict, bpt: str, pcutoff: float, ROI: list)`
**Description:** DLC 결과 기반 body part ROI 진입 감지.

---

#### `get_bodypoints_distance`
**Signature:** `get_bodypoints_distance(DLCresult: dict, bpt: str, bpt2: str, pcutoff: float = 0.95, distance_thres: float = 30)`
**Description:** 두 body part 간 거리 계산 및 접촉 이벤트 감지.

---

#### `annotate_body_part_proximity`
**Signature:** `annotate_body_part_proximity(body_part_data1, body_part_data2, points_df, body_part_name1, body_part_name2, pcutoff, d_threshold, d_threshold2 = 110.0) -> pd.DataFrame`
**Description:** 프레임별 두 body part 거리 계산 및 근접도 어노테이션 (close / nearby / far).

---

#### `check_point_in_regions`
**Signature:** `check_point_in_regions(x, y, regions)`
**Description:** 좌표가 어느 Shapely 영역에 속하는지 확인.

---

### preprocessing.py (행동 비디오 전처리)

#### `preprocess_behavioral_data`
**Signature:** `preprocess_behavioral_data(config_path: str)`
**Description:** YAML config 기반 행동 비디오 전처리 (세그먼트별 비디오 생성).

---

#### `create_video_segments_from_images`
**Signature:** `create_video_segments_from_images(image_folder: str, output_basename: str, frame_rate: int, segments: list, codec: str = 'mp4v', quality: int = 95) -> list`
**Description:** 이미지 폴더에서 인덱스 범위별 복수 비디오 세그먼트 생성.

---

### export.py (BORIS 내보내기)

#### `extract_bouts_from_annotation`
**Signature:** `extract_bouts_from_annotation(series: pd.Series, fps: int) -> list[tuple[int, int]]`
**Description:** 어노테이션 시리즈에서 'on' bout의 시작/종료 인덱스 감지.

---

#### `prepare_boris_export`
**Signature:** `prepare_boris_export(evt_dict: dict, fps: int, animal_id: str, behavior_map: dict) -> pd.DataFrame`
**Description:** 이벤트 bout에서 BORIS 내보내기 DataFrame 준비.

---

### boris/base.py

#### `process_animal`
**Signature:** `process_animal(animal_id: str, cfg: dict, output_base: Path)`
**Description:** 단일 동물의 DLC 출력 처리 및 BORIS 변환.

---

#### `main`
**Signature:** `main()`
**Description:** 기본 DLC→BORIS 변환 CLI 진입점.

---

### boris/ct.py (3-Chamber Test)

#### `process_animal`
**Signature:** `process_animal(animal: str, session: str, config: dict, output_dir: Path) -> None`
**Description:** 3CT 실험용 DLC 출력 처리 및 행동 메트릭 추출.

---

#### `run_dlc2boris_pipeline`
**Signature:** `run_dlc2boris_pipeline(config: dict) -> None`
**Description:** 전체 동물/세션에 대해 DLC→BORIS 파이프라인 실행.

---

#### `main`
**Signature:** `main() -> None`
**Description:** 3CT 변환 CLI 진입점.

---

### boris/di.py (Direct Interaction)

#### `run_dlc2boris_pipeline`
**Signature:** `run_dlc2boris_pipeline(cfg: dict)`
**Description:** Direct Interaction 실험용 DLC→BORIS 파이프라인 실행.

---

#### `process_animal`
**Signature:** `process_animal(animal_id: str, animal_data_dir: Path, output_dir: Path, cfg: dict)`
**Description:** DI 실험 단일 동물 DLC 데이터 처리.

---

---

## fp_behav.video — 영상 처리

### functions.py

#### `extract_frames`
**Signature:** `extract_frames(video_path: str, frame_indices: list, output_folder: str)`
**Description:** 지정 프레임 인덱스를 PNG로 추출.

---

#### `extract_video_slices`
**Signature:** `extract_video_slices(video_path: str, slices_df, output_folder: str)`
**Description:** DataFrame에 정의된 구간별로 개별 AVI 파일 생성.

---

#### `create_animated_chart`
**Signature:** `create_animated_chart(data: list, filename: str, interval: int, offset: float)`
**Description:** 데이터로 애니메이션 차트 생성 → MP4 저장.

---

#### `add_inset_chart`
**Signature:** `add_inset_chart(video_slice_path: str, chart_path: str, filename: str, position: tuple = ('right', 'bottom'), chart_width: int = 480)`
**Description:** 비디오에 인셋 차트 오버레이.

---

#### `VideoChopper`
**Signature:** `VideoChopper(input_file: str, tags: list = [], chunk_duration: int = 60, startingIdx: int = 0)`
**Description:** 비디오를 지정 길이 청크로 분할.

---

#### `create_video_from_images`
**Signature:** `create_video_from_images(image_folder: str, output_filename: str, frame_rate: int = 25, duration: int = 600, codec: str = 'mp4v', quality = 95)`
**Description:** 이미지 시퀀스에서 비디오 생성.

---

#### `resize_video`
**Signature:** `resize_video(input_path, output_path, scale_factor = 0.5)`
**Description:** scale_factor로 비디오 리사이징.

---

#### `Generate_montage`
**Signature:** `Generate_montage(input_folder: str, output_filename: str, rows: int = 3, cols: int = 3, frame_rate: int = 25, duration: int = 600, codec: str = 'mp4v', titles: list = [], popups: list = [], scale_factor: float = 1.0)`
**Description:** 여러 비디오를 격자 레이아웃 몽타주로 합성.

---

#### `draw_polygon_on_image`
**Signature:** `draw_polygon_on_image(image, polygon, color = (0, 255, 0), thickness = 2)`
**Description:** 이미지에 폴리곤 그리기.

---

#### `extract_first_frame_and_draw_rois`
**Signature:** `extract_first_frame_and_draw_rois(video_path: str, rois: list, output_image_path: str)`
**Description:** 비디오 첫 프레임 추출 후 ROI 표시 → 저장.

---

#### `flip_video`
**Signature:** `flip_video(input_path: str, output_path: str) -> None`
**Description:** 비디오 수직 반전.

---

---

## fp_behav.group — 그룹 수준 분석

### summary.py

#### `gather_files`
**Signature:** `gather_files(root: str, batches: list, session: str, signal: str, event: str, align: str) -> tuple`
**Description:** 특정 signal/event에 해당하는 PKL 경로 수집.

---

#### `compute_group_summary`
**Signature:** `compute_group_summary(pkl_files: list, group: str, signal: str, event: str, align: str, root: str) -> tuple`
**Description:** 그룹 평균 및 SEM 계산 → CSV 저장.

---

#### `plot_group_trace`
**Signature:** `plot_group_trace(time, mean, sem, group, signal, event, align, root, color: str = 'green', return_fig = False)`
**Description:** 그룹 평균 ± SEM 트레이스 플롯.

---

#### `plot_with_individual`
**Signature:** `plot_with_individual(time, traces, group, signal, event, align, root, color: str = 'green', return_fig = False)`
**Description:** 개별 트레이스 + 평균 오버레이 플롯.

---

#### `process_analysis`
**Signature:** `process_analysis(sig_conf: dict, global_cfg: dict, return_fig = False)`
**Description:** 단일 signal/event 분석 실행 (색상 매핑 포함).

---

#### `main`
**Signature:** `main(config_path: str)`
**Description:** config 로드 후 전체 signal/event 분석 실행.

---

#### `load_group_summaries`
**Signature:** `load_group_summaries(root_folder: str, groups: list[str], align_point: str = 'onset') -> tuple`
**Description:** 각 그룹의 time/mean/SEM 시리즈 로드.

---

#### `plot_multi_group`
**Signature:** `plot_multi_group(root_folder: str, groups: list[str], align_point: str = 'onset', colors: list[str] | None = None, save: bool = True) -> None`
**Description:** 여러 그룹 요약 플롯 (선택적 SEM 음영 포함).

---

---

## fp_behav.plot — 시각화

### functions.py

#### `plot_single_line`
**Signature:** `plot_single_line(x, y, fig_size, fig_title, x_label, y_label, x_lim, y_lim, color, save = False, ax = None)`
**Description:** 단순 단일 라인 플롯.

---

#### `plot_dual_line`
**Signature:** `plot_dual_line(x1, y1, x2, y2, fig_size = (10, 6), fig_title = '', x_label = '', y1_label = '', y2_label = '', x_lim = (), y1_lim = (), y2_lim = (), color1 = 'k', color2 = 'k', save = False, font_size = 12, title_size = 14, legend_size = 10, ax = None)`
**Description:** 두 개 Y축을 가진 이중 라인 그래프.

---

#### `plot_traces_with_mean`
**Signature:** `plot_traces_with_mean(trace_array, trace_time, ax = None, color = 'green', title = '', xlabel = '', ylabel = '', mode = 'std')`
**Description:** 개별 트라이얼 + 평균 ± SEM/STD 오버레이.

---

#### `plot_trace_heatmap`
**Signature:** `plot_trace_heatmap(traces, trace_time, vmin = None, vmax = None, ax = None, title = '', xlabel = '', ylabel = '', cmap = 'viridis')`
**Description:** NaN 지원 정렬 트레이스 히트맵.

---

#### `plot_multi_line`
**Signature:** `plot_multi_line(xy_pairs, sem_pairs = None, fig_size = (10, 6), title = '', x_label = '', y_label = None, y_labels = None, x_lim = None, y_lim = None, colors = None, line_styles = None, save = False, font_size = 12, title_size = 14, legend_size = 10, legend_loc = 'upper left', axvline = None, axhline = None, ax = None)`
**Description:** SEM 음영을 포함한 다중 라인 그래프.

---

---

## fp_behav.io — 파일 입출력

### files.py

#### `set_working_directory`
**Signature:** `set_working_directory(base_folder: str) -> str`
**Description:** 기본 경로와 하위 폴더로 작업 디렉토리 생성 및 이동.

---

#### `grab_files`
**Signature:** `grab_files(folder_path: str, ext: str = '', recursive: bool = False) -> List[str]`
**Description:** 폴더에서 특정 확장자 파일 경로 목록 반환.

---

#### `grab_folders`
**Signature:** `grab_folders(folder_path: str, recursive: bool = False, names_only: bool = False) -> List[str]`
**Description:** 디렉토리에서 하위 폴더 경로 또는 이름 목록 반환.

---

#### `get_dirname_and_basename`
**Signature:** `get_dirname_and_basename(path: str) -> PathInfo`
**Description:** 파일 경로에서 부모 폴더명과 파일명(확장자 제외) 추출.

---

#### `load_dataframes`
**Signature:** `load_dataframes(file_list: List[str], file_type: str = 'pickle', trace_start_idx: int = 1)`
**Description:** 여러 데이터 파일 로드 및 트레이스/시간 컬럼 추출.

---

#### `load_config`
**Signature:** `load_config(config_path: str) -> dict`
**Description:** YAML 파일에서 설정 로드.

---

#### `save_config_copy`
**Signature:** `save_config_copy(cfg: dict, output_dir: Path) -> None`
**Description:** 분석에 사용된 설정 사본 저장 (재현성).

---

#### `temp_chdir`
**Signature:** `temp_chdir(path)`
**Description:** 임시 작업 디렉토리 변경 컨텍스트 매니저.

---

#### `ensure_dir`
**Signature:** `ensure_dir(path: str) -> str`
**Description:** 지정 경로 디렉토리 존재 보장 (없으면 생성).

---

#### `load_yaml`
**Signature:** `load_yaml(path: str) -> Dict`
**Description:** YAML 파일 로드 → 딕셔너리 반환.

---

### report.py

#### `FP_preprocessing`
**Signature:** `FP_preprocessing(output_path: str, title: str, image_paths: list, comments: list)`
**Description:** 이미지, 코멘트, 제목, 날짜/시간을 포함한 PDF 리포트 생성.

---

---

## fp_behav.core — 공통 유틸리티

### config.py

#### `deep_merge`
**Signature:** `deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]`
**Description:** 두 딕셔너리 재귀적 병합.

---

#### `get_cfg`
**Signature:** `get_cfg(config_path: Optional[str]) -> Dict[str, Any]`
**Description:** YAML 설정 파일 로드.

---

#### `pick`
**Signature:** `pick(cli_val, cfg: Dict[str, Any], key: str, default = None)`
**Description:** CLI 값 우선, 없으면 cfg[key], 둘 다 없으면 default.

---

#### `resolve_path`
**Signature:** `resolve_path(cli_path: Optional[List[str]], cfg: Dict, config_key: str, config_section: Optional[str] = None, is_input: bool = False) -> List[str]`
**Description:** CLI 인수 또는 설정 딕셔너리에서 파일 경로 해석.

---

### logging.py

#### `setup_logging`
**Signature:** `setup_logging(log_dir = 'logs', level = logging.INFO) -> Tuple[logging.Logger, str]`
**Description:** 파일 및 콘솔 출력을 위한 프로젝트 전반 로거 설정. Snakemake 파이프라인 모드 자동 감지 지원 (`PIPELINE_LOG_FILE` 환경변수).

---

#### `setup_named_logger`
**Signature:** `setup_named_logger(logger_name: str, log_dir: str = 'logs', level = logging.INFO) -> Tuple[logging.Logger, str]`
**Description:** 타임스탬프 파일에 기록하는 이름 있는 로거 생성.

---

#### `StreamToLogger`
**Description:** `print()` 출력을 logging 모듈로 리디렉션하는 클래스. Jupyter 노트북 및 독립 스크립트에서 모두 사용 가능.

---
