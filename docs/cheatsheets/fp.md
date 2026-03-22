# FP Cheatsheet

Fiber Photometry 데이터 처리 빠른 참조.

## 주요 임포트 경로

```python
from fp_behav.fp.functions import (
    FP_preprocessing_1ch,
    FP_preprocessing_2ch_new,
    Peak_Analysis,
    Epoch_Analysis_3EVT,
    Import_manual_scoring,
    calculate_auc,
    extract_traces_with_padding,
    detect_slow_peaks,
)
```

---

## Top 5 사용 사례

### 1. 1채널 FP 전처리 (TDT 시스템)

```python
from fp_behav.fp.functions import FP_preprocessing_1ch

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

**결과:** 원시 FP 데이터 로드 → 지수함수 피팅 디트렌딩 → dF/F 정규화 → CSV/PNG 저장

---

### 2. 2채널 FP 전처리 (465nm & 560nm)

```python
from fp_behav.fp.functions import FP_preprocessing_2ch_new

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

**결과:** GCaMP(465nm)와 컨트롤(560nm) 채널을 동시에 처리

---

### 3. Epoch 분석 (이벤트 중심)

```python
from fp_behav.fp.functions import Epoch_Analysis_3EVT

Epoch_Analysis_3EVT(
    pkl_path='Final_table_raw_trace.pkl',
    evt_path='Data_DLC.csv',
    PRE_TIME=5,
    POST_TIME=10,
    FPS=25,
    Rec_duration=600,
    SaveData=True
)
```

**결과:** 행동 이벤트에 정렬된 신호 트레이스 추출 및 baseline 보정

---

### 4. Peak 감지

```python
from fp_behav.fp.functions import Peak_Analysis

Peak_Analysis(
    pkl_path='Final_table_raw_trace.pkl',
    signal2use='Zscore',
    prominence_thres=2,
    amplitude_thres=4,
    FPS=25,
    pre_window_len=3,
    post_window_len=3,
    SaveData=True
)
```

**결과:** prominence/amplitude 기준으로 칼슘 transient 감지

---

### 5. 수동 행동 스코어링 불러오기

```python
from fp_behav.fp.functions import Import_manual_scoring

events = Import_manual_scoring(
    file_path='/path/to/scoring.tsv',
    FPS=25,
    Event='Social_Contact',
    UseFilter=True,
    MinDuration=0.5,
    MinInterval=2.0
)
```

**결과:** 행동 어노테이션 로드 → duration/interval 필터링 → onset/offset 시간 반환

---

## 추가 함수

### AUC 계산

```python
from fp_behav.fp.functions import calculate_auc

auc_values = calculate_auc(
    time=time_array,
    signal=dff_trace,
    intervals=[(10, 20), (30, 40)]
)
```

### 패딩 포함 트레이스 추출

```python
from fp_behav.fp.functions import extract_traces_with_padding

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

### Slow Peak 감지

```python
from fp_behav.fp.functions import detect_slow_peaks

peaks = detect_slow_peaks(
    signal=dff_trace,
    sampling_rate=25,
    height=1.3,
    min_interval=2.0,
    min_peak_width=0.5
)
```

### RWD 시스템 데이터 로드

```python
from fp_behav.fp.loaders import load_fluorescence

settings, df = load_fluorescence('/path/to/Fluorescence.csv')
```

---

## 파이프라인 실행

```bash
# CLI
fp-preprocess     --config configs/fp_1ch.yaml
fp-preprocess-2ch --config configs/fp_2ch.yaml
fp-epoch          --config configs/epoch.yaml
fp-peak           --config configs/peak.yaml

# Snakemake
snakemake --configfile configs/fp_1ch.yaml -j 4
```

---

## Tips

1. **샘플링 레이트**: TDT 시스템은 ~1017 Hz로 기록하지만, 행동 정렬은 카메라 FPS(25 Hz) 사용
2. **디트렌딩 선택**: `'Exp_fit'` - 지수적 감쇠(일반적), `'Highpass_filter'` - 선형 드리프트
3. **Baseline 보정**: 이벤트 전 -5 ~ -1초 구간 사용 권장
4. **중간 파일 저장**: `SaveAsCSV=True` 로 QC용 중간 파일 보존

---

## 관련 문서

- [Plot Cheatsheet](plot.md)
- [Files Cheatsheet](files.md)
- [DLC Cheatsheet](dlc.md)
