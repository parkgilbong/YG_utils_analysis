# fp_behav

Fiber Photometry & Behavior Analysis Toolkit — Python 패키지. 여러 연구 프로젝트에서 Fiber Photometry(FP)와 행동 데이터 분석 코드의 재사용성, 일관성, 유지보수성을 높이기 위해 만들어졌습니다.

## 주요 기능

- **Fiber Photometry 전처리**: 1채널/2채널 FP 데이터의 노이즈 제거, 디트렌딩, 정규화
- **행동 데이터 분석**: DeepLabCut(DLC) 트래킹 → BORIS 변환, 비디오 처리
- **Epoch / Peak 분석**: YAML 기반 설정으로 구동되는 이벤트 중심 분석 파이프라인
- **Snakemake 워크플로우**: 파일 단위 실행이 가능한 재현 가능한 파이프라인
- **CLI**: 표준 분석 워크플로우를 실행하는 커맨드라인 도구

## 설치

### 표준 설치

```bash
pip install git+https://github.com/parkgilbong/YG_utils_analysis.git
```

### 개발 설치

```bash
git clone https://github.com/parkgilbong/YG_utils_analysis.git
cd YG_utils_analysis
pip install -e .
```

TDT 하드웨어를 사용한다면:

```bash
pip install -e ".[tdt]"
```

## 사용법

### 1. Python에서 함수 임포트

패키지는 도메인별 서브패키지로 구성됩니다.

```python
# Fiber Photometry
from fp_behav.fp.functions import FP_preprocessing_1ch
from fp_behav.fp.functions import Peak_Analysis, extract_traces_with_padding

# 행동 분석
from fp_behav.behavior.dlc import df_to_dic_single, get_velocity
from fp_behav.behavior.boris.ct import main as dlc2boris_ct

# 그룹 분석
from fp_behav.group.summary import process_analysis

# 시각화
from fp_behav.plot.functions import plot_traces_with_mean, plot_multi_line

# 파일/IO
from fp_behav.io.files import grab_files, grab_folders, load_config
```

### 2. 커맨드라인 도구

```bash
# 1채널 FP 전처리
fp-preprocess --config configs/fp_1ch.yaml

# 2채널 FP 전처리
fp-preprocess-2ch --config configs/fp_2ch.yaml

# Epoch 분석
fp-epoch --config configs/epoch.yaml

# Peak 분석
fp-peak --config configs/peak.yaml

# 행동 비디오 전처리
behav-preprocess --config configs/behavior.yaml

# DLC → BORIS 변환
behav-dlc2boris --config configs/behavior.yaml --variant base   # base / ct / di
```

### 3. Snakemake 파이프라인

파일 단위로 병렬 실행이 가능한 재현 가능한 파이프라인:

```bash
# 설정 파일을 지정해서 전체 파이프라인 실행
snakemake --configfile configs/fp_1ch.yaml -j 4

# dry-run으로 실행 계획 확인
snakemake --configfile configs/fp_1ch.yaml -j 4 -n

# conda 환경 자동 생성 포함
snakemake --configfile configs/fp_1ch.yaml -j 4 --use-conda
```

## 패키지 구조

```
src/fp_behav/
├── fp/                  # Fiber Photometry
│   ├── functions.py     # 핵심 신호처리 함수 (전처리, epoch, peak 등)
│   ├── loaders.py       # RWD 시스템 데이터 로더
│   ├── preprocessing.py # 1채널 전처리 파이프라인
│   ├── preprocessing_2ch.py  # 2채널 전처리 파이프라인
│   ├── epoch.py         # Epoch 분석 파이프라인
│   └── peak.py          # Peak 분석 파이프라인
│
├── behavior/            # 행동 분석
│   ├── dlc.py           # DLC 데이터 처리 함수
│   ├── preprocessing.py # 행동 비디오 전처리 파이프라인
│   ├── export.py        # BORIS 포맷 내보내기
│   └── boris/           # DLC → BORIS 변환 (실험 유형별)
│       ├── base.py      # 기본 변환
│       ├── ct.py        # 3-Chamber Test 변환
│       └── di.py        # Direct Interaction 변환
│
├── video/               # 영상 처리
│   └── functions.py     # 프레임 추출, 슬라이싱, 리사이징, 몽타주 등
│
├── group/               # 그룹 수준 분석
│   └── summary.py       # 그룹 평균/SEM, 멀티그룹 비교
│
├── plot/                # 시각화
│   └── functions.py     # 시계열, 히트맵, 멀티라인 플롯
│
├── io/                  # 파일 입출력
│   ├── files.py         # 파일/폴더 유틸리티, YAML 로더
│   └── report.py        # PDF 리포트 생성
│
├── core/                # 공통 유틸리티
│   ├── config.py        # 설정 관리 (병합, 경로 해석)
│   └── logging.py       # 로깅 설정
│
└── cli/                 # CLI 진입점
    ├── fp_commands.py
    └── behavior_commands.py

workflow/                # Snakemake 파이프라인
├── Snakefile
├── rules/
│   ├── fp.smk
│   ├── behavior.smk
│   └── video.smk
├── envs/environment.yaml
└── schemas/config.schema.yaml

configs/                 # 설정 템플릿
├── fp_1ch.yaml
├── fp_2ch.yaml
└── behavior.yaml
```

## 문서 및 Cheatsheet

| 문서 | 내용 |
|------|------|
| [FP Cheatsheet](docs/cheatsheets/fp.md) | Fiber Photometry 데이터 처리 |
| [DLC Cheatsheet](docs/cheatsheets/dlc.md) | DeepLabCut 트래킹 분석 |
| [Video Cheatsheet](docs/cheatsheets/video.md) | 비디오 처리 |
| [Plot Cheatsheet](docs/cheatsheets/plot.md) | 시각화 |
| [Files Cheatsheet](docs/cheatsheets/files.md) | 파일 I/O 및 설정 관리 |
| [Function Summary](docs/utils_summary.md) | 전체 함수 목록 및 시그니처 |
| [Examples Notebook](examples/notebooks/examples.ipynb) | 실제 사용 예시 |

## 라이선스

MIT License. 자세한 내용은 `LICENSE.txt`를 참고하세요.
