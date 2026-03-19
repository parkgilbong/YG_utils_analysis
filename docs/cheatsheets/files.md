# Files Cheatsheet

파일 I/O, 설정 관리, 디렉토리 유틸리티 빠른 참조.

## 주요 임포트 경로

```python
from fp_behav.io.files import (
    set_working_directory,
    grab_files,
    grab_folders,
    get_dirname_and_basename,
    load_dataframes,
    load_config,
    save_config_copy,
    temp_chdir,
    ensure_dir,
    load_yaml,
)

from fp_behav.core.config import (
    get_cfg,
    deep_merge,
    pick,
    resolve_path,
)

from fp_behav.core.logging import (
    setup_logging,
    setup_named_logger,
)
```

---

## Top 5 사용 사례

### 1. 작업 디렉토리 자동 생성 및 이동

```python
from fp_behav.io.files import set_working_directory

output_path = set_working_directory(
    '/path/to/project',
    'results',
    'experiment_2024',
    'group_A'
)
# /path/to/project/results/experiment_2024/group_A 생성 후 이동
```

**결과:** 중첩 디렉토리 생성 및 현재 작업 디렉토리 변경

---

### 2. 특정 확장자 파일 검색

```python
from fp_behav.io.files import grab_files

# CSV 파일만 찾기
csv_files = grab_files('/path/to/data', ext='.csv', recursive=False)

# 하위 폴더 포함 PKL 파일 검색
pkl_files = grab_files('/path/to/data', ext='.pkl', recursive=True)

print(f"Found {len(pkl_files)} pkl files")
```

**결과:** 지정 확장자의 파일 경로 목록 반환

---

### 3. YAML 설정 파일 로드

```python
from fp_behav.io.files import load_config

config = load_config('configs/fp_1ch.yaml')

tank_path = config['raw_data_path']
fps = config['fps']
rec_duration = config['rec_duration']
```

**결과:** YAML 설정 파일을 Python 딕셔너리로 로드

---

### 4. 여러 데이터 파일 일괄 로드

```python
from fp_behav.io.files import load_dataframes

file_list = [
    '/path/to/mouse1_data.pkl',
    '/path/to/mouse2_data.pkl',
    '/path/to/mouse3_data.pkl'
]

traces, time_vector, labels = load_dataframes(
    file_list=file_list,
    file_type='pickle',
    trace_start_idx=1
)

print(f"Loaded {traces.shape[0]} traces")
```

**결과:** 여러 파일 일괄 로드 및 그룹 분석용 배열로 결합

---

### 5. 임시 디렉토리 변경 (컨텍스트 매니저)

```python
from fp_behav.io.files import temp_chdir

with temp_chdir('/path/to/temporary/location'):
    # 이 블록 안에서만 해당 디렉토리 사용
    with open('temp_file.txt', 'w') as f:
        f.write('Temporary data')

# 자동으로 원래 디렉토리로 복귀
```

**결과:** 안전하게 디렉토리를 임시 변경하는 컨텍스트 매니저

---

## 추가 파일 유틸리티

### 폴더 목록 가져오기

```python
from fp_behav.io.files import grab_folders

# 전체 경로
folders = grab_folders('/path/to/data', recursive=False, names_only=False)

# 폴더 이름만
folder_names = grab_folders('/path/to/data', recursive=False, names_only=True)
```

---

### 파일 경로에서 부모 폴더/파일명 추출

```python
from fp_behav.io.files import get_dirname_and_basename

path_info = get_dirname_and_basename('/experiments/group_A/mouse_001/data.pkl')
print(f"Parent: {path_info.parent}")  # group_A
print(f"Stem: {path_info.stem}")      # data
```

---

### 디렉토리 존재 보장

```python
from fp_behav.io.files import ensure_dir

output_dir = ensure_dir('/path/to/output/directory')
# 없으면 생성, 있으면 그냥 경로 반환
```

---

### 설정 파일 사본 저장 (재현성)

```python
from fp_behav.io.files import save_config_copy, load_config
from pathlib import Path

config = load_config('configs/fp_1ch.yaml')

# 분석 실행 후 사용한 설정 보관
output_dir = Path('/path/to/results')
save_config_copy(config, output_dir)
# 저장: /path/to/results/config_used.yaml
```

---

## 설정 관리 (core.config)

```python
from fp_behav.core.config import get_cfg, deep_merge, pick

# YAML 로드
cfg = get_cfg('configs/fp_1ch.yaml')

# 두 설정 딕셔너리 병합 (update가 base를 덮어씀)
merged = deep_merge(base_cfg, user_cfg)

# CLI 인수 우선, 없으면 cfg 값, 둘 다 없으면 기본값
fps = pick(cli_fps, cfg, 'fps', default=25)
```

---

## 로깅 설정 (core.logging)

```python
from fp_behav.core.logging import setup_logging, setup_named_logger

# 기본 로거 설정
logger, log_path = setup_logging(log_dir='logs', level=logging.INFO)

# 이름 있는 로거 (타임스탬프 파일에 저장)
logger, log_path = setup_named_logger('fp_preprocess', log_dir='logs')

logger.info("Processing started")
logger.error("Something went wrong")
```

---

## 전형적인 워크플로우 패턴

```python
from fp_behav.io.files import load_config, set_working_directory, grab_files, save_config_copy
from fp_behav.core.logging import setup_logging
from pathlib import Path

# 1. 설정 로드
config = load_config('configs/fp_1ch.yaml')

# 2. 로깅 설정
setup_logging(log_dir='logs')

# 3. 출력 디렉토리 생성
output_dir = set_working_directory(config['base_folder'], config['batch_folder'])

# 4. 설정 사본 저장
save_config_copy(config, Path(output_dir))

# 5. 입력 파일 검색
input_files = grab_files(config['raw_data_path'], ext='.pkl', recursive=True)

# 6. 파일 처리
for file in input_files:
    process_file(file, config)
```

---

## Tips

1. **설정 파일 버전 관리**: YAML 파일을 git에 포함하고, 분석마다 `save_config_copy`로 사본 저장
2. **파일 명명 규칙**: `mouse001_session01_20240115.pkl` 처럼 날짜 포함
3. **배치 처리**: `grab_files + recursive=True`로 하위 폴더 전체 처리
4. **경로**: `pathlib.Path`로 크로스플랫폼 호환 경로 사용

---

## 관련 문서

- [FP Cheatsheet](fp.md)
- [DLC Cheatsheet](dlc.md)
