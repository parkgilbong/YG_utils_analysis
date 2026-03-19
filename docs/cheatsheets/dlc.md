# DLC Cheatsheet

DeepLabCut 데이터 처리 및 행동 분석 빠른 참조.

## 주요 임포트 경로

```python
from fp_behav.behavior.dlc import (
    df_to_dic_single,
    df_to_dic_multi,
    get_velocity,
    roi_entry_analysis,
    get_bodypoints_distance,
    annotate_body_part_proximity,
    PostDLC_3CT_3EVTs,
    check_point_in_regions,
)
```

---

## Top 5 사용 사례

### 1. DLC DataFrame을 딕셔너리로 변환

```python
from fp_behav.behavior.dlc import df_to_dic_single
import pandas as pd

dlc_df = pd.read_hdf('video_DLC.h5')

dlc_dict = df_to_dic_single(
    df=dlc_df,
    ignore_bodyparts='PatchCordBase'
)

# 좌표 접근
nose_x = dlc_dict['Nose']['x']
nose_y = dlc_dict['Nose']['y']
nose_likelihood = dlc_dict['Nose']['likelihood']
```

**결과:** DLC의 복잡한 DataFrame을 단순 딕셔너리 구조로 변환

---

### 2. Body Part 속도 계산

```python
from fp_behav.behavior.dlc import get_velocity

velocity_result = get_velocity(
    DLCresult=dlc_dict,
    bpt='Center',
    FPS=25,
    pcutoff=0.95
)

velocity = velocity_result['velocity']  # pixels/second
time = velocity_result['time']
```

**결과:** 위치 트래킹 데이터로부터 프레임별 속도 계산

---

### 3. ROI 진입 이벤트 분석

```python
from fp_behav.behavior.dlc import roi_entry_analysis

roi = [100, 300, 150, 350]  # [x_min, x_max, y_min, y_max]

entry_result = roi_entry_analysis(
    DLCresult=dlc_dict,
    bpt='Center',
    pcutoff=0.9,
    ROI=roi
)

in_roi = entry_result['in_ROI']          # Boolean array
entry_bouts = entry_result['entry_bouts'] # Start/end frames
```

**결과:** 동물이 ROI에 진입/이탈하는 시점 감지

---

### 4. Body Part 간 거리 측정

```python
from fp_behav.behavior.dlc import get_bodypoints_distance

distance_result = get_bodypoints_distance(
    DLCresult=dlc_dict,
    bpt='Nose',
    bpt2='Object',
    pcutoff=0.95,
    distance_thres=30
)

distance = distance_result['distance']    # 프레임별 거리
in_contact = distance_result['in_contact'] # 접촉 boolean
contact_bouts = distance_result['bouts']  # 접촉 에피소드
```

**결과:** 두 트래킹 포인트 간 거리 계산 및 접촉 이벤트 감지

---

### 5. 3-Chamber Test 분석

```python
from fp_behav.behavior.dlc import PostDLC_3CT_3EVTs

results = PostDLC_3CT_3EVTs(
    DLCresult=dlc_dict,
    destfolder='/path/to/output',
    ROI='new',
    Nose2Snout_dist=30,
    Evt1=(0.5, 2),  # Nose-poke: (min_duration, min_interval)
    Evt2=(2, 2),    # S-Zone
    Evt3=(2, 2),    # E-Zone
    FPS=25,
    SaveData=True
)

print(f"Social Preference Index: {results['SPI']}")
print(f"Time in Social Zone: {results['Time_SZone']}s")
print(f"Number of Nose-pokes: {results['N_Nosepoke']}")
```

**결과:** 3-chamber 사회적 선호도 테스트 전체 분석

---

## 고급 DLC 처리

### 멀티 동물 트래킹

```python
from fp_behav.behavior.dlc import df_to_dic_multi, get_velocity

animal1_dict, animal2_dict = df_to_dic_multi(
    df=dlc_df,
    ignore_bodyparts='PatchCordBase'
)

velocity1 = get_velocity(animal1_dict, 'Center', FPS=25)
velocity2 = get_velocity(animal2_dict, 'Center', FPS=25)
```

---

### Body Part Proximity 어노테이션

```python
from fp_behav.behavior.dlc import annotate_body_part_proximity
import pandas as pd

points_df = pd.DataFrame({'frame': range(len(dlc_dict['Nose']['x']))})

annotated_df = annotate_body_part_proximity(
    body_part_data1=animal1_dict,
    body_part_data2=animal2_dict,
    points_df=points_df,
    body_part_name1='Nose',
    body_part_name2='Nose',
    pcutoff=0.9,
    d_threshold=50,
    d_threshold2=150
)

close_frames = annotated_df[annotated_df['proximity'] == 'close']
```

---

### DLC → BORIS 변환

```python
# 3-Chamber Test 변환 (ct 변형)
from fp_behav.behavior.boris.ct import main as dlc2boris_ct
dlc2boris_ct()

# Direct Interaction 변환 (di 변형)
from fp_behav.behavior.boris.di import main as dlc2boris_di
dlc2boris_di()

# 기본 변환
from fp_behav.behavior.boris.base import main as dlc2boris_base
dlc2boris_base()
```

CLI 사용:

```bash
behav-dlc2boris --config configs/behavior.yaml --variant ct   # base / ct / di
```

---

## Tips

1. **Likelihood 필터링**: `pcutoff=0.9` 기본, 트래킹이 드문 경우 낮추기
2. **좌표 시스템**: DLC는 이미지 좌표 (0,0 = 좌상단), Y는 아래로 증가
3. **속도 계산**: 노이즈가 많으면 이동 평균으로 스무딩 후 사용
4. **ROI 정의**: 첫 프레임으로 ROI를 시각적으로 정의 → `extract_first_frame_and_draw_rois` 활용

---

## 관련 문서

- [Video Cheatsheet](video.md)
- [FP Cheatsheet](fp.md)
- [Files Cheatsheet](files.md)
