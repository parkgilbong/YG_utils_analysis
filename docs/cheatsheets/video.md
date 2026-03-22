# Video Cheatsheet

영상 처리 및 조작 빠른 참조.

## 주요 임포트 경로

```python
from fp_behav.video.functions import (
    extract_frames,
    extract_video_slices,
    flip_video,
    resize_video,
    add_inset_chart,
    create_video_from_images,
    VideoChopper,
    Generate_montage,
    extract_first_frame_and_draw_rois,
    create_animated_chart,
)
```

---

## Top 5 사용 사례

### 1. 특정 프레임 추출

```python
from fp_behav.video.functions import extract_frames

extract_frames(
    video_path='/path/to/video.mp4',
    frame_indices=[100, 500, 1000, 1500, 2000],
    output_folder='/path/to/output/frames'
)
# 출력: frame_100.png, frame_500.png, ...
```

**결과:** 지정 프레임을 PNG로 저장 (논문 figure 준비, QC용)

---

### 2. 비디오 구간 추출

```python
from fp_behav.video.functions import extract_video_slices
import pandas as pd

slices_df = pd.DataFrame({
    'start_frame': [250, 1000, 2500],
    'end_frame':   [500, 1250, 2750]
})

extract_video_slices(
    video_path='/path/to/full_video.mp4',
    slices_df=slices_df,
    output_folder='/path/to/output/slices'
)
# 출력: slice_0.avi, slice_1.avi, slice_2.avi
```

**결과:** 특정 시간 구간을 별도 비디오 파일로 추출

---

### 3. 비디오 플립 (좌우/상하 반전)

```python
from fp_behav.video.functions import flip_video

flip_video(
    input_path='/path/to/input_video.mp4',
    output_path='/path/to/flipped_video.mp4'
)
```

**결과:** 카메라 방향 표준화를 위한 비디오 수직 반전

---

### 4. 비디오 리사이징

```python
from fp_behav.video.functions import resize_video

# 50% 축소
resize_video(
    input_path='/path/to/large_video.mp4',
    output_path='/path/to/small_video.mp4',
    scale_factor=0.5
)
```

**결과:** 파일 크기 축소 또는 발표용 해상도 조정

---

### 5. FP 신호 차트 오버레이

```python
from fp_behav.video.functions import add_inset_chart

add_inset_chart(
    video_slice_path='/path/to/behavior_video.avi',
    chart_path='/path/to/calcium_trace.png',
    filename='behavior_with_calcium',
    position=('right', 'bottom'),
    chart_width=480
)
```

**결과:** 행동 영상 위에 FP 신호 플롯을 합성한 복합 비디오 생성

---

## 고급 비디오 처리

### 몽타주 생성 (격자 레이아웃)

```python
from fp_behav.video.functions import Generate_montage

Generate_montage(
    input_folder='/path/to/videos',
    output_filename='montage_3x3.mp4',
    rows=3,
    cols=3,
    frame_rate=25,
    duration=600,
    titles=['Mouse 1', 'Mouse 2', 'Mouse 3',
            'Mouse 4', 'Mouse 5', 'Mouse 6',
            'Mouse 7', 'Mouse 8', 'Mouse 9'],
    scale_factor=0.3
)
```

---

### 비디오 청크 분할

```python
from fp_behav.video.functions import VideoChopper

VideoChopper(
    input_file='/path/to/long_video.mp4',
    chunk_duration=60,
    tags=['exp1', 'trial1'],
    startingIdx=0
)
# 출력: exp1_trial1_000.mp4, exp1_trial1_001.mp4, ...
```

---

### 이미지 시퀀스로 비디오 생성

```python
from fp_behav.video.functions import create_video_from_images

create_video_from_images(
    image_folder='/path/to/frames',
    output_filename='reconstructed_video.mp4',
    frame_rate=25,
    duration=600,
    codec='mp4v',
    quality=95
)
```

---

### ROI를 첫 프레임에 그리기

```python
from fp_behav.video.functions import extract_first_frame_and_draw_rois

rois = [
    [(100, 100), (300, 100), (300, 300), (100, 300)],  # Social zone
    [(400, 400), (600, 400), (600, 600), (400, 600)]   # Object zone
]

extract_first_frame_and_draw_rois(
    video_path='/path/to/video.mp4',
    rois=rois,
    output_image_path='/path/to/arena_with_rois.png'
)
```

---

### Snakemake에서 비디오 처리

```bash
# video.smk 규칙 실행
snakemake --configfile configs/behavior.yaml results/mouse001/session01/behavior/video_flipped.avi -j 1
```

---

## Tips

1. **비디오 포맷**: 입력 `.mp4` / `.avi` 모두 지원, 출력은 공유용 `.mp4`, 후처리용 `.avi`
2. **코덱 선택**: `'mp4v'` - 범용 호환, `'XVID'` - 고품질, `'H264'` - 최고 압축
3. **리사이징**: scale_factor 0.5 = 50% 크기 = 파일 크기 약 1/4
4. **차트 오버레이**: 차트 너비는 비디오 너비의 1/3 이하 권장

---

## 관련 문서

- [DLC Cheatsheet](dlc.md)
- [FP Cheatsheet](fp.md)
