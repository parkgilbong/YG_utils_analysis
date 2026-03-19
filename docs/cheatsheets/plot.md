# Plot Cheatsheet

시각화 빠른 참조.

## 주요 임포트 경로

```python
from fp_behav.plot.functions import (
    plot_single_line,
    plot_dual_line,
    plot_traces_with_mean,
    plot_trace_heatmap,
    plot_multi_line,
)
```

---

## Top 5 사용 사례

### 1. 단일 시계열 플롯

```python
from fp_behav.plot.functions import plot_single_line
import numpy as np

time = np.linspace(0, 10, 1000)
signal = np.sin(time)

plot_single_line(
    x=time,
    y=signal,
    fig_size=(12, 4),
    fig_title='Calcium Signal',
    x_label='Time (s)',
    y_label='dF/F (z-score)',
    x_lim=(0, 10),
    y_lim=(-3, 3),
    color='green',
    save=True
)
```

**결과:** 커스터마이즈 가능한 논문용 단일 라인 플롯

---

### 2. 두 신호 이중 Y축 플롯

```python
from fp_behav.plot.functions import plot_dual_line

plot_dual_line(
    x1=time, y1=calcium_signal,
    x2=time, y2=behavior_score,
    fig_size=(14, 6),
    fig_title='FP Signal with Behavior',
    x_label='Time (s)',
    y1_label='dF/F',
    y2_label='Social Index',
    x_lim=(0, 600),
    y1_lim=(-2, 4),
    y2_lim=(0, 1),
    color1='green',
    color2='orange',
    save=True
)
```

**결과:** 독립 Y축을 가진 두 시계열 오버레이 (FP + 행동 동시 표시에 유용)

---

### 3. 평균 ± SEM 트레이스 플롯

```python
from fp_behav.plot.functions import plot_traces_with_mean
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

plot_traces_with_mean(
    trace_array=epoch_traces,  # Shape: (n_trials, n_timepoints)
    trace_time=time_vector,
    ax=ax,
    color='blue',
    title='Peri-Event Traces',
    xlabel='Time from Event (s)',
    ylabel='dF/F (z-score)',
    mode='sem'  # 'std' 도 가능
)

plt.axvline(0, color='red', linestyle='--', label='Event Onset')
plt.legend()
plt.tight_layout()
plt.savefig('epoch_analysis.png', dpi=300)
```

**결과:** 평균 ± 음영 오차 영역(SEM/SD)으로 여러 트라이얼 시각화

---

### 4. 트라이얼별 히트맵

```python
from fp_behav.plot.functions import plot_trace_heatmap
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

plot_trace_heatmap(
    traces=epoch_traces,
    trace_time=time_vector,
    vmin=-2,
    vmax=3,
    ax=ax,
    title='Trial-by-Trial Calcium Activity',
    xlabel='Time from Event (s)',
    ylabel='Trial #',
    cmap='RdYlBu_r'
)

plt.axvline(0, color='white', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300)
```

**결과:** 개별 트라이얼 반응을 시간에 따라 히트맵으로 표시

---

### 5. 멀티 그룹 비교

```python
from fp_behav.plot.functions import plot_multi_line

xy_pairs = [
    (time, control_mean, 'Control'),
    (time, drug_mean, 'Drug'),
    (time, stress_mean, 'Stress')
]

sem_pairs = [
    (time, control_sem),
    (time, drug_sem),
    (time, stress_sem)
]

plot_multi_line(
    xy_pairs=xy_pairs,
    sem_pairs=sem_pairs,
    fig_size=(12, 6),
    title='Group Comparison',
    x_label='Time (s)',
    y_label='dF/F',
    colors=['gray', 'red', 'blue'],
    save=True
)
```

**결과:** 오차 밴드를 포함한 여러 그룹 평균 오버레이

---

## 고급 플로팅

### 멀티 패널 피규어

```python
import matplotlib.pyplot as plt
from fp_behav.plot.functions import plot_single_line, plot_traces_with_mean, plot_trace_heatmap

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Raw trace
plot_single_line(
    x=time, y=raw_trace,
    fig_size=None,
    fig_title='A. Raw Signal',
    x_label='Time (s)', y_label='Fluorescence',
    x_lim=(0, 100), y_lim=None,
    color='black',
    ax=axes[0, 0]
)

# Panel B: dF/F trace
plot_single_line(
    x=time, y=dff_trace,
    fig_size=None,
    fig_title='B. dF/F Signal',
    x_label='Time (s)', y_label='dF/F',
    x_lim=(0, 100), y_lim=None,
    color='green',
    ax=axes[0, 1]
)

# Panel C: Epoch average
plot_traces_with_mean(
    trace_array=epoch_traces,
    trace_time=epoch_time,
    ax=axes[1, 0],
    title='C. Peri-Event Average',
    xlabel='Time (s)', ylabel='dF/F'
)

# Panel D: Heatmap
plot_trace_heatmap(
    traces=epoch_traces,
    trace_time=epoch_time,
    ax=axes[1, 1],
    title='D. Trial Heatmap'
)

plt.tight_layout()
plt.savefig('figure_panel.png', dpi=300)
```

---

## Tips

1. **Figure 크기**: 시계열 (12, 4), 비교 (10, 6), 히트맵 (10, 8)
2. **색상 선택**: 초록/청록 = 칼슘신호(GCaMP), 파랑 = 대조, 빨강/주황 = 이벤트, 회색 = 대조군
3. **DPI**: 논문 300, 발표 150, 빠른 확인 72
4. **이벤트 마커**: 중요 시점(t=0, 자극)은 항상 수직선으로 표시
5. **오차 밴드**: 그룹 비교에는 SEM, 개별 변동성에는 STD

---

## 관련 문서

- [FP Cheatsheet](fp.md)
- [Video Cheatsheet](video.md)
