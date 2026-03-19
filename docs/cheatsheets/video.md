# VideoFunctions Cheatsheet

Quick reference for common video processing and manipulation tasks.

## 🎬 Top 5 Most Common Use Cases

### 1. Extract Specific Frames from Video

```python
from utils.VideoFunctions import extract_frames

# Extract frames at specific indices
video_path = '/path/to/video.mp4'
frame_indices = [100, 500, 1000, 1500, 2000]
output_folder = '/path/to/output/frames'

extract_frames(
    video_path=video_path,
    frame_indices=frame_indices,
    output_folder=output_folder
)
# Output: frame_100.png, frame_500.png, etc.
```

**What it does:** Extracts and saves specific frames as PNG images for figure preparation or QC.

---

### 2. Extract Video Slices/Segments

```python
from utils.VideoFunctions import extract_video_slices
import pandas as pd

# Define time segments to extract
slices_df = pd.DataFrame({
    'start_frame': [250, 1000, 2500],  # Start frames
    'end_frame':   [500, 1250, 2750]   # End frames
})

extract_video_slices(
    video_path='/path/to/full_video.mp4',
    slices_df=slices_df,
    output_folder='/path/to/output/slices'
)
# Output: slice_0.avi, slice_1.avi, slice_2.avi
```

**What it does:** Extracts specific time segments as separate video files, useful for highlighting behaviors.

---

### 3. Flip/Mirror Videos

```python
from utils.VideoFunctions import flip_video

# Horizontally flip video (mirror)
flip_video(
    input_path='/path/to/input_video.mp4',
    output_path='/path/to/flipped_video.mp4'
)
```

**What it does:** Horizontally flips videos, useful for standardizing camera orientation.

---

### 4. Resize Videos

```python
from utils.VideoFunctions import resize_video

# Reduce video size by 50%
resize_video(
    input_path='/path/to/large_video.mp4',
    output_path='/path/to/small_video.mp4',
    scale_factor=0.5  # 0.5 = 50% of original size
)

# Enlarge video
resize_video(
    input_path='/path/to/small_video.mp4',
    output_path='/path/to/large_video.mp4',
    scale_factor=2.0  # 2x larger
)
```

**What it does:** Resizes videos to reduce file size or prepare for presentations.

---

### 5. Add FP Signal Chart Overlay to Video

```python
from utils.VideoFunctions import add_inset_chart

# Overlay calcium signal plot on video
add_inset_chart(
    video_slice_path='/path/to/behavior_video.avi',
    chart_path='/path/to/calcium_trace.png',
    filename='behavior_with_calcium',
    position=('right', 'bottom'),  # Chart position
    chart_width=480  # Width of chart in pixels
)
```

**What it does:** Creates a composite video with behavioral footage and synchronized FP signal.

---

## 🎨 Advanced Video Processing

### Create Video Montage (Grid Layout)

```python
from utils.VideoFunctions import Generate_montage

# Create 3x3 grid of videos
Generate_montage(
    input_folder='/path/to/videos',
    output_filename='montage_3x3.mp4',
    rows=3,
    cols=3,
    frame_rate=25,
    duration=600,  # seconds
    codec='mp4v',
    titles=['Mouse 1', 'Mouse 2', 'Mouse 3',
            'Mouse 4', 'Mouse 5', 'Mouse 6',
            'Mouse 7', 'Mouse 8', 'Mouse 9'],
    scale_factor=0.3  # Scale each video to 30%
)
```

**What it does:** Combines multiple videos into a grid layout for simultaneous viewing.

---

### Chop Video into Chunks

```python
from utils.VideoFunctions import VideoChopper

# Split long video into 60-second chunks
VideoChopper(
    input_file='/path/to/long_video.mp4',
    chunk_duration=60,  # seconds per chunk
    tags=['exp1', 'trial1'],  # Tags for filename
    startingIdx=0
)
# Output: exp1_trial1_000.mp4, exp1_trial1_001.mp4, etc.
```

**What it does:** Splits long recordings into manageable chunks.

---

### Create Video from Image Sequence

```python
from utils.VideoFunctions import create_video_from_images

# Combine frames into video
create_video_from_images(
    image_folder='/path/to/frames',
    output_filename='reconstructed_video.mp4',
    frame_rate=25,
    duration=600,
    codec='mp4v',
    quality=95
)
```

**What it does:** Assembles individual frames into a video file.

---

### Draw ROIs on First Frame

```python
from utils.VideoFunctions import extract_first_frame_and_draw_rois

# Define regions of interest
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

**What it does:** Creates arena diagram with ROIs marked for methods figures.

---

### Create Animated Chart

```python
from utils.VideoFunctions import create_animated_chart

# Create animated plot that syncs with video
data = [
    {'time': 0.0, 'signal': 0.5, 'baseline': 0.0},
    {'time': 0.04, 'signal': 0.6, 'baseline': 0.0},
    # ... more data points
]

create_animated_chart(
    data=data,
    filename='animated_trace.mp4',
    interval=40,  # ms between frames (25 FPS = 40 ms)
    offset=0.0
)
```

**What it does:** Creates an animated line plot that can be synced with behavioral video.

---

## 💡 Tips & Best Practices

1. **Video formats**:
   - Input: `.mp4`, `.avi` both supported
   - Output: Use `.mp4` for sharing, `.avi` for further processing

2. **Codec choices**:
   - `'mp4v'`: Good compression, widely compatible
   - `'XVID'`: Better quality, larger files
   - `'H264'`: Best compression (may need extra codecs)

3. **Frame extraction**:
   - Extract frames at key timepoints for figure panels
   - Use 300 DPI PNG for publication quality

4. **Resizing**:
   - Scale factor 0.5 = 50% size, 1/4 file size
   - Always resize before uploading/sharing

5. **Chart overlays**:
   - Position: `('right', 'bottom')` usually works best
   - Keep chart width ≤ 1/3 of video width
   - Ensure time synchronization between video and chart

6. **Processing large files**:
   - Process in chunks for very long videos (>1 hour)
   - Free disk space should be 3x video size

---

## 🔗 Related

- See [DLCFunctions Cheatsheet](cheatsheet_DLCFunctions.md) for tracking analysis
- See [FPFunctions Cheatsheet](cheatsheet_FPFunctions.md) for signal analysis
