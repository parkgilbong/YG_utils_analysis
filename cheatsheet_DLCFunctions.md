# DLCFunctions Cheatsheet

Quick reference for DeepLabCut data processing and behavioral analysis.

## 🐭 Top 5 Most Common Use Cases

### 1. Convert DLC DataFrame to Dictionary

```python
from utils.DLCFunctions import df_to_dic_single
import pandas as pd

# Load DLC output
dlc_df = pd.read_hdf('video_DLC.h5')

# Convert to dictionary for easier processing
dlc_dict = df_to_dic_single(
    df=dlc_df,
    ignore_bodyparts='PatchCordBase'  # Skip this body part
)

# Access coordinates
nose_x = dlc_dict['Nose']['x']
nose_y = dlc_dict['Nose']['y']
nose_likelihood = dlc_dict['Nose']['likelihood']
```

**What it does:** Converts DLC's complex DataFrame into a simple dictionary structure.

---

### 2. Calculate Body Part Velocity

```python
from utils.DLCFunctions import get_velocity

# Calculate center point velocity
velocity_result = get_velocity(
    DLCresult=dlc_dict,
    bpt='Center',  # Body part name
    FPS=25,
    pcutoff=0.95  # Likelihood threshold
)

# Access velocity data
velocity = velocity_result['velocity']  # Pixels/second
time = velocity_result['time']  # Time vector
```

**What it does:** Computes frame-by-frame velocity from position tracking data.

---

### 3. Analyze ROI Entry Events

```python
from utils.DLCFunctions import roi_entry_analysis

# Define rectangular ROI
roi = [100, 300, 150, 350]  # [x_min, x_max, y_min, y_max]

# Detect entries into ROI
entry_result = roi_entry_analysis(
    DLCresult=dlc_dict,
    bpt='Center',  # Body part to track
    pcutoff=0.9,
    ROI=roi
)

# Access results
in_roi = entry_result['in_ROI']  # Boolean array
entry_bouts = entry_result['entry_bouts']  # Start/end frames
```

**What it does:** Identifies when an animal enters/exits a defined region of interest.

---

### 4. Measure Distance Between Body Parts

```python
from utils.DLCFunctions import get_bodypoints_distance

# Calculate nose-to-object distance
distance_result = get_bodypoints_distance(
    DLCresult=dlc_dict,
    bpt='Nose',
    bpt2='Object',
    pcutoff=0.95,
    distance_thres=30  # Threshold for "contact" (pixels)
)

# Access results
distance = distance_result['distance']  # Frame-by-frame distance
in_contact = distance_result['in_contact']  # Boolean for contact
contact_bouts = distance_result['bouts']  # Contact episodes
```

**What it does:** Computes distance between two tracked points and identifies contact events.

---

### 5. Social Preference Analysis (3-Chamber Test)

```python
from utils.DLCFunctions import PostDLC_3CT_3EVTs

# Comprehensive 3-chamber test analysis
results = PostDLC_3CT_3EVTs(
    DLCresult=dlc_dict,
    destfolder='/path/to/output',
    ROI='new',  # ROI configuration
    Nose2Snout_dist=30,  # Distance threshold for nose-poke
    Evt1=(0.5, 2),  # Nose-poke: (min_duration, min_interval)
    Evt2=(2, 2),    # S-Zone: (min_duration, min_interval)
    Evt3=(2, 2),    # E-Zone: (min_duration, min_interval)
    FPS=25,
    SaveData=True
)

# Access results
print(f"Social Preference Index: {results['SPI']}")
print(f"Time in Social Zone: {results['Time_SZone']}s")
print(f"Number of Nose-pokes: {results['N_Nosepoke']}")
```

**What it does:** Complete analysis of 3-chamber social preference test with multiple behavioral metrics.

---

## 🔬 Advanced DLC Processing

### Multi-Animal Tracking

```python
from utils.DLCFunctions import df_to_dic_multi

# Convert multi-animal DLC data
animal1_dict, animal2_dict = df_to_dic_multi(
    df=dlc_df,
    ignore_bodyparts='PatchCordBase'
)

# Analyze each animal separately
velocity1 = get_velocity(animal1_dict, 'Center', FPS=25)
velocity2 = get_velocity(animal2_dict, 'Center', FPS=25)
```

---

### Annotate Body Part Proximity

```python
from utils.DLCFunctions import annotate_body_part_proximity
import pandas as pd

# Create points dataframe
points_df = pd.DataFrame({
    'frame': range(len(dlc_dict['Nose']['x']))
})

# Annotate proximity between two animals
annotated_df = annotate_body_part_proximity(
    body_part_data1=animal1_dict,
    body_part_data2=animal2_dict,
    points_df=points_df,
    body_part_name1='Nose',
    body_part_name2='Nose',
    pcutoff=0.9,
    d_threshold=50,  # "Close" threshold
    d_threshold2=150  # "Nearby" threshold
)

# Check proximity states
close_frames = annotated_df[annotated_df['proximity'] == 'close']
```

**What it does:** Classifies inter-animal distance into categories (close, nearby, far).

---

### Custom ROI Shapes

```python
from utils.DLCFunctions import check_point_in_regions

# Define complex ROI shapes (using Shapely)
from shapely.geometry import Polygon

social_zone = Polygon([(100, 100), (300, 100), (300, 300), (100, 300)])
empty_zone = Polygon([(400, 100), (600, 100), (600, 300), (400, 300)])

regions = {
    'social': social_zone,
    'empty': empty_zone
}

# Check which region the animal is in
for frame_idx in range(len(nose_x)):
    x, y = nose_x[frame_idx], nose_y[frame_idx]
    region = check_point_in_regions(x, y, regions)
    print(f"Frame {frame_idx}: Animal in {region}")
```

---

## 💡 Tips & Best Practices

1. **Likelihood filtering**:
   - Use `pcutoff=0.9` for good tracking
   - Use `pcutoff=0.95` for high-confidence only
   - Lower threshold if tracking is sparse

2. **Coordinate systems**:
   - DLC uses image coordinates (0,0 = top-left)
   - Y increases downward
   - Calibrate to real-world units if needed

3. **Velocity calculation**:
   - Smooth with moving average if noisy
   - Threshold velocity to detect movement vs. stillness
   - Convert pixels/frame to cm/s using calibration

4. **ROI definition**:
   - Use first frame to define ROIs visually
   - Save ROI coordinates for reproducibility
   - Account for camera distortion near edges

5. **Event detection**:
   - Set `min_duration` to filter brief artifacts
   - Set `min_interval` to merge consecutive events
   - Typical values: duration=0.5s, interval=2s

6. **Data quality**:
   - Always check likelihood distributions
   - Interpolate missing points if likelihood drops
   - Validate with manual scoring samples

---

## 🔗 Related

- See [VideoFunctions Cheatsheet](cheatsheet_VideoFunctions.md) for video processing
- See [FPFunctions Cheatsheet](cheatsheet_FPFunctions.md) for combining with FP data
- See `analysis/DLC2BORIS.py` for converting to BORIS format
