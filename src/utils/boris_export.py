# boris_export.py

import pandas as pd


def extract_bouts_from_annotation(series: pd.Series, fps: int) -> list[tuple[int, int]]:
    """
    Identify 'on' bouts in an annotation series by detecting start and end indices.
    
    Parameters:
    - series: pd.Series with 'on'/'off' annotation
    - fps: frames per second to understand time scale

    Returns:
    - List of (start_frame, end_frame) tuples
    """
    shifted = series.shift(1)
    starts = series[(series == 'on') & (shifted != 'on')].index
    ends = series[(series == 'on') & (series.shift(-1) != 'on')].index
    return list(zip(starts, ends))


def prepare_boris_export(evt_dict: dict, fps: int, animal_id: str, behavior_map: dict) -> pd.DataFrame:
    """
    Prepare BORIS export dataframe from event bouts.

    Parameters:
    - evt_dict: Dictionary {event_name -> list of (start, stop) tuples}
    - fps: Frames per second
    - animal_id: ID to annotate subject column
    - behavior_map: Dictionary mapping annotation column to behavior name

    Returns:
    - DataFrame with columns: time, subject, code, modifier, comment
    """
    behavior, behavior_type, time = [], [], []

    for evt_name, bouts in evt_dict.items():
        label = behavior_map.get(evt_name, evt_name.replace('_annot', ''))
        for start, stop in bouts:
            t0, t1 = round(start / fps, 3), round(stop / fps, 3)
            behavior += [label, label]
            behavior_type += ['START', 'STOP']
            time += [t0, t1]

    df = pd.DataFrame({
        'time': time,
        'subject': [animal_id] * len(time),
        'code': behavior,
        'modifier': [''] * len(time),
        'comment': [''] * len(time)
    })
    return df.sort_values(by='time').reset_index(drop=True)
