def df_to_dic_single(df, ignore_bodyparts:str = 'PatchCordBase'):
    """
    Convert single dlc output data to a python dictionary. The input data format should be a pandas DataFrame format. 
    """

    # Create a dictionary to map body part names to coordinate columns 
    body_part_data = {} 
    # Get unique body parts from the first level of the MultiIndex (excluding 'bodyparts')
    unique_body_parts = df.columns.get_level_values(0).unique().tolist() 
    unique_body_parts.remove('bodyparts')
    # unique_body_parts.remove(ignore_bodyparts)
        
    for part in unique_body_parts:
    # Use the MultiIndex to access the data
        body_part_data[part] = {
            "x": df.loc[:, (part, 'x')].to_numpy(),
            "y": df.loc[:, (part, 'y')].to_numpy(), 
            "likelihood": df.loc[:, (part, 'likelihood')].to_numpy()
            }
    
    for part in body_part_data.keys():
        print(f"body parts: {part}")

    return body_part_data

####################################################################################################################
####################################################################################################################
####################################################################################################################

def df_to_dic_multi(df, ignore_bodyparts:str = 'PatchCordBase'):     
    """
    Convert multi dlc output data to two python dictionaries. The input data format should be a pandas DataFrame format.
    """

    # Create a dictionary to map body part names to coordinate columns
    body_part_data1 = {}
    body_part_data2 = {}

    # Get unique body parts from the first level of the MultiIndex (excluding 'bodyparts')
    unique_body_parts = df.columns.get_level_values(0).unique().tolist()
    unique_body_parts.remove('bodyparts')
    unique_body_parts.remove(ignore_bodyparts)
        
    for part in unique_body_parts:
        # Use the MultiIndex to access the data
        body_part_data1[part] = {
            "x": df.loc[:, (part, 'x')].to_numpy(), 
            "y": df.loc[:, (part, 'y')].to_numpy(), 
            "likelihood": df.loc[:, (part, 'likelihood')].to_numpy()}
        
    for part in unique_body_parts:
        # Use the MultiIndex to access the data
        body_part_data2[part] = {
            "x": df.loc[:, (part, 'x.1')].to_numpy(), 
            "y": df.loc[:, (part, 'y.1')].to_numpy(), 
            "likelihood": df.loc[:, (part, 'likelihood.1')].to_numpy()}
    
    return body_part_data1, body_part_data2

####################################################################################################################
####################################################################################################################
####################################################################################################################

def PostDLC_3CT_3EVTs(DLCresult:dict, destfolder:str = '', ROI:str = 'new', Nose2Snout_dist: float = 30, Evt1: tuple = (0.5, 2), Evt2: tuple = (2, 2), Evt3: tuple = (2, 2), FPS: int = 25, SaveData:bool = False) -> dict:
    """
    - Social Preference Index based on time spent in different regions of interest (ROIs)
    - Identification of specific events (Nose-poke, S-Zone entry, E-Zone entry)
    
    Parameters:
    -----------
    DLCresult : dict
    Dictionary containing the DLC output data with body part coordinates and likelihoods.
    ROI : str, optional
    Region of interest setting, either 'old' or 'new'. Default is 'new'.
    Nose2Snout_dist : float, optional
    Distance threshold between Nose and Snout to consider a Nose-poke event. Default is 30.
    Evt1 : tuple, optional
        Tuple containing minimum duration and minimum interval for Nose-poke events. Default is (0.5, 2).
    Evt2 : tuple, optional
        Tuple containing minimum duration and minimum interval for S-Zone entry events. Default is (2, 2).
    Evt3 : tuple, optional
        Tuple containing minimum duration and minimum interval for E-Zone entry events. Default is (2, 2).
    FPS : int, optional
        Frames per second of the video data. Default is 25.
    SaveData : bool, optional
        If True, the processed data will be saved to a CSV file. Default is False.
    
    Returns:
    --------
    Evt_dic : dict
        Dictionary containing filtered bouts for Nose-poke, S-Zone entry, and E-Zone entry events.
    
    Example:
    --------
    >>> DLCresult = {
    ...     'Nose': {'x': [1, 2, 3], 'y': [4, 5, 6], 'likelihood': [0.99, 0.98, 0.97]},
    ...     'snout1': {'x': [1, 2, 3], 'y': [4, 5, 6], 'likelihood': [0.99, 0.98, 0.97]},
    ...     'bodypart3': {'x': [1, 2, 3], 'y': [4, 5, 6]},
    ...     'bodypart4': {'x': [1, 2, 3], 'y': [4, 5, 6]}
    ... }
    >>> PostDLC_3CT(DLCresult, ROI='new', Nose2Snout_dist=30, Evt1=(0.5, 2), Evt2=(2, 2), Evt3=(2, 2), FPS=25, SaveData=False)
        The Social Preference Index: 0.0
        Filtered Event Bouts (start index, end index):
        Filtered Event Bouts (start index, end index):
        Filtered Event Bouts (start index, end index):
    """
    
    from shapely.geometry import Point, Polygon
    import pandas as pd
    import math
    import numpy as np
    import os 

    # define ROIs 
    if ROI == 'old': 
        roi_S = Polygon([(629,218),  #roi for S_sniffing_zone
                        (631,257), 
                        (603,278), 
                        (575,278),
                        (544,272),
                        (514,258),
                        (489,241),
                        (469,217),
                        (456,197),
                        (445,172),
                        (441,151),
                        (439,130),
                        (441,105),
                        (443, 90),
                        (472, 97),
                        (483,117),
                        (500,145),
                        (525,173),
                        (550,191),
                        (575,203),
                        (601,212)])

        roi_E = Polygon([(482,943),  #roi for E_sniffing_zone
                        (448,943), 
                        (441,915), 
                        (440,891),
                        (442,863),
                        (451,832),
                        (466,804),
                        (484,780),
                        (508,762),
                        (526,750),
                        (548,741),
                        (576,734),
                        (602,733),
                        (634,763),
                        (628,795),
                        (602,800),
                        (569,813),
                        (541,830),
                        (516,856),
                        (495,887),
                        (486,922)])
    else: 
        roi_S = Polygon([(624,198),  #roi for S_sniffing_zone_2(new setting, since 2024 Dec 12nd)
                        (626,237), 
                        (593,262), 
                        (569,262),
                        (539,255),
                        (509,238),
                        (487,219),
                        (470,199),
                        (458,176),
                        (450,151),
                        (445,132),
                        (441,109),
                        (440,91),
                        (440,74),
                        (476,74),
                        (484,99),
                        (502,124),
                        (526,151),
                        (547,168),
                        (574,182),
                        (596,192)])

        roi_E = Polygon([(470,911),  #roi for E_sniffing_zone_2(new setting, since 2024 Dec 12nd)
                        (436,911), 
                        (429,883), 
                        (428,859),
                        (430,831),
                        (439,800),
                        (454,772),
                        (472,748),
                        (496,730),
                        (514,718),
                        (536,709),
                        (564,702),
                        (590,701),
                        (622,731),
                        (616,763),
                        (590,768),
                        (557,781),
                        (529,798),
                        (504,824),
                        (483,855),
                        (474,890)])

    ROIs = ['S', 'E']   # ROIs = ['S', 'E'] or ['S', 'E', 'S2', 'E2']

    ###########################################################################################################
    # ROI Analysis
    ###########################################################################################################
    # Select th bodypart that we are going to use
    x_coordinates = DLCresult[list(DLCresult.keys())[0]]['x']  # Generate x coordinates
    y_coordinates = DLCresult[list(DLCresult.keys())[0]]['y']  # create y coordinates

    # Function to check if points are within each region
    def check_point_in_regions(x, y, regions):
        points = [Point(x[i], y[i]) for i in range(len(x))]
        result = []
        for point in points:
            in_region = None
            for i, region in enumerate(regions):
                if region.contains(point):
                    in_region = ROIs[i]
                    break
            result.append(in_region)
        return result

    # 각 점을 확인하고 각 점이 ROI 중 어디에 속하는지 결정
    roi_results = check_point_in_regions(x_coordinates, y_coordinates, [roi_S, roi_E])

    # 결과를 pandas DataFrame에 추가
    points_df = pd.DataFrame({list(DLCresult.keys())[0]+'_x': DLCresult[list(DLCresult.keys())[0]]['x'],
                            list(DLCresult.keys())[0]+'_y': DLCresult[list(DLCresult.keys())[0]]['y'],
                            list(DLCresult.keys())[3]+'_x': x_coordinates, 
                            list(DLCresult.keys())[3]+'_y': y_coordinates, 
                            'roi': roi_results})

    # roi 데이터에서 None(S-zone, E-zone 외의 영역에 동물이 존재함)을 'else' 일괄 변경 
    points_df.roi = points_df.roi.fillna('else')

    # Add index when ROI transition occurs. 0:else -> S 
    transition_index = []
    prev_roi = None
    for roi in points_df.roi:
        if prev_roi is None:
            transition_index.append(None)
        else:
            if prev_roi == 'else' and roi == 'S':
                transition_index.append(str(0)) #'0' stands for else -> S-zone 
            elif prev_roi == 'S' and roi == 'else':
                transition_index.append(str(1)) #'1' stands for S-zone -> else
            elif prev_roi == 'else' and roi == 'E':
                transition_index.append(str(2)) #'2' stands for else -> E-zone
            elif prev_roi == 'E' and roi == 'else':
                transition_index.append(str(3)) #'3' stands for E-zone -> else
            else:
                transition_index.append(None)
        prev_roi = roi

    # 인덱스 열 추가
    points_df['transition_index'] = transition_index

    ###########################################################################################################
    # Calculate the social preference index using time spent in each zone
    ###########################################################################################################
    time_spent_S = (len(points_df[points_df.roi=='S']['roi']))*(1/FPS) 
    time_spent_E = (len(points_df[points_df.roi=='E']['roi']))*(1/FPS) 
    time_spent_else = (len(points_df[points_df.roi=='else']['roi']))*(1/FPS)

    SocialPreferenceIndex = (time_spent_S-time_spent_E)/(time_spent_S+time_spent_E)*100

    print("\n"+"The Social Preference Index:", round(SocialPreferenceIndex, ndigits=3))

    ###########################################################################################################
    # Measure the distance between Nose and Snout
    ###########################################################################################################
    # Nose와 Snout, 또는 (만약 Nose가 가용하지 않다면) Head와 Snout간의 거리 계산하기 
    Nose_X = DLCresult['Nose']['x']  #'Nose'의 x 좌표
    Nose_Y = DLCresult['Nose']['y']  #'Nose'의 y 좌표
    Snout_X = DLCresult['snout1']['x'] #'snout1'의 x 좌표
    Snout_Y = DLCresult['snout1']['y'] #'snout1'의 y 좌표

    N_S_dist = []
    NosePoke = []
    distance = None

    for i in range(len(Nose_X)):
        if DLCresult['snout1']['likelihood'][i]>=0.95:
            distance = math.sqrt(((Nose_X[i]-Snout_X[i])**2) + ((Nose_Y[i]-Snout_Y[i])**2))
            N_S_dist.append(distance)
            
            if distance <= Nose2Snout_dist:
                NosePoke.append("on")
            else: NosePoke.append("off") 
            
        else: 
            N_S_dist.append(None)
            NosePoke.append(None)

    points_df['N_s1_dist'] = N_S_dist
    points_df['NosePoke'] = NosePoke

    ###########################################################################################################
    # Identification of Nose-poke events (hereafter reffered as EVT1)
    ###########################################################################################################
    Evt_dic = {}
    
    min_duration = Evt1[0] * FPS  # Minimum duration of an event in samples
    min_interval = Evt1[1] * FPS  # Minimum interval between bouts in samples

    # Detect (roi transition)
    df = points_df.NosePoke
    df_shifted = df.shift(1)
    NosePoke_start_indices = df[(df == 'on') & (df_shifted != 'on')].index
    NosePoke_end_indices = df[(df == 'on') & (df.shift(-1) != 'on')].index

    # Filter bouts based on minimum duration
    bouts = [(start, end) for start, end in zip(NosePoke_start_indices, NosePoke_end_indices) if end - start + 1 >= min_duration]

    # Ensure bouts are at least 2 seconds apart
    filtered_bouts = []
    EVT_marker1 = np.zeros(len(points_df.NosePoke))

    for i, (start, end) in enumerate(bouts):
        if i == 0:
            filtered_bouts.append((start, end))
            EVT_marker1[start] = int(1)
            EVT_marker1[end] = int(2)
        else:
            prev_start, prev_end = filtered_bouts[-1]
            if start - prev_end >= min_interval:
                filtered_bouts.append((start, end))
                EVT_marker1[start] = int(1)
                EVT_marker1[end] = int(2)

    Evt_dic['NosePoke'] = filtered_bouts
    points_df['EVT1'] = np.where(EVT_marker1==0, None, EVT_marker1) 
                
    # Print the filtered bouts
    print("\n"+"Filtered Event1 (Nose-to-Snout) Bouts (start index, end index):")
    for start, end in filtered_bouts:
        print(f"Start: {start}, End: {end}, Duration: {(end - start + 1) / FPS:.2f} seconds")

    ###########################################################################################################
    # Identification of S-Zone entry (hereafter reffered as EVT2)
    ###########################################################################################################
    min_duration = Evt2[0] * FPS  # Minimum duration of 'S' events in samples
    min_interval = Evt2[1] * FPS  # Minimum interval between bouts in samples

    # Detect (roi transition)
    df = points_df.roi
    df_shifted = df.shift(1)
    bout_start_indices = df[(df == 'S') & (df_shifted != 'S')].index
    bout_end_indices = df[(df == 'S') & (df.shift(-1) != 'S')].index
    bout_start_indices, bout_end_indices

    # Filter bouts based on minimum duration
    bouts = [(start, end) for start, end in zip(bout_start_indices, bout_end_indices) if end - start + 1 >= min_duration]

    # Ensure bouts are at least 2 seconds apart
    filtered_bouts2 = []
    EVT_marker2 = np.zeros(len(points_df.roi))

    for i, (start, end) in enumerate(bouts):
        if i == 0:
            filtered_bouts2.append((start, end))
            EVT_marker2[start] = int(1)
            EVT_marker2[end] = int(2)
        else:
            prev_start, prev_end = filtered_bouts2[-1]
            if start - prev_end >= min_interval:
                filtered_bouts2.append((start, end))
                EVT_marker2[start] = int(1) 
                EVT_marker2[end] = int(2)

    Evt_dic['S_zone'] = filtered_bouts2
    points_df['EVT2'] = np.where(EVT_marker2==0, None, EVT_marker2)

    # Print the filtered bouts
    print("\n"+"Filtered Event2 (S-zone-entry) Bouts (start index, end index):")
    for start, end in filtered_bouts2:
        print(f"Start: {start}, End: {end}, Duration: {(end - start + 1) / FPS:.2f} seconds")
        
    ###########################################################################################################
    # Identification of E-Zone entry (hereafter reffered as EVT3)
    ###########################################################################################################
    min_duration = Evt3[0] * FPS  # Minimum duration of 'S' events in samples
    min_interval = Evt3[1] * FPS  # Minimum interval between bouts in samples

    # Detect (roi transition)
    df = points_df.roi
    df_shifted = df.shift(1)
    bout_start_indices = df[(df == 'E') & (df_shifted != 'E')].index
    bout_end_indices = df[(df == 'E') & (df.shift(-1) != 'E')].index
    bout_start_indices, bout_end_indices

    # Filter bouts based on minimum duration
    bouts = [(start, end) for start, end in zip(bout_start_indices, bout_end_indices) if end - start + 1 >= min_duration]

    # Ensure bouts are at least 2 seconds apart
    filtered_bouts3 = []
    EVT_marker3 = np.zeros(len(points_df.roi))

    for i, (start, end) in enumerate(bouts):
        if i == 0:
            filtered_bouts3.append((start, end))
            EVT_marker3[start] = int(1)
            EVT_marker3[end] = int(2)
        else:
            prev_start, prev_end = filtered_bouts3[-1]
            if start - prev_end >= min_interval:
                filtered_bouts3.append((start, end))
                EVT_marker3[start] = int(1) 
                EVT_marker3[end] = int(2)

    Evt_dic['E_zone'] = filtered_bouts3
    points_df['EVT3'] = np.where(EVT_marker3==0, None, EVT_marker3)

    # Print the filtered bouts
    print("\n"+"Filtered Event3 (E-zone-entry) Bouts (start index, end index):")
    for start, end in filtered_bouts3:
        print(f"Start: {start}, End: {end}, Duration: {(end - start + 1) / FPS:.2f} seconds")

    ###########################################################################################################
    # Save data
    ###########################################################################################################
    if SaveData: 
        if destfolder == '':
            destfolder = os.getcwd() + 'Data_DLC.csv'
        else:
            if not os.path.exists(destfolder):
                os.makedirs(destfolder)
            destfolder = os.path.join(destfolder, 'Data_DLC.csv')
        points_df.to_csv(destfolder, index=False)

    ###########################################################################################################
    # Return the results
    ###########################################################################################################
    return Evt_dic # Return the event dictionary containing filtered bouts for Nose-poke, S-Zone entry, and E-Zone entry events.

####################################################################################################################
####################################################################################################################
####################################################################################################################

def get_velocity(DLCresult:dict, bpt:str, FPS:int, pcutoff:float=0.95):
    """
    Calculate the velocity of a specific body part from DLC results.
    
    Parameters:
    - DLCresult (dict): Dictionary containing the DLC results.
    - bpt (str): The body part to calculate the velocity for.
    - FPS (int): Frames per second of the video.
    - pcutoff (float): Probability cutoff for considering a point valid.

    Returns:
    - velocities (list): List of velocities for the specified body part.
    """
    import numpy as np

    # Extract x and y coordinates of the specified body part
    x_coords = DLCresult[bpt]['x']
    y_coords = DLCresult[bpt]['y']

    time = np.arange(len(x_coords))/FPS  # Create time array based on FPS
    distance = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    distance = np.insert(distance, 0, 0)  # Insert 0 at the beginning to match the length of x_coords and y_coords
    velocities = []

    for i in range(len(distance)):
        if DLCresult[bpt]['likelihood'][i]>=pcutoff:
            velocities.append(distance[i]*FPS)
        else: 
            velocities.append(None)
            
    return time, velocities

####################################################################################################################
####################################################################################################################
####################################################################################################################

def roi_entry_analysis(DLCresult:dict, bpt:str, pcutoff: float, ROI:list):
    '''
    Indentify the entry of a body part into a region of interest (ROI) based on DLC results.
    The ROI is defined by a list of tuples, where each tuple contains the x and y coordinates of the polygon vertices.
    The function returns a list indicating whether the body part is inside the ROI at each frame.
    Note: The result uses 'on' if the body part is inside the ROI with sufficient likelihood, and 'off' both if the likelihood is below the cutoff or the body part is outside the ROI.

    Parameters:
    - DLCresult (dict): Dictionary containing the DLC results.
    - bpt (str): The body part to check for entry into the ROI.
    - pcutoff (float): Probability cutoff for considering a point valid.
    - ROI (list): List of tuples defining the vertices of the polygon representing the ROI.

    Returns:
    - inside_roi (list): List of 'on'/'off' values indicating whether the body part is inside the ROI at each frame.
    '''
    import numpy as np
    from shapely.geometry import Point, Polygon

    x_coords = DLCresult[bpt]['x']  # Generate x coordinates
    y_coords = DLCresult[bpt]['y']  # Create y coordinates
    likelihoods = DLCresult[bpt]['likelihood']  # Extract likelihood values

    ROI_polygon = Polygon(ROI)  # Create a polygon from the ROI coordinates

    # Check if each point is within the ROI polygon
    inside_roi = []
    for x, y, likelihood in zip(x_coords, y_coords, likelihoods):
        if likelihood >= pcutoff:  # Only consider points with sufficient likelihood
            point = Point(x, y)
            if ROI_polygon.contains(point):
                inside_roi.append('on')
            else:
                inside_roi.append('off')
        else:
            inside_roi.append('off')  # Mark as outside ROI if likelihood is below cutoff

    return inside_roi

####################################################################################################################
####################################################################################################################
####################################################################################################################

def get_bodypoints_distance(DLCresult:dict, bpt:str, bpt2:str, pcutoff:float=0.95, distance_thres:float=30):
    """
    Calculate the distance between two body parts in DLC results.
    
    Parameters:
    - DLCresult (dict): Dictionary containing the DLC results.
    - bpt (str): The first body part to calculate the distance for.
    - bpt2 (str): The second body part to calculate the distance for.
    - pcutoff (float): Probability cutoff for considering a point valid.
    - distance_thres (float): Threshold for distance to determine if the body parts are close.

    Returns:
    - distances (list): List of distances between the two body parts.
    - close (list): List of boolean values (True/False) indicating if the body parts are close, or None if the value is invalid.
    """
    import numpy as np

    # Extract x and y coordinates of the specified body parts
    x_coords1 = DLCresult[bpt]['x']
    y_coords1 = DLCresult[bpt]['y']
    
    x_coords2 = DLCresult[bpt2]['x']
    y_coords2 = DLCresult[bpt2]['y']

    distances = []

    for i in range(len(x_coords1)):
        if DLCresult[bpt]['likelihood'][i]>=pcutoff and DLCresult[bpt2]['likelihood'][i]>=pcutoff:
            distance = np.sqrt(((x_coords1[i]-x_coords2[i])**2) + ((y_coords1[i]-y_coords2[i])**2))
            distances.append(distance)
        else: 
            distances.append(np.nan)

    close = [dist <= distance_thres if not np.isnan(dist) else None for dist in distances]
    return distances, close

import math
from typing import Dict, Any
import pandas as pd
import numpy as np

def annotate_body_part_proximity(
    body_part_data1: Dict[str, Dict[str, Any]],
    body_part_data2: Dict[str, Dict[str, Any]],
    points_df: pd.DataFrame,
    body_part_name1: str,
    body_part_name2: str,
    pcutoff: float,
    d_threshold: float,
    d_threshold2: float = 110.0
) -> pd.DataFrame:
    """
    Compute distance between two body parts across frames and annotate proximity.

    Parameters:
    - body_part_data1: Dictionary with keys as body parts and values containing x, y, and likelihood arrays
    - body_part_data2: Same as body_part_data1 but can represent a different frame or subject
    - points_df: DataFrame to append the distance and annotation results
    - body_part_name1: Name of the first body part to compare
    - body_part_name2: Name of the second body part to compare
    - pcutoff: Likelihood threshold below which the value is considered unreliable
    - d_threshold: Distance threshold to decide "on" or "off"
    - d_threshold2: Threshold used for multi-body relational proximity conditions

    Returns:
    - Updated points_df with distance, annotation columns, and relevant frame indices
    """
    for name, data in [(body_part_name1, body_part_data1), (body_part_name2, body_part_data2)]:
        if name not in data:
            raise ValueError(f"'{name}' not found in provided body part data")

    bp1_x = np.array(body_part_data1[body_part_name1]['x'])
    bp1_y = np.array(body_part_data1[body_part_name1]['y'])
    bp2_x = np.array(body_part_data2[body_part_name2]['x'])
    bp2_y = np.array(body_part_data2[body_part_name2]['y'])
    likelihood = np.array(body_part_data1[body_part_name1]['likelihood'])

    dist = np.full_like(bp1_x, fill_value=np.nan, dtype=np.float64)
    annot = np.full_like(bp1_x, fill_value=None, dtype=object)
    extra_threshold = 60.0

    reliable = likelihood >= pcutoff
    dist[reliable] = np.hypot(bp1_x[reliable] - bp2_x[reliable], bp1_y[reliable] - bp2_y[reliable])
    annot[reliable] = np.where(dist[reliable] <= d_threshold, "on", "off")

    if body_part_name1 == "Nose":
        unreliable = ~reliable
        left_x = np.array(body_part_data1["Left_ear"]['x'])
        right_x = np.array(body_part_data1["Right_ear"]['x'])
        left_y = np.array(body_part_data1["Left_ear"]['y'])
        right_y = np.array(body_part_data1["Right_ear"]['y'])

        head_x = (left_x + right_x) / 2
        head_y = (left_y + right_y) / 2

        dist[unreliable] = np.hypot(head_x[unreliable] - bp2_x[unreliable], head_y[unreliable] - bp2_y[unreliable])
        annot[unreliable] = np.where(dist[unreliable] <= (d_threshold + extra_threshold), "on", "off")

    dist_col = f"{body_part_name1}2{body_part_name2}"
    annot_col = f"{dist_col}_annot"

    points_df[dist_col] = dist
    points_df[annot_col] = annot

    # Compute index sets and convert to annotation
    try:
        points_df['Nose2Body_flag'] = (
            (points_df.get('Nose2Left_ear_annot') == 'on') |
            (points_df.get('Nose2Right_ear_annot') == 'on') |
            (points_df.get('Nose2Left_fhip_annot') == 'on') |
            (points_df.get('Nose2Right_fhip_annot') == 'on')
        )

        points_df['Body2Nose_flag'] = (
            (points_df.get('Left_ear2Nose_annot') == 'on') |
            (points_df.get('Right_ear2Nose_annot') == 'on') |
            (points_df.get('Left_fhip2Nose_annot') == 'on') |
            (points_df.get('Right_fhip2Nose_annot') == 'on')
        )

        points_df['SideBySide_flag'] = (
            (points_df.get('Nose2Nose', pd.Series(np.inf, index=points_df.index)) <= d_threshold2) &
            (points_df.get('Tail_base2Tail_base', pd.Series(np.inf, index=points_df.index)) <= d_threshold2)
        )

        points_df['SideReverseSide_flag'] = (
            (points_df.get('Nose2Tail_base', pd.Series(np.inf, index=points_df.index)) <= d_threshold2) &
            (points_df.get('Tail_base2Nose', pd.Series(np.inf, index=points_df.index)) <= d_threshold2)
        )

        # Convert flags to annotation columns
        for flag_col in ['Nose2Body_flag', 'Body2Nose_flag', 'SideBySide_flag', 'SideReverseSide_flag']:
            annot_col = flag_col.replace('_flag', '_annot')
            points_df[annot_col] = np.where(points_df[flag_col], 'on', 'off')

    except Exception:
        pass

    return points_df