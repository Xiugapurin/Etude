# src/etude/structuralize/transcription_aligner.py

from typing import List, Dict

import numpy as np
from scipy.interpolate import interp1d

def compute_wp_std(time_map: list) -> float:
    """Calculates the standard deviation of the time differences in a time map (WP-Std from Music2MIDI paper)."""
    if not time_map:
        return float('inf')
    differences = [pair[0] - pair[1] for pair in time_map]
    return np.std(differences)

def create_time_map_from_downbeats(
    downbeats: List[float],
    align_result: Dict,
    feature_rate: int = 50
) -> List[List[float]]:
    """
    Creates a time map by finding the corresponding time points in the cover audio
    for each downbeat in the origin audio, using a pre-computed warping path.

    Args:
        downbeats (List[float]): A list of downbeat timestamps from the origin audio.
        align_result (Dict): The result dictionary from an AudioAligner, containing the warping path.
        feature_rate (int): The feature rate used during alignment.

    Returns:
        List[List[float]]: The time map as a list of [origin_time, cover_time] pairs.
    """
    wp = align_result['wp']
    
    # The warping path maps cover time (wp[0]) to origin time (wp[1]).
    time_origin_path = wp[1] / feature_rate
    time_cover_path = wp[0] / feature_rate
    
    interp_func = interp1d(
        time_origin_path, 
        time_cover_path, 
        kind="linear",
        bounds_error=False,
        fill_value=(time_cover_path[0], time_cover_path[-1])
    )
    
    time_map = []
    for db_time in downbeats:
        if db_time <= time_origin_path[-1]:
            corresponding_cover_time = interp_func(db_time)
            time_map.append([float(db_time), float(corresponding_cover_time)])
            
    return time_map

def weakly_align(
    transcription_notes: List[Dict],
    time_map: List[List[float]]
) -> List[Dict]:
    """
    Remaps the timestamps of a transcription based on a provided time map.
    This function implements the "weakly-alignment" logic.

    Args:
        transcription_notes (List[Dict]): The list of note objects to align.
        time_map (List[List[float]]): A list of [origin_time, cover_time] anchor points.

    Returns:
        List[Dict]: A new list of note objects with timestamps aligned to the origin's timeline.
    """
    if not time_map or not transcription_notes:
        return []

    aligned_transcription = []
    
    time_map.sort(key=lambda p: p[1])
    
    sorted_notes = sorted(transcription_notes, key=lambda n: n['onset'])

    map_idx = 0
    for note in sorted_notes:
        t_on = note["onset"]
        note_duration = note["offset"] - t_on
        
        while map_idx + 1 < len(time_map) and t_on >= time_map[map_idx + 1][1]:
            map_idx += 1
            
        t_S1, t_P1 = time_map[map_idx]
        if map_idx + 1 < len(time_map):
            t_S2, t_P2 = time_map[map_idx + 1]
        else:
            t_S2, t_P2 = t_S1 + 10, t_P1 + 10

        segment_duration_p = t_P2 - t_P1
        if segment_duration_p < 1e-6:
            continue
            
        if t_P1 <= t_on < t_P2:
            relative_pos = (t_on - t_P1) / segment_duration_p
            new_onset = t_S1 + relative_pos * (t_S2 - t_S1)
            
            aligned_note = {
                "pitch": note["pitch"],
                "onset": new_onset,
                "offset": new_onset + note_duration,
                "velocity": note["velocity"]
            }
            aligned_transcription.append(aligned_note)

    return aligned_transcription