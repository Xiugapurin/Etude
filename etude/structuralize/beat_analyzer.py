# src/etude/preprocessing/beat_analyzer.py

import json
import math
from pathlib import Path
from statistics import mode, StatisticsError
from typing import List, Dict, Union

import numpy as np


class BeatAnalyzer:
    """
    Analyzes beat and downbeat predictions to extract tempo, time signature,
    and measure information for different regions of a song.
    """

    def __init__(self, verbose: bool = False):
        """
        Initializes the BeatAnalyzer.
        
        Args:
            verbose (bool): If True, prints detailed debugging information.
        """
        self.verbose = verbose
        self.beat_pred = []
        self.downbeat_pred = []

    def analyze(self, beat_file_path: Union[str, Path]) -> List[Dict]:
        """
        Analyzes a beat/downbeat prediction file to extract structured tempo information.

        Args:
            beat_file_path (Union[str, Path]): Path to the JSON file from beat detection,
                                               containing 'beat_pred' and 'downbeat_pred' lists.

        Returns:
            List[Dict]: A list of dictionaries, each representing a stable tempo region.
                        Returns an empty list if analysis is not possible.
        """
        with open(beat_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.beat_pred = data.get('beat_pred', [])
        self.downbeat_pred = data.get('downbeat_pred', [])

        if not self.downbeat_pred:
            print("Warning: No downbeats found in the file. Cannot perform analysis.")
            return []

        filtered_beats = self._remove_close_beats()
        measures = self._compute_measures(filtered_beats)
        if not measures:
            print("Warning: Could not compute any valid measures.")
            return []

        global_time_sig = self._compute_global_time_sig(measures)
        stable_regions_indices = self._detect_stable_regions(measures)
        
        processed_regions = []
        for start_idx, end_idx, _ in stable_regions_indices:
            region_measures = measures[start_idx : end_idx + 1]
            downbeats = [m['start'] for m in region_measures]
            if end_idx + 1 < len(measures):
                downbeats.append(measures[end_idx + 1]['start'])

            durations = [downbeats[i+1] - downbeats[i] for i in range(len(downbeats) - 1)]

            if durations:
                avg_duration = sum(durations) / len(durations)
                avg_bpm = (60 * global_time_sig) / avg_duration if avg_duration > 0 else 0
                
                processed_regions.append({
                    "start_time": downbeats[0],
                    "downbeats": downbeats[:-1],
                    "avg_duration": avg_duration,
                    "bpm": avg_bpm,
                    "time_sig": global_time_sig,
                })
        
        if not processed_regions:
            print("Warning: No stable tempo regions were detected.")
            return []

        final_regions = self._patch_region_gaps(processed_regions)

        final_output = []
        for region in final_regions:
            final_output.append({
                "time_sig": region['time_sig'],
                "bpm": region['bpm'],
                "start": region['start_time'],
                "downbeats": region['downbeats']
            })
        
        if self.verbose:
            print(f"Tempo analysis complete. Found {len(final_output)} regions.")
        
        return final_output

    @staticmethod
    def save_tempo_data(tempo_data: List[Dict], output_path: Union[str, Path]):
        """Saves the structured tempo data to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tempo_data, f, indent=4)
        print(f"Tempo data saved successfully to: {output_path}")

    def _remove_close_beats(self, beat_threshold: float = 0.1) -> List[float]:
        """Filters out beat predictions that are too close to a downbeat."""
        filtered = [
            beat for beat in self.beat_pred
            if not any(abs(beat - db) < beat_threshold for db in self.downbeat_pred)
        ]
        return filtered

    def _compute_measures(self, beats: List[float], uniformity_threshold: float = 0.1) -> List[Dict]:
        """Groups beats into measures based on downbeats and analyzes their uniformity."""
        measures = []
        for i in range(len(self.downbeat_pred) - 1):
            start = self.downbeat_pred[i]
            end = self.downbeat_pred[i+1]
            
            beats_in_measure = [start] + [b for b in beats if start < b < end]
            
            is_uniform = True
            if len(beats_in_measure) > 1:
                intervals = np.diff(beats_in_measure)
                mean_interval = np.mean(intervals)
                if mean_interval > 0:
                    rel_std = np.std(intervals) / mean_interval
                    is_uniform = rel_std < uniformity_threshold
            
            measures.append({
                'start': start,
                'raw_beats': len(beats_in_measure),
                'duration': end - start,
                'uniform': is_uniform
            })
        return measures

    def _compute_global_time_sig(self, measures: List[Dict]) -> int:
        """Determines the most likely global time signature (beats per measure)."""
        valid_beat_counts = [m['raw_beats'] for m in measures if m.get('uniform', True)]
        
        if not valid_beat_counts or len(valid_beat_counts) < 10:
            return 4 # Default to 4 if data is sparse or unreliable
        
        try:
            mode_val = mode(valid_beat_counts)
        except StatisticsError:
            mode_val = valid_beat_counts[0]
        
        # Common misinterpretations (e.g., 2/4 is often detected as 2 beats)
        return 4 if mode_val == 2 else mode_val

    def _detect_stable_regions(self, measures: List[Dict], window_size: int = 4, threshold: float = 0.1) -> List:
        """Identifies regions of consecutive measures with a stable tempo."""
        stable_regions = []
        i = 0
        while i <= len(measures) - window_size:
            # Check a window of measures for consistent duration
            intervals = [measures[j+1]['start'] - measures[j]['start'] for j in range(i, i + window_size - 1)]
            if not intervals or np.std(intervals) >= threshold:
                i += 1
                continue
            
            # If stable, extend the region as long as the consistency holds
            ideal_interval = np.mean(intervals)
            region_end = i + window_size - 1
            
            j = region_end
            while j + 1 < len(measures):
                predicted_next_start = measures[j]['start'] + ideal_interval
                if abs(measures[j+1]['start'] - predicted_next_start) < threshold:
                    region_end = j + 1
                    j += 1
                else:
                    break
            
            stable_regions.append((i, region_end, ideal_interval))
            i = region_end + 1
        return stable_regions

    def _patch_region_gaps(self, processed_regions: List[Dict], tolerance: float = 0.25) -> List[Dict]:
        """Heuristically fills gaps between stable tempo regions with measures of estimated time signatures."""
        if len(processed_regions) < 2:
            return processed_regions

        # This method's internal logic is complex and domain-specific, focusing on
        # identifying if gaps between regions correspond to an integer or half-integer
        # number of measures, then inserting new "patched" regions.
        # The implementation from the original file is preserved here.
        patched_regions = []
        current_region = processed_regions[0]
        for i in range(len(processed_regions) - 1):
            patched_regions.append(current_region)
            next_region = processed_regions[i+1]
            
            last_downbeat_ts = current_region['downbeats'][-1]
            measure_duration = current_region['avg_duration']
            theoretical_end_ts = last_downbeat_ts + measure_duration
            next_region_start_ts = next_region['downbeats'][0]
            gap_duration = next_region_start_ts - theoretical_end_ts
            
            if measure_duration <= 0 or gap_duration < 0:
                current_region = next_region
                continue

            ratio = gap_duration / measure_duration
            num_full_measures = 0
            has_half_measure = False

            if abs(ratio - round(ratio)) < tolerance and round(ratio) >= 1:
                # Case N.0x: Gap is roughly N full measures
                num_full_measures = round(ratio)
            elif abs(ratio - (math.floor(ratio) + 0.5)) < tolerance:
                # Case N.5x: Gap is roughly N.5 measures
                num_full_measures = math.floor(ratio)
                has_half_measure = True

            insert_ts = theoretical_end_ts
            for _ in range(num_full_measures):
                patched_regions.append({
                    "time_sig": current_region['time_sig'], 
                    "bpm": current_region['bpm'],
                    "start_time": insert_ts, 
                    "downbeats": [insert_ts], 
                    "avg_duration": measure_duration
                })
                insert_ts += measure_duration
            
            if has_half_measure:
                patched_regions.append({
                    "time_sig": 2, 
                    "bpm": current_region['bpm'], # Assume 2/4 for a half measure
                    "start_time": insert_ts, 
                    "downbeats": [insert_ts], 
                    "avg_duration": measure_duration / 2
                })

            current_region = next_region

        patched_regions.append(current_region)
        
        # Merge adjacent regions if they have the same tempo and time signature
        merged_regions = []
        if not patched_regions: return []
        for region in patched_regions:
            if not merged_regions or \
               merged_regions[-1].get('time_sig') != region.get('time_sig') or \
               abs(merged_regions[-1].get('bpm', 0) - region.get('bpm', -1)) >= 1.0:
                merged_regions.append(region)
            else:
                merged_regions[-1]['downbeats'].extend(region.get('downbeats', []))
        
        return merged_regions