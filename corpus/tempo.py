import json
from statistics import mode, StatisticsError
import numpy as np
import librosa
import IPython.display as ipd
import math


class TempoInfoGenerator:
    def __init__(self, path_beat, verbose=False):
        with open(path_beat, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.beat_pred = data['beat_pred']
        self.downbeat_pred = data['downbeat_pred']
        self.verbose = verbose

        if self.verbose:
            print(f"num beats = {len(self.beat_pred)}")
            print(f"num downbeat = {len(self.downbeat_pred)}")


    def save_json(self, filename, events):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=4, ensure_ascii=False)


    def remove_close_beats(self, beat_threshold=0.1):
        filtered = []
        for beat in self.beat_pred:
            if any(abs(beat - db) < beat_threshold for db in self.downbeat_pred):
                continue
            filtered.append(beat)
        return filtered


    def compute_measures(self, beats, uniformity_threshold=0.1):
        measures = []
        for i in range(len(self.downbeat_pred)-1):
            start = self.downbeat_pred[i]
            end = self.downbeat_pred[i+1]
            duration = end - start
            beats_in_measure = [b for b in beats if start < b < end]
            measure_beats = [start] + beats_in_measure
            raw_beats = len(measure_beats)
            if len(measure_beats) > 1:
                intervals = [measure_beats[j+1] - measure_beats[j] for j in range(len(measure_beats)-1)]
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                rel_std = std_interval / mean_interval if mean_interval > 0 else 0
                uniform = rel_std < uniformity_threshold
            else:
                uniform = True
            measures.append({
                'start': start,
                'raw_beats': raw_beats,
                'duration': duration,
                'measure_beats': measure_beats,
                'uniform': uniform
            })
        return measures


    def compute_global_time_sig(self, measures):
        valid_beats = [m['raw_beats'] for m in measures if m.get('uniform', True)]
        
        if len(valid_beats) < 10:
            mode_val = 4
        else:
            try:
                mode_val = mode(valid_beats)
            except StatisticsError:
                mode_val = valid_beats[0] if valid_beats else 4
        
        if self.verbose:
            print(f"[DEBUG] 用於取眾數的小節拍數列表: {valid_beats}")
            print(f"[DEBUG] 取眾數拍數: {mode_val}")
        
        global_time_sig = 4 if mode_val == 2 else mode_val
        
        if self.verbose:
            print(f"[DEBUG] 全局拍號設定為: {global_time_sig}/4")
        
        for m in measures:
            m['time_sig'] = global_time_sig
        return global_time_sig


    def detect_stable_regions(self, measures, window_size=4, threshold=0.1):
        stable_regions = []
        i = 0
        while i <= len(measures) - window_size:
            intervals = [measures[j+1]['start'] - measures[j]['start'] for j in range(i, i + window_size - 1)]
            if not intervals:
                i += 1
                continue
            std_interval = np.std(intervals)
            if std_interval < threshold:
                ideal_interval = np.mean(intervals)
                region_start = i
                region_end = i + window_size - 1
                
                j = region_end
                while j + 1 < len(measures):
                    predicted = measures[j]['start'] + ideal_interval
                    if abs(measures[j+1]['start'] - predicted) < threshold:
                        region_end = j + 1
                        j += 1
                    else:
                        break
                stable_regions.append((region_start, region_end, ideal_interval))
                i = region_end + 1
            else:
                i += 1
        return stable_regions
    
    
    def _patch_region_gaps(self, processed_regions, tolerance=0.25):
        if len(processed_regions) < 2:
            return processed_regions

        patched_regions = []
        current_region = processed_regions[0]

        for i in range(len(processed_regions) - 1):
            next_region = processed_regions[i+1]
            
            # 不再修改 current_region，直接將其加入最終列表
            patched_regions.append(current_region)
            
            # === 使用您定義的正確邏輯來計算間隙 ===
            last_downbeat_ts = current_region['downbeats'][-1]
            measure_duration = current_region['avg_duration']
            
            # 1. 計算前一區域最後一個小節的 "理論結束點"
            theoretical_end_ts = last_downbeat_ts + measure_duration
            
            # 2. 計算理論結束點到下一個區域開始點的 "真實間隙"
            next_region_start_ts = next_region['downbeats'][0]
            gap_duration = next_region_start_ts - theoretical_end_ts
            
            if measure_duration <= 0 or gap_duration < 0:
                current_region = next_region
                continue
            
            ratio = gap_duration / measure_duration
            
            # === 在間隙中插入新的小節區域 ===

            # Case 1: 間隙長度約為 N.5 倍 (0.5, 1.5, 2.5...)
            if abs(ratio - (math.floor(ratio) + 0.5)) < tolerance:
                num_full_measures = math.floor(ratio)
                if self.verbose:
                    print(f"修復 Case N.5x: 在 {theoretical_end_ts:.3f}s 後的間隙中偵測到 {num_full_measures} 個 4/4 拍和 1 個 2/4 拍")
                
                insert_ts = theoretical_end_ts
                # 插入 N 個完整的 4/4 拍小節
                for _ in range(num_full_measures):
                    full_measure_region = {
                        "time_sig": current_region['time_sig'],
                        "bpm": current_region['bpm'],
                        "start_time": insert_ts,
                        "downbeats": [insert_ts],
                        "avg_duration": measure_duration
                    }
                    patched_regions.append(full_measure_region)
                    insert_ts += measure_duration
                
                # 插入最後的 2/4 拍小節
                half_measure_region = {
                    "time_sig": 2,
                    "bpm": current_region['bpm'],
                    "start_time": insert_ts,
                    "downbeats": [insert_ts],
                    "avg_duration": measure_duration / 2
                }
                patched_regions.append(half_measure_region)

            # Case 2: 間隙長度約為 N 倍 (1, 2, 3...)
            elif abs(ratio - round(ratio)) < tolerance and round(ratio) >= 1:
                num_full_measures = round(ratio)
                if self.verbose:
                    print(f"修復 Case Nx: 在 {theoretical_end_ts:.3f}s 後的間隙中偵測到 {num_full_measures} 個 4/4 拍")

                insert_ts = theoretical_end_ts
                for _ in range(num_full_measures):
                    full_measure_region = {
                        "time_sig": current_region['time_sig'],
                        "bpm": current_region['bpm'],
                        "start_time": insert_ts,
                        "downbeats": [insert_ts],
                        "avg_duration": measure_duration
                    }
                    patched_regions.append(full_measure_region)
                    insert_ts += measure_duration
            
            current_region = next_region

        # 加入最後一個區域
        patched_regions.append(current_region)
        
        # --- 合併邏輯保持不變 ---
        merged_regions = []
        if not patched_regions:
            return []

        for region in patched_regions:
            if 'bpm' not in region or 'time_sig' not in region:
                merged_regions.append(region)
                continue

            if (merged_regions and 
                'bpm' in merged_regions[-1] and 'time_sig' in merged_regions[-1] and
                merged_regions[-1]['time_sig'] == region['time_sig'] and 
                abs(merged_regions[-1]['bpm'] - region['bpm']) < 1.0):
                merged_regions[-1]['downbeats'].extend(region['downbeats'])
            else:
                merged_regions.append(region)

        return merged_regions    


    def generate_tempo_info(self, path_tempo_output: str, path_beats_output: str = ""):
        filtered_beats = self.remove_close_beats()
        measures = self.compute_measures(filtered_beats)
        if not measures:
            print("無法計算任何小節，終止處理。")
            return

        global_time_sig = self.compute_global_time_sig(measures)
        
        stable_regions_indices = self.detect_stable_regions(measures)
        
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
            print("未偵測到任何穩定區域，無法生成節奏資訊。")
            return

        final_regions = self._patch_region_gaps(processed_regions)

        final_output = []
        for region in final_regions:
            final_output.append({
                "time_sig": region['time_sig'],
                "bpm": region['bpm'],
                "start": region['start_time'],
                "downbeats": region['downbeats']
            })
            
        self.save_json(path_tempo_output, final_output)

        if self.verbose:
            print(f"tempo.json 已生成，共 {len(final_output)} 個區域。")

        if path_beats_output:
            beat_pred_data = self.events_to_beat_pred(final_output)
            self.save_json(path_beats_output, beat_pred_data)
            if self.verbose:
                print(f"beats.json 已生成。")
    
    def events_to_beat_pred(self, region_events, default_time_sig=4, default_bpm=120):
        beat_pred = []
        downbeat_pred = []
        for region in region_events:
            ts = region.get("time_sig", default_time_sig)
            bpm = region.get("bpm", default_bpm)
            if bpm is None: bpm = default_bpm
            if ts is None: ts = default_time_sig
            
            beat_interval = 60 / bpm if bpm > 0 else 0
            for db in region.get("downbeats", []):
                downbeat_pred.append(db)
                for k in range(int(ts)):
                    beat_time = db + k * beat_interval
                    if beat_time not in beat_pred:
                        beat_pred.append(beat_time)

        beat_pred.sort()
        downbeat_pred.sort()
        return {"beat_pred": beat_pred, "downbeat_pred": downbeat_pred}


    def play_audio_with_clicks(self, audio_path, beat_pred_data):
        audio, sr = librosa.load(audio_path, sr=None)
        beats = beat_pred_data.get("beat_pred", [])
        downbeats = beat_pred_data.get("downbeat_pred", [])
        beats_click = librosa.clicks(times=beats, sr=sr, click_freq=1000.0, click_duration=0.1, length=len(audio))
        downbeats_click = librosa.clicks(times=downbeats, sr=sr, click_freq=1500.0, click_duration=0.15, length=len(audio))
        mixed_audio = 0.4 * audio + 0.6 * beats_click + 0.6 * downbeats_click
        return ipd.Audio(mixed_audio, rate=sr)