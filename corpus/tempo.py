import os
import json
from statistics import mode, StatisticsError, median
import numpy as np
import librosa
import IPython.display as ipd


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

    def _harmonize_tempo_regions(self, processed_regions, global_time_sig):
        """
        修正 BPM 加倍或減半的區域。
        """
        if len(processed_regions) < 2:
            return processed_regions

        # 使用中位數作為參考時長，更穩健
        all_durations = [r['avg_duration'] for r in processed_regions if r['avg_duration'] > 0]
        if not all_durations:
            return processed_regions
        reference_duration = median(all_durations)

        if self.verbose:
            print("\n--- 開始進行速度統一化（Harmonization）---")
            print(f"[DEBUG] 參考平均時長 (中位數): {reference_duration:.4f}")

        for region in processed_regions:
            avg_duration = region['avg_duration']
            
            # 避免除以零的錯誤
            if reference_duration <= 0 or avg_duration <= 0:
                continue

            # 確保總是大除以小，得到一個大於等於1的比例
            ratio = avg_duration / reference_duration if avg_duration > reference_duration else reference_duration / avg_duration
            
            # 檢查比例是否落在 [1.9, 2.1] 的區間內，並且當前區域是慢速區
            if 1.9 < ratio < 2.1 and avg_duration > reference_duration:
                if self.verbose:
                    print(f"[INFO] 檢測到慢速區域 (時長 {avg_duration:.4f}, 與參考時長比例 {ratio:.2f})，進行小節切分...")
                
                original_downbeats = region['downbeats']
                new_downbeats = []
                
                # 將該區域的每個小節從中切開，產生新的 downbeat
                for i in range(len(original_downbeats) - 1):
                    db1 = original_downbeats[i]
                    db2 = original_downbeats[i+1]
                    midpoint = db1 + (db2 - db1) / 2
                    new_downbeats.append(db1)
                    new_downbeats.append(midpoint)
                new_downbeats.append(original_downbeats[-1]) # 加入最後一個原始 downbeat
                
                region['downbeats'] = sorted(new_downbeats)
                
                # 重新計算該區域的 avg_duration 和 bpm
                new_durations = [region['downbeats'][i+1] - region['downbeats'][i] for i in range(len(region['downbeats']) - 1)]
                if new_durations:
                    region['avg_duration'] = sum(new_durations) / len(new_durations)
                    region['bpm'] = (60 * global_time_sig) / region['avg_duration']
                    if self.verbose:
                        print(f"[INFO] 修正後: 新平均時長={region['avg_duration']:.4f}, 新BPM={region['bpm']:.2f}")

        if self.verbose:
            print("--- 速度統一化完成 ---\n")
            
        return processed_regions

    def _merge_similar_regions(self, processed_regions, global_time_sig, threshold=0.1):
        """
        合併 BPM (平均時長) 相近的相鄰區域。
        """
        if self.verbose:
            print("\n--- 開始合併相似速度區域 ---")
        
        while True:
            merged_in_this_pass = False
            i = 0
            while i < len(processed_regions) - 1:
                region1 = processed_regions[i]
                region2 = processed_regions[i+1]

                # 當兩個相鄰區域的平均時長差距小於閾值時，合併
                if abs(region1['avg_duration'] - region2['avg_duration']) <= threshold:
                    if self.verbose:
                        print(f"[INFO] 合併區域 {i} (時長 {region1['avg_duration']:.4f}) 和 {i+1} (時長 {region2['avg_duration']:.4f})")
                    
                    # 合併 downbeats 列表
                    merged_downbeats = sorted(region1['downbeats'] + region2['downbeats'])
                    
                    # 創建新區域並重新計算 BPM
                    new_region = {
                        "start_time": merged_downbeats[0],
                        "downbeats": merged_downbeats,
                        "time_sig": global_time_sig
                    }
                    
                    merged_durations = [merged_downbeats[j+1] - merged_downbeats[j] for j in range(len(merged_downbeats) - 1)]
                    if merged_durations:
                        new_avg_duration = sum(merged_durations) / len(merged_durations)
                        new_region['avg_duration'] = new_avg_duration
                        new_region['bpm'] = (60 * global_time_sig) / new_avg_duration
                    else:
                        new_region['avg_duration'] = 0
                        new_region['bpm'] = 0

                    # 替換掉舊的區域
                    processed_regions = processed_regions[:i] + [new_region] + processed_regions[i+2:]
                    merged_in_this_pass = True
                    break # 完成一次合併後，從頭開始重新掃描
                i += 1
            
            if not merged_in_this_pass:
                break # 如果完整掃描一遍都沒有發生合併，則結束

        if self.verbose:
            print(f"--- 合併完成，剩餘 {len(processed_regions)} 個區域 ---\n")
        return processed_regions

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

        processed_regions = self._harmonize_tempo_regions(processed_regions, global_time_sig)
        processed_regions = self._merge_similar_regions(processed_regions, global_time_sig)

        final_output = []
        for region in processed_regions:
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