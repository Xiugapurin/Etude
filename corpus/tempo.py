import os
import json
from statistics import mode, StatisticsError
import numpy as np
import librosa
import IPython.display as ipd


class TempoInfoGenerator:
    def __init__(self, path_beat, verbose = False):
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
        """
        根據 downbeats 與 beats，計算每個小節資訊：
        - 'start': 小節起始時間（downbeat 時間）
        - 'raw_beats': 小節內拍數（downbeat 與該小節內其他 beat 的數量）
        - 'duration': 小節時長（當前 downbeat 到下一個 downbeat）
        - 'measure_beats': 小節內所有 beat 的時間列表（包含 downbeat）
        - 'uniform': 檢查該小節內 beat 間隔是否均勻，若均勻則為 True，不均勻則為 False
            判定方式：計算各拍間隔的標準差與平均值的比例，若比例低於 uniformity_threshold（預設 0.1），則認為均勻
        """
        measures = []
        for i in range(len(self.downbeat_pred)-1):
            start = self.downbeat_pred[i]
            end = self.downbeat_pred[i+1]
            duration = end - start
            # 取得落在該小節內的 beat（不包含 downbeat）
            beats_in_measure = [b for b in beats if start < b < end]
            # 小節內 beat 時間列表，包含 downbeat
            measure_beats = [start] + beats_in_measure
            raw_beats = len(measure_beats)
            # 檢查 beat 間隔是否均勻
            if len(measure_beats) > 1:
                intervals = [measure_beats[j+1] - measure_beats[j] for j in range(len(measure_beats)-1)]
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                rel_std = std_interval / mean_interval if mean_interval > 0 else 0
                uniform = rel_std < uniformity_threshold
            else:
                uniform = True  # 只有一個 beat 時視為合理
            measures.append({
                'start': start,
                'raw_beats': raw_beats,
                'duration': duration,
                'measure_beats': measure_beats,
                'uniform': uniform
            })
        return measures


    def compute_global_time_sig(self, measures):
        """
        根據所有小節的原始拍數（raw_beats）取眾數作為全局拍號，
        但僅考慮每個小節內 beat 分布均勻的（uniform 為 True）小節；
        若沒有任何小節通過檢查，則使用所有小節的 raw_beats。
        若眾數為 2，則將全局拍號轉為4 (4/4)。
        同時對每個 measure 設定 time_sig 欄位：
        - 若全局拍號為4，但某 measure 原始為2，則標記 temporary 為 True，
            並強制將該 measure 的 time_sig 更新為 4，以便 BPM 計算時乘以2。
        BPM 的計算公式為： BPM = (60 * time_sig) / duration
        """
        # 只使用均勻的小節參與眾數計算
        valid_beats = [m['raw_beats'] for m in measures if m.get('uniform', True)]
        
        if len(valid_beats) < 10:
            mode_val = 4
        else:
            try:
                mode_val = mode(valid_beats)
            except StatisticsError:
                mode_val = valid_beats[0]
        
        if self.verbose:
            print(f"[DEBUG] 用於取眾數的小節拍數列表: {valid_beats}")
            print(f"[DEBUG] 取眾數拍數: {mode_val}")
        
        # 若眾數為2，轉換為4/4
        if mode_val == 2:
            global_time_sig = 4
        else:
            global_time_sig = mode_val
        
        if self.verbose:
            print(f"[DEBUG] 全局拍號設定為: {global_time_sig}/4")
        
        # 更新每個 measure 的 time_sig 與 temporary 標記，並重新計算 BPM
        for m in measures:
            if global_time_sig == 4 and m['raw_beats'] == 2:
                m['temporary'] = True
                m['time_sig'] = global_time_sig  # 強制更新為4，確保 BPM = (60*4)/duration
            else:
                m['temporary'] = False
                m['time_sig'] = global_time_sig
            if m['duration'] > 0:
                m['bpm'] = (60 * m['time_sig']) / m['duration']
            else:
                m['bpm'] = 0
        return global_time_sig


    def detect_stable_regions(self, measures, window_size=4, threshold=0.1):
        """
        掃描 measures（依 downbeat 的 start 時間），尋找連續 window_size 小節間隔標準差小於 threshold 的區域，
        並嘗試向後延伸。
        
        回傳 stable_regions 列表，每一項為 (start_index, end_index, ideal_interval)
        """
        stable_regions = []
        i = 0
        while i <= len(measures) - window_size:
            # 計算窗口內各相鄰小節間隔
            intervals = [measures[j+1]['start'] - measures[j]['start'] for j in range(i, i+window_size-1)]
            std_interval = np.std(intervals)
            if std_interval < threshold:
                ideal_interval = np.mean(intervals)
                region_start = i
                region_end = i + window_size - 1

                if self.verbose:
                    print(f"[DEBUG] 在索引 {i} 發現候選穩定區域，初始窗口間隔: {intervals}, std={std_interval:.3f}, ideal_interval={ideal_interval:.3f}")
                
                # 向後延伸檢查
                j = region_end
                while j + 1 < len(measures):
                    predicted = measures[j]['start'] + ideal_interval
                    if abs(measures[j+1]['start'] - predicted) < threshold:
                        region_end = j + 1
                        j += 1
                    else:
                        break
                stable_regions.append((region_start, region_end, ideal_interval))

                if self.verbose:
                    print(f"[DEBUG] 穩定區域確定為索引 {region_start} 到 {region_end}，延伸後理想間隔 = {ideal_interval:.3f}")

                i = region_end + 1
            else:
                i += 1
        return stable_regions


    def partition_unstable_regions(self, total_measures, stable_regions):
        """
        根據穩定區域列表，將 measures 中不屬於任何穩定區域的連續區段劃分為不穩定區域。
        回傳 unstable_regions 列表，每一項為 (start_index, end_index)
        """
        unstable_regions = []
        stable_idx = set()
        for reg in stable_regions:
            start, end, _ = reg
            stable_idx.update(range(start, end+1))
        i = 0
        while i < total_measures:
            if i not in stable_idx:
                start_un = i
                while i < total_measures and i not in stable_idx:
                    i += 1
                end_un = i - 1
                unstable_regions.append((start_un, end_un))
            else:
                i += 1

        if self.verbose:
            print(f"[DEBUG] 檢測到 {len(unstable_regions)} 個不穩定區域: {unstable_regions}")

        return unstable_regions


    def adjust_unstable_region(self, measures, un_region, prev_stable_idx, next_stable_idx, global_time_sig, epsilon=0.1):
        """
        根據不穩定區域（un_region = (start, end)）及其前後穩定區參考值，
        嘗試依照下列邏輯修正：
        - 設 A 為 prev_stable（最後一個穩定區小節），其時長 d_a
        - 設 B 為不穩定區域第一個小節，且 next 為該區後第一個穩定區的小節
        - 計算 d = next.start - A.start, ratio = d / d_a
        - 若 global_time_sig 為 4 且 ratio 的小數部分接近 0.5 (情形3.1)：
            將該不穩定區域內最後一個小節的時間調整為 next.start - (0.5 * d_a)
            並根據 ratio 的整數部分（假設為 n）重新線性分配區間內 downbeat 時間。
        - 若 ratio 的小數部分接近 0 (情形3.2)：
            以 A.start 與 next.start 為端點，將不穩定區域 downbeat 均勻分佈。
        - 其他情況（情形3.3）：放棄調整
        若不穩定區域長度超過 4 則放棄調整。
        
        此函數直接修改 measures 中對應的 start 時間，並印出調整細節。
        """
        start_un, end_un = un_region
        region_length = end_un - start_un + 1

        if self.verbose:
            print(f"[DEBUG] 處理不穩定區域索引 {start_un} 到 {end_un}，共 {region_length} 個小節")

        if region_length > 4:
            if self.verbose:
                print("[DEBUG] 不穩定區域長度超過4，放棄修正")
            return

        if prev_stable_idx is None or next_stable_idx is None:
            if self.verbose:
                print("[DEBUG] 不穩定區域缺少前後穩定區參考，放棄修正")
            return

        A = measures[prev_stable_idx]  # 前一穩定區最後一小節
        next_measure = measures[next_stable_idx]  # 後一穩定區第一小節
        d_a = A['duration']
        d = next_measure['start'] - A['start']
        ratio = d_a / d
        n_int = int(ratio)
        frac = ratio - n_int

        if self.verbose:
            print(f"[DEBUG] A.start={A['start']:.3f}, A.duration={d_a:.3f}, next.start={next_measure['start']:.3f}")
            print(f"[DEBUG] d = {d:.3f}, d/d_a = {ratio:.3f} (整數部分 = {n_int}, 小數部分 = {frac:.3f})")

        # 情形3.1：判定 x 的小數部分接近 0.5，代表該區域內有一個臨時 2/4 拍的小節
        if global_time_sig == 4 and abs(frac - 0.5) < epsilon:
            if self.verbose:
                print("[DEBUG] 判定為情形 3.1：存在一個臨時 2/4 拍的小節")
            # 將不穩定區域最後一小節調整為 next_measure.start - (0.5*d_a)
            new_last = next_measure['start'] - (0.5 * d_a)
            if self.verbose:
                print(f"[DEBUG] 將不穩定區域最後小節從原始時間 {measures[end_un]['start']:.3f} 調整為 {new_last:.3f}")
            measures[end_un]['start'] = new_last
            # 記錄該小節為臨時 2/4 拍
            measures[end_un]['time_sig'] = 2
            measures[end_un]['temporary'] = True
            # 理想上，該區域包含 n_int 個完整小節加上一個臨時小節，故總數為 n_int + 0.5
            ideal_total = n_int + 0.5
            total_gap = new_last - A['start']
            # 線性插值重新分配不穩定區內 downbeat（不包含 A，但包含最後已修正的值）
            for count, idx in enumerate(range(start_un, end_un), start=1):
                new_time = A['start'] + (total_gap / ideal_total) * count
                print(f"[DEBUG] 調整 measure[{idx}].start 從 {measures[idx]['start']:.3f} 調整為 {new_time:.3f}")
                measures[idx]['start'] = new_time
        # 情形3.2：若 x 的小數部分接近 0，則認為 downbeat 錯置，進行均勻分佈
        elif abs(frac) < epsilon:
            if self.verbose:
                print("[DEBUG] 判定為情形 3.2：downbeat 錯置，進行均勻分佈")
            total_gap = next_measure['start'] - A['start']
            step = total_gap / (n_int + 1)
            for count, idx in enumerate(range(start_un, end_un + 1), start=1):
                new_time = A['start'] + step * count
                if self.verbose:
                    print(f"[DEBUG] 均勻分配 measure[{idx}].start 從 {measures[idx]['start']:.3f} 調整為 {new_time:.3f}")
                measures[idx]['start'] = new_time
        else:
            if self.verbose:
                print("[DEBUG] 該不穩定區域不符合修正條件，放棄調整")


    def generate_tempo_events(self, measures, stable_regions, global_time_sig):
        """
        依據 measures 與穩定區域資訊生成 tempo.json 所需的事件列表。
        - 每個小節的 downbeat皆產生一個 "bar" 事件
        - 在每個穩定區域的起始小節產生一次 "time_sig" 與 "bpm" 事件（取該區域所有小節 BPM 的平均值）
        """
        events = []
        # 每個小節產生 bar 事件
        for m in measures:
            events.append({
                "e": "bar",
                "t": m['start'],
                "v": 0
            })
        # 對每個穩定區域產生 bpm 與 time_sig 事件
        for reg in stable_regions:
            start_idx, end_idx, _ = reg
            region_measures = measures[start_idx:end_idx+1]
            bpms = [m['bpm'] for m in region_measures]
            avg_bpm = sum(bpms)/len(bpms) if bpms else 0
            avg_bpm = max(60, min(300, avg_bpm))

            if self.verbose:
                print(f"[DEBUG] 穩定區域 {start_idx}-{end_idx}，平均 BPM = {avg_bpm:.2f}")
            event_time = region_measures[0]['start']
            events.append({
                "e": "time_sig",
                "t": event_time,
                "v": global_time_sig
            })
            events.append({
                "e": "bpm",
                "t": event_time,
                "v": avg_bpm
            })
        # 依時間排序
        events.sort(key=lambda x: x['t'])
        return events


    def generate_region_events_from_events(self, events):
        """
        根據已排序好的 events（包含 "bpm"、"time_sig" 與 "bar" 事件）生成區域事件，
        新區域從讀到第一個 bpm 或 time_sig 事件開始（包含該事件），直到遇到下一組 bpm/time_sig 事件為止。
        
        回傳格式為：
        [
            {
                "time_sig": 該區域的拍號,
                "bpm": 該區域的 bpm,
                "start": 該區域的起始時間,
                "downbeats": [該區域的所有 downbeat 時間]
            },
            ...
        ]
        """
        # 先將事件依照時間排序
        priority = {"time_sig": 0, "bpm": 0, "bar": 1}
        sorted_events = sorted(events, key=lambda x: (x["t"], priority.get(x["e"], 2)))
        
        regions = []
        current_region = None
        for evt in sorted_events:
            if evt["e"] in ["bpm", "time_sig"]:
                # 若已有當前區域且此 bpm/time_sig 事件的時間大於目前區域的起始時間，
                # 表示遇到新一組設定，則先保存舊區域
                if current_region is not None and evt["t"] > current_region["start"]:
                    regions.append(current_region)
                    current_region = None
                # 若沒有當前區域，則以此事件的時間作為新區域的起始
                if current_region is None:
                    current_region = {
                        "time_sig": None,
                        "bpm": None,
                        "start": evt["t"],
                        "downbeats": []
                    }
                # 更新區域設定：若為 time_sig 則更新拍號；若為 bpm 則更新 bpm 值
                if evt["e"] == "time_sig":
                    current_region["time_sig"] = evt["v"]
                elif evt["e"] == "bpm":
                    current_region["bpm"] = evt["v"]
            elif evt["e"] == "bar":
                # 若目前尚無區域，則以預設值建立一個區域（此情形通常發生在最前面沒有 bpm/time_sig 事件時）
                if current_region is None:
                    current_region = {
                        "time_sig": 4,
                        "bpm": 120,
                        "start": evt["t"],
                        "downbeats": []
                    }
                # 將 bar 事件時間加入當前區域的 downbeats
                current_region["downbeats"].append(evt["t"])
        if current_region is not None:
            regions.append(current_region)
        return regions


    def events_to_beat_pred(self, region_events, default_time_sig=4, default_bpm=120):
        """
        根據新的區域事件格式轉換為 beat_pred.json 的資料結構：
        - downbeat_pred：合併所有區域內的 downbeat 時間
        - beat_pred：對每個區域內每個 downbeat，
                    先加入該 downbeat時間，再根據區域內的 BPM 與拍號補齊該小節內其他 beat 的時間
                    
        若某區域缺少 BPM 或拍號資訊，則使用預設值。
        """
        beat_pred = []
        downbeat_pred = []
        for region in region_events:
            ts = region.get("time_sig", default_time_sig)
            bpm = region.get("bpm", default_bpm)
            # 若 BPM 或拍號未設定，採用預設值
            if bpm is None:
                bpm = default_bpm
            if ts is None:
                ts = default_time_sig
            beat_interval = 60 / bpm if bpm > 0 else 0
            for db in region.get("downbeats", []):
                # 將 downbeat 加入
                beat_pred.append(db)
                downbeat_pred.append(db)
                # 補齊該小節內其他 beat（downbeat 為第一拍，其餘拍數 ts-1）
                for k in range(1, int(ts)):
                    beat_time = db + k * beat_interval
                    beat_pred.append(beat_time)
        beat_pred.sort()
        downbeat_pred.sort()
        return {"beat_pred": beat_pred, "downbeat_pred": downbeat_pred}


    def play_audio_with_clicks(self, audio_path, beat_pred_data):
        """
        根據傳入的音訊檔與 beat_pred.json 資料，在原始音訊上加入 click 音效以檢查 beat 與 downbeat 時間是否正確
        參數：
        audio_path: 原始音訊檔路徑（例如 '../origin.wav'）
        beat_pred_data: 包含 "beat_pred" 與 "downbeat_pred" 的字典
        回傳：
        IPython.display.Audio 物件，可直接在 notebook 中播放
        """
        # 載入原始音訊，保留原始取樣率
        audio, sr = librosa.load(audio_path, sr=None)
        
        beats = beat_pred_data.get("beat_pred", [])
        downbeats = beat_pred_data.get("downbeat_pred", [])
        
        # 產生 beat click 音效（較低頻率）
        beats_click = librosa.clicks(times=beats, sr=sr, click_freq=1000.0, click_duration=0.1, length=len(audio))
        # 產生 downbeat click 音效（較高頻率）
        downbeats_click = librosa.clicks(times=downbeats, sr=sr, click_freq=1500.0, click_duration=0.15, length=len(audio))
        
        # 將原始音訊與 click 混合
        mixed_audio = 0.4 * audio + 0.6 * beats_click + 0.6 * downbeats_click
        return ipd.Audio(mixed_audio, rate=sr)


    def generate_tempo_info(self, path_tempo_output: str, path_beats_output: str = ""):
        # 去除與 downbeat 接近的 beat
        filtered_beats = self.remove_close_beats(beat_threshold=0.1)
        if self.verbose:
            print("[DEBUG] 篩選後 beat 數量 =", len(filtered_beats))
        
        # 初步計算小節資訊
        measures = self.compute_measures(filtered_beats, uniformity_threshold=0.1)

        if self.verbose:
            print(f"[DEBUG] 初步計算出 {len(measures)} 個小節")
        
        # 決定全局拍號，並更新各 measure 的 BPM
        global_time_sig = self.compute_global_time_sig(measures)
        
        # 檢測穩定區域（至少4個小節）
        stable_regions = self.detect_stable_regions(measures, window_size=4, threshold=0.1)
        
        # 分割出不穩定區域
        unstable_regions = self.partition_unstable_regions(len(measures), stable_regions)
        
        # 對每個不穩定區域進行檢查與可能的修正
        # 此處假設每個不穩定區域的前後均有穩定區參考；若缺失則跳過
        for un_region in unstable_regions:
            start_un, end_un = un_region
            prev_stable_idx = start_un - 1 if start_un - 1 >= 0 else None
            # 尋找下一個穩定區：在 stable_regions 中找到第一個區間，其起始索引大於 end_un
            next_stable_idx = None
            for reg in stable_regions:
                if reg[0] > end_un:
                    next_stable_idx = reg[0]
                    break
            self.adjust_unstable_region(measures, un_region, prev_stable_idx, next_stable_idx, global_time_sig, epsilon=0.1)
        
        # 依照最新 measures 重新計算各小節 BPM（若 downbeat 時間被調整）
        for i in range(len(measures)-1):
            measures[i]['duration'] = measures[i+1]['start'] - measures[i]['start']
            measures[i]['bpm'] = (60 * measures[i]['time_sig']) / measures[i]['duration'] if measures[i]['duration'] > 0 else 0
        # 最後一個小節 BPM 不更新
        
        # 生成 tempo 事件
        events = self.generate_tempo_events(measures, stable_regions, global_time_sig)
        region_events = self.generate_region_events_from_events(events)
        
        # 輸出 tempo.json
        self.save_json(path_tempo_output, region_events)

        if self.verbose:
            print(f"tempo.json 已生成，共 {len(region_events)} 條事件。")

        if path_beats_output:
            beat_pred_data = self.events_to_beat_pred(region_events)
            self.save_json(path_beats_output, beat_pred_data)