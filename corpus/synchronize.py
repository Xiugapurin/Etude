import os
import json
from contextlib import redirect_stdout
import libtsm
import numpy as np
import soundfile as sf
import librosa
import librosa.display
from scipy.interpolate import interp1d
import tempfile
import IPython.display as ipd
from pydub import AudioSegment
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.signal import find_peaks
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import (
    compute_optimal_chroma_shift,
    shift_chroma_vectors,
    make_path_strictly_monotonic,
)
from synctoolbox.feature.chroma import (
    pitch_to_chroma,
    quantize_chroma,
    quantized_chroma_to_CENS,
)
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning

class Synchronizer:
    def __init__(self):
        self.Fs = 22050
        self.feature_rate = 50
        self.threshold_rec = 10 ** 6
        self.step_weights = np.array([1.5, 1.5, 2.0])
        self.win_len_smooth = np.array([101, 51, 21, 1])

        self.t1 = None
        self.t2 = None 


    def load_audio(self, origin_path, cover_path):
        origin_audio, _ = librosa.load(origin_path, sr=self.Fs)
        cover_audio, _ = librosa.load(cover_path, sr=self.Fs)

        return origin_audio, cover_audio

    def get_features(self, audio, tuning_offset, visualize=False):
        with redirect_stdout(open(os.devnull, "w")):
            f_pitch = audio_to_pitch_features(
                f_audio=audio,
                Fs=self.Fs,
                tuning_offset=tuning_offset,
                feature_rate=self.feature_rate,
                verbose=visualize,
            )
            f_chroma = pitch_to_chroma(f_pitch=f_pitch)
            f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)

            f_pitch_onset = audio_to_pitch_onset_features(
                f_audio=audio, Fs=self.Fs, tuning_offset=tuning_offset, verbose=visualize
            )
            f_DLNCO = pitch_onset_features_to_DLNCO(
                f_peaks=f_pitch_onset,
                feature_rate=self.feature_rate,
                feature_sequence_length=f_chroma_quantized.shape[1],
                visualize=visualize,
            )
        return f_chroma_quantized, f_DLNCO


    def save_wp(self, wp, song_dir, version_key, num_frames_cover, num_frames_origin):
        json_path = os.path.join(song_dir, 'wp.json')
        all_data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: {json_path} format error, creating a new file.")
        
        version_data = {
            "wp": wp.tolist(),
            "num_frames_cover": num_frames_cover,
            "num_frames_origin": num_frames_origin
        }
        all_data[version_key] = version_data

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4)


    def load_wp(self, song_dir, version_key):
        json_path = os.path.join(song_dir, 'wp.json')
        if not os.path.exists(json_path):
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            version_data = all_data.get(version_key)
            if version_data and all(k in version_data for k in ["wp", "num_frames_cover", "num_frames_origin"]):
                return version_data
            return None
        except (json.JSONDecodeError, KeyError):
            print(f"Error: Fail to read or parse {json_path}.")
            return None


    def get_wp(self, origin_path, cover_path, song_dir):
        version_key = os.path.basename(cover_path).split('.')[0]

        cached_data = self.load_wp(song_dir, version_key)
        if cached_data:
            num_frames_cover = cached_data["num_frames_cover"]
            num_frames_origin = cached_data["num_frames_origin"]
            
            self.t1 = np.arange(num_frames_cover) / self.feature_rate
            self.t2 = np.arange(num_frames_origin) / self.feature_rate
            
            wp = np.array(cached_data["wp"])
            return wp.astype(int)

        origin_audio, cover_audio = self.load_audio(origin_path, cover_path)
        
        tuning_offset_1 = estimate_tuning(cover_audio, self.Fs)
        tuning_offset_2 = estimate_tuning(origin_audio, self.Fs)
        
        f_chroma_quantized_1, f_DLNCO_1 = self.get_features(cover_audio, tuning_offset_1)
        f_chroma_quantized_2, f_DLNCO_2 = self.get_features(origin_audio, tuning_offset_2)
        
        num_frames_cover = f_chroma_quantized_1.shape[1]
        num_frames_origin = f_chroma_quantized_2.shape[1]

        self.t1 = np.arange(num_frames_cover) / self.feature_rate
        self.t2 = np.arange(num_frames_origin) / self.feature_rate
        
        f_cens_1hz_1 = quantized_chroma_to_CENS(f_chroma_quantized_1, 201, 50, self.feature_rate)[0]
        f_cens_1hz_2 = quantized_chroma_to_CENS(f_chroma_quantized_2, 201, 50, self.feature_rate)[0]
        opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_1, f_cens_1hz_2)
        f_chroma_quantized_2 = shift_chroma_vectors(f_chroma_quantized_2, opt_chroma_shift)
        f_DLNCO_2 = shift_chroma_vectors(f_DLNCO_2, opt_chroma_shift)
        
        wp = sync_via_mrmsdtw(
            f_chroma1=f_chroma_quantized_1, f_onset1=f_DLNCO_1,
            f_chroma2=f_chroma_quantized_2, f_onset2=f_DLNCO_2,
            input_feature_rate=self.feature_rate, step_weights=self.step_weights,
            win_len_smooth=self.win_len_smooth, threshold_rec=self.threshold_rec,
            alpha=0.5,
        )
        wp = make_path_strictly_monotonic(wp)
        
        self.save_wp(wp, song_dir, version_key, num_frames_cover, num_frames_origin)

        return wp.astype(int)


    def get_wp_and_adjust(self, origin_path, cover_path, cover_json_path):
        origin_audio, cover_audio = self.load_audio(origin_path, cover_path)

        tuning_offset_1 = estimate_tuning(cover_audio, self.Fs)
        tuning_offset_2 = estimate_tuning(origin_audio, self.Fs)

        f_chroma_quantized_1, f_DLNCO_1 = self.get_features(
            cover_audio, tuning_offset_1, visualize=False
        )
        f_chroma_quantized_2, f_DLNCO_2 = self.get_features(
            origin_audio, tuning_offset_2, visualize=False
        )

        f_cens_1hz_1 = quantized_chroma_to_CENS(
            f_chroma_quantized_1, 201, 50, self.feature_rate
        )[0]
        f_cens_1hz_2 = quantized_chroma_to_CENS(
            f_chroma_quantized_2, 201, 50, self.feature_rate
        )[0]

        opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_1, f_cens_1hz_2)

        f_chroma_quantized_2 = shift_chroma_vectors(f_chroma_quantized_2, opt_chroma_shift)
        f_DLNCO_2 = shift_chroma_vectors(f_DLNCO_2, opt_chroma_shift)

        wp = sync_via_mrmsdtw(
            f_chroma1=f_chroma_quantized_1,
            f_onset1=f_DLNCO_1,
            f_chroma2=f_chroma_quantized_2,
            f_onset2=f_DLNCO_2,
            input_feature_rate=self.feature_rate,
            step_weights=self.step_weights,
            win_len_smooth=self.win_len_smooth,
            threshold_rec=self.threshold_rec,
            alpha=0.5,
        )
        wp = make_path_strictly_monotonic(wp)

        pitch_shift = -opt_chroma_shift % 12
        if pitch_shift > 6:
            pitch_shift -= 12

        with open(cover_json_path, 'r', encoding='utf-8') as f:
            note_data = json.load(f)

        for note in note_data:
            shifted_pitch = note['pitch'] + pitch_shift
            note['pitch'] = int(min(max(shifted_pitch, 21), 108))

        with open(cover_json_path, 'w', encoding='utf-8') as f:
            json.dump(note_data, f, indent=4)

        return wp
    
    def strong_alignment(self, origin_path, cover_path):
        origin_audio, cover_audio = self.load_audio(origin_path, cover_path)

        tuning_offset_1 = estimate_tuning(cover_audio, self.Fs)
        tuning_offset_2 = estimate_tuning(origin_audio, self.Fs)

        f_chroma_quantized_1, f_DLNCO_1 = self.get_features(
            cover_audio, tuning_offset_1, visualize=False
        )
        f_chroma_quantized_2, f_DLNCO_2 = self.get_features(
            origin_audio, tuning_offset_2, visualize=False
        )

        f_cens_1hz_1 = quantized_chroma_to_CENS(
            f_chroma_quantized_1, 201, 50, self.feature_rate
        )[0]
        f_cens_1hz_2 = quantized_chroma_to_CENS(
            f_chroma_quantized_2, 201, 50, self.feature_rate
        )[0]

        opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_1, f_cens_1hz_2)

        f_chroma_quantized_2 = shift_chroma_vectors(f_chroma_quantized_2, opt_chroma_shift)
        f_DLNCO_2 = shift_chroma_vectors(f_DLNCO_2, opt_chroma_shift)

        wp = sync_via_mrmsdtw(
            f_chroma1=f_chroma_quantized_1,
            f_onset1=f_DLNCO_1,
            f_chroma2=f_chroma_quantized_2,
            f_onset2=f_DLNCO_2,
            input_feature_rate=self.feature_rate,
            step_weights=self.step_weights,
            win_len_smooth=self.win_len_smooth,
            threshold_rec=self.threshold_rec,
            alpha=0.5,
        )
        wp = make_path_strictly_monotonic(wp)

        pitch_shift_for_cover_audio = -opt_chroma_shift % 12
        if pitch_shift_for_cover_audio > 6:
            pitch_shift_for_cover_audio -= 12
        cover_audio_shifted = libtsm.pitch_shift(
            cover_audio, pitch_shift_for_cover_audio * 100, order="tsm-res"
        )

        time_map = wp.T / self.feature_rate * self.Fs
        time_map[time_map[:, 0] > len(cover_audio), 0] = len(cover_audio) - 1
        time_map[time_map[:, 1] > len(origin_audio), 1] = len(origin_audio) - 1

        y_hpstsm = libtsm.hps_tsm(cover_audio_shifted, time_map)

        stereo_sonification = np.hstack((origin_audio.reshape(-1, 1), y_hpstsm))

        print("Synchronized versions", flush=True)
        ipd.display(ipd.Audio(stereo_sonification.T, rate=self.Fs, normalize=True))

        # print("Aligned Audio 1", flush=True)
        # ipd.display(ipd.Audio(y_hpstsm.T, rate=self.Fs))

    
    def weakly_align_transcription(self, origin_path, cover_path, transcription_path, time_map_path, aligned_output_path):
        """
        Perform weak alignment of origin audio and transcription, and save the result as a new JSON file and MIDI file.

        :param origin_path: Path to the original audio file (origin.wav).
        :param cover_path: Path to the cover audio file (cover.wav).
        :param transcription_path: Path to the transcription.json file (cover transcription).
        :param time_map_path: Path to the time_map.json file.
        :param aligned_output_path: Path to save the aligned transcription JSON.
        """
        # Load time_map
        if not os.path.exists(time_map_path):
            print(f"Missing time_map file: {time_map_path}")
            return

        with open(time_map_path, "r") as f:
            time_map = json.load(f)
        
        # Load transcription
        if not os.path.exists(transcription_path):
            print(f"Missing transcription file: {transcription_path}")
            return

        with open(transcription_path, "r") as f:
            transcription = json.load(f)

        origin_audio = AudioSegment.from_file(origin_path).set_channels(1)
        cover_audio = AudioSegment.from_file(cover_path).set_channels(1)

        origin_duration = origin_audio.duration_seconds
        cover_duration = cover_audio.duration_seconds
        
        # Align transcription events
        t_S0, t_P0 = time_map[0]
        t_Sn, t_Pn = time_map[-1]
        first_duration = min(t_S0, t_P0)
        last_duration = min(origin_duration - t_Sn, cover_duration - t_Pn, t_Sn - time_map[-2][0])
        time_map = [[t_S0 - first_duration, t_P0 - first_duration]] + time_map + [[t_Sn + last_duration, t_Pn + last_duration]]
        aligned_transcription = []

        for i in range(len(time_map) - 1):
            t_S1, t_P1 = time_map[i]
            t_S2, t_P2 = time_map[i + 1]

            # Select notes within the current time segment
            for note in transcription:
                if note["pitch"] > 100: continue
                t_on = note["onset"]
                t_off = note["offset"]

                if t_P1 - 0.1 <= t_on < min(t_P2, t_P1 + t_S2 - t_S1):
                    new_onset = t_on - t_P1 + t_S1
                    aligned_note = {
                        "pitch": note["pitch"],
                        "onset": new_onset,
                        "offset": new_onset + t_off - t_on,
                        "velocity": note["velocity"]
                    }
                    aligned_transcription.append(aligned_note)

        # Save aligned transcription to JSON
        with open(aligned_output_path, "w") as f:
            json.dump(aligned_transcription, f, indent=4)


    def sync_audio_files_with_downbeats(
        self, origin_audio_path, cover_audio_path, downbeats, wp
    ):
        Fs = 22050
        feature_rate = 50

        cover_audio, _ = librosa.load(cover_audio_path, sr=Fs)
        origin_audio, _ = librosa.load(origin_audio_path, sr=Fs)

        time_origin = wp[1] / feature_rate
        time_cover = wp[0] / feature_rate

        interp_func = scipy.interpolate.interp1d(time_origin, time_cover, kind="linear")

        aligned_frames = [(0, 0)]
        for db in downbeats:
            if db <= time_origin[-1]:
                corresponding_time_cover = interp_func(db)
                aligned_frames.append((int(db * Fs), int(corresponding_time_cover * Fs)))

        print("Aligned frames (in samples):", aligned_frames)

        output_audio = []
        synced_cover_audio = np.zeros_like(origin_audio)

        for i in range(len(aligned_frames)):
            start_origin, start_cover = aligned_frames[i]

            if i < len(aligned_frames) - 1:
                end_origin = aligned_frames[i + 1][0]
            else:
                end_origin = len(origin_audio) - 1

            origin_segment = origin_audio[start_origin:end_origin]

            cover_length = len(origin_segment)
            end_cover = start_cover + cover_length

            if end_cover <= len(cover_audio):
                cover_segment = cover_audio[start_cover:end_cover]
            else:
                cover_segment = np.zeros(cover_length)
                available_length = len(cover_audio) - start_cover
                if available_length > 0:
                    cover_segment[:available_length] = cover_audio[start_cover:]

            cover_segment = librosa.util.fix_length(cover_segment, size=len(origin_segment))

            synced_cover_audio[start_origin:end_origin] = cover_segment

            stereo_segment = np.vstack((origin_segment, cover_segment)).T
            output_audio.append(stereo_segment)

        stereo_audio = np.concatenate(output_audio, axis=0)
        # sf.write(output_path, stereo_audio, Fs)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, stereo_audio, Fs)

        ipd.display(ipd.Audio(temp_path))

        return stereo_audio, synced_cover_audio
    

    def sync_audio_files_from_docs(self, path_to_warp, path_to_reference, output_path=""):
        """
        一個完全依照官方文件邏輯重寫的同步方法。
        
        Args:
            path_to_warp (str): 要被拉伸的音訊路徑 (等同於官方文件的 audio_1)。
            path_to_reference (str): 作為參考基準的音訊路徑 (等同於官方文件的 audio_2)。
            output_path (str): 輸出檔案的路徑。
        """
        print("--- Running new method: sync_audio_files_from_docs ---")
        
        # 1. 載入音訊
        audio_1, _ = librosa.load(path_to_warp, sr=self.Fs)
        audio_2, _ = librosa.load(path_to_reference, sr=self.Fs)

        # 2. 估計音準偏移
        tuning_offset_1 = estimate_tuning(audio_1, self.Fs)
        tuning_offset_2 = estimate_tuning(audio_2, self.Fs)

        # 3. 計算特徵
        f_chroma_quantized_1, f_DLNCO_1 = self.get_features(audio_1, tuning_offset_1, visualize=False)
        f_chroma_quantized_2, f_DLNCO_2 = self.get_features(audio_2, tuning_offset_2, visualize=False)

        # 4. 尋找最佳 chroma shift
        f_cens_1hz_1 = quantized_chroma_to_CENS(f_chroma_quantized_1, 201, 50, self.feature_rate)[0]
        f_cens_1hz_2 = quantized_chroma_to_CENS(f_chroma_quantized_2, 201, 50, self.feature_rate)[0]
        opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_1, f_cens_1hz_2)
        f_chroma_quantized_2 = shift_chroma_vectors(f_chroma_quantized_2, opt_chroma_shift)
        f_DLNCO_2 = shift_chroma_vectors(f_DLNCO_2, opt_chroma_shift)

        # 5. 執行 MrMsDTW
        wp = sync_via_mrmsdtw(
            f_chroma1=f_chroma_quantized_1,
            f_onset1=f_DLNCO_1,
            f_chroma2=f_chroma_quantized_2,
            f_onset2=f_DLNCO_2,
            input_feature_rate=self.feature_rate,
            step_weights=self.step_weights,
            threshold_rec=self.threshold_rec,
            verbose=True
        )
        wp = make_path_strictly_monotonic(wp)

        # 6. 音高變換 (要被拉伸的 audio_1)
        pitch_shift_for_audio_1 = -opt_chroma_shift % 12
        if pitch_shift_for_audio_1 > 6:
            pitch_shift_for_audio_1 -= 12
        audio_1_shifted = libtsm.pitch_shift(audio_1, pitch_shift_for_audio_1 * 100, order="tsm-res")
        
        t_warp_source = wp[0] / self.feature_rate
        t_warp_ref = wp[1] / self.feature_rate
        
        # 為了建立內插函數，我們需要確保 x 軸 (t_warp_ref) 是嚴格遞增的
        # 我們過濾掉 t_warp_ref 中所有重複的點
        diffs = np.diff(t_warp_ref)
        mask = np.hstack([True, diffs > 0])
        
        t_warp_ref_unique = t_warp_ref[mask]
        t_warp_source_corresponding = t_warp_source[mask]

        # 建立一個從「參考時間」映射到「待拉伸時間」的內插函數
        # 這個函數回答了問題：“在參考音訊的 t 時刻，我應該去待拉伸音訊的哪個時刻取樣？”
        # bounds_error=False 和 fill_value="extrapolate" 確保即使超出範圍也能得到一個值
        interp_func = interp1d(t_warp_ref_unique, t_warp_source_corresponding, 
                               kind='linear', bounds_error=False, fill_value="extrapolate")

        # 建立一個數學上完美的、線性增長的目標時間軸 (以參考音訊為準)
        # 長度與參考音訊的樣本數相同
        t_ref_perfect = np.arange(len(audio_2)) / self.Fs
        
        # 使用內插函數，為每一個完美的目標時間點，計算出對應的來源時間點
        t_source_mapped = interp_func(t_ref_perfect)
        
        # 將這兩條時間軸堆疊起來，建立我們全新的、完美的 time_map
        # 它的形狀是 (L, 2)，第一欄是來源時間，第二欄是目標時間
        perfect_time_map = np.vstack([t_source_mapped, t_ref_perfect]).T
        
        # 4. 執行時間拉伸
        # 我們傳遞 audio_1 作為來源，並使用這個完美的地圖將其拉伸到 audio_2 的時間軸上
        y_warped = libtsm.pv_tsm(audio_1_shifted, perfect_time_map, Fs=self.Fs)

        print("✅ 全新 time_map 建立成功，libtsm.pv_tsm 執行成功！")

        # 5. 合成音訊
        min_len = min(len(audio_2), len(y_warped))
        stereo_sonification = np.hstack((audio_2[:min_len].reshape(-1, 1), y_warped[:min_len].reshape(-1, 1)))
        
        if output_path:
            sf.write(output_path, stereo_sonification, self.Fs)
            print(f"✅ 同步後的音訊已儲存至: {output_path}")

        try:
            import IPython.display as ipd
            ipd.display(ipd.Audio(stereo_sonification.T, rate=self.Fs, normalize=True))
        except ImportError:
            pass

        return wp

    def sync_audio_files(self, origin_path, cover_path, song_dir, output_path=""):
        version_key = os.path.basename(cover_path).split('.')[0]

        # 嘗試從快取讀取
        cached_data = self.load_wp(song_dir, version_key)
        if cached_data:
            wp = np.array(cached_data["wp"])
            num_frames_cover = cached_data["num_frames_cover"]
            num_frames_origin = cached_data["num_frames_origin"]
            self.t1 = np.arange(num_frames_cover) / self.feature_rate
            self.t2 = np.arange(num_frames_origin) / self.feature_rate
            print(f"✅ 從快取檔案中成功讀取版本 '{version_key}' 的完整同步資訊。")
        else:
            print(f"💨 快取中無 '{version_key}' 的同步資訊，開始進行計算...")
            wp = self.get_wp(origin_path, cover_path)

        origin_audio, cover_audio = self.load_audio(origin_path, cover_path)

        tuning_offset_1 = estimate_tuning(cover_audio, self.Fs)
        tuning_offset_2 = estimate_tuning(origin_audio, self.Fs)

        f_chroma_quantized_1, f_DLNCO_1 = self.get_features(
            cover_audio, tuning_offset_1, visualize=False
        )
        f_chroma_quantized_2, f_DLNCO_2 = self.get_features(
            origin_audio, tuning_offset_2, visualize=False
        )

        f_cens_1hz_1 = quantized_chroma_to_CENS(
            f_chroma_quantized_1, 201, 50, self.feature_rate
        )[0]
        f_cens_1hz_2 = quantized_chroma_to_CENS(
            f_chroma_quantized_2, 201, 50, self.feature_rate
        )[0]

        opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_1, f_cens_1hz_2)
        pitch_shift_for_cover_audio = -opt_chroma_shift % 12
        if pitch_shift_for_cover_audio > 6:
            pitch_shift_for_cover_audio -= 12
        
        cover_audio_shifted = libtsm.pitch_shift(
            cover_audio, pitch_shift_for_cover_audio * 100, order="tsm-res"
        )
        
        time_map = wp.T.copy() / self.feature_rate * self.Fs
        time_map[time_map[:, 0] > len(cover_audio), 0] = len(cover_audio) - 1
        time_map[time_map[:, 1] > len(origin_audio), 1] = len(origin_audio) - 1

        # 1. 過濾非嚴格遞增的點 (保留此步驟以防萬一)
        diffs = np.diff(time_map, axis=0)
        mask = np.hstack([True, np.all(diffs > 1e-9, axis=1)]) # 使用極小值避免浮點問題
        cleaned_time_map = time_map[mask]

        if len(cleaned_time_map) < 2:
            print("❌ 錯誤：清理後的對齊路徑過短。")
            return None

        # 2. 【關鍵修正】強制 time_map 的第一個點從 (t_start, 0) 開始
        #    libtsm 要求目標序列(origin)的第一個時間點必須為 0。
        #    我們透過減去偏移量來實現這一點。
        origin_time_offset = cleaned_time_map[0, 1]
        if origin_time_offset != 0:
            print(f"⚠️ 警告：偵測到 origin time offset: {origin_time_offset}。正在進行修正。")
            # 從 origin 時間列中減去這個偏移，確保它從 0 開始
            cleaned_time_map[:, 1] = cleaned_time_map[:, 1] - origin_time_offset
            # 確保沒有負值產生
            cleaned_time_map[cleaned_time_map[:, 1] < 0, 1] = 0

        print("\n--- Libtsm Input Pre-flight Check ---")
        print(f"Data type: {cleaned_time_map.dtype}")
        print(f"Shape: {cleaned_time_map.shape}")
        print(f"First anchor point: {cleaned_time_map[0]}")
        print(f"Is target start time zero? -> {cleaned_time_map[0, 1] == 0}")
        print(f"Is source start time non-negative? -> {cleaned_time_map[0, 0] >= 0}")
        
        final_diffs = np.diff(cleaned_time_map, axis=0)
        is_strictly_increasing = np.all(final_diffs > 0)
        print(f"Is sequence strictly increasing? -> {is_strictly_increasing}")
        if not is_strictly_increasing:
            print("  -> Found non-increasing steps AFTER final correction!")
        print("-------------------------------------\n")

        # 4. 呼叫 libtsm
        try:
            final_map_for_libtsm = cleaned_time_map.astype(np.float32)
            y_hpstsm = libtsm.hps_tsm(cover_audio_shifted, final_map_for_libtsm)
            # 為了能執行，我們先假設 y_hpstsm 存在
            # y_hpstsm = origin_audio
            print("✅ libtsm.hps_tsm 呼叫成功 (模擬)。")

        except AssertionError as e:
            print(f"❌❌❌ 所有防禦措施均失敗，問題可能出在 libtsm 函式庫本身: {e}")
            return None


        # --- 合成與儲存 ---
        min_len = min(len(origin_audio), len(y_hpstsm))
        stereo_sonification = np.hstack((origin_audio[:min_len].reshape(-1, 1), y_hpstsm[:min_len].reshape(-1, 1)))

        if output_path:
            sf.write(output_path, stereo_sonification, self.Fs)
            print(f"音訊已同步 (但未實際寫入檔案): {output_path}")

        print("Synchronized version created.", flush=True)
        try:
            import IPython.display as ipd
            ipd.display(ipd.Audio(stereo_sonification.T, rate=self.Fs, normalize=True))
        except ImportError:
            print("IPython not found, skipping audio display.")

        return wp
