# etude/transcription/hft_transformer.py

import io
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from ..models import amt_apc

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
        if module.startswith('model'):
            new_module = 'etude.models.amt_apc'
            return getattr(sys.modules[new_module], name)
        return super().find_class(module, name)

class HFT_Transcriber:
    """
    A fully integrated transcriber based on the hFT-Transformer pipeline.
    """
    def __init__(self, config: Dict, model_path: str, device: str = 'auto', verbose: bool = False):
        self.verbose = verbose

        if device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.config = config
        
        if self.verbose:
            print(f"    > Loading hFT-Transformer model from: {model_path}")

        with open(model_path, "rb") as f:
            self.model = CustomUnpickler(f).load()
        
        self.model = self.model.to(self.device)
        self.model.eval()

        try:
            encoder = self.model.encoder_spec2midi
            decoder = self.model.decoder_spec2midi

            if hasattr(encoder, 'scale_freq'):
                encoder.scale_freq = encoder.scale_freq.to(self.device)
            if hasattr(decoder, 'scale_time'):
                decoder.scale_time = decoder.scale_time.to(self.device)

            for module in self.model.modules():
                if isinstance(module, amt_apc.MultiHeadAttentionLayer):
                    if hasattr(module, 'scale'):
                        module.scale = module.scale.to(self.device)
        except AttributeError as e:
            print(f"    > [WARN] Could not perform manual device fix for sub-tensors: {e}")

        if self.verbose:
            print(f"    > hFT-Transformer model loaded successfully on device: {self.device}")

    def transcribe(
        self,
        input_wav_path: Union[str, Path],
        output_json_path: Union[str, Path],
        inference_params: Dict
    ):
        """
        Performs the complete transcription of a single audio file.

        Args:
            input_wav_path (Union[str, Path]): Path to the input .wav file.
            output_json_path (Union[str, Path]): Path to save the output note list as a .json file.
            inference_params (Dict): A dictionary with inference settings like thresholds and stride.
        """
        feature = self._wav2feature(input_wav_path)

        n_stride = inference_params.get('n_stride', 0)
        mode = inference_params.get('mode', 'combination')
        
        if n_stride > 0:
            predictions = self._transcript_stride(feature, n_stride, mode=mode)
        else:
            predictions = self._transcript(feature, mode=mode)

        if mode == 'combination':
            onset, offset, mpe, velocity = predictions[4], predictions[5], predictions[6], predictions[7]
        else:
            onset, offset, mpe, velocity = predictions[0], predictions[1], predictions[2], predictions[3]
            
        notes = self._mpe2note(
            a_onset=onset,
            a_offset=offset,
            a_mpe=mpe,
            a_velocity=velocity,
            thred_onset=inference_params['thred_onset'],
            thred_offset=inference_params['thred_offset'],
            thred_mpe=inference_params['thred_mpe'],
        )

        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=4)


    def _wav2feature(self, f_wav: Union[str, Path]) -> torch.Tensor:
        """Loads an audio file and converts it into a log-Mel spectrogram."""
        wave, sr = torchaudio.load(f_wav)
        wave_mono = torch.mean(wave, dim=0)
        tr_fsconv = torchaudio.transforms.Resample(sr, self.config["feature"]["sr"])
        wave_mono_16k = tr_fsconv(wave_mono)
        tr_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config["feature"]["sr"],
            n_fft=self.config["feature"]["fft_bins"],
            win_length=self.config["feature"]["window_length"],
            hop_length=self.config["feature"]["hop_sample"],
            pad_mode=self.config["feature"]["pad_mode"],
            n_mels=self.config["feature"]["mel_bins"],
            norm="slaney",
        )
        mel_spec = tr_mel(wave_mono_16k)
        a_feature = (torch.log(mel_spec + self.config["feature"]["log_offset"])).T

        return a_feature

    def _transcript(self, a_feature: torch.Tensor, mode="combination") -> tuple:
        """Processes the feature through the model in non-overlapping segments."""
        a_feature = np.array(a_feature, dtype=np.float32)

        a_tmp_b = np.full(
            [self.config["input"]["margin_b"], self.config["feature"]["n_bins"]],
            self.config["input"]["min_value"],
            dtype=np.float32,
        )
        len_s = (
            int(
                np.ceil(a_feature.shape[0] / self.config["input"]["num_frame"])
                * self.config["input"]["num_frame"]
            )
            - a_feature.shape[0]
        )
        a_tmp_f = np.full(
            [
                len_s + self.config["input"]["margin_f"],
                self.config["feature"]["n_bins"],
            ],
            self.config["input"]["min_value"],
            dtype=np.float32,
        )
        a_input = torch.from_numpy(
            np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0)
        )
        # a_input: [margin_b+a_feature.shape[0]+len_s+margin_f, n_bins]

        a_output_onset_A = np.zeros(
            (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
            dtype=np.float32,
        )
        a_output_offset_A = np.zeros(
            (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
            dtype=np.float32,
        )
        a_output_mpe_A = np.zeros(
            (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
            dtype=np.float32,
        )
        a_output_velocity_A = np.zeros(
            (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]), dtype=np.int8
        )

        if mode == "combination":
            a_output_onset_B = np.zeros(
                (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
                dtype=np.float32,
            )
            a_output_offset_B = np.zeros(
                (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
                dtype=np.float32,
            )
            a_output_mpe_B = np.zeros(
                (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
                dtype=np.float32,
            )
            a_output_velocity_B = np.zeros(
                (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
                dtype=np.int8,
            )

        self.model.eval()
        for i in range(0, a_feature.shape[0], self.config["input"]["num_frame"]):
            input_spec = (
                (
                    a_input[
                        i : i
                        + self.config["input"]["margin_b"]
                        + self.config["input"]["num_frame"]
                        + self.config["input"]["margin_f"]
                    ]
                )
                .T.unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                if mode == "combination":
                    (
                        output_onset_A,
                        output_offset_A,
                        output_mpe_A,
                        output_velocity_A,
                        attention,
                        output_onset_B,
                        output_offset_B,
                        output_mpe_B,
                        output_velocity_B,
                    ) = self.model(input_spec)
                else:
                    output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = (
                        self.model(input_spec)
                    )

            a_output_onset_A[i : i + self.config["input"]["num_frame"]] = (
                (output_onset_A.squeeze(0)).to("cpu").detach().numpy()
            )
            a_output_offset_A[i : i + self.config["input"]["num_frame"]] = (
                (output_offset_A.squeeze(0)).to("cpu").detach().numpy()
            )
            a_output_mpe_A[i : i + self.config["input"]["num_frame"]] = (
                (output_mpe_A.squeeze(0)).to("cpu").detach().numpy()
            )
            a_output_velocity_A[i : i + self.config["input"]["num_frame"]] = (
                (output_velocity_A.squeeze(0).argmax(2)).to("cpu").detach().numpy()
            )

            if mode == "combination":
                a_output_onset_B[i : i + self.config["input"]["num_frame"]] = (
                    (output_onset_B.squeeze(0)).to("cpu").detach().numpy()
                )
                a_output_offset_B[i : i + self.config["input"]["num_frame"]] = (
                    (output_offset_B.squeeze(0)).to("cpu").detach().numpy()
                )
                a_output_mpe_B[i : i + self.config["input"]["num_frame"]] = (
                    (output_mpe_B.squeeze(0)).to("cpu").detach().numpy()
                )
                a_output_velocity_B[i : i + self.config["input"]["num_frame"]] = (
                    (output_velocity_B.squeeze(0).argmax(2)).to("cpu").detach().numpy()
                )

        if mode == "combination":
            return (
                a_output_onset_A,
                a_output_offset_A,
                a_output_mpe_A,
                a_output_velocity_A,
                a_output_onset_B,
                a_output_offset_B,
                a_output_mpe_B,
                a_output_velocity_B,
            )
        else:
            return (
                a_output_onset_A,
                a_output_offset_A,
                a_output_mpe_A,
                a_output_velocity_A,
            )

    def _transcript_stride(self, a_feature: torch.Tensor, n_offset: int, mode="combination") -> tuple:
        """Processes the feature through the model in overlapping segments (stride)."""
        # a_feature: [num_frame, n_mels]
        a_feature = np.array(a_feature, dtype=np.float32)

        half_frame = int(self.config["input"]["num_frame"] / 2)
        a_tmp_b = np.full(
            [
                self.config["input"]["margin_b"] + n_offset,
                self.config["feature"]["n_bins"],
            ],
            self.config["input"]["min_value"],
            dtype=np.float32,
        )
        tmp_len = (
            a_feature.shape[0]
            + self.config["input"]["margin_b"]
            + self.config["input"]["margin_f"]
            + half_frame
        )
        len_s = int(np.ceil(tmp_len / half_frame) * half_frame) - tmp_len
        a_tmp_f = np.full(
            [
                len_s + self.config["input"]["margin_f"] + (half_frame - n_offset),
                self.config["feature"]["n_bins"],
            ],
            self.config["input"]["min_value"],
            dtype=np.float32,
        )

        a_input = torch.from_numpy(
            np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0)
        )
        # a_input: [n_offset+margin_b+a_feature.shape[0]+len_s+(half_frame-n_offset)+margin_f, n_bins]

        a_output_onset_A = np.zeros(
            (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
            dtype=np.float32,
        )
        a_output_offset_A = np.zeros(
            (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
            dtype=np.float32,
        )
        a_output_mpe_A = np.zeros(
            (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
            dtype=np.float32,
        )
        a_output_velocity_A = np.zeros(
            (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]), dtype=np.int8
        )

        if mode == "combination":
            a_output_onset_B = np.zeros(
                (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
                dtype=np.float32,
            )
            a_output_offset_B = np.zeros(
                (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
                dtype=np.float32,
            )
            a_output_mpe_B = np.zeros(
                (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
                dtype=np.float32,
            )
            a_output_velocity_B = np.zeros(
                (a_feature.shape[0] + len_s, self.config["midi"]["num_note"]),
                dtype=np.int8,
            )

        self.model.eval()
        for i in range(0, a_feature.shape[0], half_frame):
            input_spec = (
                (
                    a_input[
                        i : i
                        + self.config["input"]["margin_b"]
                        + self.config["input"]["num_frame"]
                        + self.config["input"]["margin_f"]
                    ]
                )
                .T.unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                if mode == "combination":
                    (
                        output_onset_A,
                        output_offset_A,
                        output_mpe_A,
                        output_velocity_A,
                        attention,
                        output_onset_B,
                        output_offset_B,
                        output_mpe_B,
                        output_velocity_B,
                    ) = self.model(input_spec)
                else:
                    output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = (
                        self.model(input_spec)
                    )

            a_output_onset_A[i : i + half_frame] = (
                (output_onset_A.squeeze(0)[n_offset : n_offset + half_frame])
                .to("cpu")
                .detach()
                .numpy()
            )
            a_output_offset_A[i : i + half_frame] = (
                (output_offset_A.squeeze(0)[n_offset : n_offset + half_frame])
                .to("cpu")
                .detach()
                .numpy()
            )
            a_output_mpe_A[i : i + half_frame] = (
                (output_mpe_A.squeeze(0)[n_offset : n_offset + half_frame])
                .to("cpu")
                .detach()
                .numpy()
            )
            a_output_velocity_A[i : i + half_frame] = (
                (
                    output_velocity_A.squeeze(0)[
                        n_offset : n_offset + half_frame
                    ].argmax(2)
                )
                .to("cpu")
                .detach()
                .numpy()
            )

            if mode == "combination":
                a_output_onset_B[i : i + half_frame] = (
                    (output_onset_B.squeeze(0)[n_offset : n_offset + half_frame])
                    .to("cpu")
                    .detach()
                    .numpy()
                )
                a_output_offset_B[i : i + half_frame] = (
                    (output_offset_B.squeeze(0)[n_offset : n_offset + half_frame])
                    .to("cpu")
                    .detach()
                    .numpy()
                )
                a_output_mpe_B[i : i + half_frame] = (
                    (output_mpe_B.squeeze(0)[n_offset : n_offset + half_frame])
                    .to("cpu")
                    .detach()
                    .numpy()
                )
                a_output_velocity_B[i : i + half_frame] = (
                    (
                        output_velocity_B.squeeze(0)[
                            n_offset : n_offset + half_frame
                        ].argmax(2)
                    )
                    .to("cpu")
                    .detach()
                    .numpy()
                )

        if mode == "combination":
            return (
                a_output_onset_A,
                a_output_offset_A,
                a_output_mpe_A,
                a_output_velocity_A,
                a_output_onset_B,
                a_output_offset_B,
                a_output_mpe_B,
                a_output_velocity_B,
            )
        else:
            return (
                a_output_onset_A,
                a_output_offset_A,
                a_output_mpe_A,
                a_output_velocity_A,
            )

    def _mpe2note(
        self,
        a_onset=None,
        a_offset=None,
        a_mpe=None,
        a_velocity=None,
        thred_onset=0.5,
        thred_offset=0.5,
        thred_mpe=0.5,
        mode_velocity="ignore_zero",
        mode_offset="shorter",
    ):
        a_note = []
        hop_sec = float(
            self.config["feature"]["hop_sample"] / self.config["feature"]["sr"]
        )

        for j in range(self.config["midi"]["num_note"]):
            # find local maximum
            a_onset_detect = []
            for i in range(len(a_onset)):
                if a_onset[i][j] >= thred_onset:
                    left_flag = True
                    for ii in range(i - 1, -1, -1):
                        if a_onset[i][j] > a_onset[ii][j]:
                            left_flag = True
                            break
                        elif a_onset[i][j] < a_onset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i + 1, len(a_onset)):
                        if a_onset[i][j] > a_onset[ii][j]:
                            right_flag = True
                            break
                        elif a_onset[i][j] < a_onset[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):
                        if (i == 0) or (i == len(a_onset) - 1):
                            onset_time = i * hop_sec
                        else:
                            if a_onset[i - 1][j] == a_onset[i + 1][j]:
                                onset_time = i * hop_sec
                            elif a_onset[i - 1][j] > a_onset[i + 1][j]:
                                onset_time = i * hop_sec - (
                                    hop_sec
                                    * 0.5
                                    * (a_onset[i - 1][j] - a_onset[i + 1][j])
                                    / (a_onset[i][j] - a_onset[i + 1][j])
                                )
                            else:
                                onset_time = i * hop_sec + (
                                    hop_sec
                                    * 0.5
                                    * (a_onset[i + 1][j] - a_onset[i - 1][j])
                                    / (a_onset[i][j] - a_onset[i - 1][j])
                                )
                        a_onset_detect.append({"loc": i, "onset_time": onset_time})
            a_offset_detect = []
            for i in range(len(a_offset)):
                if a_offset[i][j] >= thred_offset:
                    left_flag = True
                    for ii in range(i - 1, -1, -1):
                        if a_offset[i][j] > a_offset[ii][j]:
                            left_flag = True
                            break
                        elif a_offset[i][j] < a_offset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i + 1, len(a_offset)):
                        if a_offset[i][j] > a_offset[ii][j]:
                            right_flag = True
                            break
                        elif a_offset[i][j] < a_offset[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):
                        if (i == 0) or (i == len(a_offset) - 1):
                            offset_time = i * hop_sec
                        else:
                            if a_offset[i - 1][j] == a_offset[i + 1][j]:
                                offset_time = i * hop_sec
                            elif a_offset[i - 1][j] > a_offset[i + 1][j]:
                                offset_time = i * hop_sec - (
                                    hop_sec
                                    * 0.5
                                    * (a_offset[i - 1][j] - a_offset[i + 1][j])
                                    / (a_offset[i][j] - a_offset[i + 1][j])
                                )
                            else:
                                offset_time = i * hop_sec + (
                                    hop_sec
                                    * 0.5
                                    * (a_offset[i + 1][j] - a_offset[i - 1][j])
                                    / (a_offset[i][j] - a_offset[i - 1][j])
                                )
                        a_offset_detect.append({"loc": i, "offset_time": offset_time})

            time_next = 0.0
            time_offset = 0.0
            time_mpe = 0.0
            for idx_on in range(len(a_onset_detect)):
                # onset
                loc_onset = a_onset_detect[idx_on]["loc"]
                time_onset = a_onset_detect[idx_on]["onset_time"]

                if idx_on + 1 < len(a_onset_detect):
                    loc_next = a_onset_detect[idx_on + 1]["loc"]
                    # time_next = loc_next * hop_sec
                    time_next = a_onset_detect[idx_on + 1]["onset_time"]
                else:
                    loc_next = len(a_mpe)
                    time_next = (loc_next - 1) * hop_sec

                # offset
                loc_offset = loc_onset + 1
                flag_offset = False
                # time_offset = 0###
                for idx_off in range(len(a_offset_detect)):
                    if loc_onset < a_offset_detect[idx_off]["loc"]:
                        loc_offset = a_offset_detect[idx_off]["loc"]
                        time_offset = a_offset_detect[idx_off]["offset_time"]
                        flag_offset = True
                        break
                if loc_offset > loc_next:
                    loc_offset = loc_next
                    time_offset = time_next

                # offset by MPE
                # (1frame longer)
                loc_mpe = loc_onset + 1
                flag_mpe = False
                # time_mpe = 0###
                for ii_mpe in range(loc_onset + 1, loc_next):
                    if a_mpe[ii_mpe][j] < thred_mpe:
                        loc_mpe = ii_mpe
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                """
                # (right algorighm)
                loc_mpe = loc_onset
                flag_mpe = False
                for ii_mpe in range(loc_onset+1, loc_next+1):
                    if a_mpe[ii_mpe][j] < thred_mpe:
                        loc_mpe = ii_mpe-1
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                """
                pitch_value = int(j + self.config["midi"]["note_min"])
                velocity_value = int(a_velocity[loc_onset][j])

                if (flag_offset is False) and (flag_mpe is False):
                    offset_value = float(time_next)
                elif (flag_offset is True) and (flag_mpe is False):
                    offset_value = float(time_offset)
                elif (flag_offset is False) and (flag_mpe is True):
                    offset_value = float(time_mpe)
                else:
                    if mode_offset == "offset":
                        ## (a) offset
                        offset_value = float(time_offset)
                    elif mode_offset == "longer":
                        ## (b) longer
                        if loc_offset >= loc_mpe:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_mpe)
                    else:
                        ## (c) shorter
                        if loc_offset <= loc_mpe:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_mpe)
                if mode_velocity != "ignore_zero":
                    a_note.append(
                        {
                            "pitch": pitch_value,
                            "onset": float(time_onset),
                            "offset": offset_value,
                            "velocity": velocity_value,
                        }
                    )
                else:
                    if velocity_value > 0:
                        a_note.append(
                            {
                                "pitch": pitch_value,
                                "onset": float(time_onset),
                                "offset": offset_value,
                                "velocity": velocity_value,
                            }
                        )

                if (
                    (len(a_note) > 1)
                    and (
                        a_note[len(a_note) - 1]["pitch"]
                        == a_note[len(a_note) - 2]["pitch"]
                    )
                    and (
                        a_note[len(a_note) - 1]["onset"]
                        < a_note[len(a_note) - 2]["offset"]
                    )
                ):
                    a_note[len(a_note) - 2]["offset"] = a_note[len(a_note) - 1]["onset"]

        a_note = sorted(
            sorted(a_note, key=lambda x: x["pitch"]), key=lambda x: x["onset"]
        )
        return a_note