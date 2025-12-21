# etude/data/extractor.py

"""
Audio-to-Note Extraction Pipeline for the 'Extract' Stage.

This module provides the main class, `AMTAPC_Extractor`, which uses the
AMT-APC model architecture as an "extractor". Its primary function is to
convert a raw audio waveform into a sequence of piano cover MIDI notes.

Source: The logic is a refactored and merged version based on the AMT-APC project.
https://github.com/misya11p/amt-apc
Original files: /models/_models.py and /models/hFT_Transformer/amt.py
"""

from pathlib import Path
from typing import Union, Optional

import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import pretty_midi
from tqdm import tqdm

from ..config.schema import ExtractorConfig
from ..models.amt_apc import (
    Encoder_SPEC2MIDI as Encoder,
    Decoder_SPEC2MIDI as Decoder,
    Model_SPEC2MIDI as BaseSpec2MIDI,
)


class _Spec2MIDI(BaseSpec2MIDI):
    """A private wrapper around the base model to handle style vectors (currently disabled)."""
    def __init__(self, encoder, decoder, sv_dim: int = 0):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        delattr(self, "encoder_spec2midi")
        delattr(self, "decoder_spec2midi")
        self.sv_dim = sv_dim
        if sv_dim:
            hidden_size = encoder.hid_dim
            self.fc_sv = nn.Linear(sv_dim, hidden_size)
            self.gate_sv = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )

    def forward(self, x, sv=None):
        h = self.encode(x, sv)
        y = self.decode(h)
        return y

    def encode(self, x, sv=None):
        h = self.encoder(x)
        if self.sv_dim and (sv is not None):
            sv = self.fc_sv(sv)
            _, n_frames, n_bin, _ = h.shape
            sv = sv.unsqueeze(1).unsqueeze(2)
            sv = sv.repeat(1, n_frames, n_bin, 1)
            z = self.gate_sv(h)
            h = z * h + (1 - z) * sv
        return h

    def decode(self, h):
        onset_f, offset_f, mpe_f, velocity_f, attention, \
        onset_t, offset_t, mpe_t, velocity_t = self.decoder(h)
        return (
            onset_f, offset_f, mpe_f, velocity_f, attention,
            onset_t, offset_t, mpe_t, velocity_t
        )


def _load_model(config: ExtractorConfig, path_model: Union[str, Path], device: torch.device) -> _Spec2MIDI:
    """Helper function to construct and load the transcription model."""
    encoder = Encoder(
        n_margin=config.input.margin_b,
        n_frame=config.input.num_frame,
        n_bin=config.feature.n_bins,
        cnn_channel=config.model.cnn_channel,
        cnn_kernel=config.model.cnn_kernel,
        hid_dim=config.model.transformer_hid_dim,
        n_layers=config.model.encoder_n_layer,
        n_heads=config.model.encoder_n_head,
        pf_dim=config.model.transformer_pf_dim,
        dropout=config.model.dropout,
        device=device,
    )

    decoder = Decoder(
        n_frame=config.input.num_frame,
        n_bin=config.feature.n_bins,
        n_note=config.midi.num_note,
        n_velocity=config.midi.num_velocity,
        hid_dim=config.model.transformer_hid_dim,
        n_layers=config.model.decoder_n_layer,
        n_heads=config.model.decoder_n_head,
        pf_dim=config.model.transformer_pf_dim,
        dropout=config.model.dropout,
        device=device,
    )

    model = _Spec2MIDI(encoder, decoder, sv_dim=0)  # sv disabled
    state_dict = torch.load(path_model, weights_only=True, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


class AMTAPC_Extractor:
    """
    A pipeline for converting audio files into musical note representations (JSON/MIDI).
    """

    def __init__(
        self,
        config: ExtractorConfig,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "auto",
    ):
        """
        Initializes the extractor.

        Args:
            config: ExtractorConfig containing all extraction parameters.
            model_path: Path to the pre-trained model checkpoint.
            device: The device to run on ('cuda', 'mps', 'cpu', or 'auto').
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.config = config
        self.model = _load_model(self.config, model_path, self.device)

    def extract(self, audio_path: str, output_json_path: str, output_midi_path: Optional[str] = None):
        """
        Performs audio-to-note transcription.

        Args:
            audio_path (str): Path to the input audio file.
            output_json_path (str): Path to save the output JSON file.
            output_midi_path (Optional[str]): If provided, saves the output MIDI file to this path.
        """
        # Step 1: Convert audio waveform to feature representation
        feature = self._wav2feature(audio_path)
        
        # Step 2: Run the model to get frame-wise predictions
        _, _, _, _, onset, offset, frame, velocity = self._transcript(feature)
        
        # Step 3: Convert frame-wise predictions to a list of discrete notes
        notes = self._mpe2note(
            onset, offset, frame, velocity,
            thred_onset=self.config.infer.onset_threshold,
            thred_offset=self.config.infer.offset_threshold,
            thred_mpe=self.config.infer.frame_threshold,
        )

        # Step 4: Save notes to JSON and optionally to MIDI
        min_duration = self.config.infer.min_duration
        self._note2json(notes, output_json_path, min_duration)
        
        if output_midi_path:
            self._note2midi(notes, output_midi_path, min_duration)

    def _wav2feature(self, audio_path: str) -> torch.Tensor:
        """Loads an audio file and converts it into a log-Mel spectrogram."""
        wave, sr = torchaudio.load(audio_path)
        wave_mono = torch.mean(wave, dim=0)

        resampler = torchaudio.transforms.Resample(sr, self.config.feature.sr)
        wave_resampled = resampler(wave_mono)

        mel_transformer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.feature.sr,
            n_fft=self.config.feature.fft_bins,
            win_length=self.config.feature.window_length,
            hop_length=self.config.feature.hop_sample,
            n_mels=self.config.feature.mel_bins,
            norm="slaney",
        )
        mel_spec = mel_transformer(wave_resampled)

        log_mel_spec = torch.log(mel_spec + self.config.feature.log_offset)
        return log_mel_spec.T

    def _transcript(self, a_feature, sv=None, silent=True, mode="combination", ablation_flag=False):
        # a_feature: [num_frame, n_mels]
        a_feature = np.array(a_feature, dtype=np.float32)

        margin_b = self.config.input.margin_b
        margin_f = self.config.input.margin_f
        num_frame = self.config.input.num_frame
        n_bins = self.config.feature.n_bins
        min_value = self.config.input.min_value
        num_note = self.config.midi.num_note

        a_tmp_b = np.full([margin_b, n_bins], min_value, dtype=np.float32)
        len_s = int(np.ceil(a_feature.shape[0] / num_frame) * num_frame) - a_feature.shape[0]
        a_tmp_f = np.full([len_s + margin_f, n_bins], min_value, dtype=np.float32)
        a_input = torch.from_numpy(np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0))

        a_output_onset_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
        a_output_offset_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
        a_output_mpe_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
        a_output_velocity_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.int8)

        if mode == "combination":
            a_output_onset_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
            a_output_offset_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
            a_output_mpe_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
            a_output_velocity_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.int8)

        self.model.eval()
        for i in tqdm(range(0, a_feature.shape[0], num_frame), desc="Processing each segment", disable=silent):
            input_spec = (a_input[i : i + margin_b + num_frame + margin_f]).T.unsqueeze(0).to(self.device)

            with torch.no_grad():
                if mode == "combination":
                    if ablation_flag is True:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec, sv)
                    else:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec, sv)
                else:
                    output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = self.model(input_spec)

            a_output_onset_A[i : i + num_frame] = output_onset_A.squeeze(0).cpu().detach().numpy()
            a_output_offset_A[i : i + num_frame] = output_offset_A.squeeze(0).cpu().detach().numpy()
            a_output_mpe_A[i : i + num_frame] = output_mpe_A.squeeze(0).cpu().detach().numpy()
            a_output_velocity_A[i : i + num_frame] = output_velocity_A.squeeze(0).argmax(2).cpu().detach().numpy()

            if mode == "combination":
                a_output_onset_B[i : i + num_frame] = output_onset_B.squeeze(0).cpu().detach().numpy()
                a_output_offset_B[i : i + num_frame] = output_offset_B.squeeze(0).cpu().detach().numpy()
                a_output_mpe_B[i : i + num_frame] = output_mpe_B.squeeze(0).cpu().detach().numpy()
                a_output_velocity_B[i : i + num_frame] = output_velocity_B.squeeze(0).argmax(2).cpu().detach().numpy()

        if mode == "combination":
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A, a_output_onset_B, a_output_offset_B, a_output_mpe_B, a_output_velocity_B
        else:
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A


    def _mpe2note(self, a_onset=None, a_offset=None, a_mpe=None, a_velocity=None, thred_onset=0.5, thred_offset=0.5, thred_mpe=0.5, mode_velocity="ignore_zero", mode_offset="shorter"):
        # mode_velocity: 'org' (0-127) or 'ignore_zero' (exclude velocity 0)
        # mode_offset: 'shorter', 'longer', or 'offset'

        a_note = []
        hop_sec = float(self.config.feature.hop_sample / self.config.feature.sr)
        num_note = self.config.midi.num_note
        note_min = self.config.midi.note_min

        for j in range(num_note):
            # find local maximum
            a_onset_detect = []
            for i in range(len(a_onset)):
                if a_onset[i][j] >= thred_onset:
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if a_onset[i][j] > a_onset[ii][j]:
                            left_flag = True
                            break
                        elif a_onset[i][j] < a_onset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(a_onset)):
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
                            if a_onset[i-1][j] == a_onset[i+1][j]:
                                onset_time = i * hop_sec
                            elif a_onset[i-1][j] > a_onset[i+1][j]:
                                onset_time = (i * hop_sec - (hop_sec * 0.5 * (a_onset[i-1][j] - a_onset[i+1][j]) / (a_onset[i][j] - a_onset[i+1][j])))
                            else:
                                onset_time = (i * hop_sec + (hop_sec * 0.5 * (a_onset[i+1][j] - a_onset[i-1][j]) / (a_onset[i][j] - a_onset[i-1][j])))
                        a_onset_detect.append({'loc': i, 'onset_time': onset_time})
            a_offset_detect = []
            for i in range(len(a_offset)):
                if a_offset[i][j] >= thred_offset:
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if a_offset[i][j] > a_offset[ii][j]:
                            left_flag = True
                            break
                        elif a_offset[i][j] < a_offset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(a_offset)):
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
                            if a_offset[i-1][j] == a_offset[i+1][j]:
                                offset_time = i * hop_sec
                            elif a_offset[i-1][j] > a_offset[i+1][j]:
                                offset_time = (i * hop_sec - (hop_sec * 0.5 * (a_offset[i-1][j] - a_offset[i+1][j]) / (a_offset[i][j] - a_offset[i+1][j])))
                            else:
                                offset_time = (i * hop_sec + (hop_sec * 0.5 * (a_offset[i+1][j] - a_offset[i-1][j]) / (a_offset[i][j] - a_offset[i-1][j])))
                        a_offset_detect.append({'loc': i, 'offset_time': offset_time})

            time_next = 0.0
            time_offset = 0.0
            time_mpe = 0.0
            for idx_on in range(len(a_onset_detect)):
                # onset
                loc_onset = a_onset_detect[idx_on]['loc']
                time_onset = a_onset_detect[idx_on]['onset_time']

                if idx_on + 1 < len(a_onset_detect):
                    loc_next = a_onset_detect[idx_on+1]['loc']
                    #time_next = loc_next * hop_sec
                    time_next = a_onset_detect[idx_on+1]['onset_time']
                else:
                    loc_next = len(a_mpe)
                    time_next = (loc_next-1) * hop_sec

                # offset
                loc_offset = loc_onset+1
                flag_offset = False
                #time_offset = 0###
                for idx_off in range(len(a_offset_detect)):
                    if loc_onset < a_offset_detect[idx_off]['loc']:
                        loc_offset = a_offset_detect[idx_off]['loc']
                        time_offset = a_offset_detect[idx_off]['offset_time']
                        flag_offset = True
                        break
                if loc_offset > loc_next:
                    loc_offset = loc_next
                    time_offset = time_next

                # offset by MPE
                # (1frame longer)
                loc_mpe = loc_onset+1
                flag_mpe = False
                #time_mpe = 0###
                for ii_mpe in range(loc_onset+1, loc_next):
                    if a_mpe[ii_mpe][j] < thred_mpe:
                        loc_mpe = ii_mpe
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                '''
                # (right algorighm)
                loc_mpe = loc_onset
                flag_mpe = False
                for ii_mpe in range(loc_onset+1, loc_next+1):
                    if a_mpe[ii_mpe][j] < thred_mpe:
                        loc_mpe = ii_mpe-1
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                '''
                pitch_value = int(j + note_min)
                velocity_value = int(a_velocity[loc_onset][j])

                if (flag_offset is False) and (flag_mpe is False):
                    offset_value = float(time_next)
                elif (flag_offset is True) and (flag_mpe is False):
                    offset_value = float(time_offset)
                elif (flag_offset is False) and (flag_mpe is True):
                    offset_value = float(time_mpe)
                else:
                    if mode_offset == 'offset':
                        ## (a) offset
                        offset_value = float(time_offset)
                    elif mode_offset == 'longer':
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
                if mode_velocity != 'ignore_zero':
                    a_note.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})
                else:
                    if velocity_value > 0:
                        a_note.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})

                if (len(a_note) > 1) and \
                   (a_note[len(a_note)-1]['pitch'] == a_note[len(a_note)-2]['pitch']) and \
                   (a_note[len(a_note)-1]['onset'] < a_note[len(a_note)-2]['offset']):
                    a_note[len(a_note)-2]['offset'] = a_note[len(a_note)-1]['onset']

        a_note = sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['onset'])
        
        return a_note


    def _note2midi(self, notes, path_output, min_length=0.):
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        for note in notes:
            if note['offset'] - note['onset'] < min_length:
                continue
            instrument.notes.append(pretty_midi.Note(velocity=note['velocity'], pitch=note['pitch'], start=note['onset'], end=note['offset']))
        midi.instruments.append(instrument)
        midi.write(path_output)
    
    
    def _note2json(self, notes, path_output, min_length=0.):
        filtered = []
        for note in notes:
            duration = note['offset'] - note['onset']
            if duration < min_length:
                continue
            filtered.append({
                'onset': note['onset'],
                'offset': note['offset'],
                'pitch': note['pitch'],
                'velocity': note['velocity']
            })

        with open(path_output, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)