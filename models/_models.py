"""
Source: https://github.com/misya11p/amt-apc  
This code was originally taken from the above GitHub repository.
Original file: /models/_models.py

Changes: style vector (sv) have been disabled as per project requirements; rewrote wav2midi to output JSON and MIDI (optional).
"""

from pathlib import Path
import sys
from collections import OrderedDict
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from .hFT_Transformer.amt import AMT
from .hFT_Transformer.model_spec2midi import (
    Encoder_SPEC2MIDI as Encoder,
    Decoder_SPEC2MIDI as Decoder,
    Model_SPEC2MIDI as BaseSpec2MIDI,
)
from utils import config


DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Array = List | Tuple | np.ndarray | torch.Tensor


class Pipeline(AMT):
    def __init__(
        self,
        path_model: str | None = None,
        device: torch.device = DEVICE_DEFAULT,
        amt: bool = False,
        with_sv: bool = False, # Modify from the original (defaults to True)
        no_load: bool = False,
        no_model: bool = False,
    ):
        """
        Pipeline for converting audio to MIDI. Contains some methods for
        converting audio to MIDI, models, and configurations.

        Args:
            path_model (str, optional):
                Path to the model. If None, use the default model
                (CONFIG.PATH.AMT or CONFIG.PATH.APC). Defaults to None.
            device (torch.device, optional):
                Device to use for the model. Defaults to auto (CUDA if
                available else CPU).
            amt (bool, optional):
                Whether to use the AMT model. Defaults to False (use the
                APC model).
            with_sv (bool, optional):
                Whether to use the style vector. Defaults to False.
            no_load (bool, optional):
                Do not load the model. Defaults to False.
            no_model (bool, optional):
                Do not own the model. Defaults to False.
        """
        self.device = device
        self.with_sv = with_sv
        if no_model:
            self.model = None
        else:
            self.model = load_model(
                device=self.device,
                amt=amt,
                path_model=path_model,
                with_sv=with_sv,
                no_load=no_load,
            )
        self.config = config.data

    # This function has been rewritten to add JSON output functionality.
    def wav2midi(
        self,
        path_input: str,
        path_output_json: str,
        path_output_midi: str = ""
    ):
        """
        Convert audio to JSON and/or MIDI.

        Args:
            path_input (str): Path to the input audio file.
            path_output_json (str): Path to the output JSON file (notes).
            path_output_midi (str, optional): Path to the output MIDI file. If empty, MIDI is not written.
        """
        # if sv is not None:
        #     sv = torch.tensor(sv)
        #     if sv.dim() == 1:
        #         sv = sv.unsqueeze(0)
        #     if sv.dim() == 2:
        #         pass
        #     else:
        #         raise ValueError(f"Invalid shape of style vector: {sv.shape}")
        #     sv = sv.to(self.device).to(torch.float32)

        feature = self.wav2feature(path_input)
        _, _, _, _, onset, offset, frame, velocity = self.transcript(feature)

        notes = self.mpe2note(
            onset, offset, frame, velocity,
            thred_onset=config.infer.threshold.onset,
            thred_offset=config.infer.threshold.offset,
            thred_mpe=config.infer.threshold.frame,
        )

        self.note2json(notes, path_output_json, min_length=config.infer.min_duration)

        if path_output_midi:
            self.note2midi(notes, path_output_midi, min_length=config.infer.min_duration)


class Spec2MIDI(BaseSpec2MIDI):
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


def load_model(
    path_model: str | None = None,
    device: torch.device = DEVICE_DEFAULT,
    amt: bool = False,
    with_sv: bool = False, # Modify from the original (defaults to True)
    no_load: bool = False,
) -> Spec2MIDI:
    """
    Load the model.

    Args:
        path_model (str, optional):
            Path to the model. If None, use the default model
            (CONFIG.PATH.AMT or CONFIG.PATH.APC). Defaults to None.
        device (torch.device, optional):
            Device to use for the model. Defaults to auto (CUDA if
            available else CPU).
        amt (bool, optional):
            Whether to use the AMT model. Defaults to False (use the
            APC model).
        with_sv (bool, optional):
            Whether to use the style vector. Defaults to False.
        no_load (bool, optional):
            Do not load the model. Defaults to False.
    Returns:
        Spec2MIDI: The model.
    """
    if amt:
        path_model = path_model or str(ROOT / config.path.amt)
    else:
        path_model = path_model or str(ROOT / config.path.apc)

    encoder = Encoder(
        n_margin=config.data.input.margin_b,
        n_frame=config.data.input.num_frame,
        n_bin=config.data.feature.n_bins,
        cnn_channel=config.model.cnn.channel,
        cnn_kernel=config.model.cnn.kernel,
        hid_dim=config.model.transformer.hid_dim,
        n_layers=config.model.transformer.encoder.n_layer,
        n_heads=config.model.transformer.encoder.n_head,
        pf_dim=config.model.transformer.pf_dim,
        dropout=config.model.dropout,
        device=device,
    )
    decoder = Decoder(
        n_frame=config.data.input.num_frame,
        n_bin=config.data.feature.n_bins,
        n_note=config.data.midi.num_note,
        n_velocity=config.data.midi.num_velocity,
        hid_dim=config.model.transformer.hid_dim,
        n_layers=config.model.transformer.decoder.n_layer,
        n_heads=config.model.transformer.decoder.n_head,
        pf_dim=config.model.transformer.pf_dim,
        dropout=config.model.dropout,
        device=device,
    )
    sv_dim = config.model.sv_dim if with_sv else 0
    model = Spec2MIDI(encoder, decoder, sv_dim=sv_dim)
    if not no_load:
        state_dict = torch.load(
            path_model,
            weights_only=True,
            map_location=device
        )
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def save_model(model: nn.Module, path: str) -> None:
    """
    Save the model.

    Args:
        model (nn.Module): Model to save.
        path (str): Path to save the model.
    """
    state_dict = model.state_dict()
    correct_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.", "")
            # If the model has been compiled with `torch.compile()`,
            # "_orig_mod." is appended to the key
        key = key.replace("module.", "")
            # If the model is saved with `torch.nn.DataParallel()`,
            # "module." is appended to the key
        correct_state_dict[key] = value
    torch.save(correct_state_dict, path)
