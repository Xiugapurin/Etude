# etude/utils/model_loader.py

import torch
from collections import OrderedDict
from pathlib import Path
from typing import Union

from ..models.etude_decoder import EtudeDecoder, EtudeDecoderConfig
from .logger import logger


def load_etude_decoder(
    config_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    device: str = "auto",
) -> EtudeDecoder:
    """
    Initializes an EtudeDecoder model and loads a checkpoint.

    Args:
        config_path: Path to the model configuration JSON file.
        checkpoint_path: Path to the model checkpoint file.
        device: Device to load the model on ('auto', 'cuda', 'mps', 'cpu').

    Returns:
        Loaded and initialized EtudeDecoder model.
    """
    logger.step("Loading decoder model")

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.substep("Loading model configuration...")
    config = EtudeDecoderConfig.from_json_file(str(config_path))

    model = EtudeDecoder(config)

    logger.substep("Loading checkpoint weights...")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Extract the model's state dict if it's in a payload
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Clean the '_orig_mod.' prefix from all keys
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        cleaned_state_dict[k.replace("_orig_mod.", "")] = v

    # Load with strict=True to ensure a perfect match
    model.load_state_dict(cleaned_state_dict, strict=True)

    model.to(device)
    model.eval()
    return model
