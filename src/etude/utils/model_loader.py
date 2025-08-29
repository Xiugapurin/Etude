# src/etude/utils/model_loader.py

import torch
from collections import OrderedDict
from pathlib import Path
from typing import Union, Optional

from ..models.etude_decoder import EtudeDecoder, EtudeDecoderConfig

def load_etude_decoder(
    config_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "auto"
) -> EtudeDecoder:
    """
    Initializes an EtudeDecoder model and loads a checkpoint.
    This version handles the '_orig_mod.' prefix and uses strict loading.
    """
    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"           > Loading model configuration from: {config_path}")
    config = EtudeDecoderConfig.from_json_file(str(config_path))
    
    model = EtudeDecoder(config)
    print(f"           > Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    if checkpoint_path:
        print(f"           > Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Extract the model's state dict if it's in a payload
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # Clean the '_orig_mod.' prefix from all keys
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            cleaned_state_dict[k.replace('_orig_mod.', '')] = v
        
        # Load with strict=True to ensure a perfect match
        model.load_state_dict(cleaned_state_dict, strict=True)
        print("           > Checkpoint loaded successfully (Strict Mode).")
            
    model.to(device)
    model.eval()
    return model