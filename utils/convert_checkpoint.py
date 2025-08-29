# scripts/utils/convert_checkpoint.py

import argparse
from pathlib import Path
from collections import OrderedDict
import torch

def convert_etude_decoder_checkpoint(input_path: str, output_path: str):
    """
    Converts an old EtudeDecoder checkpoint to the new, refactored format by
    precisely remapping layer names. This is the definitive, corrected version.
    """
    print(f"Loading original checkpoint from: {input_path}")
    payload = torch.load(input_path, map_location='cpu')
    
    # Extract the actual model state dictionary
    state_dict = payload.get('model_state_dict', payload)

    new_state_dict = OrderedDict()
    
    # --- This is the definitive mapping from old names to new names ---
    RENAME_MAP = {
        'avg_note_overlap_embeddings': 'pitch_overlap_embeddings',
        'rel_note_per_pos_embeddings': 'polyphony_embeddings',
        'rel_avg_duration_embeddings': 'note_sustain_embeddings',
        'rel_pos_density_embeddings': 'rhythm_intensity_embeddings',
        'attribute_projection_layer': 'attribute_projection'
    }

    print("Starting key conversion...")
    for old_key, value in state_dict.items():
        # First, remove the '_orig_mod.' prefix
        key = old_key.replace('_orig_mod.', '')
        
        # Explicitly skip the redundant 'embed_in' layer
        # if 'transformer.embed_in.weight' in key:
        #     print(f"  - Skipping redundant key: {old_key}")
        #     continue

        # Assume the key will not be changed unless a match is found
        new_key = key
        
        # Check if any of the layer names need to be remapped
        for old_name, new_name in RENAME_MAP.items():
            if key.startswith(old_name):
                # Replace the old name prefix with the new name prefix
                new_key = key.replace(old_name, new_name, 1)
                break
        
        if old_key != new_key:
            print(f"  - Remapping '{old_key}' -> '{new_key}'")
            
        new_state_dict[new_key] = value

    # Save the new state dict to the output path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, output_path)
    
    print("\n[SUCCESS] Conversion complete.")
    print(f"New checkpoint for the refactored model saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert old EtudeDecoder checkpoint keys.")
    parser.add_argument("--input", required=True, help="Path to the old checkpoint file (.pth).")
    parser.add_argument("--output", required=True, help="Path to save the new, converted checkpoint file (.pth).")
    args = parser.parse_args()
    convert_etude_decoder_checkpoint(args.input, args.output)

if __name__ == "__main__":
    main()