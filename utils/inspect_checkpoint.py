# scripts/utils/inspect_checkpoint.py

import argparse
import torch
from collections import OrderedDict

def inspect_checkpoint(checkpoint_path: str):
    """
    Loads a PyTorch checkpoint and prints a detailed report of its contents,
    focusing on the model's state_dict.
    """
    print("="*80)
    print(f"Inspecting Checkpoint: {checkpoint_path}")
    print("="*80)

    try:
        # Load the entire file to CPU to avoid GPU memory issues
        payload = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"FATAL: Could not load the file. Error: {e}")
        return

    state_dict = None
    if isinstance(payload, OrderedDict) or isinstance(payload, dict):
        # Case 1: The file is the state_dict itself.
        if 'model_state_dict' not in payload:
            print("File seems to be a raw model state_dict.")
            state_dict = payload
        # Case 2: The file is a payload dictionary containing the state_dict.
        else:
            print("File is a payload dictionary. Extracting 'model_state_dict'.")
            state_dict = payload['model_state_dict']
            print(f"Other keys found in payload: {[k for k in payload.keys() if k != 'model_state_dict']}")
    else:
        print(f"ERROR: Loaded object is not a dictionary, but a {type(payload)}. Cannot inspect.")
        return

    if not state_dict:
        print("ERROR: Could not find a model state_dict to inspect.")
        return

    total_params = 0
    key_count = 0
    
    print("\n--- Model State Dictionary Keys and Shapes ---")
    for key, tensor in state_dict.items():
        key_count += 1
        num_params = tensor.numel()
        total_params += num_params
        print(f"- Key {key_count:03d}: {key:<80} | Shape: {str(list(tensor.shape)):<25} | Params: {num_params:,}")
        
    print("\n--- Summary ---")
    print(f"Total number of keys: {key_count}")
    print(f"Total number of parameters: {total_params:,}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch checkpoint file (.pth) and print its contents."
    )
    parser.add_argument("checkpoint_path", help="Path to the checkpoint file to inspect.")
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_path)

if __name__ == "__main__":
    main()