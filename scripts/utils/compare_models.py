# scripts/utils/compare_models.py

import argparse
import sys
from pathlib import Path
from collections import OrderedDict
import torch
import yaml

# --- Helper to load the NEW, refactored model ---
# We assume the new structure is the primary one
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.etude.utils.model_loader import load_etude_decoder
from src.etude.models.etude_decoder import EtudeDecoder

# --- Helper to load the OLD, original model ---
# This part is tricky as it needs to simulate the old environment.
# It assumes the old 'models' and 'corpus' directories exist at the project root.
def load_old_model_for_comparison(config_path, checkpoint_path, vocab_path, device):
    print("\n--- Attempting to load model using OLD logic ---")
    try:
        # Temporarily add project root to path, similar to old scripts
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        
        # These are the imports from the original _infer.py and models.py
        from models import load_model as old_load_model_func 
        from corpus import Vocab as OldVocab
        
        # The old load_model function was simpler and part of the models package
        model = old_load_model_func(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device
        )
        print("[SUCCESS] Loaded model using OLD logic.")
        return model
    except Exception as e:
        print(f"[FAILED] Could not load model using OLD logic: {e}")
        print("         Please ensure your original 'models' and 'corpus' directories are present.")
        return None

def compare_models(model_old: torch.nn.Module, model_new: torch.nn.Module):
    """Performs a deep comparison of two model state_dicts."""
    print("\n" + "="*80)
    print("                 Model Comparison Report")
    print("="*80)

    state_old = model_old.state_dict()
    state_new = model_new.state_dict()

    keys_old = set(state_old.keys())
    keys_new = set(state_new.keys())

    # 1. Compare Architectures (by comparing the set of keys)
    print("\n--- 1. Architecture (Layer Name) Comparison ---")
    if keys_old == keys_new:
        print("✅ [SUCCESS] Model architectures appear IDENTICAL. All layer names match.")
        print(f"          - Total layers: {len(keys_old)}")
    else:
        print("❌ [FAILURE] Model architectures are DIFFERENT.")
        if missing_in_new := keys_old - keys_new:
            print(f"  - Layers MISSING in NEW model: {missing_in_new}")
        if missing_in_old := keys_new - keys_old:
            print(f"  - Layers EXTRA in NEW model: {missing_in_old}")
        # We stop here if architectures differ, as weight comparison is meaningless.
        return

    # 2. Compare Weights (Tensor Values)
    print("\n--- 2. Weight (Tensor Value) Comparison ---")
    mismatched_tensors = []
    for key in keys_old:
        tensor_old = state_old[key]
        tensor_new = state_new[key]
        if not torch.equal(tensor_old, tensor_new):
            mismatched_tensors.append({
                "key": key,
                "shape_old": list(tensor_old.shape),
                "shape_new": list(tensor_new.shape),
                "diff_abs_sum": torch.sum(torch.abs(tensor_old - tensor_new)).item()
            })
    
    if not mismatched_tensors:
        print("✅ [SUCCESS] All model weights are IDENTICAL.")
    else:
        print(f"❌ [FAILURE] Found {len(mismatched_tensors)} mismatched weight tensors!")
        print("--- Details of First 5 Mismatches ---")
        for i, mismatch in enumerate(mismatched_tensors[:5]):
            print(f"  - Mismatch #{i+1}:")
            print(f"    Key:      {mismatch['key']}")
            print(f"    Shapes:   {mismatch['shape_old']} (OLD) vs {mismatch['shape_new']} (NEW)")
            print(f"    Abs Diff: {mismatch['diff_abs_sum']:.6f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Deep compare models from old and new loading logic.")
    # Assuming standard paths from our refactoring
    parser.add_argument("--old_config", default="checkpoints/decoder/etude_decoder_config.json")
    parser.add_argument("--old_checkpoint", default="checkpoints/decoder/90.pth")
    parser.add_argument("--new_config", default="checkpoints/decoder/etude_decoder_config.json")
    parser.add_argument("--new_checkpoint", default="checkpoints/decoder/latest.pth")
    parser.add_argument("--vocab", default="dataset/tokenized/vocab.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the new model using our refactored, robust loader
    print("--- Attempting to load model using NEW logic ---")
    model_new = load_etude_decoder(args.new_config, args.new_checkpoint, device)
    
    # Load the old model by simulating the old environment
    # Note: This requires your original 'models' and 'corpus' directories to be accessible
    model_old = load_old_model_for_comparison(args.old_config, args.old_checkpoint, args.vocab, device)
    
    if model_old and model_new:
        compare_models(model_old, model_new)
    else:
        print("\nCould not load both models. Comparison aborted.")

if __name__ == "__main__":
    main()