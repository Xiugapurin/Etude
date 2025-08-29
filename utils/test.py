# scripts/utils/compare_sequences.py

import json
from pathlib import Path
from typing import List, Any

def compare_json_lists(file1_path: Path, file2_path: Path):
    """
    Compares the contents of two JSON files that are expected to contain lists of numbers.
    Provides a detailed report of any differences found.
    """
    print("="*60)
    print("      Running Sequence Comparison Script")
    print("="*60)
    print(f"[INFO] Comparing OLD file: {file1_path}")
    print(f"[INFO] Comparing NEW file: {file2_path}")
    print("-"*60)

    # --- Load Data ---
    try:
        with open(file1_path, 'r') as f:
            data_old: List[Any] = json.load(f)
        with open(file2_path, 'r') as f:
            data_new: List[Any] = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ [ERROR] File not found: {e.filename}")
        print("          Please make sure you have generated both files correctly.")
        return
    except json.JSONDecodeError as e:
        print(f"❌ [ERROR] Could not parse JSON in one of the files: {e}")
        return

    # --- Perform Comparison ---
    if data_old == data_new:
        print("✅ [SUCCESS] The two sequences are IDENTICAL.")
        print(f"          - Length: {len(data_old)}")
        print("          - This suggests the MidiTokenizer's encoding logic is likely correct.")
        return

    print("❌ [FAILURE] The two sequences are NOT identical.")
    
    # --- Detailed Difference Analysis ---
    print("\n--- Difference Analysis ---")
    
    # 1. Compare lengths
    len_old, len_new = len(data_old), len(data_new)
    if len_old != len_new:
        print(f"- Length Mismatch:")
        print(f"  - OLD sequence length: {len_old}")
        print(f"  - NEW sequence length: {len_new}")
    else:
        print(f"- Lengths are identical: {len_old}")

    # 2. Find the first mismatch
    first_mismatch_found = False
    for i, (item_old, item_new) in enumerate(zip(data_old, data_new)):
        if item_old != item_new:
            print(f"- First Mismatch Found at index {i}:")
            print(f"  - OLD value: {item_old}")
            print(f"  - NEW value: {item_new}")
            first_mismatch_found = True
            break
    
    if not first_mismatch_found and len_old != len_new:
        print("- No mismatches found within the common length.")

    # 3. Count total mismatches (if lengths are the same)
    if len_old == len_new:
        mismatch_count = sum(1 for item_old, item_new in zip(data_old, data_new) if item_old != item_new)
        print(f"- Total Mismatches: {mismatch_count} out of {len_old} tokens ({mismatch_count/len_old:.2%})")
    
    print("--- End of Report ---")

def main():
    # Define the paths based on our previous debugging steps
    # We assume this script is run from the project's root directory
    project_root = Path.cwd()
    old_file = project_root / "condition_ids_OLD.json"
    # Assuming the default output path for the new inference script
    new_file = project_root / "condition_ids_NEW.json"

    compare_json_lists(old_file, new_file)

if __name__ == "__main__":
    main()