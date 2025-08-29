# scripts/utils/compare_files.py

import argparse
import difflib
import sys
from pathlib import Path

def compare_text_files(file1_path: Path, file2_path: Path):
    """
    Compares two text files line by line and prints the result.

    If the files are identical, it prints a success message.
    If they differ, it prints a detailed diff report to the console.

    Args:
        file1_path (Path): Path to the first file.
        file2_path (Path): Path to the second file.
    """
    print("="*60)
    print("      Running File Content Comparison")
    print("="*60)
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print("-"*60)

    try:
        # Read the content of both files, splitting by lines
        with open(file1_path, 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
        with open(file2_path, 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()

    except FileNotFoundError as e:
        print(f"❌ [ERROR] File not found: {e.filename}", file=sys.stderr)
        print("          Please ensure both file paths are correct.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ [ERROR] An unexpected error occurred while reading the files: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Perform Comparison ---
    if lines1 == lines2:
        print("✅ [SUCCESS] The two files are IDENTICAL.")
        print(f"          - Both files contain {len(lines1)} lines.")
    else:
        print("❌ [FAILURE] The two files have DIFFERENT content.")
        print("\n--- Detailed Differences (Diff Report) ---")
        print("  Legend: ")
        print("    - : Line only in File 1")
        print("    + : Line only in File 2")
        print("    ? : Line with formatting differences")
        print("------------------------------------------")
        
        # Generate a unified diff report
        diff = difflib.unified_diff(
            lines1,
            lines2,
            fromfile=str(file1_path),
            tofile=str(file2_path),
            lineterm=''
        )
        
        # Print the diff report to the console
        for line in diff:
            print(line)
        
        print("\n--- End of Report ---")

def main():
    """Main function to parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two text files and show the differences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("file1", type=Path, help="Path to the first text file.")
    parser.add_argument("file2", type=Path, help="Path to the second text file.")
    args = parser.parse_args()
    
    compare_text_files(args.file1, args.file2)

if __name__ == "__main__":
    main()