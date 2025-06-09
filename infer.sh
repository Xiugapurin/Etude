#!/bin/bash

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.
set -o pipefail # Return value of a pipeline is the status of the last command to exit with a non-zero status, or zero if no command exited with a non-zero status.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

INFER_DIR="$SCRIPT_DIR/infer"
BEAT_DETECTION_DIR="$SCRIPT_DIR/beat_detection"
SRC_DIR="$INFER_DIR/src"
AUDIO_TARGET_PATH="$SRC_DIR/origin.wav"
SEP_OUTPUT_PATH="$SRC_DIR/sep.npy"

DOWNLOAD_SCRIPT="$SCRIPT_DIR/utils/_download.py"
SEP_SCRIPT="$SCRIPT_DIR/corpus/seperate.py"

BEAT_DETECTION_SCRIPT="$BEAT_DETECTION_DIR/beat_detection.py"
BEAT_JSON_OUTPUT_NAME="beat_pred.json"
BEAT_JSON_OUTPUT_PATH="$SRC_DIR/$BEAT_JSON_OUTPUT_NAME"

MAIN_INFER_SCRIPT="$INFER_DIR/_infer.py"

# --- Conda Environment Name ---
# This is the *required*, pre-existing environment name
REQUIRED_CONDA_ENV_NAME="py38_spleeter"
BEAT_DETECTION_CONDA_ENV_NAME="py38_madmom"

# --- Step 0: Check for Conda ---
echo "--- Step 0: Checking for Conda ---"
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not found in PATH." >&2
    echo "Please install Miniconda or Anaconda: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi
echo "Conda found."

# --- Step 1: Prepare Source Audio ---
echo ""
echo "--- Step 1: Preparing Source Audio ---"

# Check if at least one argument (URL/path) is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <youtube_url_or_file_path> [additional_args_for_generation_script...]" >&2
    echo "Example: $0 \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" --temp 0.9" >&2
    exit 1
fi

origin_input="${1}" # First argument is the URL or file path
# Shift the first argument, so "$@" now contains only the additional arguments
shift

mkdir -p "$SRC_DIR"
echo "Ensured directory exists: $SRC_DIR"

# (Same logic as before: handles URL, empty input, or local file path)
if [[ -n "$origin_input" && ("$origin_input" == http://* || "$origin_input" == https://*) ]]; then
    echo "Input is a URL: $origin_input"
    if [ ! -f "$DOWNLOAD_SCRIPT" ]; then echo "Error: Download script missing: $DOWNLOAD_SCRIPT" >&2; exit 1; fi
    echo "Downloading audio..."
    python "$DOWNLOAD_SCRIPT" "$origin_input" "$SRC_DIR" # Use system/base python for download
    if [ ! -f "$AUDIO_TARGET_PATH" ]; then echo "Error: Download failed: $AUDIO_TARGET_PATH not found" >&2; exit 1; fi
    echo "Audio downloaded to $AUDIO_TARGET_PATH"
elif [[ -z "$origin_input" ]]; then
    echo "Input empty. Assuming audio exists at $AUDIO_TARGET_PATH"
    if [ ! -f "$AUDIO_TARGET_PATH" ]; then echo "Error: Default file not found: $AUDIO_TARGET_PATH" >&2; exit 1; fi
    echo "Using existing file: $AUDIO_TARGET_PATH"
else
    echo "Input '$origin_input' assumed local path."
    if [ ! -f "$origin_input" ]; then echo "Error: Local file not found: $origin_input" >&2; exit 1; fi
    echo "Copying '$origin_input' to '$AUDIO_TARGET_PATH'..."
    cp "$origin_input" "$AUDIO_TARGET_PATH"
    if [ $? -ne 0 ]; then echo "Error: Failed to copy file" >&2; exit 1; fi
    echo "File copied."
fi


# --- Step 2: Check and Use Existing Conda Env for Spleeter ---
echo ""
echo "--- Step 2: Locating and Using Spleeter Conda Environment ---"

# Check if the required conda environment exists
echo "Checking for Conda environment: '$REQUIRED_CONDA_ENV_NAME'..."
if ! conda env list | grep -E "^${REQUIRED_CONDA_ENV_NAME}\s+"; then
    # Use >&2 to print error messages to standard error
    echo "######################################################################" >&2
    echo "# Error: Required Conda environment '$REQUIRED_CONDA_ENV_NAME' not found." >&2
    echo "#" >&2
    echo "# This script requires a pre-existing Conda environment named '$REQUIRED_CONDA_ENV_NAME'" >&2
    echo "# containing Python 3.8.20 and a working Spleeter installation." >&2
    echo "#" >&2
    echo "# Please create and configure it manually before running this script." >&2
    echo "# Example creation steps (adjust dependencies as needed):" >&2
    echo "# 1. conda create -n $REQUIRED_CONDA_ENV_NAME python=3.8.20 -y" >&2
    echo "# 2. conda activate $REQUIRED_CONDA_ENV_NAME" >&2
    echo "# 3. conda install -c conda-forge ffmpeg -y  # OR ensure ffmpeg is installed system-wide" >&2
    echo "# 4. pip install spleeter==2.3.2 librosa" >&2
    echo "# 5. conda deactivate" >&2
    echo "######################################################################" >&2
    exit 1 # Exit the script because the required environment is missing
else
    echo "Found required Conda environment '$REQUIRED_CONDA_ENV_NAME'."
    # Optional quick checks (uncomment if needed for debugging)
    # echo "Verifying Python version..."
    # conda run -n "$REQUIRED_CONDA_ENV_NAME" python --version
    # echo "Verifying spleeter import..."
    # if ! conda run -n "$REQUIRED_CONDA_ENV_NAME" python -c "import spleeter; print('Spleeter import OK')" &> /dev/null; then
    #    echo "Warning: Could not import 'spleeter' in the '$REQUIRED_CONDA_ENV_NAME' environment. It might be misconfigured." >&2
    # fi
fi

# Check if the separation script itself exists
if [ ! -f "$SEP_SCRIPT" ]; then
    echo "Error: Audio separation script not found at $SEP_SCRIPT" >&2
    exit 1
fi

# Execute the separation script using the specified Conda environment
echo "Running separation script '$SEP_SCRIPT' in '$REQUIRED_CONDA_ENV_NAME' environment..."
# Use 'conda run' which executes a command inside the environment without permanently activating it
# Assumes audio_seperate.py uses argparse and defaults correctly, or pass args here if needed:
# conda run -n "$REQUIRED_CONDA_ENV_NAME" python "$SEP_SCRIPT" --input_audio "$AUDIO_TARGET_PATH" --output_npy "$SEP_OUTPUT_PATH"
conda run -n "$REQUIRED_CONDA_ENV_NAME" python "$SEP_SCRIPT"

sep_exit_code=$? # Capture the exit code of the separation script

if [ $sep_exit_code -ne 0 ]; then
    echo "Error: Audio separation script failed with exit code $sep_exit_code." >&2
    exit 1
fi

# Check if the expected output file was created
if [ ! -f "$SEP_OUTPUT_PATH" ]; then
    echo "Error: Separation script finished but did not produce the expected output file: $SEP_OUTPUT_PATH" >&2
    exit 1
fi
echo "Audio separation completed successfully. Output: $SEP_OUTPUT_PATH"


# --- Step 2.5: Beat and Downbeat Detection (in Conda Env: $BEAT_DETECTION_CONDA_ENV_NAME) ---
echo ""
echo "--- Step 2.5: Beat and Downbeat Detection ---"

echo "Checking for Beat Detection Conda environment: '$BEAT_DETECTION_CONDA_ENV_NAME'..."
if ! conda env list | grep -E "^${BEAT_DETECTION_CONDA_ENV_NAME}\s+"; then
    echo "######################################################################" >&2
    echo "# Error: Required Conda environment '$BEAT_DETECTION_CONDA_ENV_NAME' for Beat Detection not found." >&2
    echo "# Please create and configure it manually with Python 3.8, PyTorch, Madmom, etc." >&2
    echo "# Example: conda create -n $BEAT_DETECTION_CONDA_ENV_NAME python=3.8 -y && conda activate $BEAT_DETECTION_CONDA_ENV_NAME && pip install torch madmom numpy" >&2
    echo "######################################################################" >&2
    exit 1
else
    echo "Found Beat Detection Conda environment '$BEAT_DETECTION_CONDA_ENV_NAME'."
fi

if [ ! -f "$BEAT_DETECTION_SCRIPT" ]; then
    echo "Error: Beat detection script ('$BEAT_DETECTION_SCRIPT') not found" >&2
    exit 1
fi

# Check if the input for beat detection (output from Spleeter) exists
if [ ! -f "$SEP_OUTPUT_PATH" ]; then
    echo "Error: Input file for beat detection '$SEP_OUTPUT_PATH' not found. Skipping beat detection." >&2
    exit 1
fi

echo "Running beat detection script '$BEAT_DETECTION_SCRIPT' in '$BEAT_DETECTION_CONDA_ENV_NAME' environment..."
conda run -n "$BEAT_DETECTION_CONDA_ENV_NAME" python "$BEAT_DETECTION_SCRIPT" --input_dir "$SRC_DIR" --output_file_name "$BEAT_JSON_OUTPUT_NAME"
# The --checkpoint_dir argument in beat_detection.py defaults to '../checkpoint' relative to itself.

beat_exit_code=$?
if [ $beat_exit_code -ne 0 ]; then
    echo "Error: Beat detection script failed with exit code $beat_exit_code." >&2
    exit 1
fi

# The beat detection script removes sep.npy on success.
if [ ! -f "$BEAT_JSON_OUTPUT_PATH" ]; then
    echo "Error: Beat detection script finished but did not produce the expected output JSON: $BEAT_JSON_OUTPUT_PATH" >&2
    exit 1
fi
echo "Beat detection completed successfully. Output: $BEAT_JSON_OUTPUT_PATH"
if [ -f "$SEP_OUTPUT_PATH" ]; then
    echo "Warning: Source NPY file '$SEP_OUTPUT_PATH' was NOT removed by the beat detection script." >&2
else
    echo "Source NPY file '$SEP_OUTPUT_PATH' was successfully removed by the beat detection script."
fi


# --- Step 3: Run Main Music Generation Script (Uses Original/Default Environment) ---
echo ""
echo "--- Step 3: Running Main Music Generation Script ($MAIN_INFER_SCRIPT) ---"

if [ ! -f "$MAIN_INFER_SCRIPT" ]; then
    echo "Error: Main music generation script not found: $MAIN_INFER_SCRIPT" >&2
    exit 1
fi

echo "Executing $MAIN_INFER_SCRIPT (using default Python environment)..."
# Pass the output of beat detection as the condition file for music generation.
# Other arguments for _infer.py will use their defaults unless specified here.
python "$MAIN_INFER_SCRIPT" 
    # Add other arguments for _infer.py here if needed, e.g.:
    # --output_note_file "$INFER_DIR/output/generated_music.json" \
    # --output_score_file "$INFER_DIR/output/generated_music.musicxml" \
    # --checkpoint "$SCRIPT_DIR/checkpoint/decoder/your_decoder_model.pth" \
    # --vocab "$SCRIPT_DIR/dataset/tokenized/your_vocab.json" \
    # --config "$SCRIPT_DIR/dataset/tokenized/your_decoder_config.json" \
    # --temp 0.85 \
    # --top_p 0.95

main_infer_exit_code=$?
if [ $main_infer_exit_code -ne 0 ]; then
    echo "Error: Main music generation script failed with exit code $main_infer_exit_code." >&2;
    exit 1;
fi

echo ""
echo "--- Inference pipeline completed successfully! ---"

exit 0