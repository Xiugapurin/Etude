# Etude

This project is an academic research framework for controllable piano cover generation. It follows a three-stage pipeline: **Extract**, **Structuralize**, and **Decode**. Given a piece of source audio, it can generate a piano cover in a style controlled by high-level musical attributes.

The core of the project is the **Decode** stage, which utilizes a Transformer-based model to generate piano cover based on musical context and user-defined attributes. The **Extract** and **Structuralize** stages are handled by pre-existing models to provide the necessary input for the decoder.

- [Demo Page](https://xiugapurin.github.io/Etude/)
- [Paper](https://arxiv.org/abs/2509.16522)

## Requirements

- **Linux (Ubuntu)**: Recommended for production use
- **macOS (Apple Silicon)**: Experimental support with MPS acceleration
- GPU with at least 16GB of VRAM (CUDA) or Apple Silicon with 16GB+ unified memory (MPS)
- `ffmpeg` is required

## Environment Setup

### Install ffmpeg

#### Ubuntu:
```bash
sudo apt-get update && sudo apt-get install ffmpeg
```

#### macOS:
```bash
brew install ffmpeg
```

### Setup the Main Environment

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e "."
```

### Beat Detection Dependencies

The **Structuralize** stage requires audio source separation for beat detection. There are two backend options:

#### Spleeter (Default, Recommended)

**Spleeter** provides the best beat detection accuracy but requires a separate conda environment.

Create the `spleeter` environment (the environment name must match `configs/project_config.yaml`):

```bash
conda create --name py38_spleeter python=3.8.20 -y
conda activate py38_spleeter
pip install spleeter==2.3.2 librosa
conda deactivate
```

#### Demucs (Experimental)

For **macOS** users who cannot install Spleeter, **Demucs** is available as an alternative backend.

> **⚠️ WARNING:** Using **Demucs** with the **Beat-Transformer** produces less accurate beat information, which may affect the quality of the final output.

To use **Demucs**:

1. Install the **Demucs** dependency:
   ```bash
   pip install -e ".[demucs]"
   ```

2. Modify `configs/project_config.yaml`:
   ```yaml
   env:
     separation_backend: "demucs"  # Change from "spleeter" to "demucs"
   ```

### Download Pre-trained Models

Download the pre-trained model checkpoints and place them in their respective directories.

```bash
wget -O checkpoints.zip "https://github.com/Xiugapurin/Etude/releases/download/latest/checkpoints.zip"
unzip checkpoints.zip
rm checkpoints.zip
```

After downloading checkpoints, verify that the files have been placed correctly. Your project's `checkpoints/` directory should have the following structure:

```
checkpoints/
├── beat_detector/
│   └── latest.pt
├── decoder/
│   └── latest.pth
├── extractor/
│   └── latest.pth
└── hft_transformer/
    ├── config.json
    └── latest.pkl
```

### Generate Your Piano Cover

Once the environments are set up and checkpoints are in place, you can generate a piano cover with a single command.

Provide a YouTube URL:

```bash
python infer.py --input "https://youtu.be/dQw4w9WgXcQ"
```

Or, provide a local audio file path:

```bash
python infer.py --input "path/to/my/song.wav"
```

## Inference Guide

The **Etude** framework offers controllable piano cover generation. You can adjust three high-level musical attributes to steer the style of the output. The value for each attribute ranges from 0 (low intensity) to 2 (high intensity), with 1 being the default neutral value.

### Controllable Attributes

- **Polyphony**: Controls the density of the musical texture.
- **Rhythm Intensity**: Controls the rhythmic complexity and activity.
- **Note Sustain**: Controls the average duration of notes (articulation).

You can specify these attributes as command-line arguments to creatively guide the generation.

Example: Generate a cover that is harmonically simple (`--polyphony 0`), has a neutral rhythm (`--rhythm 1`), and is very smooth and connected (`--sustain 2`):

```bash
python infer.py --input "https://youtu.be/dQw4w9WgXcQ" --polyphony 0 --rhythm 1 --sustain 2
```

### Fast Experimentation

The full pipeline executes three stages: `extract`, `structuralize`, and `decode`. After you have successfully processed a song once, the intermediate files are saved. You can then use the `--decode-only` flag to skip the time-consuming `extract` and `structuralize` stages, allowing you to rapidly test different musical styles for the same song.

Example: After running a song once, re-generate it with maximum polyphony and rhythm:

```bash
python infer.py --decode-only --polyphony 2 --rhythm 2 --sustain 1
```

## Evaluation

The evaluate.py script is a command-line tool for calculating and analyzing various performance metrics for different model versions.

### Full Evaluation

This command will calculate all metrics for all versions specified in your configuration file and generate a full report.

```bash
python evaluate.py --config configs/evaluate_config.yaml
```

### Partial & Specific Evaluation

The script provides flags to flexibly run only the parts you are interested in.

Example: Calculate only the RGC and IPE metrics:

```bash
python evaluate.py --metrics rgc ipe
```

Example: Evaluate only your own model (etude_d) and the human performance (human):

```bash
python evaluate.py --versions etude_d human
```

Example: Run all calculations but only save the raw data to a CSV without printing reports:

```bash
python evaluate.py --no-report --output-csv "my_results.csv"
```

## Training

This project involves two main models that can be trained: the Extractor and the Decoder.

### Training an Extractor (AMT-APC)

The extractor model is responsible for the initial `audio-to-MIDI` transcription. This project uses a pre-trained model based on the `AMT-APC` architecture. If you wish to train your own extractor from scratch, please refer to the detailed instructions provided in the original [AMT-APC](https://github.com/misya11p/amt-apc) project repository.

### Training a Decoder

The core of this project is the `EtudeDecoder` model. To train your own decoder, you first need to prepare a dataset.

#### Prepare Your Dataset

The data preparation pipeline is designed to work with a dataset format similar to that provided by the [pop2piano](https://github.com/sweetcocoa/pop2piano) project.

You will need a CSV file that lists pairs of YouTube video IDs: one for the original song (`pop_ids`) and one for the corresponding piano cover (`piano_ids`). An example is provided in `asset/dataset.csv`.

#### Run the Data Preparation Pipeline

Once your dataset CSV is ready, you can run a single script to perform all necessary preparation steps (download, preprocess, align, extract and tokenize).

**To run the full pipeline from start to finish**:

```bash
python prepare.py
```

This script is designed to be resumable. If it's interrupted, you can run it again, and it will skip already completed steps.

**To control the execution flow**:

You can use flags to run only specific parts of the pipeline, which is useful for debugging or re-running a single stage.

Use the `--start-from` flag to begin execution at a specific stage:

```bash
# Skip the 'download' stage and start from 'preprocess'
python scripts/prepare_dataset.py --start-from preprocess
```

Use the `--run-only` flag to execute only a single stage:

```bash
# Run only the final 'tokenize' stage
python scripts/prepare_dataset.py --run-only tokenize
```

#### Run the Training Script

Once your dataset has been successfully prepared (i.e., the `dataset/tokenized/` directory is populated), execute the following command to start training your custom `EtudeDecoder` model:

```bash
python train.py
```

You can modify all training settings, such as learning rate, batch size, and number of epochs, in the `configs/training_config.yaml` file.

#### Use Your New Model for Inference

After training is complete, a new run directory will be created (e.g., `outputs/train/your_run_id/`). Inside, you will find your new model weights (`latest.pth`) and the corresponding configuration file (`etude_decoder_config.json`).

To test your new model, remember to update the `configs/inference_config.yaml` file to point to these newly generated files:

```YAML
# In configs/inference_config.yaml
decoder:
  model_path: "outputs/train/your_run_id/latest.pth"
  config_path: "outputs/train/your_run_id/etude_decoder_config.json"
  vocab_path: "dataset/vocab.json"
  # ...
```

## Debug & Logging

The project uses a unified logging system controlled by the `LOG_LEVEL` environment variable.

### Log Levels

| Level | Description |
|-------|-------------|
| `DEBUG` | Detailed information for debugging (file processing, cache hits, etc.) |
| `INFO` | Standard progress information (default) |
| `WARN` | Warnings about skipped files or potential issues |
| `ERROR` | Error messages |

### Usage

```bash
# Standard output (INFO level, default)
python prepare.py

# Show detailed debug messages
LOG_LEVEL=DEBUG python prepare.py

# Suppress all but warnings and errors
LOG_LEVEL=WARN python infer.py --input "song.wav"
```

### Disable Colors

To disable colored output (useful for log files or CI environments):

```bash
NO_COLOR=1 python prepare.py
```

---

## License

This project is released under a dual-license system.

### Source Code

The **source code** of this project is licensed under the **MIT License**. You can find the full license text in the `LICENSE` file at the root of this repository.

### Pre-trained Models & Checkpoints

All pre-trained model checkpoints (files ending in `.pth`, `.pkl`, etc.) located in the `checkpoints/` directory are made available under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**.

[![CC BY-NC-SA 4.0](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This means you are free to share and adapt these models for **non-commercial research and artistic purposes**, provided you give appropriate credit and distribute any derivative works under the same license.

**Use of the pre-trained models for commercial purposes is strictly prohibited under this license.** For inquiries about commercial licensing, please contact the project owner.
