# Etude: A Controllable Music Generation Project

This project is an academic research framework for controllable music generation. It follows a three-stage pipeline: **Extract**, **Structuralize**, and **Decode**. Given a piece of source audio, it can generate a new piano performance in a style controlled by high-level musical attributes.

The core of the project is the **Decode** stage, which utilizes a custom Transformer-based model (`EtudeDecoder`) to generate music based on musical context and user-defined attributes. The **Extract** and **Structuralize** stages are handled by pre-existing models to provide the necessary input for the decoder.

---

## Setup and Installation

TODO: Complete this section

---

## Data Preparation & Training

All commands should be executed from the project's root directory while the main conda environment (`etude_env`) is activated.

### 1. Data Preparation (for Training)

TODO: Complete this section

### 2. Training

This step trains the core `EtudeDecoder` model using the prepared dataset.Before running, ensure all parameters in the specified config file are correct.

```bash
python train.py --config configs/training_config.yaml
```

To resume from a previous checkpoint, set the resume_from_checkpoint key in your training_config.yaml to the desired run_id.

## Inference

The infer.py script is the primary entry point for generating music. It provides a complete, end-to-end pipeline from a source audio file to a final MIDI output.

### Full Pipeline

This command executes all stages: it downloads the audio, runs all preprocessing and analysis, and then generates the final music based on the provided attributes.

Example using a local audio file and controlling the generation style:

```bash
python infer.py --input "/path/to/my/song.wav" --output_name "my_cover" --polyphony 2 --rhythm 1 --sustain 0
```

Example using a YouTube URL:

```bash
python infer.py --input "https://youtu.be/dQw4w9WgXcQ" --output_name "my_cover" --polyphony 2 --rhythm 1 --sustain 0
```

Intermediate files from the preprocessing stages are saved to outputs/inference/temp/.
The final MIDI file is saved to outputs/inference/.

### Decode-Only (Fast Experimentation)

After running the full pipeline once for a song, you can use the --decode-only flag to skip all time-consuming preprocessing steps. This allows you to rapidly test different creative attributes using the already-generated intermediate files.

Example of re-generating with different attributes:

```bash
python infer.py --decode-only --polyphony 1 --rhythm 1 --sustain 2
```

## Evaluation

The evaluate.py script is a command-line tool for calculating and analyzing various performance metrics for different model versions.

### Full Evaluation

This command will calculate all metrics for all versions specified in your configuration file and generate a full report.

```bash
python evaluate.py --config configs/evaluation_config.yaml
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
