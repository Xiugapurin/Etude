import sys, pathlib
import argparse
import json
import time
import torch
import pretty_midi
from pathlib import Path

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
print(f"Project ROOT set to: {ROOT}")

# --- Imports from project modules ---
from corpus import extract
from corpus import Vocab, MidiTokenizer
from corpus.tempo import TempoInfoGenerator
from models import load_model

# --- Constants and Defaults ---
DEFAULT_SRC_DIR = ROOT / "infer" / "src"
DEFAULT_GENERATION_OUTPUT_DIR = ROOT / "infer" / "output"

DEFAULT_BEAT_PRED_FILE = ROOT / "infer" / "src" / "beat_pred.json"
DEFAULT_EXTRACT_INPUT_AUDIO_NAME = "origin.wav"
DEFAULT_EXTRACT_OUTPUT_JSON_NAME = "extract.json"
DEFAULT_EXTRACT_OUTPUT_MIDI_NAME = "etude_e.mid"
DEFAULT_EXTRACTOR_MODEL_PATH = ROOT / "checkpoint" / "extractor" / "9.pth"

DEFAULT_GENERATION_VOCAB_PATH = ROOT / "dataset" / "tokenized" / "vocab.json"
DEFAULT_GENERATION_CHECKPOINT_PATH = ROOT / "checkpoint" / "decoder" / "90.pth"
DEFAULT_GENERATION_CONFIG_PATH = ROOT / "checkpoint" / "decoder" / "etude_decoder_config.json"
DEFAULT_GENERATION_TEMPO_FILE = DEFAULT_SRC_DIR / "tempo.json"
DEFAULT_GENERATION_VOLUME_FILE = DEFAULT_SRC_DIR / "volume.json"
DEFAULT_GENERATION_OUTPUT_NOTE_FILE = DEFAULT_GENERATION_OUTPUT_DIR / "output.json"
DEFAULT_GENERATION_OUTPUT_MIDI_FILE = DEFAULT_GENERATION_OUTPUT_DIR / "etude_d_d.mid"
DEFAULT_GENERATION_OUTPUT_SCORE_FILE = DEFAULT_GENERATION_OUTPUT_DIR / "output.musicxml"

DEFAULT_MAX_OUTPUT_TOKENS = 25600
DEFAULT_MAX_BAR_TOKEN_LIMIT = 512
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_BINS_TOTAL = 3

# --- Utility Functions ---
def _split_into_bars(id_sequence: list[int], bar_bos_id: int, bar_eos_id: int) -> list[list[int]]:
    bars = []; current_bar = []; in_bar = False
    if bar_bos_id < 0 or bar_eos_id < 0: raise ValueError("Invalid Bar BOS/EOS IDs for splitting")
    for token_id in id_sequence:
        if token_id == bar_bos_id:
            if in_bar and current_bar: bars.append(current_bar)
            current_bar = [token_id]; in_bar = True
        elif token_id == bar_eos_id:
            if in_bar:
                current_bar.append(token_id)
                bars.append(current_bar)
                current_bar = []
                in_bar = False
        elif in_bar:
            current_bar.append(token_id)
    if current_bar and in_bar:
        if current_bar[-1] != bar_eos_id: current_bar.append(bar_eos_id)
        bars.append(current_bar)
    return [b for b in bars if len(b) > 1 and b[0] == bar_bos_id and b[-1] == bar_eos_id]


def save_notes_to_json(notes: list[dict], output_path: pathlib.Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        print(f"Generated note list saved to: {output_path}")
    except Exception as e:
        print(f"Error saving notes: {e}", file=sys.stderr)


def note_to_midi(note_list, output_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for note in note_list:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=note["velocity"],
                pitch=note["pitch"],
                start=note["onset"],
                end=note["offset"],
            )
        )

    midi.instruments.append(instrument)
    midi.write(output_path)


# --- Main Music Generation Logic ---
def run_music_generation(args):
    print("\n--- Initializing Music Generation ---")
    
    paths_to_check = { "Condition file": args.gen_condition_file, "Model config": args.gen_config_path,
                       "Vocab file": args.gen_vocab_path, "Model Checkpoint": args.gen_checkpoint_path }
    for name, path_obj in paths_to_check.items():
        if not path_obj.exists():
            print(f"Error: Required file '{name}' not found at: {path_obj}", file=sys.stderr); sys.exit(1)

    args.gen_output_note_file.parent.mkdir(parents=True, exist_ok=True)
    args.gen_output_score_file.parent.mkdir(parents=True, exist_ok=True)

    vocab = Vocab.load(args.gen_vocab_path)
    tokenizer = MidiTokenizer(tempo_path=str(args.gen_tempo_file) if args.gen_tempo_file.exists() else None)
    model = load_model(config_path=str(args.gen_config_path), checkpoint_path=str(args.gen_checkpoint_path), device=args.gen_device)

    print("Preprocessing initial condition...")
    condition_events = tokenizer.encode(str(args.gen_condition_file), with_grace_note=False)
    initial_condition_ids = vocab.encode_sequence(condition_events)
    if not initial_condition_ids:
        print("Error: Initial condition sequence is empty after encoding.", file=sys.stderr); sys.exit(1)

    num_x_bars = len(_split_into_bars(initial_condition_ids, vocab.get_bar_bos_id(), vocab.get_bar_eos_id())) or 1
    print(f"Model will attempt to generate {num_x_bars} target bar(s).")

    print(f"{args.gen_avg_note_overlap_bin = } \n {args.gen_rel_note_per_pos_bin = } \n {args.gen_rel_avg_duration_bin = } \n {args.gen_rel_pos_density_bin = } \n ")

    target_attributes_per_bar_list = []
    for _ in range(num_x_bars):
        target_attributes_per_bar_list.append({
            "note_overlap_bin": args.gen_avg_note_overlap_bin,
            "note_per_pos_bin": args.gen_rel_note_per_pos_bin,
            "avg_dur_bin": args.gen_rel_avg_duration_bin,
            "pos_dens_bin": args.gen_rel_pos_density_bin
        })

    print(f"\nStarting music generation with {num_x_bars} target attribute sets.")
    start_time = time.time()
    
    generated_event_sequence = model.generate(
        vocab=vocab, initial_condition_token_ids=initial_condition_ids,
        target_attributes_per_bar=target_attributes_per_bar_list, max_output_tokens=args.gen_max_tokens,
        max_bar_token_limit=args.gen_max_bar_token_limit, temperature=args.gen_temp, top_p=args.gen_top_p
    )
    print(f"Music generation process took {time.time() - start_time:.2f} seconds.")

    if generated_event_sequence:
        print(f"Generated {len(generated_event_sequence)} events.")
        note_list = tokenizer.decode_to_notes(events=generated_event_sequence, volume_map_path=DEFAULT_GENERATION_VOLUME_FILE)
        if note_list: 
            # save_notes_to_json(note_list, args.gen_output_note_file)
            note_to_midi(note_list, args.gen_output_midi_file)
        
        # try:
            # tokenizer.decode_to_score(generated_event_sequence, path_out=args.gen_output_score_file)
        #     print(f"Score saved to {args.gen_output_score_file}")
        # except Exception as e:
        #     print(f"An unexpected error occurred during score generation: {e}", file=sys.stderr)
    else:
        print("Music generation failed or produced an empty sequence.", file=sys.stderr)

    print("\nMusic generation part finished.")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: Extract MIDI, generate tempo, and Generate Music.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--extract_path_input_audio", type=pathlib.Path, default=DEFAULT_SRC_DIR / DEFAULT_EXTRACT_INPUT_AUDIO_NAME)
    extract_group = parser.add_argument_group('MIDI Extraction Options')
    extract_group.add_argument("--extract_path_output_json", type=pathlib.Path, default=DEFAULT_SRC_DIR / DEFAULT_EXTRACT_OUTPUT_JSON_NAME)
    extract_group.add_argument("--extract_path_output_midi", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_DIR / DEFAULT_EXTRACT_OUTPUT_MIDI_NAME)
    extract_group.add_argument("--extract_path_model", type=pathlib.Path, default=DEFAULT_EXTRACTOR_MODEL_PATH)

    gen_group = parser.add_argument_group('Music Generation Options')
    gen_group.add_argument("--gen_beat_pred_file_for_tempo", type=pathlib.Path, default=DEFAULT_BEAT_PRED_FILE)
    gen_group.add_argument("--gen_tempo_file", type=pathlib.Path, default=DEFAULT_GENERATION_TEMPO_FILE)
    gen_group.add_argument("--gen_output_note_file", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_NOTE_FILE)
    gen_group.add_argument("--gen_output_midi_file", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_MIDI_FILE)
    gen_group.add_argument("--gen_output_score_file", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_SCORE_FILE)
    gen_group.add_argument("--gen_config_path", type=pathlib.Path, default=DEFAULT_GENERATION_CONFIG_PATH)
    gen_group.add_argument("--gen_vocab_path", type=pathlib.Path, default=DEFAULT_GENERATION_VOCAB_PATH)
    gen_group.add_argument("--gen_checkpoint_path", type=pathlib.Path, default=DEFAULT_GENERATION_CHECKPOINT_PATH)
    gen_group.add_argument("--gen_max_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    gen_group.add_argument("--gen_max_bar_token_limit", type=int, default=DEFAULT_MAX_BAR_TOKEN_LIMIT)
    gen_group.add_argument("--gen_temp", type=float, default=DEFAULT_TEMPERATURE)
    gen_group.add_argument("--gen_top_p", type=float, default=DEFAULT_TOP_P)
    gen_group.add_argument("--gen_device", type=str, default=DEVICE, choices=["cuda", "cpu"])
    
    gen_group.add_argument("--gen_avg_note_overlap_bin", type=int, default=2, choices=range(NUM_BINS_TOTAL), help=f"Target bin for note overlap (0-{NUM_BINS_TOTAL-1})")
    gen_group.add_argument("--gen_rel_note_per_pos_bin", type=int, default=1, choices=range(NUM_BINS_TOTAL), help=f"Target bin for note density (0-{NUM_BINS_TOTAL-1})")
    gen_group.add_argument("--gen_rel_avg_duration_bin", type=int, default=1, choices=range(NUM_BINS_TOTAL), help=f"Target bin for relative average duration (0-{NUM_BINS_TOTAL-1})")
    gen_group.add_argument("--gen_rel_pos_density_bin", type=int, default=1, choices=range(NUM_BINS_TOTAL), help=f"Target bin for positional density (0-{NUM_BINS_TOTAL-1})")

    args = parser.parse_args()
    
    print("Validating paths for MIDI extraction...")
    if not args.extract_path_input_audio.exists(): sys.exit(f"Error: Input audio not found: {args.extract_path_input_audio}")
    if not args.extract_path_model.exists(): sys.exit(f"Error: Extractor model not found: {args.extract_path_model}")
    args.extract_path_output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.extract_path_output_midi: args.extract_path_output_midi.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Step 1: MIDI Extraction ---")
    extract(str(args.extract_path_input_audio), str(args.extract_path_output_json),
            str(args.extract_path_output_midi) if args.extract_path_output_midi else "",
            str(args.extract_path_model))

    print(f"\n--- Intermediate Step: Tempo Generation ---")
    if args.gen_beat_pred_file_for_tempo.exists():
        try:
            tg = TempoInfoGenerator(path_beat=str(args.gen_beat_pred_file_for_tempo), verbose=False)
            tg.generate_tempo_info(str(args.gen_tempo_file))
            print(f"Tempo info saved to: {args.gen_tempo_file}")
        except Exception as e:
            print(f"Warning: Could not generate tempo info: {e}. MidiTokenizer may use defaults.", file=sys.stderr)
            args.gen_tempo_file = Path("/nonexistent/path")
    else:
        print(f"Warning: Beat prediction file not found. Cannot generate tempo info.", file=sys.stderr)
        args.gen_tempo_file = Path("/nonexistent/path")

    print("\n--- Step 2: Music Generation ---")
    args.gen_condition_file = args.extract_path_output_json
    run_music_generation(args)

    print("\n--- Full pipeline completed successfully! ---")