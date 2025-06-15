import sys, pathlib
import argparse
import json
import time
import torch

# --- Path Setup ---
ROOT = pathlib.Path(__file__).parent.parent.resolve()
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
DEFAULT_EXTRACTOR_MODEL_PATH = ROOT / "checkpoint" / "extractor" / "15.pth"

DEFAULT_GENERATION_CONFIG_PATH = ROOT / "dataset" / "tokenized" / "etude_decoder_config.json"
DEFAULT_GENERATION_VOCAB_PATH = ROOT / "dataset" / "tokenized" / "vocab.json"
DEFAULT_GENERATION_CHECKPOINT_PATH = ROOT / "checkpoint" / "decoder" / "latest.pth"
DEFAULT_GENERATION_TEMPO_FILE = DEFAULT_SRC_DIR / "tempo.json"
DEFAULT_GENERATION_OUTPUT_NOTE_FILE = DEFAULT_GENERATION_OUTPUT_DIR / "output.json"
DEFAULT_GENERATION_OUTPUT_SCORE_FILE = DEFAULT_GENERATION_OUTPUT_DIR / "output.musicxml"

# [MODIFIED] Update max generation length
DEFAULT_MAX_OUTPUT_TOKENS = 25600
DEFAULT_MAX_BAR_TOKEN_LIMIT = 512
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [MODIFIED] All attributes now use 3 bins. The middle bin (1) is the neutral default.
NUM_BINS_TOTAL = 3
DEFAULT_ATTRIBUTE_BIN = NUM_BINS_TOTAL // 2 # = 1


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

# --- Main Music Generation Logic ---
def run_music_generation(args):
    print("\n--- Initializing Music Generation ---")
    
    condition_file_path = args.gen_condition_file
    tempo_file_path = args.gen_tempo_file
    output_note_file_path = args.gen_output_note_file
    output_score_file_path = args.gen_output_score_file
    config_path = args.gen_config_path
    vocab_path = args.gen_vocab_path
    checkpoint_path = args.gen_checkpoint_path

    paths_to_check = { "Generation Condition file": condition_file_path, "Generation Model config": config_path,
                       "Generation Vocab file": vocab_path, "Generation Model Checkpoint file": checkpoint_path }
    if tempo_file_path and tempo_file_path.exists():
        paths_to_check["Generation Tempo file"] = tempo_file_path

    for name, path_obj in paths_to_check.items():
        if not path_obj.exists():
            print(f"Error: Required file '{name}' not found at: {path_obj}", file=sys.stderr); sys.exit(1)

    output_note_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_score_file_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading vocabulary...")
    vocab = Vocab.load(vocab_path)
    print("Initializing tokenizer...")
    tokenizer_tempo_file_str = str(tempo_file_path) if tempo_file_path and tempo_file_path.exists() else None
    tokenizer = MidiTokenizer(tempo_path=tokenizer_tempo_file_str)

    print(f"Loading EtudeDecoder model...")
    model = load_model(config_path=str(config_path), checkpoint_path=str(checkpoint_path), device=args.gen_device)

    print("Preprocessing initial condition...")
    condition_events = tokenizer.encode(str(condition_file_path))
    initial_condition_ids = vocab.encode_sequence(condition_events)
    if not initial_condition_ids:
        print(f"Error: Initial condition sequence is empty after encoding.", file=sys.stderr); sys.exit(1)

    temp_bar_bos_id = vocab.get_bar_bos_id()
    temp_bar_eos_id = vocab.get_bar_eos_id()
    num_x_bars = len(_split_into_bars(initial_condition_ids, temp_bar_bos_id, temp_bar_eos_id))
    if num_x_bars == 0:
        print("Warning: Could not determine number of bars. Assuming 1.", file=sys.stderr)
        num_x_bars = 1

    print(f"Model will attempt to generate {num_x_bars} target Y bar(s).")

    target_attributes_per_bar_list = []
    for _ in range(num_x_bars):
        target_attributes_per_bar_list.append({
            "avg_note_overlap_bin": args.gen_avg_note_overlap_bin,
            "pitch_coverage_bin": args.gen_pitch_coverage_bin,
            "note_per_pos_bin": args.gen_note_per_pos_bin,
            "pitch_class_entropy_bin": args.gen_pitch_class_entropy_bin
        })

    print(f"\nStarting music generation with {num_x_bars} target attribute sets.")
    start_time = time.time()
    
    generated_event_sequence = model.generate(
        vocab=vocab,
        initial_condition_token_ids=initial_condition_ids,
        target_attributes_per_bar=target_attributes_per_bar_list,
        max_output_tokens=args.gen_max_tokens,
        max_bar_token_limit=args.gen_max_bar_token_limit,
        temperature=args.gen_temp,
        top_p=args.gen_top_p,
        context_overlap_ratio=args.gen_context_overlap_ratio
    )
    end_time = time.time()
    print(f"Music generation process took {end_time - start_time:.2f} seconds.")

    if generated_event_sequence:
        print(f"Generated {len(generated_event_sequence)} events.")
        note_list = tokenizer.decode_to_notes(generated_event_sequence)
        if note_list: save_notes_to_json(note_list, output_note_file_path)
        try:
            # tokenizer.decode_to_score(generated_event_sequence, path_out=output_score_file_path)
            print(f"Score saved to {output_score_file_path}")
        except Exception as e:
            print(f"Error decoding to score: {e}", file=sys.stderr)
    else:
        print("Music generation failed or produced an empty sequence.", file=sys.stderr)

    print("\nMusic generation part finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: Extract MIDI, generate tempo, and Generate Music.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--extract_path_input_audio", type=pathlib.Path, default=DEFAULT_SRC_DIR / DEFAULT_EXTRACT_INPUT_AUDIO_NAME, help="Path to input audio file for MIDI extraction.")
    extract_group = parser.add_argument_group('MIDI Extraction Options')
    extract_group.add_argument("--extract_path_output_json", type=pathlib.Path, default=DEFAULT_SRC_DIR / DEFAULT_EXTRACT_OUTPUT_JSON_NAME, help="Path to output JSON file from MIDI extraction.")
    extract_group.add_argument("--extract_path_output_midi", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_DIR / DEFAULT_EXTRACT_OUTPUT_MIDI_NAME, help="Path to output MIDI file from MIDI extraction.")
    extract_group.add_argument("--extract_path_model", type=pathlib.Path, default=DEFAULT_EXTRACTOR_MODEL_PATH, help="Path to MIDI extraction model checkpoint.")

    gen_group = parser.add_argument_group('Music Generation Options')
    gen_group.add_argument("--gen_beat_pred_file_for_tempo", type=pathlib.Path, default=DEFAULT_BEAT_PRED_FILE, help="Path to beat prediction JSON file for TempoInfoGenerator.")
    gen_group.add_argument("--gen_tempo_file", type=pathlib.Path, default=DEFAULT_GENERATION_TEMPO_FILE, help="Path to tempo JSON file.")
    gen_group.add_argument("--gen_output_note_file", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_NOTE_FILE, help="Path to save the generated notes as JSON.")
    gen_group.add_argument("--gen_output_score_file", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_SCORE_FILE, help="Path to save the generated score as MusicXML.")
    gen_group.add_argument("--gen_config_path", type=pathlib.Path, default=DEFAULT_GENERATION_CONFIG_PATH, help="Path to EtudeDecoder's JSON configuration file.")
    gen_group.add_argument("--gen_vocab_path", type=pathlib.Path, default=DEFAULT_GENERATION_VOCAB_PATH, help="Path to generation model's vocabulary file.")
    gen_group.add_argument("--gen_checkpoint_path", type=pathlib.Path, default=DEFAULT_GENERATION_CHECKPOINT_PATH, help="Path to EtudeDecoder's checkpoint (.pth) file.")
    gen_group.add_argument("--gen_max_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Global maximum number of tokens to generate.")
    gen_group.add_argument("--gen_max_bar_token_limit", type=int, default=DEFAULT_MAX_BAR_TOKEN_LIMIT, help="Token limit per individual bar during generation.")
    gen_group.add_argument("--gen_temp", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature (0 for greedy).")
    gen_group.add_argument("--gen_top_p", type=float, default=DEFAULT_TOP_P, help="Nucleus sampling (top-p) probability.")
    gen_group.add_argument("--gen_device", type=str, default=DEVICE, choices=["cuda", "cpu"], help="Device to use for generation.")
    gen_group.add_argument("--gen_context_overlap_ratio", type=float, default=0.5, help="Context overlap ratio for truncation in model.generate().")
    
    # [MODIFIED] All attribute arguments now use the same range (0-2) and default.
    gen_group.add_argument("--gen_avg_note_overlap_bin", type=int, default=DEFAULT_ATTRIBUTE_BIN,
                           choices=range(NUM_BINS_TOTAL),
                           help=f"Target bin ID for average note overlap ratio (0-{NUM_BINS_TOTAL-1}).")
    gen_group.add_argument("--gen_pitch_class_entropy_bin", type=int, default=DEFAULT_ATTRIBUTE_BIN,
                           choices=range(NUM_BINS_TOTAL),
                           help=f"Target bin ID for relative pitch_class_entropy ratio (0-{NUM_BINS_TOTAL-1}).")
    gen_group.add_argument("--gen_pitch_coverage_bin", type=int, default=DEFAULT_ATTRIBUTE_BIN,
                           choices=range(NUM_BINS_TOTAL), 
                           help=f"Target bin ID for relative pitch coverage (0-{NUM_BINS_TOTAL-1}).")
    gen_group.add_argument("--gen_note_per_pos_bin", type=int, default=DEFAULT_ATTRIBUTE_BIN,
                           choices=range(NUM_BINS_TOTAL),
                           help=f"Target bin ID for relative note_per_pos ratio (0-{NUM_BINS_TOTAL-1}).")

    args = parser.parse_args()
    
    print("Validating paths for MIDI extraction...")
    if not args.extract_path_input_audio.exists():
        print(f"Error: Input audio not found: {args.extract_path_input_audio}", file=sys.stderr); sys.exit(1)
    if not args.extract_path_model.exists():
        print(f"Error: Extractor model not found: {args.extract_path_model}", file=sys.stderr); sys.exit(1)
    args.extract_path_output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.extract_path_output_midi: args.extract_path_output_midi.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Step 1: MIDI Extraction ---")
    extract(str(args.extract_path_input_audio), str(args.extract_path_output_json),
            str(args.extract_path_output_midi) if args.extract_path_output_midi else "",
            str(args.extract_path_model))

    print(f"\nGenerating tempo info...")
    if args.gen_beat_pred_file_for_tempo.exists():
        try:
            tg = TempoInfoGenerator(path_beat=args.gen_beat_pred_file_for_tempo, verbose=False)
            tg.generate_tempo_info(args.gen_tempo_file)
            print(f"Tempo info saved to: {args.gen_tempo_file}")
        except Exception as e:
            print(f"Warning: Could not generate tempo info: {e}. MidiTokenizer will use defaults.", file=sys.stderr)
            args.gen_tempo_file = None
    else:
        print(f"Warning: Beat prediction file not found. Cannot generate tempo info.", file=sys.stderr)
        args.gen_tempo_file = None

    print("\n--- Step 2: Music Generation ---")
    args.gen_condition_file = args.extract_path_output_json
    run_music_generation(args)

    print("\n--- Full pipeline completed successfully! ---")