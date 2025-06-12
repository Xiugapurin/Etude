import sys, pathlib
import argparse
import json
import time
import torch
import torch.nn.functional as F # Model.generate might use it internally
import traceback

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

# Defaults for MIDI Extraction (Step 1)
DEFAULT_BEAT_PRED_FILE = ROOT / "infer" / "src" / "beat_pred.json"
DEFAULT_EXTRACT_INPUT_AUDIO_NAME = "origin.wav"
DEFAULT_EXTRACT_OUTPUT_JSON_NAME = "extract.json"
DEFAULT_EXTRACT_OUTPUT_MIDI_NAME = "etude_e.mid"
DEFAULT_EXTRACTOR_MODEL_PATH = ROOT / "checkpoint" / "extractor" / "15.pth"

# Defaults for Music Generation (Step 2)
DEFAULT_GENERATION_CONFIG_PATH = ROOT / "dataset" / "tokenized" / "etude_decoder_config.json" # EtudeDecoderConfig JSON
DEFAULT_GENERATION_VOCAB_PATH = ROOT / "dataset" / "tokenized" / "vocab.json"
DEFAULT_GENERATION_CHECKPOINT_PATH = ROOT / "checkpoint" / "decoder" / "latest.pth" # EtudeDecoder checkpoint
DEFAULT_GENERATION_TEMPO_FILE = DEFAULT_SRC_DIR / "tempo.json" # Output of TempoInfoGenerator
DEFAULT_GENERATION_OUTPUT_NOTE_FILE = DEFAULT_GENERATION_OUTPUT_DIR / "output.json"
DEFAULT_GENERATION_OUTPUT_SCORE_FILE = DEFAULT_GENERATION_OUTPUT_DIR / "output.musicxml"

DEFAULT_MAX_OUTPUT_TOKENS = 10000
DEFAULT_MAX_BAR_TOKEN_LIMIT = 512
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default target attribute bins for generation
NUM_BINS_TOTAL = 5 # Changed from 7 to 5 to match EtudeDataset
DEFAULT_PITCH_COVERAGE_BIN = NUM_BINS_TOTAL // 2
DEFAULT_NOTE_PER_POS_BIN = NUM_BINS_TOTAL // 2
DEFAULT_PITCH_CLASS_ENTROPY_BIN = NUM_BINS_TOTAL // 2 # New attribute


# --- Utility Functions ---
def _split_into_bars(id_sequence: list[int], bar_bos_id: int, bar_eos_id: int) -> list[list[int]]:
    # This version is from your EtudeInfer-v2.py, kept for calculating num_x_bars
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
    if current_bar and in_bar: # If sequence ends mid-bar after a BOS
        # Ensure last bar also ends with EOS if it was started
        if current_bar[-1] != bar_eos_id: current_bar.append(bar_eos_id)
        bars.append(current_bar)
    # Filter for bars that strictly start with BOS and end with EOS, and have content
    return [b for b in bars if len(b) > 1 and b[0] == bar_bos_id and b[-1] == bar_eos_id]

def save_notes_to_json(notes: list[dict], output_path: pathlib.Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        print(f"Generated note list saved to: {output_path}")
    except TypeError as te:
        print(f"TypeError saving notes: {te}. Data (first 200 chars): {str(notes)[:200]}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving notes: {e}", file=sys.stderr)

# --- Main Music Generation Logic ---
def run_music_generation(args):
    print("\n--- Initializing Music Generation ---")
    
    condition_file_path = args.gen_condition_file # This is extract_path_output_json
    tempo_file_path = args.gen_tempo_file       # This is DEFAULT_GENERATION_TEMPO_FILE (output of TempoInfoGenerator)
    output_note_file_path = args.gen_output_note_file
    output_score_file_path = args.gen_output_score_file
    config_path = args.gen_config_path          # Path to EtudeDecoderConfig JSON
    vocab_path = args.gen_vocab_path
    checkpoint_path = args.gen_checkpoint_path  # Path to EtudeDecoder .pth checkpoint

    # File existence checks
    paths_to_check = {
        "Generation Condition file": condition_file_path,
        "Generation Model config": config_path,
        "Generation Vocab file": vocab_path,
        "Generation Model Checkpoint file": checkpoint_path
    }
    if tempo_file_path and tempo_file_path.exists(): # Tempo file is optional for tokenizer
        paths_to_check["Generation Tempo file"] = tempo_file_path
    elif tempo_file_path: 
         print(f"Warning: Expected tempo file for generation at {tempo_file_path} but it does not exist. "
               "MidiTokenizer might use defaults or fail if it strictly requires it.", file=sys.stderr)
    else: 
        print("Note: No tempo file path resolved for generation tokenizer, using MidiTokenizer defaults.", file=sys.stderr)

    for name, path_obj in paths_to_check.items():
        if not path_obj.exists():
            print(f"Error: Required file for generation - '{name}' not found at: {path_obj}", file=sys.stderr)
            sys.exit(1)

    output_note_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_score_file_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading vocabulary for generation...")
    vocab = Vocab.load(vocab_path)

    print("Initializing tokenizer for generation...")
    tokenizer_tempo_file_str = str(tempo_file_path) if tempo_file_path and tempo_file_path.exists() else None
    try:
        tokenizer = MidiTokenizer(tempo_path=tokenizer_tempo_file_str)
    except Exception as e_tok:
        print(f"Error initializing MidiTokenizer with tempo_path='{tokenizer_tempo_file_str}': {e_tok}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr); sys.exit(1)

    print(f"Loading EtudeDecoder model from config: {config_path} and checkpoint: {checkpoint_path}...")
    # Using load_model from EtudeModel.py, which handles EtudeDecoderConfig and state_dict loading
    model = load_model(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=args.gen_device
    )

    print("Preprocessing initial condition for generation...")
    try:
        condition_events = tokenizer.encode(str(condition_file_path))
        initial_condition_ids = vocab.encode_sequence(condition_events)
        if not initial_condition_ids:
            print(f"Error: Initial condition sequence from {condition_file_path} is empty after encoding.", file=sys.stderr)
            sys.exit(1)
        print(f"Full initial condition sequence (all X bars) contains {len(initial_condition_ids)} tokens.")
    except Exception as e:
        print(f"Error preprocessing initial condition from {condition_file_path}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr); sys.exit(1)

    # --- 準備 target_attributes_per_bar ---
    # 列表長度應等於 initial_condition_ids 中 X bar 的數量
    temp_bar_bos_id = vocab.get_bar_bos_id()
    temp_bar_eos_id = vocab.get_bar_eos_id()
    num_x_bars = 0
    if temp_bar_bos_id != -1 and temp_bar_eos_id != -1:
        # Use the _split_into_bars defined in this script
        num_x_bars = len(_split_into_bars(initial_condition_ids, temp_bar_bos_id, temp_bar_eos_id))
    
    if num_x_bars == 0: # If no bars found in condition, or BOS/EOS invalid
        print("Warning: Could not determine number of X bars from initial_condition_ids or no X bars found. "
              "Attempting to generate a fixed number of bars if specified, or defaulting to 1.", file=sys.stderr)
        # Fallback to a predefined number of bars if gen_num_target_bars was an arg,
        # or simply generate for 1 bar if that was the previous default.
        # For now, let's assume we always want to generate Y bars corresponding to X bars.
        # If initial_condition_ids has content but no bar structure, model.generate might handle it as one large X.
        # This case might need more specific handling based on desired behavior.
        if initial_condition_ids: # If there are tokens, but not bar-structured
            print("  Treating entire initial condition as a single segment for one target bar generation.", file=sys.stderr)
            num_x_bars = 1 # Generate one Y for the whole X
        else: # No condition tokens at all
            print("  Initial condition is empty. Cannot generate.", file=sys.stderr)
            sys.exit(1)


    print(f"Based on initial condition, model will attempt to generate {num_x_bars} target Y bar(s).")

    target_attributes_per_bar_list = []
    for _ in range(num_x_bars): # 為每個 Xi 對應的 Yi 準備目標屬性
        target_attributes_per_bar_list.append({
            # 使用更新後的屬性鍵名 (與 EtudeModel.generate 期望的字典鍵匹配)
            "pitch_coverage_bin": args.gen_pitch_coverage_bin,
            "note_per_pos_bin": args.gen_note_per_pos_bin,
            "pitch_class_entropy_bin": args.gen_pitch_class_entropy_bin # 新增
        })

    print(f"\nStarting music generation using model.generate() with {num_x_bars} target attribute sets.")
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
        print(f"model.generate() produced {len(generated_event_sequence)} events.")
        print("Decoding generated events to note list...")
        note_list = tokenizer.decode_to_notes(generated_event_sequence)
        print(f"Decoded into {len(note_list)} notes.")
        if note_list:
            save_notes_to_json(note_list, output_note_file_path)
        else:
            print("Warning: Note list is empty after decoding generated events.", file=sys.stderr)

        print(f"Decoding generated events to score and saving to {output_score_file_path}...")
        try:
            tokenizer.decode_to_score(generated_event_sequence, path_out=output_score_file_path)
            print(f"Score saved to {output_score_file_path}")
        except Exception as e:
            print(f"Error decoding to score: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    else:
        print("Music generation (model.generate()) failed or produced an empty sequence.", file=sys.stderr)

    print("\nMusic generation part finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full pipeline: Extract MIDI, generate tempo, and Generate Music.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--extract_path_input_audio", type=pathlib.Path,
                        default=DEFAULT_SRC_DIR / DEFAULT_EXTRACT_INPUT_AUDIO_NAME,
                        help="Path to input audio file for MIDI extraction.")

    extract_group = parser.add_argument_group('MIDI Extraction Options')
    extract_group.add_argument("--extract_path_output_json", type=pathlib.Path,
                               default=DEFAULT_SRC_DIR / DEFAULT_EXTRACT_OUTPUT_JSON_NAME,
                               help="Path to output JSON file from MIDI extraction (used as condition for generation)")
    extract_group.add_argument("--extract_path_output_midi", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_DIR / DEFAULT_EXTRACT_OUTPUT_MIDI_NAME,
                               help="Path to output MIDI file from MIDI extraction (optional)")
    extract_group.add_argument("--extract_path_model", type=pathlib.Path, default=DEFAULT_EXTRACTOR_MODEL_PATH,
                               help="Path to MIDI extraction model checkpoint")

    gen_group = parser.add_argument_group('Music Generation Options')
    gen_group.add_argument("--gen_beat_pred_file_for_tempo", type=pathlib.Path, default=DEFAULT_BEAT_PRED_FILE,
                           help="Path to beat prediction JSON file (input for TempoInfoGenerator).")
    gen_group.add_argument("--gen_tempo_file", type=pathlib.Path, default=DEFAULT_GENERATION_TEMPO_FILE,
                           help="Path to tempo JSON file (output of TempoInfoGenerator, input for MidiTokenizer).")
    gen_group.add_argument("--gen_output_note_file", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_NOTE_FILE,
                           help="Path to save the generated notes as JSON")
    gen_group.add_argument("--gen_output_score_file", type=pathlib.Path, default=DEFAULT_GENERATION_OUTPUT_SCORE_FILE,
                           help="Path to save the generated score as MusicXML")
    gen_group.add_argument("--gen_config_path", type=pathlib.Path, default=DEFAULT_GENERATION_CONFIG_PATH,
                           help="Path to EtudeDecoder's JSON configuration file")
    gen_group.add_argument("--gen_vocab_path", type=pathlib.Path, default=DEFAULT_GENERATION_VOCAB_PATH,
                           help="Path to generation model's vocabulary file")
    gen_group.add_argument("--gen_checkpoint_path", type=pathlib.Path, default=DEFAULT_GENERATION_CHECKPOINT_PATH,
                           help="Path to EtudeDecoder's checkpoint (.pth) file")
    
    gen_group.add_argument("--gen_max_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
                           help="Global maximum number of tokens to generate for the entire target sequence")
    gen_group.add_argument("--gen_max_bar_token_limit", type=int, default=DEFAULT_MAX_BAR_TOKEN_LIMIT,
                           help="Token limit per individual bar during generation for model.generate()")
    gen_group.add_argument("--gen_temp", type=float, default=DEFAULT_TEMPERATURE,
                           help="Sampling temperature for generation (0 for greedy, >0 for randomness)")
    gen_group.add_argument("--gen_top_p", type=float, default=DEFAULT_TOP_P,
                           help="Nucleus sampling (top-p) probability. Set to >1 or 0 to disable.")
    gen_group.add_argument("--gen_device", type=str, default=DEVICE, choices=["cuda", "cpu"],
                           help="Device to use for generation ('cuda' or 'cpu')")
    
    # 更新的屬性 bin ID (與 EtudeDecoderConfig 和 EtudeDataset 一致, 5 bins total, ID 0-4)
    gen_group.add_argument("--gen_pitch_class_entropy_bin", type=int, default=DEFAULT_PITCH_CLASS_ENTROPY_BIN,
                           choices=range(NUM_BINS_TOTAL),
                           help=f"Target bin ID for relative pitch_class_entropy ratio (0-{NUM_BINS_TOTAL-1})")
    gen_group.add_argument("--gen_pitch_coverage_bin", type=int, default=DEFAULT_PITCH_COVERAGE_BIN,
                           choices=range(NUM_BINS_TOTAL), 
                           help=f"Target bin ID for relative pitch coverage (0-{NUM_BINS_TOTAL-1})")
    gen_group.add_argument("--gen_note_per_pos_bin", type=int, default=DEFAULT_NOTE_PER_POS_BIN,
                           choices=range(NUM_BINS_TOTAL),
                           help=f"Target bin ID for relative note_per_pos ratio (0-{NUM_BINS_TOTAL-1})")
        
    gen_group.add_argument("--gen_num_past_bars_context", type=int, default=2, # Default to 2 as per EtudeModel's config
                           help="Number of past (X_prev+Y_prev) bar pairs for context in model.generate(). Should match model config if not overriding.")
    gen_group.add_argument("--gen_context_overlap_ratio", type=float, default=0.5,
                           help="Context overlap ratio for truncation in model.generate().")

    args = parser.parse_args()
    
    # --- Path Validation for Extraction ---
    print("Validating paths for MIDI extraction...")
    if not args.extract_path_input_audio.exists():
        print(f"Error: Input audio for extraction not found: {args.extract_path_input_audio}", file=sys.stderr)
        sys.exit(1)
    if not args.extract_path_input_audio.is_file():
        print(f"Error: Input audio for extraction is not a file: {args.extract_path_input_audio}", file=sys.stderr)
        sys.exit(1)
    if not args.extract_path_model.exists():
        print(f"Error: Extractor model checkpoint not found: {args.extract_path_model}", file=sys.stderr)
        sys.exit(1)
    args.extract_path_output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.extract_path_output_midi:
        args.extract_path_output_midi.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: MIDI Extraction ---
    print("\n--- Step 1: MIDI Extraction ---")
    print(f"Input audio: {args.extract_path_input_audio}")
    print(f"Output JSON: {args.extract_path_output_json}")
    if args.extract_path_output_midi: print(f"Output MIDI: {args.extract_path_output_midi}")
    print(f"Extractor Model: {args.extract_path_model}")
    try:
        extract(
            str(args.extract_path_input_audio),
            str(args.extract_path_output_json),
            str(args.extract_path_output_midi) if args.extract_path_output_midi else "",
            str(args.extract_path_model)
        )
        print("MIDI extraction finished successfully.")
    except Exception as e:
        print(f"Error during MIDI extraction: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr); sys.exit(1)

    # --- Intermediate Step: Tempo Generation ---
    print(f"\nGenerating tempo info using beat prediction file: {args.gen_beat_pred_file_for_tempo}")
    if args.gen_beat_pred_file_for_tempo.exists():
        try:
            tg = TempoInfoGenerator(path_beat=args.gen_beat_pred_file_for_tempo, verbose=False)
            tg.generate_tempo_info(args.gen_tempo_file) 
            print(f"Tempo info generated and saved to: {args.gen_tempo_file}")
        except Exception as e_tempo:
            print(f"Error generating tempo info: {e_tempo}. Proceeding with potentially missing tempo file.", file=sys.stderr)
            args.gen_tempo_file = None 
    else:
        print(f"Warning: Beat prediction file {args.gen_beat_pred_file_for_tempo} not found. Cannot generate tempo.json.", file=sys.stderr)
        args.gen_tempo_file = None

    # --- Step 2: Music Generation ---
    args.gen_condition_file = args.extract_path_output_json 

    try:
        run_music_generation(args)
    except Exception as e:
        print(f"Critical error during music generation step: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print("\n--- Full pipeline completed successfully! ---")
