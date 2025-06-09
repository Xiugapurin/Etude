# infer.py

import torch
import torch.nn.functional as F
import json
from pathlib import Path
import time
from tqdm import tqdm
import argparse
import traceback

from corpus import Vocab, Event, MidiTokenizer, COND_CLASS_ID, TGT_CLASS_ID
from models import EtudeDecoder, EtudeDecoderConfig

# --- Configuration ---
ATTRIBUTE_PAD_ID = 0
# Paths
INPUT_DIR = Path("./infer/src/")
OUTPUT_DIR = Path("./infer/output/")
DEFAULT_CONFIG_PATH = Path("./dataset/tokenized/etude_decoder_config.json") # 確保此文件名與 train.py 中保存的一致
DEFAULT_VOCAB_PATH = Path("./dataset/tokenized/vocab.json")
DEFAULT_CHECKPOINT_PATH = Path("./checkpoint/decoder/latest.pth")

DEFAULT_CONDITION_FILE = INPUT_DIR / "extract.json"
DEFAULT_TEMPO_FILE = INPUT_DIR / "tempo.json"
DEFAULT_OUTPUT_NOTE_FILE = OUTPUT_DIR / "output.json"
DEFAULT_OUTPUT_SCORE_FILE = OUTPUT_DIR / "output.musicxml"

DEFAULT_MAX_OUTPUT_TOKENS = 10000
DEFAULT_MAX_BAR_TOKEN_LIMIT = 512
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_BIN_FOR_INFERENCE = 2
NUM_BINS_FOR_MODEL_ATTRIBUTES = 5

# --- Utility Functions (_split_into_bars, save_notes_to_json 保持不變) ---
def _split_into_bars(id_sequence: list[int], bar_bos_id: int, bar_eos_id: int) -> list[list[int]]:
    # ... (與您上一版本中的 _split_into_bars 邏輯相同) ...
    bars = []; current_bar = []; in_bar = False
    if bar_bos_id < 0 or bar_eos_id < 0: raise ValueError("Invalid Bar BOS/EOS IDs")
    for token_id in id_sequence:
        if token_id == bar_bos_id:
            if in_bar and current_bar: bars.append(current_bar)
            current_bar = [token_id]; in_bar = True
        elif token_id == bar_eos_id:
            if in_bar: current_bar.append(token_id); bars.append(current_bar); current_bar = []; in_bar = False
        elif in_bar: current_bar.append(token_id)
    if current_bar and in_bar: bars.append(current_bar)
    return [b for b in bars if len(b) > 1]


def save_notes_to_json(notes: list[dict], output_path: Path):
    # ... (與您上一版本中的 save_notes_to_json 邏輯相同) ...
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(notes, f, indent=2, ensure_ascii=False)
        print(f"Generated note list saved to: {output_path}")
    except TypeError as te: print(f"TypeError saving notes: {te}. Data: {str(notes)[:200]}")
    except Exception as e: print(f"Error saving notes: {e}")


# --- Main Generation Function (修改後) ---
@torch.no_grad()
def generate_music(
    model: EtudeDecoder,
    vocab: Vocab,
    condition_ids: list[int],

    density_bin_target: int,
    pitch_class_coverage_bin_target: int,
    pitch_centroid_ratio_bin_target: int,

    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_bar_token_limit: int = DEFAULT_MAX_BAR_TOKEN_LIMIT,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    device: str = DEVICE
) -> list[Event]:
    model.eval()
    model.to(device)

    try:
        bar_bos_id = vocab.get_bar_bos_id(); bar_eos_id = vocab.get_bar_eos_id()
        if bar_bos_id == -1 or bar_eos_id == -1: raise ValueError("Bar BOS/EOS not in vocab.")
        pad_id = vocab.get_pad_id()
    except Exception as e: print(f"Error getting special tokens: {e}"); return []

    condition_bars = _split_into_bars(condition_ids, bar_bos_id, bar_eos_id)
    if not condition_bars: print("Error: Could not split condition."); return []
    print(f"Condition split into {len(condition_bars)} bars.")

    generated_events_final: list[Event] = []
    total_generated_target_tokens = 0
    kv_cache = None

    for i, cond_bar_ids in enumerate(tqdm(condition_bars, desc="Generating Bars")):
        prompt_ids_list = cond_bar_ids + [bar_bos_id]
        prompt_class_ids_list = ([COND_CLASS_ID] * len(cond_bar_ids)) + [TGT_CLASS_ID]

        prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=device)
        prompt_class_ids = torch.tensor([prompt_class_ids_list], dtype=torch.long, device=device)
        prompt_attention_mask = torch.ones_like(prompt_ids)

        # --- 為 prompt 創建屬性 bin ID 張量，使用指定的目標值 ---
        prompt_density_bin_ids = torch.full_like(prompt_ids, density_bin_target)
        prompt_pitch_class_coverage_bin_ids = torch.full_like(prompt_ids, pitch_class_coverage_bin_target)
        prompt_pitch_centroid_bin_ids = torch.full_like(prompt_ids, pitch_centroid_ratio_bin_target)

        current_generated_target_ids = []
        last_generated_token_id = bar_bos_id
        generated_len_this_bar = 0

        while True: # Inner loop for generating one target bar
            if total_generated_target_tokens >= max_output_tokens or \
               generated_len_this_bar >= max_bar_token_limit:
                # ... (安全限制日誌) ...
                if generated_len_this_bar >= max_bar_token_limit: print(f"\nBar {i} limit hit.")
                if total_generated_target_tokens >= max_output_tokens: print(f"\nMax output tokens hit.")
                break

            try:
                if generated_len_this_bar == 0: # First token for this target bar
                    model_input_ids = prompt_ids
                    model_class_ids = prompt_class_ids
                    model_attention_mask = prompt_attention_mask
                    # 使用為 prompt 創建的屬性 IDs
                    model_density_bin_ids = prompt_density_bin_ids
                    model_pitch_class_coverage_bin_ids = prompt_pitch_class_coverage_bin_ids
                    model_pitch_centroid_bin_ids = prompt_pitch_centroid_bin_ids
                    current_kv_cache = None
                else: # Subsequent tokens in the target bar
                    model_input_ids = torch.tensor([[last_generated_token_id]], dtype=torch.long, device=device)
                    model_class_ids = torch.tensor([[TGT_CLASS_ID]], dtype=torch.long, device=device)
                    current_seq_len = prompt_ids.shape[1] + generated_len_this_bar
                    model_attention_mask = torch.ones((1, current_seq_len), dtype=torch.long, device=device)
                    # 為新生成的 token 使用指定的屬性 bin ID
                    model_density_bin_ids = torch.full_like(model_input_ids, density_bin_target)
                    model_pitch_class_coverage_bin_ids = torch.full_like(model_input_ids, pitch_class_coverage_bin_target)
                    model_pitch_centroid_bin_ids = torch.full_like(model_input_ids, pitch_centroid_ratio_bin_target)
                    current_kv_cache = kv_cache

                outputs = model(
                    input_ids=model_input_ids,
                    class_ids=model_class_ids,
                    attention_mask=model_attention_mask,
                    density_bin_ids=model_density_bin_ids,
                    pitch_class_coverage_bin_ids=model_pitch_class_coverage_bin_ids, # 傳遞
                    pitch_centroid_bin_ids=model_pitch_centroid_bin_ids,   # 傳遞
                    past_key_values=current_kv_cache,
                    use_cache=True, return_dict=True
                )
                next_token_logits = outputs.logits[:, -1, :]
                kv_cache = outputs.past_key_values

            except Exception as gen_e:
                 print(f"\nError in model forward for bar {i}: {gen_e}"); traceback.print_exc(); break

            # --- Sampling (保持不變) ---
            if temperature != 1.0: next_token_logits = next_token_logits / temperature
            if top_p is not None and 0.0 < top_p < 1.0:
                 sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                 cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                 sorted_indices_to_remove = cumulative_probs > top_p
                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                 sorted_indices_to_remove[..., 0] = 0
                 indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                 next_token_logits[indices_to_remove] = -float("Inf")
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze().item()

            current_generated_target_ids.append(next_token_id)
            last_generated_token_id = next_token_id
            generated_len_this_bar += 1
            total_generated_target_tokens += 1
            if next_token_id == bar_eos_id: break
        # --- End of Inner Loop ---

        try: # Append generated target bar events
             if current_generated_target_ids:
                  generated_events_final.append(Event(type_="Bar", value="BOS")) # 手動添加 BOS
                  decoded_target_events = []
                  for tid in current_generated_target_ids: # current_generated_target_ids 不包含初始 BOS
                       if tid == pad_id: continue
                       token_str = vocab.decode(tid)
                       parts = token_str.split('_', 1)
                       type_, value_str = parts[0], parts[1] if len(parts) > 1 else ''
                       try: value = int(value_str) if type_ in ["Note", "Pos", "TimeSig"] else value_str
                       except ValueError: value = value_str
                       decoded_target_events.append(Event(type_=type_, value=value))
                  generated_events_final.extend(decoded_target_events)
        except Exception as dec_e: print(f"\nWarning: Error decoding bar {i}: {dec_e}")
        if total_generated_target_tokens >= max_output_tokens: break # Exit outer loop
    print(f"\nGeneration finished. Total target tokens: {total_generated_target_tokens}")
    return generated_events_final

# --- Main Execution (修改 argparse) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music with specified attributes.")
    parser.add_argument("--condition_file", type=str, default=str(DEFAULT_CONDITION_FILE))
    parser.add_argument("--tempo_file", type=str, default=str(DEFAULT_TEMPO_FILE))
    parser.add_argument("--output_note_file", type=str, default=str(DEFAULT_OUTPUT_NOTE_FILE))
    parser.add_argument("--output_score_file", type=str, default=str(DEFAULT_OUTPUT_SCORE_FILE))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--vocab", type=str, default=str(DEFAULT_VOCAB_PATH))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--device", type=str, default=DEVICE)

    # --- 新增：為每個屬性分別指定 bin ID ---
    parser.add_argument("--density_bin", type=int, default=DEFAULT_BIN_FOR_INFERENCE,
                        help=f"Bin ID for relative note density (0-4, default: {DEFAULT_BIN_FOR_INFERENCE}).")
    parser.add_argument("--pitch_class_cov_bin", type=int, default=DEFAULT_BIN_FOR_INFERENCE,
                        help=f"Bin ID for relative pitch class coverage (0-4, default: {DEFAULT_BIN_FOR_INFERENCE}).")
    parser.add_argument("--pitch_centroid_bin", type=int, default=DEFAULT_BIN_FOR_INFERENCE,
                        help=f"Bin ID for relative pitch centriod (0-4, default: {DEFAULT_BIN_FOR_INFERENCE}).")
    args = parser.parse_args()

    paths_to_check = {
        "Condition file": Path(args.condition_file), "Tempo file": Path(args.tempo_file),
        "Model config": Path(args.config), "Vocab file": Path(args.vocab),
        "Checkpoint file": Path(args.checkpoint)
    }
    for name, path_obj in paths_to_check.items():
        if not path_obj.exists(): print(f"Error: {name} not found: {path_obj}"); exit()
    output_path = Path(args.output_note_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_score_path = Path(args.output_score_file)

    # --- 加載組件 (修改 model_config 加載以確保屬性配置存在) ---
    print("Loading vocabulary..."); vocab = Vocab.load(Path(args.vocab))
    pad_id = vocab.get_pad_id()
    print("Initializing tokenizer..."); tokenizer = MidiTokenizer(str(Path(args.tempo_file)))

    print("Loading model configuration...")
    try:
        with open(Path(args.config), 'r') as f: config_dict = json.load(f)
        config_dict['vocab_size'] = len(vocab)
        config_dict['pad_token_id'] = pad_id
        default_num_bins = 8 # 假設模型預期8個 bins
        default_emb_dim = config_dict.get('hidden_size', 512) # 默認與 hidden_size 一致或一個固定值

        attr_configs = [
            ("num_density_bins", "density_emb_dim"),
            ("num_pitch_class_coverage_bins", "pitch_class_coverage_emb_dim"),
            ("num_pitch_centroid_bins", "pitch_centroid_emb_dim")
        ]

        for bins_key, emb_dim_key in attr_configs:
            if bins_key not in config_dict: config_dict[bins_key] = default_num_bins
            if emb_dim_key not in config_dict: config_dict[emb_dim_key] = default_emb_dim
        if 'attribute_pad_id' not in config_dict: config_dict['attribute_pad_id'] = ATTRIBUTE_PAD_ID
            
        model_config = EtudeDecoderConfig(**config_dict)
    except Exception as e: print(f"Error loading model config: {e}"); traceback.print_exc(); exit()

    print(f"Loading model from checkpoint: {args.checkpoint}...")
    try:
         model = EtudeDecoder(model_config) # 使用 EtudeDecoder
         state_dict = torch.load(Path(args.checkpoint), map_location=args.device)
         model_state = state_dict.get('model_state_dict', state_dict.get('model', state_dict))
         load_result = model.load_state_dict(model_state, strict=True) # 嘗試 True
         print(f"Model loaded. Load result: {load_result}")
    except Exception as e: print(f"Error loading checkpoint: {e}"); traceback.print_exc(); exit()

    # --- 預處理輸入 (保持不變) ---
    print("Preprocessing input condition...")
    try:
        condition_events = tokenizer.encode(str(Path(args.condition_file)))
        condition_ids = vocab.encode_sequence(condition_events)
        if not condition_ids: print("Error: Condition sequence empty."); exit()
        print(f"Condition sequence: {len(condition_ids)} tokens.")
    except Exception as e: print(f"Error preprocessing input: {e}"); traceback.print_exc(); exit()

    # --- 生成音樂 (傳入新的屬性 bin ID) ---
    print("\nStarting music generation (target events only)...")
    start_time = time.time()
    generated_event_sequence = generate_music(
        model=model, vocab=vocab, condition_ids=condition_ids,
        density_bin_target=args.density_bin,
        pitch_class_coverage_bin_target=args.pitch_class_cov_bin,
        pitch_centroid_ratio_bin_target=args.pitch_centroid_bin,
        max_output_tokens=args.max_tokens, temperature=args.temp,
        top_p=args.top_p, device=args.device
    )
    end_time = time.time()
    print(f"Event generation took {end_time - start_time:.2f} seconds.")

    if generated_event_sequence:
        print(f"Generated {len(generated_event_sequence)} target events.")
        print("Decoding generated events to note list...")

        note_list = tokenizer.decode_to_notes(generated_event_sequence)
        print(f"Decoded into {len(note_list)} notes.")
        save_notes_to_json(note_list, output_path)

        tokenizer.decode_to_score(generated_event_sequence, path_out=output_score_path)
    else:
        print("Generation failed or produced empty sequence.")
    print("\nInference finished.")