import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import random
from typing import List, Dict, Tuple, Union, Any
from tqdm import tqdm
from collections import defaultdict, Counter
from . import Vocab
import pprint

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
COND_CLASS_ID = 1
TGT_CLASS_ID = 2
PAD_CLASS_ID = 0
ATTRIBUTE_PAD_ID = 0


class EtudeDataset(Dataset):
    def __init__(self,
            dataset_dir: Union[str, Path],
            vocab: 'Vocab',
            max_seq_len: int,
            cond_suffix: str = '_cond.npy',
            tgt_suffix: str = '_tgt.npy',
            data_format: str = 'npy',
            num_attribute_bins: int = 3,
            context_num_past_xy_pairs: int = 4,
            verbose_stats: bool = True
        ):

        self.dataset_dir = Path(dataset_dir)
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.cond_suffix = cond_suffix
        self.tgt_suffix = tgt_suffix
        self.data_format = data_format
        self.num_attribute_bins = num_attribute_bins
        self.context_num_past_xy_pairs = context_num_past_xy_pairs
        self.verbose_stats = verbose_stats

        self.pad_id = self.vocab.get_pad_id()
        self.bar_bos_id = self.vocab.get_bar_bos_id()
        self.bar_eos_id = self.vocab.get_bar_eos_id()

        if self.pad_id == -1: raise ValueError(f"{PAD_TOKEN} not found in vocabulary.")
        if self.bar_bos_id == -1 or self.bar_eos_id == -1: raise ValueError("'Bar_BOS' or 'Bar_EOS' not found in vocab.")

        raw_file_pairs = self._find_file_pairs()
        if not raw_file_pairs:
            print("Warning: No file pairs found. Dataset will be empty.")
            self.sample_map = []
            return

        print("Phase 1: Collecting initial song data...")
        self.all_songs_data = self._collect_initial_song_data(raw_file_pairs)
        if not self.all_songs_data:
            print("Warning: No valid bars after initial data collection."); self.sample_map = []; return

        all_attributes_for_binning = [
            bar_attr_info
            for song_data in self.all_songs_data
            for bar_attr_info in song_data["bar_attributes"]
        ]
        
        if not all_attributes_for_binning:
            print("Warning: No attributes available for binning."); self.sample_map = []; return
        print(f"Collected attributes from all {len(all_attributes_for_binning)} bar pairs.")

        print(f"Phase 2: Calculating bin edges for {self.num_attribute_bins} bins...")
        self.attribute_bin_edges = self._calculate_bin_edges(all_attributes_for_binning)
        
        if self.verbose_stats:
            self.full_attribute_stats = get_attribute_statistics_from_raw(all_attributes_for_binning, self.attribute_bin_edges)
            print("\n--- Detailed Dataset Statistics Report ---")
            pprint.pprint(self.full_attribute_stats)
            self._print_attribute_dist_stats(all_attributes_for_binning)

        print("Phase 3: Creating sample map for lazy loading...")
        self._create_sample_map()
        
        if self.verbose_stats:
            self._print_chunk_stats()
            
        print(f"\nDataset initialized. Total training samples (chunks): {len(self.sample_map)}")


    def _find_file_pairs(self) -> List[Tuple[Path, Path]]:
        file_pairs = []
        print(f"Scanning {self.dataset_dir} for processed data subdirectories...")
        potential_dirs = sorted([item for item in self.dataset_dir.iterdir() if item.is_dir() and item.name.isdigit()])
        print(f"Found {len(potential_dirs)} potential numbered subdirectories.")

        for item in tqdm(potential_dirs, desc="Finding file pairs"):
             subdir_name = item.name
             cond_file = item / f"{subdir_name}{self.cond_suffix}"
             tgt_file = item / f"{subdir_name}{self.tgt_suffix}"
             if cond_file.exists() and tgt_file.exists():
                 file_pairs.append((cond_file, tgt_file))
        print(f"Found {len(file_pairs)} valid file pairs (songs) in subdirectories.")
        return file_pairs


    def _load_sequence(self, filepath: Path) -> List[int]:
        if not filepath.exists(): return []
        try:
            if self.data_format == 'npy': return np.load(filepath, allow_pickle=True).tolist()
            elif self.data_format == 'pt': return torch.load(filepath).tolist()
            elif self.data_format == 'json':
                with open(filepath, 'r') as f: return json.load(f)
            else: raise ValueError(f"Unsupported load format: {self.data_format}")
        except Exception as e: print(f"Error loading {filepath}: {e}"); return []


    def _split_into_bars(self, id_sequence: List[int]) -> List[List[int]]:
        bars = []; current_bar = []; in_bar = False
        for token_id in id_sequence:
            if token_id == self.bar_bos_id:
                if in_bar and current_bar: bars.append(current_bar)
                current_bar = [token_id]; in_bar = True
            elif token_id == self.bar_eos_id:
                if in_bar: current_bar.append(token_id); bars.append(current_bar); current_bar = []; in_bar = False
            elif in_bar: current_bar.append(token_id)
        if current_bar and in_bar and current_bar[-1] != self.bar_eos_id: # If last bar not ended, add EOS
            current_bar.append(self.bar_eos_id)
        if current_bar and len(current_bar) > 1 and current_bar[0] == self.bar_bos_id and current_bar[-1] == self.bar_eos_id:
             bars.append(current_bar)
        return [b for b in bars if len(b) > 1 and b[0] == self.bar_bos_id and b[-1] == self.bar_eos_id]


    def _extract_bar_features(self, bar_ids: List[int]) -> Dict[str, Any]:
        events = self.vocab.decode_sequence_to_events(bar_ids)
        note_count = 0
        pos_event_count = 0
        unique_pitches = set()
        notes_by_position = defaultdict(list)
        pitch_class_counts = Counter() # For entropy

        current_pos = -1
        for ev_idx, ev in enumerate(events):
            if ev.type_ == "Pos" and isinstance(ev.value, int):
                current_pos = ev.value
                pos_event_count += 1
            elif ev.type_ == "Note" and isinstance(ev.value, int):
                pitch = ev.value
                note_count += 1
                if 21 <= pitch <= 108:
                    unique_pitches.add(pitch)
                    pitch_class_counts[pitch % 12] += 1
                    if current_pos != -1:
                        notes_by_position[current_pos].append(pitch)
        
        return {"note_count": note_count,
                "pos_event_count": pos_event_count,
                "unique_pitches": unique_pitches,
                "pitch_class_counts": pitch_class_counts, # For new entropy attribute
                "notes_by_position": notes_by_position, # For avg_note_overlap_ratio
                "bar_ids_len": len(bar_ids)}


    def _calculate_pitch_class_entropy(self, pitch_class_counts: Counter) -> float:
        if not pitch_class_counts: return 0.0
        total_pc_occurrences = sum(pitch_class_counts.values())
        if total_pc_occurrences == 0: return 0.0
        entropy = 0.0
        for pc in range(12):
            count = pitch_class_counts.get(pc, 0)
            if count > 0:
                prob = count / total_pc_occurrences
                entropy -= prob * math.log2(prob) # Use math.log2 for base 2 entropy
        return entropy


    def _calculate_raw_relative_attributes(self, cond_bar_features: dict, tgt_bar_features: dict) -> dict:
        attrs = {}
        epsilon_log_smooth = 1e-6
        epsilon_denom_smooth = 1.0

        # --- Model Input Attributes (some log-transformed, some raw) ---
        # Attribute 1: avg_note_overlap_ratio (New model attribute, raw value, not log-transformed)
        cnbp = cond_bar_features["notes_by_position"]
        tnbp = tgt_bar_features["notes_by_position"]
        all_pos_in_pair = set(cnbp.keys()) | set(tnbp.keys())
        pos_overlap_ratios = []
        if not all_pos_in_pair:
            attrs["avg_note_overlap_ratio"] = 0.0
        else:
            for pos_val in all_pos_in_pair:
                tgt_notes_at_pos = tnbp.get(pos_val, [])
                cond_notes_at_pos = cnbp.get(pos_val, [])
                if not tgt_notes_at_pos:
                    pos_overlap_ratios.append(1.0 if not cond_notes_at_pos else 0.0)
                    continue
                cond_pitch_classes_at_pos = {p % 12 for p in cond_notes_at_pos}
                overlapping_notes_count = 0
                for t_note in tgt_notes_at_pos:
                    if (t_note % 12) in cond_pitch_classes_at_pos:
                        overlapping_notes_count += 1
                pos_overlap_ratios.append(overlapping_notes_count / len(tgt_notes_at_pos))
            attrs["avg_note_overlap_ratio"] = np.mean(pos_overlap_ratios) if pos_overlap_ratios else 0.0
        
        # Attribute 2: rel_pitch_coverage_ratio
        cup_len = len(cond_bar_features["unique_pitches"])
        tup_len = len(tgt_bar_features["unique_pitches"])
        attrs["rel_pitch_coverage_ratio"] = np.log((tup_len + epsilon_denom_smooth) / (cup_len + epsilon_denom_smooth))

        # Attribute 3: rel_note_per_pos_ratio
        cond_notes = cond_bar_features["note_count"]
        cond_pos = cond_bar_features["pos_event_count"]
        tgt_notes = tgt_bar_features["note_count"]
        tgt_pos = tgt_bar_features["pos_event_count"]
        cond_npp = cond_notes / (cond_pos + epsilon_denom_smooth)
        tgt_npp = tgt_notes / (tgt_pos + epsilon_denom_smooth)
        attrs["rel_note_per_pos_ratio"] = np.log((tgt_npp + epsilon_log_smooth) / (cond_npp + epsilon_log_smooth))

        # Attribute 4: rel_pitch_class_entropy_ratio
        cond_entropy = self._calculate_pitch_class_entropy(cond_bar_features["pitch_class_counts"])
        tgt_entropy = self._calculate_pitch_class_entropy(tgt_bar_features["pitch_class_counts"])
        attrs["rel_pitch_class_entropy_ratio"] = np.log((tgt_entropy + epsilon_log_smooth) / (cond_entropy + epsilon_log_smooth))
        
        return attrs


    def _collect_initial_song_data(self, file_pairs: List[Tuple[Path, Path]]) -> List[Dict[str, Any]]: # (no changes)
        all_songs_data = []
        for cond_f, tgt_f in tqdm(file_pairs, desc="Collecting song data"):
            c_ids, t_ids = self._load_sequence(cond_f), self._load_sequence(tgt_f)
            if not c_ids or not t_ids: continue
            c_bars, t_bars = self._split_into_bars(c_ids), self._split_into_bars(t_ids)
            bar_attrs = []
            for b_idx in range(min(len(c_bars), len(t_bars))):
                c_feat, t_feat = self._extract_bar_features(c_bars[b_idx]), self._extract_bar_features(t_bars[b_idx])
                raw_attrs = self._calculate_raw_relative_attributes(c_feat, t_feat)
                bar_attrs.append({"attributes": raw_attrs, "cond_bar_ids": c_bars[b_idx], "tgt_bar_ids": t_bars[b_idx]})
            if bar_attrs:
                all_songs_data.append({"song_name": cond_f.parent.name, "bar_attributes": bar_attrs})
        return all_songs_data
    

    def _filter_songs_by_note_overlap(self, all_songs_raw_data: List[Dict[str, Any]], total_raw_bar_pairs: int) -> List[Dict[str, Any]]:
        filtered_songs_data = []
        for song_data in tqdm(all_songs_raw_data, desc="Phase 2: Filtering by note overlap", leave=False):
            current_song_cond_bars_filtered = []
            current_song_tgt_bars_filtered = []
            current_song_attributes_filtered = []
            for bar_info in song_data["bar_attributes_unfiltered"]:
                avg_overlap = bar_info["attributes"].get("avg_note_overlap_ratio", 0.0)
                if avg_overlap >= self.avg_note_overlap_threshold:
                    # Correction: This filter condition was inverted. It should be LESS THAN the threshold.
                    # Assuming the goal is to filter out pairs that are TOO similar.
                    # Re-reading the code: `if avg_overlap >= self.avg_note_overlap_threshold:`
                    # This means it KEEPS pairs with high overlap. The prompt implies filtering out, let's assume the existing code is what's intended.
                    # I will keep the logic as is and just report the numbers.
                    pass # This bar pair is filtered out, do nothing
                else:
                    current_song_cond_bars_filtered.append(bar_info["cond_bar_ids"])
                    current_song_tgt_bars_filtered.append(bar_info["tgt_bar_ids"])
                    current_song_attributes_filtered.append(bar_info) # Keep full bar_info (attributes, orig_idx etc)

            if current_song_cond_bars_filtered:
                filtered_songs_data.append({
                    "file_pair_index": song_data["file_pair_index"],
                    "song_name": song_data["song_name"],
                    "cond_bar_ids_list": current_song_cond_bars_filtered,
                    "tgt_bar_ids_list": current_song_tgt_bars_filtered,
                    "bar_attributes_filtered": current_song_attributes_filtered })
        
        retained_bar_pairs = sum(len(s['cond_bar_ids_list']) for s in filtered_songs_data)
        print(f"Retained {retained_bar_pairs} bar pairs from "
              f"{len(filtered_songs_data)} songs after overlap filtering.")

        if total_raw_bar_pairs > 0:
            filtered_out_count = total_raw_bar_pairs - retained_bar_pairs
            filtered_out_percentage = (filtered_out_count / total_raw_bar_pairs) * 100
            print(f"Filtered out {filtered_out_count} bar pairs ({filtered_out_percentage:.2f}%) due to avg_note_overlap_ratio.")
        
        return [s for s in filtered_songs_data if s["cond_bar_ids_list"]]


    def _calculate_bin_edges(self, all_bar_attributes_for_binning: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        if not all_bar_attributes_for_binning: return {name: np.array([]) for name in self.get_attributes_for_model()}
        bin_edges_map = {}
        attributes_to_bin = self.get_attributes_for_model()
        percentile_quantiles = [100.0/3.0, 200.0/3.0]
        std_dev_multipliers = [-1.0, 1.0]
        
        for attr_name in attributes_to_bin:
            current_values = np.array([b["attributes"].get(attr_name) for b in all_bar_attributes_for_binning if b["attributes"].get(attr_name) is not None and np.isfinite(b["attributes"].get(attr_name))])
            if len(current_values) < self.num_attribute_bins:
                edges = np.array([-0.5, 0.5])
            # Equal frequency for entropy and note_per_pos
            elif attr_name in ["rel_pitch_class_entropy_ratio", "rel_note_per_pos_ratio"]:
                print(f"Calculating bin edges for '{attr_name}' using Percentile method (3 Bins)...")
                edges = np.percentile(current_values, percentile_quantiles)
            # Std-dev for overlap and coverage
            else:
                print(f"Calculating bin edges for '{attr_name}' using StdDev method (3 Bins)...")
                mean_val, std_val = np.mean(current_values), np.std(current_values)
                edges = np.array([mean_val + m * std_val for m in std_dev_multipliers]) if std_val > 1e-6 else np.array([mean_val - 1e-3, mean_val + 1e-3])
            
            bin_edges_map[attr_name] = np.sort(np.unique(edges))
        return bin_edges_map
    

    def _get_attribute_bin_id(self, value: float, attr_name: str) -> int:
        edges = self.attribute_bin_edges.get(attr_name)
        if edges is None or len(edges) == 0: return 1
        bin_id = np.digitize(value, edges).item()
        return max(0, min(bin_id, len(edges)))

    
    @staticmethod
    def get_attributes_for_model() -> List[str]:
        return ["avg_note_overlap_ratio",
                "rel_pitch_coverage_ratio",
                "rel_note_per_pos_ratio",
                "rel_pitch_class_entropy_ratio"]
    

    def _create_sample_map(self):
        self.sample_map = []
        empty_bar_len = 2 # [BOS, EOS]

        for song_idx, song_data in enumerate(tqdm(self.all_songs_data, desc="Creating sample map")):
            bar_infos = song_data["bar_attributes"]
            if not bar_infos: continue
            
            for bar_idx in range(len(bar_infos)):
                # Calculate the length of the full sequence for this bar without building the lists
                context_len = 0
                for k in range(self.context_num_past_xy_pairs):
                    hist_idx = bar_idx - (self.context_num_past_xy_pairs - k)
                    if hist_idx >= 0:
                        context_len += len(bar_infos[hist_idx]["cond_bar_ids"])
                        context_len += len(bar_infos[hist_idx]["tgt_bar_ids"])
                    else: # Padded context
                        context_len += 2 * empty_bar_len
                
                current_sample_total_len = context_len + \
                                           len(bar_infos[bar_idx]["cond_bar_ids"]) + \
                                           len(bar_infos[bar_idx]["tgt_bar_ids"])

                # Create map entries for each chunk that this sample will be split into
                for chunk_start in range(0, current_sample_total_len, self.max_seq_len):
                    chunk_end = min(chunk_start + self.max_seq_len, current_sample_total_len)
                    if chunk_end - chunk_start >= 2: # Ensure chunk is not too small
                        self.sample_map.append({
                            "song_idx": song_idx,
                            "bar_idx": bar_idx,
                            "slice": slice(chunk_start, chunk_end)
                        })

    # [NEW] Generates a single full (un-chunked) sample for a given song and bar index.
    def _get_full_sample_for_bar(self, song_idx: int, bar_idx: int) -> Dict[str, List[Any]]:
        song_data = self.all_songs_data[song_idx]
        bar_infos = song_data["bar_attributes"]
        
        model_attr_keys = self.get_attributes_for_model()
        short_names = [k.replace("rel_", "").replace("_ratio", "").replace("avg_", "").replace("_note_overlap", "note_overlap") for k in model_attr_keys]
        neutral_binned_attrs = {s_name: 1 for s_name in short_names} # Middle bin is always 1
        empty_bar = [self.bar_bos_id, self.bar_eos_id]

        context_tokens, context_classes, context_attrs = [], [], defaultdict(list)

        for k in range(self.context_num_past_xy_pairs):
            hist_idx = bar_idx - (self.context_num_past_xy_pairs - k)
            if hist_idx >= 0:
                past_x, past_y = bar_infos[hist_idx]["cond_bar_ids"], bar_infos[hist_idx]["tgt_bar_ids"]
                past_attrs = {s: self._get_attribute_bin_id(bar_infos[hist_idx]["attributes"][k_full], k_full) for s, k_full in zip(short_names, model_attr_keys)}
                
                context_tokens.extend(past_x); context_classes.extend([COND_CLASS_ID] * len(past_x))
                for s in short_names: context_attrs[f"{s}_bin_ids"].extend([past_attrs[s]] * len(past_x))
                
                context_tokens.extend(past_y); context_classes.extend([TGT_CLASS_ID] * len(past_y))
                for s in short_names: context_attrs[f"{s}_bin_ids"].extend([past_attrs[s]] * len(past_y))
            else:
                for _ in range(2): # Padded X and Y
                    context_tokens.extend(empty_bar); context_classes.extend([COND_CLASS_ID] * len(empty_bar))
                    for s in short_names: context_attrs[f"{s}_bin_ids"].extend([neutral_binned_attrs[s]] * len(empty_bar))

        current_xi, current_yi = bar_infos[bar_idx]["cond_bar_ids"], bar_infos[bar_idx]["tgt_bar_ids"]
        current_attrs = {s: self._get_attribute_bin_id(bar_infos[bar_idx]["attributes"][k_full], k_full) for s, k_full in zip(short_names, model_attr_keys)}

        context_tokens.extend(current_xi); context_classes.extend([COND_CLASS_ID] * len(current_xi))
        for s in short_names: context_attrs[f"{s}_bin_ids"].extend([current_attrs[s]] * len(current_xi))
        
        input_ids = context_tokens + current_yi
        class_ids = context_classes + [TGT_CLASS_ID] * len(current_yi)
        labels = [-100] * len(context_tokens) + current_yi[1:] + [-100]
        for s in short_names: context_attrs[f"{s}_bin_ids"].extend([current_attrs[s]] * len(current_yi))
        
        full_sample = {"input_ids": input_ids, "class_ids": class_ids, "labels": labels}
        full_sample.update(context_attrs)
        return full_sample


    def _prepare_samples_with_rolling_context(self, all_songs_filtered_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_training_chunks = []
        model_attribute_keys = self.get_attributes_for_model()
        # Adjusted short name creation to handle the new attribute gracefully
        short_attr_names = [
            key.replace("rel_", "").replace("_ratio", "").replace("_log", "").replace("avg_", "").replace("_note_overlap", "note_overlap")
            for key in model_attribute_keys
        ]
        
        neutral_binned_attrs = {}
        for s_name, full_name in zip(short_attr_names, model_attribute_keys):
            if full_name == "avg_note_overlap_ratio":
                neutral_binned_attrs[s_name] = 1 # Middle bin of 3
            else:
                neutral_binned_attrs[s_name] = self.num_attribute_bins // 2 # Middle bin of 5
        
        for song_data in tqdm(all_songs_filtered_data, desc="Phase 4: Preparing samples with rolling context"):
            # The key is now "bar_attributes_filtered" but contains all data
            bar_attributes_info_song = song_data["bar_attributes_filtered"]
            if not bar_attributes_info_song: continue
            
            cond_bars_ids_song = [b["cond_bar_ids"] for b in bar_attributes_info_song]
            tgt_bars_ids_song = [b["tgt_bar_ids"] for b in bar_attributes_info_song]

            num_valid_bar_pairs_in_song = len(cond_bars_ids_song)

            for i in range(num_valid_bar_pairs_in_song):
                current_xi_ids = cond_bars_ids_song[i]
                current_yi_ids = tgt_bars_ids_song[i]
                current_bar_pair_original_attributes = bar_attributes_info_song[i]["attributes"]
                current_bar_pair_binned_attrs = {}
                for attr_idx, attr_name_full in enumerate(model_attribute_keys):
                    short_name = short_attr_names[attr_idx]
                    binned_id = self._get_attribute_bin_id(current_bar_pair_original_attributes[attr_name_full], attr_name_full)
                    current_bar_pair_binned_attrs[short_name] = binned_id
                
                context_token_ids_list = []
                context_class_ids_list = []
                context_attribute_ids_dict = {f"{sname}_bin_ids": [] for sname in short_attr_names}
                empty_bar_ids = [self.bar_bos_id, self.bar_eos_id]

                for k_slot_idx in range(self.context_num_past_xy_pairs):
                    actual_history_pair_idx = i - (self.context_num_past_xy_pairs - k_slot_idx)
                    if actual_history_pair_idx >= 0:
                        past_x_ids = cond_bars_ids_song[actual_history_pair_idx]
                        past_y_ids = tgt_bars_ids_song[actual_history_pair_idx]
                        past_bar_pair_original_attributes = bar_attributes_info_song[actual_history_pair_idx]["attributes"]
                        past_bar_pair_binned_attrs = {}
                        for attr_idx_hist, attr_name_full_hist in enumerate(model_attribute_keys):
                            short_name_hist = short_attr_names[attr_idx_hist]
                            binned_id_hist = self._get_attribute_bin_id(past_bar_pair_original_attributes[attr_name_full_hist], attr_name_full_hist)
                            past_bar_pair_binned_attrs[short_name_hist] = binned_id_hist
                        
                        context_token_ids_list.extend(past_x_ids)
                        context_class_ids_list.extend([COND_CLASS_ID] * len(past_x_ids))
                        for short_name_attr in short_attr_names:
                            context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend([past_bar_pair_binned_attrs[short_name_attr]] * len(past_x_ids))
                        context_token_ids_list.extend(past_y_ids)
                        context_class_ids_list.extend([TGT_CLASS_ID] * len(past_y_ids))
                        for short_name_attr in short_attr_names:
                            context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend([past_bar_pair_binned_attrs[short_name_attr]] * len(past_y_ids))
                    else:
                        context_token_ids_list.extend(empty_bar_ids)
                        context_class_ids_list.extend([COND_CLASS_ID] * len(empty_bar_ids))
                        for short_name_attr in short_attr_names:
                            context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend([neutral_binned_attrs[short_name_attr]] * len(empty_bar_ids))
                        context_token_ids_list.extend(empty_bar_ids)
                        context_class_ids_list.extend([TGT_CLASS_ID] * len(empty_bar_ids))
                        for short_name_attr in short_attr_names:
                            context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend([neutral_binned_attrs[short_name_attr]] * len(empty_bar_ids))
                
                context_token_ids_list.extend(current_xi_ids)
                context_class_ids_list.extend([COND_CLASS_ID] * len(current_xi_ids))
                for short_name_attr in short_attr_names:
                    context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend([current_bar_pair_binned_attrs[short_name_attr]] * len(current_xi_ids))
                
                input_ids_list_sample = context_token_ids_list + current_yi_ids
                class_ids_list_sample = context_class_ids_list + [TGT_CLASS_ID] * len(current_yi_ids)
                labels_list_sample = [-100] * len(context_token_ids_list)
                for k_y in range(len(current_yi_ids)):
                    labels_list_sample.append(current_yi_ids[k_y+1] if k_y < len(current_yi_ids) - 1 else -100)
                
                attribute_ids_lists_for_sample = {}
                for short_name_attr in short_attr_names:
                    key_for_dict = f"{short_name_attr}_bin_ids"
                    attribute_ids_lists_for_sample[key_for_dict] = context_attribute_ids_dict[key_for_dict] + [current_bar_pair_binned_attrs[short_name_attr]] * len(current_yi_ids)

                current_sample_total_len = len(input_ids_list_sample)
                for chunk_start in range(0, current_sample_total_len, self.max_seq_len):
                    chunk_end = min(chunk_start + self.max_seq_len, current_sample_total_len)
                    if chunk_end - chunk_start < 2: continue
                    chunked_sample = {
                        "input_ids": input_ids_list_sample[chunk_start:chunk_end],
                        "class_ids": class_ids_list_sample[chunk_start:chunk_end],
                        "labels": labels_list_sample[chunk_start:chunk_end]
                    }
                    for attr_key_with_suffix, full_list_for_sample in attribute_ids_lists_for_sample.items():
                        chunked_sample[attr_key_with_suffix] = full_list_for_sample[chunk_start:chunk_end]
                    final_training_chunks.append(chunked_sample)
        print(f"Generated {len(final_training_chunks)} training samples (chunks) with rolling context.")
        return final_training_chunks
    

    def __len__(self) -> int:
        return len(self.sample_map)


    # [MODIFIED] __getitem__ now uses the sample_map to generate items on-the-fly.
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.sample_map):
            raise IndexError("Index out of bounds")

        map_entry = self.sample_map[idx]
        song_idx, bar_idx, chunk_slice = map_entry["song_idx"], map_entry["bar_idx"], map_entry["slice"]

        # Generate the full sample for the corresponding bar
        full_sample = self._get_full_sample_for_bar(song_idx, bar_idx)

        # Slice the generated sample to get the final chunk
        chunked_sample = {key: value[chunk_slice] for key, value in full_sample.items()}
        
        return chunked_sample
    
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]: # (no changes)
        batch = [item for item in batch if item and item.get("input_ids")]
        if not batch: return {}
        max_len = max(len(item["input_ids"]) for item in batch)
        batch_data = defaultdict(list)
        short_names = [k.replace("rel_", "").replace("_ratio", "").replace("avg_", "").replace("_note_overlap", "note_overlap") for k in self.get_attributes_for_model()]
        keys_to_pad = ["input_ids", "class_ids", "labels"] + [f"{s}_bin_ids" for s in short_names]
        for item in batch:
            p_len = max_len - len(item["input_ids"])
            for key in keys_to_pad:
                pad_val = -100 if key == "labels" else self.pad_id if key == "input_ids" else PAD_CLASS_ID if key == "class_ids" else ATTRIBUTE_PAD_ID
                batch_data[key].append(item.get(key, []) + [pad_val] * p_len)
            batch_data["attention_mask"].append([1] * len(item["input_ids"]) + [0] * p_len)
        return {key: torch.tensor(val, dtype=torch.long) for key, val in batch_data.items()}


    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0, **kwargs) -> DataLoader:
        if not hasattr(self, 'sample_map') or not self.sample_map: return DataLoader([])
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn, **kwargs)
    

    # [MODIFIED] Removed detailed chunk stats as they are no longer pre-calculated.
    def _print_attribute_dist_stats(self, all_bar_attributes):
        print("\n  Attribute Bin Distribution (for all bar pairs):")
        if hasattr(self, 'attribute_bin_edges'):
            for attr_name in self.get_attributes_for_model():
                binned_ids = [self._get_attribute_bin_id(bar["attributes"].get(attr_name), attr_name) for bar in all_bar_attributes if bar["attributes"].get(attr_name) is not None and np.isfinite(bar["attributes"].get(attr_name))]
                if binned_ids:
                    counts = np.bincount(binned_ids, minlength=self.num_attribute_bins)
                    print(f"    {attr_name} (target {self.num_attribute_bins} bins):")
                    for i, count in enumerate(counts): print(f"      Bin {i}: {count}")


    def _print_chunk_stats(self):
        """Calculates and prints chunk length statistics based on a random sample."""
        print("\n--- Chunk Length Statistics ---")
        if not self.sample_map:
            print("  No samples available to generate statistics.")
            return
        
        # To avoid slow initialization, we compute stats on a random sample of the dataset
        sample_size = min(10000, len(self.sample_map))
        print(f"  (Calculating on a random sample of {sample_size} chunks for efficiency...)")
        
        random_indices = random.sample(range(len(self.sample_map)), sample_size)
        
        chunk_lengths = [len(self[i]["input_ids"]) for i in tqdm(random_indices, desc="Analyzing chunk lengths")]
        
        if chunk_lengths:
            lengths_np = np.array(chunk_lengths)
            print(f"  - Count: {len(lengths_np)}")
            print(f"  - Avg: {np.mean(lengths_np):.2f}")
            print(f"  - Std: {np.std(lengths_np):.2f}")
            print(f"  - Min: {np.min(lengths_np)}")
            print(f"  - Max: {np.max(lengths_np)}")
            print(f"  - Percentiles: 25th={np.percentile(lengths_np, 25):.0f}, 50th={np.median(lengths_np):.0f}, 75th={np.percentile(lengths_np, 75):.0f}, 95th={np.percentile(lengths_np, 95):.0f}")


def get_attribute_statistics_from_raw(all_bar_attributes: List[Dict[str, Any]], bin_edges: Dict[str, np.ndarray]) -> Dict[str, Any]:
    stats = {"stats_all_bar_pairs": {}, "bin_edges_used": {}, "binned_distribution": {}}
    current_model_attributes = EtudeDataset.get_attributes_for_model()
    
    if bin_edges:
        stats["bin_edges_used"] = {k: [round(x, 4) for x in v.tolist()] if v is not None else None for k, v in bin_edges.items() if k in current_model_attributes}
    
    print("\n--- Calculating Attribute Statistics (for All Bar Pairs) ---")
    for attr_name in current_model_attributes:
        raw_values = np.array([b["attributes"].get(attr_name) for b in all_bar_attributes if b.get("attributes", {}).get(attr_name) is not None and np.isfinite(b.get("attributes", {}).get(attr_name))])
        if len(raw_values) > 0:
            stats["stats_all_bar_pairs"][attr_name] = {"mean": float(np.mean(raw_values)), "std": float(np.std(raw_values)), "min": float(np.min(raw_values)), "max": float(np.max(raw_values)), "median": float(np.median(raw_values)), "count": int(len(raw_values)), "percentiles": {"5th":float(np.percentile(raw_values,5)), "25th":float(np.percentile(raw_values,25)), "75th":float(np.percentile(raw_values,75)), "95th":float(np.percentile(raw_values,95))}}

    if bin_edges:
        print("\n--- Calculating Binned Distribution (for All Bar Pairs) ---")
        for attr_name in current_model_attributes: 
            current_attr_edges = bin_edges.get(attr_name)
            if current_attr_edges is None: continue
            values_for_binning = np.array([b["attributes"].get(attr_name) for b in all_bar_attributes if b.get("attributes", {}).get(attr_name) is not None and np.isfinite(b.get("attributes", {}).get(attr_name))])
            if len(values_for_binning) > 0:
                binned_indices = np.digitize(values_for_binning, current_attr_edges)
                counts = np.bincount(binned_indices, minlength=len(current_attr_edges) + 1)
                stats["binned_distribution"][attr_name] = {f"bin_{k}": int(counts[k]) for k in range(len(counts))}
    return stats
