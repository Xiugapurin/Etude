import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
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
                 num_attribute_bins: int = 5, # Defaulting to 5 as per request
                 avg_note_overlap_threshold: float = 0.5,
                 context_num_past_xy_pairs: int = 2,
                 verbose_stats: bool = True
                 ):

        self.dataset_dir = Path(dataset_dir)
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.cond_suffix = cond_suffix
        self.tgt_suffix = tgt_suffix
        self.data_format = data_format
        self.num_attribute_bins = num_attribute_bins
        self.avg_note_overlap_threshold = avg_note_overlap_threshold
        self.context_num_past_xy_pairs = context_num_past_xy_pairs
        self.verbose_stats = verbose_stats

        self.pad_id = self.vocab.get_pad_id()
        self.bar_bos_id = self.vocab.get_bar_bos_id()
        self.bar_eos_id = self.vocab.get_bar_eos_id()

        if self.pad_id == -1: raise ValueError(f"{PAD_TOKEN} not found in vocabulary.")
        if self.bar_bos_id == -1 or self.bar_eos_id == -1: raise ValueError("'Bar_BOS' or 'Bar_EOS' not found in vocab.")

        raw_file_pairs = self._find_file_pairs()
        if not raw_file_pairs:
            print("Warning: No file pairs (songs) found. Dataset will be empty.")
            self.processed_samples = []; self.full_attribute_stats = {}; self.total_dataset_tokens = 0; return

        print("Phase 1: Collecting initial song data (bar IDs and raw attributes)...")
        all_songs_raw_data = self._collect_initial_song_data(raw_file_pairs)
        if not all_songs_raw_data:
            print("Warning: No valid bars after initial data collection.")
            self.processed_samples = []; self.full_attribute_stats = {}; self.total_dataset_tokens = 0; return
        
        print(f"Phase 2: Filtering bar pairs by avg_note_overlap_ratio < {self.avg_note_overlap_threshold}...")
        self.all_songs_filtered_data = self._filter_songs_by_note_overlap(all_songs_raw_data)
        if not self.all_songs_filtered_data:
            print("Warning: All songs or bar pairs filtered out by note overlap filter.")
            self.processed_samples = []; self.full_attribute_stats = {}; self.total_dataset_tokens = 0; return

        all_attributes_for_binning = [] 
        for song_data in self.all_songs_filtered_data:
            for bar_attr_info in song_data["bar_attributes_filtered"]:
                all_attributes_for_binning.append(bar_attr_info) 
        
        if not all_attributes_for_binning:
            print("Warning: No attributes remaining for binning after filtering.")
            self.processed_samples = []; self.full_attribute_stats = {}; self.total_dataset_tokens = 0; return
        print(f"Collected attributes from {len(all_attributes_for_binning)} filtered bar pairs for binning.")

        print(f"Phase 3: Calculating bin edges for {self.num_attribute_bins} bins (Hybrid strategy)...") # 新日誌
        self.attribute_bin_edges = self._calculate_bin_edges(all_attributes_for_binning)
        
        if self.verbose_stats:
            all_raw_attributes_for_stats_report = []
            for song_data in all_songs_raw_data: # Use unfiltered for overall raw stats
                 for bar_attr_unfiltered in song_data["bar_attributes_unfiltered"]:
                    all_raw_attributes_for_stats_report.append(bar_attr_unfiltered)

            self.full_attribute_stats = get_attribute_statistics_from_raw(
                all_raw_attributes_for_stats_report, # All bar pairs before overlap filter
                all_attributes_for_binning,          # Filtered bar pairs used for binning
                self.attribute_bin_edges,
                self.num_attribute_bins
            )
        else: self.full_attribute_stats = {}

        print("Phase 4: Preparing final training samples with rolling context...")
        self.processed_samples = self._prepare_samples_with_rolling_context(self.all_songs_filtered_data)
        
        self.total_dataset_tokens = 0
        if self.processed_samples:
            for sample in self.processed_samples:
                self.total_dataset_tokens += len(sample.get("input_ids", []))

        if self.verbose_stats and hasattr(self, 'full_attribute_stats') and self.full_attribute_stats:
            print("\n--- Detailed Dataset Statistics Report (from EtudeDataset init) ---")
            pprint.pprint(self.full_attribute_stats)
            self._print_additional_stats(all_attributes_for_binning) 

        print(f"Dataset initialized. Total training samples (chunks with context): {len(self.processed_samples)}")
        if self.total_dataset_tokens > 0: # Only print if there are tokens
             print(f"Total tokens in the dataset (sum of input_ids lengths in all chunks): {self.total_dataset_tokens:,}")


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
        epsilon_log_smooth = 1e-6 # For log(value + eps)
        epsilon_denom_smooth = 1.0 # For (num + eps) / (den + eps) before log, or for raw ratios if den is 0

        # 1. avg_note_overlap_ratio (Re-added with octave tolerance)
        cnbp = cond_bar_features["notes_by_position"]
        tnbp = tgt_bar_features["notes_by_position"]
        all_pos_in_pair = set(cnbp.keys()) | set(tnbp.keys())
        pos_overlap_ratios = []
        if not all_pos_in_pair: attrs["avg_note_overlap_ratio"] = 0.0
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
        
        # --- Model Input Attributes (Log Transformed) ---
        # Attribute 1: rel_pitch_coverage_ratio (based on unique pitches, kept)
        cup_len = len(cond_bar_features["unique_pitches"])
        tup_len = len(tgt_bar_features["unique_pitches"])
        attrs["rel_pitch_coverage_ratio"] = np.log((tup_len + epsilon_denom_smooth) / (cup_len + epsilon_denom_smooth))

        # Attribute 2: rel_note_per_pos_ratio (kept)
        cond_notes = cond_bar_features["note_count"]
        cond_pos = cond_bar_features["pos_event_count"]
        tgt_notes = tgt_bar_features["note_count"]
        tgt_pos = tgt_bar_features["pos_event_count"]
        cond_npp = cond_notes / (cond_pos + epsilon_denom_smooth)
        tgt_npp = tgt_notes / (tgt_pos + epsilon_denom_smooth)
        attrs["rel_note_per_pos_ratio"] = np.log((tgt_npp + epsilon_log_smooth) / (cond_npp + epsilon_log_smooth)) # Using smaller epsilon for log stability

        # Attribute 3: rel_pitch_class_entropy_ratio (New, replaces old pitch class coverage)
        cond_entropy = self._calculate_pitch_class_entropy(cond_bar_features["pitch_class_counts"])
        tgt_entropy = self._calculate_pitch_class_entropy(tgt_bar_features["pitch_class_counts"])
        attrs["rel_pitch_class_entropy_ratio"] = np.log((tgt_entropy + epsilon_log_smooth) / (cond_entropy + epsilon_log_smooth))
        
        return attrs


    def _collect_initial_song_data(self, file_pairs: List[Tuple[Path, Path]]) -> List[Dict[str, Any]]:
        all_songs_data = []
        for pair_idx, (cond_f, tgt_f) in enumerate(tqdm(file_pairs, desc="Phase 1: Collecting initial song data", leave=False)):
            c_ids_full_song, t_ids_full_song = self._load_sequence(cond_f), self._load_sequence(tgt_f)
            if not c_ids_full_song or not t_ids_full_song: continue
            c_bars_song = self._split_into_bars(c_ids_full_song)
            t_bars_song = self._split_into_bars(t_ids_full_song)
            song_bar_attributes_unfiltered = []
            min_num_bars = min(len(c_bars_song), len(t_bars_song))
            if min_num_bars == 0: continue
            for b_idx in range(min_num_bars):
                c_b_ids, t_b_ids = c_bars_song[b_idx], t_bars_song[b_idx]
                c_feat = self._extract_bar_features(c_b_ids)
                t_feat = self._extract_bar_features(t_b_ids)
                raw_attrs = self._calculate_raw_relative_attributes(c_feat, t_feat)
                song_bar_attributes_unfiltered.append({
                    "attributes": raw_attrs, "cond_bar_ids": c_b_ids, 
                    "tgt_bar_ids": t_b_ids, "original_bar_index_in_song": b_idx })
            if song_bar_attributes_unfiltered:
                all_songs_data.append({
                    "file_pair_index": pair_idx, "song_name": cond_f.parent.name, 
                    "bar_attributes_unfiltered": song_bar_attributes_unfiltered })
        return all_songs_data
    

    def _filter_songs_by_note_overlap(self, all_songs_raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered_songs_data = []
        for song_data in tqdm(all_songs_raw_data, desc="Phase 2: Filtering by note overlap", leave=False):
            current_song_cond_bars_filtered = []
            current_song_tgt_bars_filtered = []
            current_song_attributes_filtered = []
            for bar_info in song_data["bar_attributes_unfiltered"]:
                avg_overlap = bar_info["attributes"].get("avg_note_overlap_ratio", 0.0)
                if avg_overlap >= self.avg_note_overlap_threshold:
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
        print(f"Retained {sum(len(s['cond_bar_ids_list']) for s in filtered_songs_data)} bar pairs from "
              f"{len(filtered_songs_data)} songs after overlap filtering.")
        return [s for s in filtered_songs_data if s["cond_bar_ids_list"]]


    def _calculate_bin_edges(self, all_bar_attributes_for_binning: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Calculates bin edges using a hybrid strategy:
        - For 'rel_pitch_class_entropy_ratio': uses percentile-based (equal frequency) binning.
        - For other attributes: uses standard deviation-based binning.
        All attributes target self.num_attribute_bins (e.g., 5 bins).
        """
        if not all_bar_attributes_for_binning:
            # Fallback for empty data: create (self.num_attribute_bins - 1) arbitrary edges
            num_edges = self.num_attribute_bins - 1
            default_edges = np.linspace(-0.75, 0.75, num_edges + 2)[1:-1] if num_edges > 0 else np.array([])
            return {name: default_edges for name in self.get_attributes_for_model()}

        bin_edges_map = {}
        attributes_to_bin = self.get_attributes_for_model()
        
        # For percentile-based binning (5 bins need 4 edges at 20, 40, 60, 80 percentiles)
        percentile_quantiles = np.linspace(0, 100, self.num_attribute_bins + 1)[1:-1]

        # For std-dev based binning (5 bins need 4 edges at mu +/- 0.2sigma, mu +/- 0.9sigma)
        std_dev_multipliers = [-0.9, -0.2, 0.2, 0.9] 
        if self.num_attribute_bins != (len(std_dev_multipliers) + 1):
            # This warning ensures consistency if self.num_attribute_bins is ever changed from 5
            # without updating std_dev_multipliers. For now, they match.
            print(f"Warning: self.num_attribute_bins ({self.num_attribute_bins}) does not match "
                  f"the std-dev multiplier count which implies {len(std_dev_multipliers) + 1} bins. "
                  f"Std-dev binning might not produce exactly {self.num_attribute_bins} bins.")


        for attr_name in attributes_to_bin:
            current_values = np.array([b_info["attributes"].get(attr_name) for b_info in all_bar_attributes_for_binning
                                       if b_info["attributes"].get(attr_name) is not None and 
                                       np.isfinite(b_info["attributes"].get(attr_name))])
            
            edges = np.array([]) # Initialize edges

            if len(current_values) < self.num_attribute_bins: # Not enough data for robust binning
                print(f"Warning: Not enough finite data points for '{attr_name}' ({len(current_values)}) to create "
                      f"{self.num_attribute_bins} distinct bins reliably. Using simplified fallback strategy.")
                unique_sorted_vals = np.sort(np.unique(current_values))
                if len(unique_sorted_vals) > 1 :
                    # Try to create num_attribute_bins-1 edges using available unique values
                    if len(unique_sorted_vals) <= self.num_attribute_bins -1 :
                        # Fewer unique values than edges needed, use midpoints if possible
                        edges = np.array([(unique_sorted_vals[j] + unique_sorted_vals[j+1])/2.0 
                                          for j in range(len(unique_sorted_vals)-1)])
                    else: # More unique values than edges needed, pick some as edges
                        indices = np.linspace(0, len(unique_sorted_vals)-1, self.num_attribute_bins).astype(int) # num_edges = num_bins -1
                        # Make sure indices are for edges, so num_bins points for N-1 edges
                        # np.linspace for N-1 edges would be num_attribute_bins points, then select from them.
                        # Example: 5 bins -> 4 edges. linspace(0, len-1, 5) -> pick inner 4 if len allows.
                        # This part is tricky. Simpler: just take some percentiles of the few unique values.
                        edges = np.percentile(unique_sorted_vals, percentile_quantiles)

                elif len(unique_sorted_vals) == 1: # Only one unique value
                    val = unique_sorted_vals[0]
                    eps_spacing = 1e-3 * (abs(val) if abs(val) > 1e-6 else 1.0)
                    # Create artificial edges around this single value
                    if self.num_attribute_bins == 5:
                         edges = np.array([val - 2*eps_spacing, val - eps_spacing, val + eps_spacing, val + 2*eps_spacing])
                    else: # Generic fallback for other num_bins
                         edges = np.linspace(val - eps_spacing*2, val + eps_spacing*2, self.num_attribute_bins +1)[1:-1]
                else: # No values at all
                    print(f"No valid values found for attribute '{attr_name}'. Using broad default edges.")
                    edges = np.linspace(-1, 1, self.num_attribute_bins+1)[1:-1] if self.num_attribute_bins > 1 else np.array([])


            elif attr_name == "rel_pitch_class_entropy_ratio":
                print(f"Calculating bin edges for '{attr_name}' using Percentile (Equal Frequency) based method...")
                edges = np.percentile(current_values, percentile_quantiles)
            else: # For other attributes, use standard deviation based method
                print(f"Calculating bin edges for '{attr_name}' using Standard Deviation based method...")
                mean_val = np.mean(current_values)
                std_val = np.std(current_values)

                if std_val < 1e-6: 
                    print(f"  Warning: Std is ~0 for '{attr_name}'. Values are likely constant (Mean={mean_val:.4f}). Creating artificial small bins around the mean.")
                    eps_std = 1e-3 * (abs(mean_val) if abs(mean_val) > 1e-6 else 1.0)
                    # Ensure we create self.num_attribute_bins - 1 edges
                    if self.num_attribute_bins == 5: # Specific for 5 bins / 4 std_multipliers
                        edges = np.array([mean_val + m * eps_std for m in std_dev_multipliers])
                        if np.allclose(edges, mean_val): # If still collapsed
                            edges = np.array([mean_val - 2*eps_std, mean_val - eps_std, mean_val + eps_std, mean_val + 2*eps_std])
                    else: # Generic fallback for other num_bins
                        edges = np.linspace(mean_val - eps_std*2, mean_val + eps_std*2, self.num_attribute_bins+1)[1:-1]
                else: 
                    # Ensure std_dev_multipliers length matches num_attribute_bins - 1
                    if len(std_dev_multipliers) == self.num_attribute_bins -1 :
                        edges = np.array([mean_val + m * std_val for m in std_dev_multipliers])
                    else:
                        print(f"  Warning: Mismatch between num_attribute_bins ({self.num_attribute_bins}) and std_dev_multipliers. Using linear spacing for std-dev edges.")
                        # Fallback to a linear spacing of std deviations if multipliers don't match
                        num_edges_needed = self.num_attribute_bins -1
                        # Create num_edges_needed points, e.g. from -1.5*std to 1.5*std
                        edge_multipliers_fallback = np.linspace(-1.5, 1.5, num_edges_needed) if num_edges_needed > 0 else []
                        edges = np.array([mean_val + m * std_val for m in edge_multipliers_fallback])
            
            bin_edges_map[attr_name] = np.sort(np.unique(edges)) # Ensure edges are unique and sorted
            effective_num_edges = len(bin_edges_map[attr_name])
            target_num_edges = self.num_attribute_bins - 1

            print(f"  Bin edges for {attr_name}: {np.round(bin_edges_map[attr_name], 4)}")
            print(f"  (Target {self.num_attribute_bins} bins from {target_num_edges} edges, "
                  f"got {effective_num_edges + 1} effective bins from {effective_num_edges} unique edges)")
            
            if effective_num_edges != target_num_edges and len(current_values) >= self.num_attribute_bins:
                 print(f"    Warning: Could not create exactly {target_num_edges} unique edges for {attr_name}. "
                       f"This might be due to data distribution (e.g., many identical values at calculated edge points).")
        return bin_edges_map


    def _get_attribute_bin_id(self, value: float, attr_name: str) -> int:
        edges = self.attribute_bin_edges.get(attr_name)
        if edges is None or len(edges) == 0: return self.num_attribute_bins // 2
        bin_id = np.digitize(value, edges).item()
        return max(0, min(bin_id, len(edges))) 

    
    @staticmethod
    def get_attributes_for_model() -> List[str]:
        return ["rel_pitch_coverage_ratio",
                "rel_note_per_pos_ratio",
                "rel_pitch_class_entropy_ratio"]
    

    def _prepare_samples_with_rolling_context(self, all_songs_filtered_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_training_chunks = []
        model_attribute_keys = self.get_attributes_for_model()
        short_attr_names = [key.replace("rel_","").replace("_ratio","").replace("_log","") for key in model_attribute_keys]
        
        empty_bar_ids = [self.bar_bos_id, self.bar_eos_id]
        neutral_binned_attrs = {s_name: self.num_attribute_bins // 2 for s_name in short_attr_names}

        for song_data in tqdm(all_songs_filtered_data, desc="Phase 4: Preparing samples with rolling context"):
            cond_bars_ids_song = song_data["cond_bar_ids_list"] 
            tgt_bars_ids_song = song_data["tgt_bar_ids_list"]   
            bar_attributes_info_song = song_data["bar_attributes_filtered"] 

            if not cond_bars_ids_song: continue 
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

                for k_slot_idx in range(self.context_num_past_xy_pairs):
                    actual_history_pair_idx = i - (self.context_num_past_xy_pairs - k_slot_idx)
                    if actual_history_pair_idx >= 0: 
                        past_x_ids = cond_bars_ids_song[actual_history_pair_idx]
                        past_y_ids = tgt_bars_ids_song[actual_history_pair_idx]
                        # Use "attributes" key which holds the dict of raw attribute values
                        past_bar_pair_original_attributes = bar_attributes_info_song[actual_history_pair_idx]["attributes"]
                        past_bar_pair_binned_attrs = {} 
                        for attr_idx_hist, attr_name_full_hist in enumerate(model_attribute_keys):
                            short_name_hist = short_attr_names[attr_idx_hist]
                            binned_id_hist = self._get_attribute_bin_id(past_bar_pair_original_attributes[attr_name_full_hist], attr_name_full_hist)
                            past_bar_pair_binned_attrs[short_name_hist] = binned_id_hist
                        
                        context_token_ids_list.extend(past_x_ids)
                        context_class_ids_list.extend([COND_CLASS_ID] * len(past_x_ids))
                        for short_name_attr in short_attr_names:
                            context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend(
                                [past_bar_pair_binned_attrs[short_name_attr]] * len(past_x_ids))
                        context_token_ids_list.extend(past_y_ids)
                        context_class_ids_list.extend([TGT_CLASS_ID] * len(past_y_ids))
                        for short_name_attr in short_attr_names:
                            context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend(
                                [past_bar_pair_binned_attrs[short_name_attr]] * len(past_y_ids))
                    else: 
                        context_token_ids_list.extend(empty_bar_ids)
                        context_class_ids_list.extend([COND_CLASS_ID] * len(empty_bar_ids))
                        for short_name_attr in short_attr_names:
                            context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend(
                                [neutral_binned_attrs[short_name_attr]] * len(empty_bar_ids))
                        context_token_ids_list.extend(empty_bar_ids)
                        context_class_ids_list.extend([TGT_CLASS_ID] * len(empty_bar_ids))
                        for short_name_attr in short_attr_names:
                            context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend(
                                [neutral_binned_attrs[short_name_attr]] * len(empty_bar_ids))
                
                context_token_ids_list.extend(current_xi_ids)
                context_class_ids_list.extend([COND_CLASS_ID] * len(current_xi_ids))
                for short_name_attr in short_attr_names:
                    context_attribute_ids_dict[f"{short_name_attr}_bin_ids"].extend(
                        [current_bar_pair_binned_attrs[short_name_attr]] * len(current_xi_ids))
                
                input_ids_list_sample = context_token_ids_list + current_yi_ids
                class_ids_list_sample = context_class_ids_list + [TGT_CLASS_ID] * len(current_yi_ids)
                labels_list_sample = [-100] * len(context_token_ids_list)
                for k_y in range(len(current_yi_ids)):
                    if k_y == len(current_yi_ids) - 1: labels_list_sample.append(-100)
                    else: labels_list_sample.append(current_yi_ids[k_y+1])
                
                attribute_ids_lists_for_sample = {}
                for short_name_attr in short_attr_names:
                    key_for_dict = f"{short_name_attr}_bin_ids"
                    attr_lookup_key = short_name_attr 
                    attribute_ids_lists_for_sample[key_for_dict] = \
                        context_attribute_ids_dict[key_for_dict] + \
                        [current_bar_pair_binned_attrs[attr_lookup_key]] * len(current_yi_ids)

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
        return len(self.processed_samples)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.processed_samples): raise IndexError("Out of bounds")
        return self.processed_samples[idx]
    
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]: # Unchanged
        batch = [item for item in batch if item and item.get("input_ids")]
        model_attribute_keys = self.get_attributes_for_model() # Get full names
        # Create short names for dict keys from full attribute names
        short_attr_names = [key.replace("rel_","").replace("_ratio","").replace("_log","")
                            for key in model_attribute_keys]
        model_attr_bin_id_keys = [f"{sname}_bin_ids" for sname in short_attr_names]

        all_keys_for_empty_batch = ["input_ids", "attention_mask", "class_ids", "labels"] + model_attr_bin_id_keys

        if not batch: return {key: torch.empty(0,0,dtype=torch.long) for key in all_keys_for_empty_batch}
        
        current_max_len_in_batch = max(len(item["input_ids"]) for item in batch)
        batch_data = defaultdict(list)
        keys_from_getitem = ["input_ids", "class_ids", "labels"] + model_attr_bin_id_keys

        for item in batch:
            s_len = len(item["input_ids"])
            p_len = current_max_len_in_batch - s_len 
            for key in keys_from_getitem:
                if key not in item: 
                    pad_val = -100 if key=="labels" else (self.pad_id if key=="input_ids" else (PAD_CLASS_ID if key=="class_ids" else ATTRIBUTE_PAD_ID))
                    batch_data[key].append([pad_val]*current_max_len_in_batch); continue
                pad_val_map = {"labels":-100, "input_ids":self.pad_id, "class_ids":PAD_CLASS_ID}
                pad_val = pad_val_map.get(key, ATTRIBUTE_PAD_ID)
                batch_data[key].append(item[key] + [pad_val]*p_len)
            batch_data["attention_mask"].append([1]*s_len + [0]*p_len)
        
        if not batch_data["input_ids"]: return self.collate_fn([])
        return {key: torch.tensor(val, dtype=torch.long) for key, val in batch_data.items()}


    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0, **kwargs) -> DataLoader:
        if not self.processed_samples: return DataLoader([]) # Handle empty dataset
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=self.collate_fn, pin_memory=torch.cuda.is_available() and num_workers > 0,
            **kwargs
        )
    

    def _print_additional_stats(self, all_bar_attributes_for_stats: List[Dict[str, Any]]): # Unchanged from last version
        print("\n--- Additional Dataset Statistics (based on filtered bar pairs before context rolling/chunking) ---")
        cond_bar_lengths = [b["cond_bar_len"] for b in all_bar_attributes_for_stats if "cond_bar_len" in b]
        tgt_bar_lengths = [b["tgt_bar_len"] for b in all_bar_attributes_for_stats if "tgt_bar_len" in b]

        if cond_bar_lengths:
             print(f"  Cond Bar Lens (Cnt:{len(cond_bar_lengths)}): Mean={np.mean(cond_bar_lengths):.1f},Std={np.std(cond_bar_lengths):.1f},Min={np.min(cond_bar_lengths)},Max={np.max(cond_bar_lengths)},Med={np.median(cond_bar_lengths)}")
        if tgt_bar_lengths:
             print(f"  Tgt Bar Lens (Cnt:{len(tgt_bar_lengths)}): Mean={np.mean(tgt_bar_lengths):.1f},Std={np.std(tgt_bar_lengths):.1f},Min={np.min(tgt_bar_lengths)},Max={np.max(tgt_bar_lengths)},Med={np.median(tgt_bar_lengths)}")
        
        interleaved_bar_pair_lens = [b["cond_bar_len"] + b["tgt_bar_len"] for b in all_bar_attributes_for_stats if "cond_bar_len" in b and "tgt_bar_len" in b]
        if interleaved_bar_pair_lens:
            print(f"  Original Interleaved Bar-Pair Lens (Cnt:{len(interleaved_bar_pair_lens)}): Mean={np.mean(interleaved_bar_pair_lens):.1f},Std={np.std(interleaved_bar_pair_lens):.1f},Min={np.min(interleaved_bar_pair_lens)},Max={np.max(interleaved_bar_pair_lens)},Med={np.median(interleaved_bar_pair_lens)}")
        
        if hasattr(self, 'attribute_bin_edges') and all_bar_attributes_for_stats:
            print("  Attribute Bin Distribution (for filtered bar pairs, based on StdDev bins):")
            model_attribute_keys = self.get_attributes_for_model()
            short_attr_names = [key.replace("rel_","").replace("_ratio","").replace("_log","").replace("_entropy","") 
                                for key in model_attribute_keys] 
            
            for idx, attr_name_model in enumerate(model_attribute_keys):
                display_name = short_attr_names[idx]
                binned_ids_for_attr = []
                for bar_info in all_bar_attributes_for_stats:
                    raw_value = bar_info["attributes"].get(attr_name_model)
                    if raw_value is not None and np.isfinite(raw_value):
                        binned_ids_for_attr.append(self._get_attribute_bin_id(raw_value, attr_name_model))
                
                if binned_ids_for_attr:
                    counts = np.bincount(binned_ids_for_attr, minlength=self.num_attribute_bins)
                    print(f"    {display_name} (target {self.num_attribute_bins} bins, ID 0-{self.num_attribute_bins-1}):")
                    for b_idx, count in enumerate(counts):
                        if b_idx < self.num_attribute_bins: print(f"      Bin {b_idx}: {count}")
                else:
                    print(f"    No valid data for {display_name} ({attr_name_model}) to show bin distribution.")
        
        if hasattr(self, 'total_dataset_tokens'):
            print(f"  Total tokens in final processed_samples (sum of input_ids lengths): {self.total_dataset_tokens:,}")


def get_attribute_statistics_from_raw( 
    all_bar_attributes_raw: List[Dict[str, Any]], 
    filtered_bar_attributes_for_binning: List[Dict[str, Any]], 
    bin_edges: Dict[str, np.ndarray] = None,
    num_bins_target: int = 5 
) -> Dict[str, Any]: # Unchanged (but will reflect new attributes)
    stats = {
        "raw_stats_all_bar_pairs_before_filtering": {}, 
        "stats_of_data_used_for_binning_after_filtering": {},
        "bin_edges_used": {}, 
        "binned_distribution_of_filtered_data": {}
    }
    current_model_attributes = EtudeDataset.get_attributes_for_model()
    attributes_to_calculate_raw_stats_for = current_model_attributes + ["avg_note_overlap_ratio"]
    
    if bin_edges:
        for k, v_array in bin_edges.items():
            if k in current_model_attributes: 
                if v_array is not None and hasattr(v_array, 'tolist'): stats["bin_edges_used"][k] = [round(x,4) for x in v_array.tolist()]
                elif v_array is not None: stats["bin_edges_used"][k] = [round(x,4) for x in v_array]
                else: stats["bin_edges_used"][k] = None
    
    print("\n--- Calculating Raw Attribute Statistics (All Bar Pairs, BEFORE any filtering) ---")
    for attr_name in attributes_to_calculate_raw_stats_for:
        raw_values = np.array([b_info["attributes"].get(attr_name) for b_info in all_bar_attributes_raw 
                               if b_info.get("attributes", {}).get(attr_name) is not None and 
                               np.isfinite(b_info.get("attributes", {}).get(attr_name))])
        if len(raw_values) > 0:
            stats["raw_stats_all_bar_pairs_before_filtering"][attr_name] = {
                "mean": float(np.mean(raw_values)), "std": float(np.std(raw_values)),
                "min": float(np.min(raw_values)), "max": float(np.max(raw_values)),
                "median": float(np.median(raw_values)), "count": int(len(raw_values)),
                "percentiles": {"5th":float(np.percentile(raw_values,5)), "25th":float(np.percentile(raw_values,25)),
                                "75th":float(np.percentile(raw_values,75)), "95th":float(np.percentile(raw_values,95))} }
        else: print(f"No valid raw values for {attr_name} before filtering.")

    if filtered_bar_attributes_for_binning:
        print("\n--- Calculating Statistics of Data Used for Binning (AFTER overlap filtering) ---")
        for attr_name in attributes_to_calculate_raw_stats_for: 
            filtered_values = np.array([b_info["attributes"].get(attr_name) for b_info in filtered_bar_attributes_for_binning 
                                        if b_info.get("attributes", {}).get(attr_name) is not None and
                                        np.isfinite(b_info.get("attributes", {}).get(attr_name))])
            if len(filtered_values) > 0:
                stats["stats_of_data_used_for_binning_after_filtering"][attr_name] = {
                    "mean": float(np.mean(filtered_values)), "std": float(np.std(filtered_values)),
                    "min": float(np.min(filtered_values)), "max": float(np.max(filtered_values)),
                    "median": float(np.median(filtered_values)), "count": int(len(filtered_values)),
                    "percentiles": {"5th":float(np.percentile(filtered_values,5)),"25th":float(np.percentile(filtered_values,25)),
                                    "75th":float(np.percentile(filtered_values,75)),"95th":float(np.percentile(filtered_values,95))} }
            else: print(f"No valid filtered values for {attr_name} for binning stats.")

    if bin_edges and filtered_bar_attributes_for_binning:
        print("\n--- Calculating Binned Distribution (for model attributes, based on data AFTER overlap filtering, using StdDev bins) ---")
        for attr_name in current_model_attributes: 
            current_attr_edges = bin_edges.get(attr_name)
            if current_attr_edges is None or not hasattr(current_attr_edges, '__len__') or len(current_attr_edges) == 0: 
                print(f"No valid edges for {attr_name}, skipping binned distribution.")
                continue
            
            values_for_binning = np.array([b_info["attributes"].get(attr_name) for b_info in filtered_bar_attributes_for_binning 
                                           if b_info.get("attributes", {}).get(attr_name) is not None and
                                           np.isfinite(b_info.get("attributes", {}).get(attr_name))])
            if len(values_for_binning) == 0: 
                print(f"No valid values for {attr_name} to bin, skipping binned distribution.")
                continue

            binned_indices = np.digitize(values_for_binning, current_attr_edges)
            actual_num_bins_from_edges = len(current_attr_edges) + 1 
            
            if num_bins_target != actual_num_bins_from_edges :
                 print(f"Warning: Target bins ({num_bins_target}) for stats of {attr_name} "
                       f"does not match effective bins from edges ({actual_num_bins_from_edges}). "
                       f"Using effective bins for stats count: {actual_num_bins_from_edges}")
            
            counts = np.bincount(binned_indices, minlength=actual_num_bins_from_edges)
            stats["binned_distribution_of_filtered_data"][attr_name] = {
                f"bin_{k}": int(counts[k]) for k in range(actual_num_bins_from_edges) 
            }
    return stats