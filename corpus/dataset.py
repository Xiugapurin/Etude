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
            self.sample_map = []
            return

        print("Phase 1: Collecting initial song data...")
        self.all_songs_data = self._collect_initial_song_data(raw_file_pairs)
        if not self.all_songs_data:
            self.sample_map = []
            return

        all_attributes_for_binning = [
            bar_attr_info
            for song_data in self.all_songs_data
            for bar_attr_info in song_data["bar_attributes"]
        ]
        
        if not all_attributes_for_binning:
            self.sample_map = []
            return
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


    def __len__(self) -> int:
        return len(self.sample_map)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.sample_map):
            raise IndexError("Index out of bounds")

        map_entry = self.sample_map[idx]
        song_idx, bar_idx, chunk_slice = map_entry["song_idx"], map_entry["bar_idx"], map_entry["slice"]

        full_sample = self._get_full_sample_for_bar(song_idx, bar_idx)

        chunked_sample = {key: value[chunk_slice] for key, value in full_sample.items()}
        
        return chunked_sample

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
        note_count, pos_event_count, total_duration_in_16ths = 0, 0, 0
        notes_by_position, pos_values = defaultdict(list), []

        for i, ev in enumerate(events):
            if ev.type_ == "Pos" and isinstance(ev.value, int):
                pos_event_count += 1
                current_pos = ev.value
                pos_values.append(ev.value)
            elif ev.type_ == "Note" and isinstance(ev.value, int):
                note_count += 1
                if 'current_pos' in locals():
                    notes_by_position[current_pos].append(ev.value)
            elif ev.type_ == "Duration" and isinstance(ev.value, int):
                total_duration_in_16ths += ev.value
        
        return {
            "note_count": note_count, 
            "pos_event_count": pos_event_count, 
            "notes_by_position": notes_by_position,
            "total_duration_in_16ths": total_duration_in_16ths, 
            "pos_values": sorted(list(set(pos_values)))
        }
    
    @staticmethod
    def _calculate_avg_silence(pos_values: list, max_pos: int) -> float:
        if not pos_values and max_pos == 0: return 0.0
        
        positions = pos_values.copy()
        if not positions or positions[0] != 0: positions.insert(0, 0)
        if positions[-1] != max_pos: positions.append(max_pos)
        
        gaps = np.diff(positions)
        return np.mean(gaps) if len(gaps) > 0 else 0.0
    

    def _calculate_raw_relative_attributes(self, cond_bar_features: dict, tgt_bar_features: dict) -> dict:
        attrs = {}
        
        # Attribute 1: avg_note_overlap_ratio (unchanged)
        cnbp, tnbp = cond_bar_features["notes_by_position"], tgt_bar_features["notes_by_position"]
        all_pos = set(cnbp.keys()) | set(tnbp.keys())
        if not all_pos: attrs["avg_note_overlap_ratio"] = 0.0
        else:
            ratios = [(sum(1 for t in tnbp.get(p,[]) if t%12 in {c%12 for c in cnbp.get(p,[])}) / len(tnbp[p])) if tnbp.get(p) else (1.0 if not cnbp.get(p) else 0.0) for p in all_pos]
            attrs["avg_note_overlap_ratio"] = np.mean(ratios) if ratios else 0.0
        
        # Attribute 2: rel_note_per_pos_ratio (no log, new 0-handling)
        cond_npp = cond_bar_features["note_count"] / cond_bar_features["pos_event_count"] if cond_bar_features["pos_event_count"] > 0 else 0
        tgt_npp = tgt_bar_features["note_count"] / tgt_bar_features["pos_event_count"] if tgt_bar_features["pos_event_count"] > 0 else 0
        if cond_npp == 0 and tgt_npp == 0: attrs["rel_note_per_pos_ratio"] = 1.0
        elif cond_npp == 0 or tgt_npp == 0: attrs["rel_note_per_pos_ratio"] = 0.0
        else: attrs["rel_note_per_pos_ratio"] = tgt_npp / cond_npp

        # Attribute 3: rel_avg_duration_ratio (new, no log)
        avg_dur_cond = cond_bar_features["total_duration_in_16ths"] / cond_bar_features["note_count"] if cond_bar_features["note_count"] > 0 else 0
        avg_dur_tgt = tgt_bar_features["total_duration_in_16ths"] / tgt_bar_features["note_count"] if tgt_bar_features["note_count"] > 0 else 0
        if avg_dur_cond == 0 and avg_dur_tgt == 0: attrs["rel_avg_duration_ratio"] = 1.0
        elif avg_dur_cond == 0 or avg_dur_tgt == 0: attrs["rel_avg_duration_ratio"] = 0.0
        else: attrs["rel_avg_duration_ratio"] = avg_dur_tgt / avg_dur_cond

        # Attribute 4: rel_avg_silence_ratio (new, no log)
        cond_pos, tgt_pos = cond_bar_features["pos_values"], tgt_bar_features["pos_values"]
        max_pos = 0
        if cond_pos: max_pos = max(max_pos, cond_pos[-1])
        if tgt_pos: max_pos = max(max_pos, tgt_pos[-1])
        avg_silence_cond = self._calculate_avg_silence(cond_pos, max_pos)
        avg_silence_tgt = self._calculate_avg_silence(tgt_pos, max_pos)
        if avg_silence_cond == 0 and avg_silence_tgt == 0: attrs["rel_avg_silence_ratio"] = 1.0
        elif avg_silence_cond == 0 or avg_silence_tgt == 0: attrs["rel_avg_silence_ratio"] = 0.0
        else: attrs["rel_avg_silence_ratio"] = avg_silence_tgt / avg_silence_cond
        
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
        
        for attr_name in attributes_to_bin:
            # --- Select std multipliers based on attribute name ---
            if attr_name in ["rel_note_per_pos_ratio", "rel_avg_silence_ratio"]:
                std_dev_multipliers = [-0.5, 0.5]
                print(f"Calculating bin edges for '{attr_name}' using StdDev method (±0.5 std)...")
            elif attr_name == "rel_avg_duration_ratio" or attr_name == "avg_note_overlap_ratio":
                std_dev_multipliers = [-0.7, 0.7]
                print(f"Calculating bin edges for '{attr_name}' using StdDev method (±0.7 std)...")
            else:
                std_dev_multipliers = [-1.0, 1.0]
                print(f"Calculating bin edges for '{attr_name}' using StdDev method (±1.0 std)...")

            current_values = np.array([b["attributes"].get(attr_name) for b in all_bar_attributes_for_binning if b["attributes"].get(attr_name) is not None and np.isfinite(b["attributes"].get(attr_name))])
            
            if len(current_values) < self.num_attribute_bins:
                edges = np.array([-0.5, 0.5]) # Fallback
            else:
                mean_val, std_val = np.mean(current_values), np.std(current_values)
                if std_val < 1e-6:
                    eps = 1e-3 * (abs(mean_val) if abs(mean_val) > 1e-6 else 1.0)
                    edges = np.array([mean_val - eps, mean_val + eps])
                else:
                    edges = np.array([mean_val + m * std_val for m in std_dev_multipliers])
            
            bin_edges_map[attr_name] = np.sort(np.unique(edges))
        return bin_edges_map
    

    def _get_attribute_bin_id(self, value: float, attr_name: str) -> int:
        edges = self.attribute_bin_edges.get(attr_name)
        if edges is None or len(edges) == 0: return 1
        bin_id = np.digitize(value, edges).item()
        return max(0, min(bin_id, len(edges)))

    
    @staticmethod
    def get_attributes_for_model() -> List[str]:
        return [
            "avg_note_overlap_ratio",
            "rel_note_per_pos_ratio",
            "rel_avg_duration_ratio",
            "rel_avg_silence_ratio"
        ]
    

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


    def _get_full_sample_for_bar(self, song_idx: int, bar_idx: int) -> Dict[str, List[Any]]:
        song_data = self.all_songs_data[song_idx]
        bar_infos = song_data["bar_attributes"]
        model_attr_keys = self.get_attributes_for_model()
        short_name_map = {
            "avg_note_overlap_ratio": "note_overlap", 
            "rel_note_per_pos_ratio": "note_per_pos",
            "rel_avg_duration_ratio": "avg_dur", 
            "rel_avg_silence_ratio": "avg_sil"
        }

        short_names = [short_name_map[k] for k in model_attr_keys]
        neutral_binned_attrs = {s_name: 1 for s_name in short_names}
        empty_bar = [self.bar_bos_id, self.bar_eos_id]
        context_tokens, context_classes, context_attrs = [], [], defaultdict(list)

        for k in range(self.context_num_past_xy_pairs):
            hist_idx = bar_idx - (self.context_num_past_xy_pairs - k)
            if hist_idx >= 0:
                past_x, past_y = bar_infos[hist_idx]["cond_bar_ids"], bar_infos[hist_idx]["tgt_bar_ids"]
                past_attrs = {s: self._get_attribute_bin_id(bar_infos[hist_idx]["attributes"][k_full], k_full) for s, k_full in zip(short_names, model_attr_keys)}
                for item_ids, class_id in [(past_x, COND_CLASS_ID), (past_y, TGT_CLASS_ID)]:
                    context_tokens.extend(item_ids); 
                    context_classes.extend([class_id] * len(item_ids))
                    for s in short_names: 
                        context_attrs[f"{s}_bin_ids"].extend([past_attrs[s]] * len(item_ids))
            else:
                for _ in range(2):
                    context_tokens.extend(empty_bar); 
                    context_classes.extend([COND_CLASS_ID] * len(empty_bar))
                    for s in short_names: 
                        context_attrs[f"{s}_bin_ids"].extend([neutral_binned_attrs[s]] * len(empty_bar))

        current_xi, current_yi = bar_infos[bar_idx]["cond_bar_ids"], bar_infos[bar_idx]["tgt_bar_ids"]
        current_attrs = {s: self._get_attribute_bin_id(bar_infos[bar_idx]["attributes"][k_full], k_full) for s, k_full in zip(short_names, model_attr_keys)}
        
        context_tokens.extend(current_xi); 
        context_classes.extend([COND_CLASS_ID] * len(current_xi))

        for s in short_names: 
            context_attrs[f"{s}_bin_ids"].extend([current_attrs[s]] * len(current_xi))
        
        full_sample = {
            "input_ids": context_tokens + current_yi, 
            "class_ids": context_classes + [TGT_CLASS_ID] * len(current_yi),
            "labels": [-100] * len(context_tokens) + current_yi[1:] + [-100]
        }

        for s in short_names: 
            context_attrs[f"{s}_bin_ids"].extend([current_attrs[s]] * len(current_yi))
        full_sample.update(context_attrs)

        return full_sample

    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = [item for item in batch if item and item.get("input_ids")]
        if not batch: return {}
        max_len = max(len(item["input_ids"]) for item in batch)
        batch_data = defaultdict(list)
        short_name_map = {
            "avg_note_overlap_ratio": "note_overlap", 
            "rel_note_per_pos_ratio": "note_per_pos",
            "rel_avg_duration_ratio": "avg_dur", 
            "rel_avg_silence_ratio": "avg_sil"
        }

        short_names = [short_name_map[k] for k in self.get_attributes_for_model()]
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
        print("\n--- Chunk Length Statistics ---")
        if not self.sample_map:
            print("  No samples available to generate statistics.")
            return
        sample_size = min(10000, len(self.sample_map))
        print(f"  (Calculating on a random sample of {sample_size} chunks for efficiency...)")
        random_indices = random.sample(range(len(self.sample_map)), sample_size)
        chunk_lengths = [len(self[i]["input_ids"]) for i in tqdm(random_indices, desc="Analyzing chunk lengths")]
        if chunk_lengths:
            lengths_np = np.array(chunk_lengths)
            print(f"  - Count: {len(lengths_np)}")
            print(f"  - Avg: {np.mean(lengths_np):.2f}, Std: {np.std(lengths_np):.2f}")
            print(f"  - Min: {np.min(lengths_np)}, Max: {np.max(lengths_np)}")
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
