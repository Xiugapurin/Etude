# etude/data/dataset.py

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict
import pprint

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .vocab import Vocab

SRC_CLASS_ID = 1
TGT_CLASS_ID = 2
PAD_CLASS_ID = 0
ATTRIBUTE_PAD_ID = 0


class EtudeDataset(Dataset):
    """
    The EtudeDataset class is responsible for loading, preprocessing, and serving paired musical sequences for model training.
    """
    
    _MODEL_ATTRIBUTES = [
        "relative_polyphony",
        "relative_rhythmic_intensity",
        "relative_note_sustain",
        "pitch_overlap_ratio"
    ]
    
    _ATTRIBUTE_SHORT_NAME_MAP = {
        "relative_polyphony": "polyphony",
        "relative_rhythmic_intensity": "rhythm_intensity",
        "relative_note_sustain": "sustain",
        "pitch_overlap_ratio": "pitch_overlap"
    }

    def __init__(self,
            dataset_dir: Union[str, Path],
            vocab: 'Vocab',
            max_seq_len: int,
            src_suffix: str = '_src.npy',
            tgt_suffix: str = '_tgt.npy',
            data_format: str = 'npy',
            num_attribute_bins: int = 3,
            context_num_past_xy_pairs: int = 4,
            verbose: bool = False
        ):
        """
        Initializes the Dataset.

        Args:
            dataset_dir (Union[str, Path]): Path to the directory containing processed data.
            vocab (Vocab): The vocabulary object.
            max_seq_len (int): The maximum sequence length for each training chunk.
            src_suffix (str): File suffix for the condition sequence.
            tgt_suffix (str): File suffix for the target sequence.
            data_format (str): Storage format of sequence files ('npy', 'pt', 'json').
            num_attribute_bins (int): The number of bins for quantizing musical attributes.
            context_num_past_xy_pairs (int): The number of past (X, Y) bar pairs to use as context.
            verbose (bool): If True, prints detailed dataset statistics during initialization.
        """
        # --- Parameter Setup ---
        self.dataset_dir = Path(dataset_dir)
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.src_suffix = src_suffix
        self.tgt_suffix = tgt_suffix
        self.data_format = data_format
        self.num_attribute_bins = num_attribute_bins
        self.context_num_past_xy_pairs = context_num_past_xy_pairs
        self.verbose = verbose

        # --- Fetch Special Token IDs from Vocabulary ---
        self.pad_id = self.vocab.get_pad_id()
        self.bar_bos_id = self.vocab.get_bar_bos_id()
        self.bar_eos_id = self.vocab.get_bar_eos_id()

        if self.pad_id == -1: 
            raise ValueError("'<PAD>' not found in vocabulary.")
        if self.bar_bos_id == -1 or self.bar_eos_id == -1: 
            raise ValueError("'Bar_BOS' or 'Bar_EOS' not found in vocab.")

        # --- Initialization Pipeline ---
        # Phase 1: Scan files and load metadata and raw attributes for all songs.
        print("Phase 1: Finding file pairs and collecting song metadata...")
        file_pairs = self._find_file_pairs()
        if not file_pairs:
            self._songs = []
            self.sample_map = []
            print("Warning: No valid data file pairs found.")
            return

        self._songs = self._load_and_preprocess_songs(file_pairs)
        if not self._songs:
            self.sample_map = []
            print("Warning: No valid songs could be processed.")
            return

        all_bar_data = [
            bar_data for song in self._songs for bar_data in song["bars"]
        ]
        
        # Phase 2: Calculate quantization bin edges for attributes based on global statistics.
        print(f"Phase 2: Calculating bin edges for {self.num_attribute_bins} bins...")
        self.attribute_bin_edges = self._calculate_bin_edges(all_bar_data)
        
        if self.verbose:
            self._print_dataset_statistics(all_bar_data)

        # Phase 3: Create the sample map for fast lookup in __getitem__.
        print("Phase 3: Creating sample map for lazy loading...")
        self._create_sample_map()
        
        if self.verbose and self.sample_map:
            self._print_chunk_stats()
            
        print(f"\nDataset initialized. Total training samples (chunks): {len(self.sample_map)}")


    def __len__(self) -> int:
        return len(self.sample_map)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Dynamically constructs and returns a training chunk based on its index."""
        if idx >= len(self.sample_map):
            raise IndexError("Index out of bounds")

        map_entry = self.sample_map[idx]
        song_idx, bar_idx, chunk_slice = map_entry["song_idx"], map_entry["bar_idx"], map_entry["slice"]

        # Dynamically generate the full, un-chunked sample for the bar.
        full_sample = self._get_full_sample_for_bar(song_idx, bar_idx)

        # Apply the slice to get the required chunk.
        chunked_sample = {key: value[chunk_slice] for key, value in full_sample.items()}
        
        return chunked_sample

    @classmethod
    def get_attributes_for_model(cls) -> List[str]:
        return cls._MODEL_ATTRIBUTES
    
    # --------------------------------------------------------------------------
    # Helper Methods for Initialization (__init__)
    # --------------------------------------------------------------------------

    def _find_file_pairs(self) -> List[Tuple[Path, Path]]:
        """Scans the data directory to find all valid (condition, target) file pairs."""
        print(f"Scanning {self.dataset_dir} for data subdirectories...")
        potential_dirs = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        
        file_pairs = []
        for item in tqdm(potential_dirs, desc="Finding file pairs"):
             subdir_name = item.name
             src_file = item / f"{subdir_name}{self.src_suffix}"
             tgt_file = item / f"{subdir_name}{self.tgt_suffix}"
             if src_file.exists() and tgt_file.exists():
                 file_pairs.append((src_file, tgt_file))
        print(f"Found {len(file_pairs)} valid file pairs (songs).")
        return file_pairs

    def _load_sequence(self, filepath: Path) -> List[int]:
        """Loads a token sequence from a single file."""
        if not filepath.exists(): return []
        try:
            if self.data_format == 'npy':
                return np.load(filepath, allow_pickle=True).tolist()
            elif self.data_format == 'pt':
                return torch.load(filepath).tolist()
            elif self.data_format == 'json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported data format: {self.data_format}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []

    def _split_into_bars(self, id_sequence: List[int]) -> List[List[int]]:
        """Splits a full token sequence into bars based on Bar_BOS/EOS tokens."""
        bars = []
        current_bar = []
        in_bar = False
        for token_id in id_sequence:
            if token_id == self.bar_bos_id:
                if in_bar and current_bar:
                    current_bar.append(self.bar_eos_id)
                    bars.append(current_bar)
                current_bar = [token_id]
                in_bar = True
            elif token_id == self.bar_eos_id:
                if in_bar:
                    current_bar.append(token_id)
                    bars.append(current_bar)
                    current_bar = []
                    in_bar = False
            elif in_bar:
                current_bar.append(token_id)
        
        if in_bar and current_bar:
            current_bar.append(self.bar_eos_id)
            bars.append(current_bar)
            
        return [b for b in bars if len(b) > 2] # Filter out malformed bars

    def _extract_bar_features(self, bar_ids: List[int]) -> Dict[str, Any]:
        """Extracts basic musical features from a bar's token IDs."""
        events = self.vocab.decode_sequence_to_events(bar_ids)
        note_count, pos_event_count, total_duration_in_16ths = 0, 0, 0
        notes_by_position = defaultdict(list)
        current_pos = -1

        for ev in events:
            if ev.type_ == "Pos" and isinstance(ev.value, int):
                pos_event_count += 1
                current_pos = ev.value
            elif ev.type_ == "Note" and isinstance(ev.value, int) and current_pos != -1:
                note_count += 1
                notes_by_position[current_pos].append(ev.value)
            elif ev.type_ == "Duration" and isinstance(ev.value, int):
                total_duration_in_16ths += ev.value
        
        return {
            "note_count": note_count, 
            "pos_event_count": pos_event_count, 
            "notes_by_position": notes_by_position,
            "total_duration_in_16ths": total_duration_in_16ths
        }
    
    def _compute_musical_attributes(self, src_features: dict, tgt_features: dict) -> dict:
        """Computes relative musical attributes between a condition bar and a target bar."""
        attrs = {}
        safe_div = lambda n, d, default = 0.0: n / d if d else default
        
        # Attribute 1: Relative Polyphony
        # Measures the change in the average number of notes per unique time event.
        # Reflects a shift in harmonic texture (e.g., towards denser chords or sparser arpeggios).
        src_npp = safe_div(src_features["note_count"], src_features["pos_event_count"])
        tgt_npp = safe_div(tgt_features["note_count"], tgt_features["pos_event_count"])
        attrs["relative_polyphony"] = safe_div(tgt_npp, src_npp, default=1.0)

        # Attribute 2: Relative Rhythmic Intensity
        # Measures the change in the density of rhythmic events (time positions) over time.
        # A higher value results in a more complex and rhythmically active phrase.
        attrs["relative_rhythmic_intensity"] = safe_div(tgt_features["pos_event_count"], src_features["pos_event_count"], default=1.0)
        
        # Attribute 3: Relative Note Sustain
        # Measures the change in the average note duration.
        # Controls the output's articulation character from legato (high) to staccato (low).
        src_avg_dur = safe_div(src_features["total_duration_in_16ths"], src_features["note_count"])
        tgt_avg_dur = safe_div(tgt_features["total_duration_in_16ths"], tgt_features["note_count"])
        attrs["relative_note_sustain"] = safe_div(tgt_avg_dur, src_avg_dur, default=1.0)

        # Attribute 4: Pitch Overlap Ratio (Auxiliary)
        # Measures the ratio of pitch classes in the target bar that are also present
        # at the same time positions in the condition bar.
        cnbp, tnbp = src_features["notes_by_position"], tgt_features["notes_by_position"]
        all_pos = set(cnbp.keys()) | set(tnbp.keys())
        if not all_pos:
            attrs["pitch_overlap_ratio"] = 0.0
        else:
            ratios = []
            for p in all_pos:
                if p in tnbp:
                    src_pitches = {c % 12 for c in cnbp.get(p, [])}
                    overlap_count = sum(1 for t in tnbp[p] if (t % 12) in src_pitches)
                    ratios.append(safe_div(overlap_count, len(tnbp[p])))
                else: 
                    ratios.append(0.0 if p in cnbp else 1.0)
            attrs["pitch_overlap_ratio"] = np.mean(ratios) if ratios else 0.0
        
        return attrs

    def _load_and_preprocess_songs(self, file_pairs: List[Tuple[Path, Path]]) -> List[Dict[str, Any]]:
        """Loads all songs and preprocesses them to compute attributes for each bar."""
        songs = []
        for src_f, tgt_f in tqdm(file_pairs, desc="Preprocessing songs"):
            c_ids, t_ids = self._load_sequence(src_f), self._load_sequence(tgt_f)
            if not c_ids or not t_ids: continue

            c_bars, t_bars = self._split_into_bars(c_ids), self._split_into_bars(t_ids)
            
            bar_data = []
            num_bars = min(len(c_bars), len(t_bars))
            for i in range(num_bars):
                c_feat = self._extract_bar_features(c_bars[i])
                t_feat = self._extract_bar_features(t_bars[i])
                raw_attrs = self._compute_musical_attributes(c_feat, t_feat)
                bar_data.append({
                    "attributes": raw_attrs,
                    "src_bar_ids": c_bars[i],
                    "tgt_bar_ids": t_bars[i]
                })

            if bar_data:
                songs.append({"song_name": src_f.parent.name, "bars": bar_data})

        return songs
    
    def _calculate_bin_edges(self, all_bar_data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Calculates the bin edges for attribute quantization based on the global distribution."""
        if not all_bar_data: 
            return {name: np.array([]) for name in self.get_attributes_for_model()}
        
        bin_edges_map = {}
        # Configuration for binning based on standard deviation multipliers.
        std_multipliers_config = {
            "relative_rhythmic_intensity": [-0.2, 0.2],
            "relative_polyphony": [-0.5, 0.5],
            "relative_note_sustain": [-0.7, 0.7],
            "pitch_overlap_ratio": [-0.7, 0.7]
        }
        default_multipliers = [-1.0, 1.0]

        for attr_name in self.get_attributes_for_model():
            multipliers = std_multipliers_config.get(attr_name, default_multipliers)
            
            values = np.array([
                bar["attributes"].get(attr_name) for bar in all_bar_data 
                if bar["attributes"].get(attr_name) is not None and np.isfinite(bar["attributes"][attr_name])
            ])
            
            if len(values) < 2:
                edges = np.array([-0.5, 0.5])
            else:
                mean, std = np.mean(values), np.std(values)
                if std < 1e-6:
                    eps = 1e-3 * (abs(mean) if abs(mean) > 1e-6 else 1.0)
                    edges = np.array([mean - eps, mean + eps])
                else:
                    edges = np.array([mean + m * std for m in multipliers])
            
            bin_edges_map[attr_name] = np.sort(np.unique(edges))

        return bin_edges_map

    def _get_attribute_bin_id(self, value: float, attr_name: str) -> int:
        """Converts a raw attribute value to its corresponding bin ID."""
        edges = self.attribute_bin_edges.get(attr_name)
        if edges is None or len(edges) == 0: return 1 # Default bin
        return np.digitize(value, edges).item()
    
    def _create_sample_map(self):
        """Creates a map from a flat integer index to a specific (song, bar, chunk) location."""
        self.sample_map = []
        empty_bar_len = 2  # [BOS, EOS]

        for song_idx, song in enumerate(tqdm(self._songs, desc="Creating sample map")):
            bars = song["bars"]
            if not bars: continue
            
            for bar_idx in range(len(bars)):
                # Step 1: Calculate the total length of the full sample for this bar (including context).
                context_len = sum(
                    len(bars[hist_idx]["src_bar_ids"]) + len(bars[hist_idx]["tgt_bar_ids"])
                    if (hist_idx := bar_idx - (self.context_num_past_xy_pairs - k)) >= 0
                    else 2 * empty_bar_len
                    for k in range(self.context_num_past_xy_pairs)
                )
                
                full_sample_len = context_len + len(bars[bar_idx]["src_bar_ids"]) + len(bars[bar_idx]["tgt_bar_ids"])

                # Step 2: Chunk the full sample and create a map entry for each chunk.
                for chunk_start in range(0, full_sample_len, self.max_seq_len):
                    chunk_end = min(chunk_start + self.max_seq_len, full_sample_len)
                    if chunk_end - chunk_start >= 2: # Ensure chunk is not too small
                        self.sample_map.append({
                            "song_idx": song_idx,
                            "bar_idx": bar_idx,
                            "slice": slice(chunk_start, chunk_end)
                        })

    # --------------------------------------------------------------------------
    # Helper Methods for Data Retrieval (__getitem__)
    # --------------------------------------------------------------------------

    def _get_full_sample_for_bar(self, song_idx: int, bar_idx: int) -> Dict[str, List[Any]]:
        """Dynamically constructs the full, un-chunked training sample for a given bar, including its historical context."""
        song = self._songs[song_idx]
        bars = song["bars"]
        short_names = [self._ATTRIBUTE_SHORT_NAME_MAP[k] for k in self.get_attributes_for_model()]
        empty_bar = [self.bar_bos_id, self.bar_eos_id]

        # --- 1. Construct historical context ---
        context_tokens, context_classes = [], []
        context_attrs = defaultdict(list)
        
        for k in range(self.context_num_past_xy_pairs):
            hist_idx = bar_idx - (self.context_num_past_xy_pairs - k)
            
            if hist_idx >= 0: # Real historical bar
                past_bar = bars[hist_idx]
                past_x, past_y = past_bar["src_bar_ids"], past_bar["tgt_bar_ids"]
                past_attrs = {
                    s_name: self._get_attribute_bin_id(past_bar["attributes"][k_full], k_full)
                    for s_name, k_full in zip(short_names, self.get_attributes_for_model())
                }
                
                for item_ids, class_id in [(past_x, SRC_CLASS_ID), (past_y, TGT_CLASS_ID)]:
                    context_tokens.extend(item_ids)
                    context_classes.extend([class_id] * len(item_ids))
                    for s_name in short_names:
                        context_attrs[f"{s_name}_bin_ids"].extend([past_attrs[s_name]] * len(item_ids))
            else: # Padded historical bar
                neutral_binned_attrs = {s_name: 1 for s_name in short_names} # Use middle bin as neutral
                for class_id in [SRC_CLASS_ID, TGT_CLASS_ID]:
                    context_tokens.extend(empty_bar)
                    context_classes.extend([class_id] * len(empty_bar))
                    for s_name in short_names:
                        context_attrs[f"{s_name}_bin_ids"].extend([neutral_binned_attrs[s_name]] * len(empty_bar))
                        
        # --- 2. Prepare the current (X, Y) bar ---
        current_bar = bars[bar_idx]
        current_xi, current_yi = current_bar["src_bar_ids"], current_bar["tgt_bar_ids"]
        current_attrs = {
            s_name: self._get_attribute_bin_id(current_bar["attributes"][k_full], k_full)
            for s_name, k_full in zip(short_names, self.get_attributes_for_model())
        }

        # --- 3. Assemble the full sample ---
        all_tokens = context_tokens + current_xi + current_yi
        all_classes = context_classes + [SRC_CLASS_ID] * len(current_xi) + [TGT_CLASS_ID] * len(current_yi)
        
        for s_name in short_names:
            attr_key = f"{s_name}_bin_ids"
            context_attrs[attr_key].extend([current_attrs[s_name]] * (len(current_xi) + len(current_yi)))
        
        # Labels for language modeling objective (-100 is the ignore index in PyTorch CrossEntropyLoss)
        labels = [-100] * (len(context_tokens) + len(current_xi)) + current_yi[1:] + [-100]

        full_sample = {"input_ids": all_tokens, "class_ids": all_classes, "labels": labels}
        full_sample.update(context_attrs)
        
        return full_sample

    # --------------------------------------------------------------------------
    # DataLoader Related Methods
    # --------------------------------------------------------------------------
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pads samples in a batch to the same length and converts them to PyTorch Tensors."""
        batch = [item for item in batch if item and "input_ids" in item]
        if not batch: return {}

        max_len = max(len(item["input_ids"]) for item in batch)
        
        padded_batch = defaultdict(list)
        short_names = [self._ATTRIBUTE_SHORT_NAME_MAP[k] for k in self.get_attributes_for_model()]
        keys_to_pad = ["input_ids", "class_ids", "labels"] + [f"{s_name}_bin_ids" for s_name in short_names]

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            
            for key in keys_to_pad:
                if key == "labels": pad_val = -100
                elif key == "input_ids": pad_val = self.pad_id
                elif key == "class_ids": pad_val = PAD_CLASS_ID
                else: pad_val = ATTRIBUTE_PAD_ID
                padded_batch[key].append(item.get(key, []) + [pad_val] * pad_len)
            
            padded_batch["attention_mask"].append([1] * len(item["input_ids"]) + [0] * pad_len)
            
        return {key: torch.tensor(val, dtype=torch.long) for key, val in padded_batch.items()}

    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0, **kwargs) -> DataLoader:
        """Creates and returns a DataLoader for this dataset."""
        if not hasattr(self, 'sample_map') or not self.sample_map:
            print("Warning: Dataset is empty, returning an empty DataLoader.")
            return DataLoader([])
        
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=self.collate_fn, **kwargs
        )
    
    # --------------------------------------------------------------------------
    # Statistics and Debugging Methods
    # --------------------------------------------------------------------------

    def _print_dataset_statistics(self, all_bar_data: List[Dict[str, Any]]):
        """Prints a detailed statistical report of the dataset's musical attributes."""
        print("\n--- Detailed Dataset Statistics Report ---")
        stats = _get_attribute_statistics(all_bar_data, self.attribute_bin_edges)
        pprint.pprint(stats)
        
        print("\n  Attribute Bin Distribution (for all bar pairs):")
        for attr_name in self.get_attributes_for_model():
            values = [bar["attributes"].get(attr_name) for bar in all_bar_data 
                      if bar["attributes"].get(attr_name) is not None and np.isfinite(bar["attributes"][attr_name])]
            if values:
                binned_ids = [self._get_attribute_bin_id(v, attr_name) for v in values]
                counts = np.bincount(binned_ids, minlength=self.num_attribute_bins)
                print(f"    {attr_name}:")
                for i, count in enumerate(counts):
                    print(f"      Bin {i}: {count} samples")

    def _print_chunk_stats(self):
        """Prints statistics about the lengths of the training chunks."""
        print("\n--- Chunk Length Statistics ---")
        sample_size = min(10000, len(self.sample_map))
        print(f"  (Calculating on a random sample of {sample_size} chunks for efficiency...)")
        
        indices = random.sample(range(len(self.sample_map)), sample_size)
        lengths = [len(self[i]["input_ids"]) for i in tqdm(indices, desc="Analyzing chunk lengths")]
        
        if lengths:
            lengths_np = np.array(lengths)
            print(f"  - Count: {len(lengths_np)}")
            print(f"  - Avg: {np.mean(lengths_np):.2f}, Std: {np.std(lengths_np):.2f}")
            print(f"  - Min: {np.min(lengths_np)}, Max: {np.max(lengths_np)}")
            percentiles = {p: np.percentile(lengths_np, p) for p in [25, 50, 75, 95]}
            print(f"  - Percentiles: 25th={percentiles[25]:.0f}, 50th(Median)={percentiles[50]:.0f}, 75th={percentiles[75]:.0f}, 95th={percentiles[95]:.0f}")


def _get_attribute_statistics(
    all_bar_data: List[Dict[str, Any]],
    bin_edges: Dict[str, np.ndarray],
    model_attributes: List[str] = EtudeDataset.get_attributes_for_model()
) -> Dict[str, Any]:
    """Calculates and returns a dictionary of detailed statistics for musical attributes."""
    stats = {"raw_stats": {}, "bin_edges": {}, "binned_distribution": {}}
    stats["bin_edges"] = {k: [round(x, 4) for x in v.tolist()] for k, v in bin_edges.items()}

    for attr_name in model_attributes:
        values = np.array([
            b["attributes"].get(attr_name) for b in all_bar_data
            if b.get("attributes", {}).get(attr_name) is not None and np.isfinite(b["attributes"][attr_name])
        ])
        
        if len(values) > 0:
            stats["raw_stats"][attr_name] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "min": float(np.min(values)), "max": float(np.max(values)),
                "median": float(np.median(values)), "count": int(len(values)),
                "percentiles": {p: float(np.percentile(values, p)) for p in [5, 25, 75, 95]}
            }
            
            current_edges = bin_edges.get(attr_name)
            if current_edges is not None:
                indices = np.digitize(values, current_edges)
                counts = np.bincount(indices, minlength=len(current_edges) + 1)
                stats["binned_distribution"][attr_name] = {
                    f"bin_{k}": int(counts[k]) for k in range(len(counts))
                }
    return stats