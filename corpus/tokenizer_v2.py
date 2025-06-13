import json
import copy
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Union

import torch
import numpy as np
from music21 import stream, note, meter, chord, clef, instrument, metadata, duration, tempo, key


PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
PAD_CLASS_ID = 0
COND_CLASS_ID = 1
TGT_CLASS_ID = 2

IDX_2_POS = {0: 0.0, 1: 1/6, 2: 1/4, 3: 1/3, 4: 1/2, 5: 2/3, 6: 3/4, 7: 5/6}
ALLOWED_DURATION = [0.0, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 1.0, 1.5, 2.0, 3.0, 4.0]
ALLOWED_DURATIONS_IN_16THS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

@dataclass
class Event:
    type_: str
    # value: str | int
    value: Union[str, int]

    def __str__(self) -> str:
        return f"{self.type_}_{self.value}"

    def __repr__(self) -> str:
        return f"Event(type={self.type_}, value={self.value})"


class Vocab:
    def __init__(self, special_tokens: List[str] = [PAD_TOKEN]):
        """
        Initializes the vocabulary.
        Args:
            special_tokens (List[str]): A list of special tokens to add first.
                                         PAD_TOKEN is usually required and often ID 0.
        """
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.special_tokens = special_tokens

        # Add special tokens first to ensure consistent IDs
        for token in self.special_tokens:
            self._add_token(token)

        print(f"Initialized Vocab with special tokens: {self.special_tokens}")

    def _add_token(self, token: str) -> int:
        """Adds a token to the vocabulary if it doesn't exist."""
        if token not in self.token_to_id:
            token_id = len(self.id_to_token)
            self.token_to_id[token] = token_id
            self.id_to_token.append(token)
            return token_id
        return self.token_to_id[token]

    def build_from_events(self, event_sequences: List[List[Event]]):
        """
        Builds or updates the vocabulary from a list of event sequences.
        Args:
            event_sequences (List[List[Event]]): A list containing multiple sequences,
                                                 where each sequence is a list of Events.
        """
        print("Building vocabulary from event sequences...")
        token_count = 0
        for seq in event_sequences:
            for event in seq:
                token_str = str(event) # Use the __str__ representation
                self._add_token(token_str)
                token_count += 1
        print(f"Vocabulary built. Total unique tokens (incl. special): {len(self)}")
        print(f"Processed {token_count} total event tokens.")

    def encode(self, token: Union[str, Event]) -> int:
        """Converts a token (string or Event) to its integer ID."""
        token_str = str(token) # Convert Event to string if necessary
        # Handle unknown tokens if UNK_TOKEN is defined
        if UNK_TOKEN in self.special_tokens:
             return self.token_to_id.get(token_str, self.token_to_id[UNK_TOKEN])
        elif token_str in self.token_to_id:
             return self.token_to_id[token_str]
        else:
             raise ValueError(f"Token '{token_str}' not found in vocabulary and UNK_TOKEN is not defined.")


    def decode(self, token_id: int) -> str:
        """Converts an integer ID back to its string token."""
        if 0 <= token_id < len(self.id_to_token):
            return self.id_to_token[token_id]
        else:
            raise ValueError(f"Invalid token ID: {token_id}")


    def decode_to_event(self, token_id: int) -> Event:
        """Converts an integer ID back to its Event object."""
        token_str = self.decode(token_id)
        if token_str == PAD_TOKEN or token_str == UNK_TOKEN: # Handle special tokens
            return Event(type_=token_str, value='')

        parts = token_str.split('_', 1)
        type_ = parts[0]
        value_str = parts[1] if len(parts) > 1 else ''
        
        # Attempt to convert value to int if appropriate for known types
        try:
            if type_ in ["Note", "Pos", "TimeSig"] and value_str: # Add other int types if any
                value = int(value_str)
            elif type_ == "Bar" and value_str in ["BOS", "EOS"]: # Specific string values
                value = value_str
            elif value_str: # Default to string if not a known int type or specific string
                value = value_str
            else: # Handle cases where value might be empty for some event types
                value = ''
        except ValueError:
            value = value_str # Keep as string if conversion fails
        
        return Event(type_=type_, value=value)

    def decode_sequence_to_events(self, id_sequence: List[int]) -> List[Event]:
        """Converts a sequence of integer IDs back to a sequence of Event objects."""
        return [self.decode_to_event(token_id) for token_id in id_sequence if token_id != self.get_pad_id()]

    def encode_sequence(self, sequence: List[Union[str, Event]]) -> List[int]:
        """Converts a sequence of tokens/Events to a sequence of integer IDs."""
        return [self.encode(token) for token in sequence]

    def decode_sequence(self, id_sequence: List[int]) -> List[str]:
        """Converts a sequence of integer IDs back to a sequence of string tokens."""
        # Filter out PAD tokens during decoding if needed, or handle them later
        return [self.decode(token_id) for token_id in id_sequence if token_id != self.token_to_id[PAD_TOKEN]]

    def encode_and_save_sequence(self,
                                 sequence: List[Union[str, Event]],
                                 filepath: Union[str, Path],
                                 format: str = 'npy'):
        """
        Encodes a sequence and saves the resulting ID list to a file.

        Args:
            sequence (List[Union[str, Event]]): The sequence of tokens or Events.
            filepath (Union[str, Path]): The path to save the encoded sequence.
            format (str): The format to save in ('npy', 'pt', 'json').
                          'npy' uses NumPy, 'pt' uses PyTorch tensors, 'json' uses JSON list.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        id_sequence = self.encode_sequence(sequence)

        if format == 'npy':
            np.save(filepath, np.array(id_sequence, dtype=np.int32))
        elif format == 'pt':
            torch.save(torch.tensor(id_sequence, dtype=torch.long), filepath)
        elif format == 'json':
             with open(filepath, 'w') as f:
                 json.dump(id_sequence, f)
        else:
            raise ValueError(f"Unsupported save format: {format}. Choose 'npy', 'pt', or 'json'.")

    def save(self, filepath: Union[str, Path]):
        """Saves the vocabulary mapping (token_to_id) to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # We only need to save token_to_id, id_to_token can be rebuilt
        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens
            # id_to_token can be reconstructed from token_to_id if needed
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary mapping saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Vocab':
        """Loads the vocabulary mapping from a JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        # Create a new Vocab instance
        instance = cls(special_tokens=vocab_data.get('special_tokens', [PAD_TOKEN])) # Load special tokens

        # Manually set the loaded mappings
        instance.token_to_id = vocab_data['token_to_id']
        # Reconstruct id_to_token based on loaded token_to_id
        instance.id_to_token = [""] * len(instance.token_to_id)
        for token, token_id in instance.token_to_id.items():
             if token_id >= len(instance.id_to_token):
                 # Adjust list size if necessary (shouldn't happen with correct save)
                 instance.id_to_token.extend([""] * (token_id - len(instance.id_to_token) + 1))
             instance.id_to_token[token_id] = token

        print(f"Vocabulary loaded from {filepath}. Size: {len(instance)}")
        # Verify PAD token ID is 0 if it exists
        if PAD_TOKEN in instance.token_to_id and instance.token_to_id[PAD_TOKEN] != 0:
             print(f"Warning: Loaded vocabulary has {PAD_TOKEN} with ID {instance.token_to_id[PAD_TOKEN]}, expected 0.")

        return instance

    def __len__(self) -> int:
        """Returns the number of unique tokens in the vocabulary."""
        return len(self.id_to_token)

    def get_pad_id(self) -> int:
        """Returns the ID of the PAD token."""
        return self.token_to_id.get(PAD_TOKEN, -1) # Return -1 if PAD not defined

    def get_bar_bos_id(self) -> int:
        """Returns the ID of the 'Bar_BOS' token."""
        # Important: Assumes your Event string format is 'Bar_BOS'
        return self.token_to_id.get('Bar_BOS', -1)

    def get_bar_eos_id(self) -> int:
        """Returns the ID of the 'Bar_EOS' token."""
        # Important: Assumes your Event string format is 'Bar_EOS'
        return self.token_to_id.get('Bar_EOS', -1)


class MidiTokenizer:
    def __init__(self, tempo_path: str):
        self.all_events = []
        with open(tempo_path, 'r') as f:
            self.tempo_data = json.load(f)
        self._create_measures()


    def _detect_and_link_grace_notes(self, midi_data: list[dict]) -> list[dict]:
        """
        Detects grace notes, adds 'grace_info' to the main note, and removes the original grace note.
        """
        if not midi_data:
            return []
        
        # Sort by onset time, then pitch, to have a deterministic order
        notes = sorted(midi_data, key=lambda x: (x['onset'], x['pitch']))
        
        notes_to_keep = [True] * len(notes)
        
        for i in range(len(notes) - 1):
            if not notes_to_keep[i]:
                continue
            
            # The potential grace note
            grace_candidate = notes[i]
            
            # Look for a main note immediately after
            for j in range(i + 1, len(notes)):
                main_candidate = notes[j]

                # If the time difference is too large, stop searching for this grace_candidate
                onset_diff = main_candidate['onset'] - grace_candidate['onset']
                if onset_diff >= 0.1:
                    break

                # Check conditions
                pitch_diff = main_candidate['pitch'] - grace_candidate['pitch']
                
                if 1e-6 < onset_diff < 0.1 and abs(pitch_diff) == 1:
                    # Found a grace note!
                    # Add grace_info to the main note. Value is 1 or -1.
                    # grace_info = -pitch_diff doesn't work. It should be based on which is higher.
                    grace_value = 1 if grace_candidate['pitch'] > main_candidate['pitch'] else -1
                    main_candidate['grace_info'] = grace_value

                    # Mark the grace note for removal
                    notes_to_keep[i] = False
                    
                    # A grace note can only be attached to one main note, so break the inner loop
                    break

        # Create the final list of notes
        final_notes = [notes[i] for i in range(len(notes)) if notes_to_keep[i]]
        return final_notes

    def _map_duration_to_token_value(self, duration_sec: float, bpm: float) -> int:
        """
        Calculates duration in 16th notes and maps it to the closest allowed duration token.
        """
        if duration_sec <= 0 or bpm <= 0:
            return ALLOWED_DURATIONS_IN_16THS[0]
        
        seconds_per_beat = 60.0 / bpm
        seconds_per_16th_note = seconds_per_beat / 4.0
        
        duration_in_16ths = duration_sec / seconds_per_16th_note
        
        closest_duration = min(ALLOWED_DURATIONS_IN_16THS, key=lambda x: abs(x - duration_in_16ths))
        
        return closest_duration
    

    def _compute_rel_pos(self, note_onset: float, measure_start: float, measure_end: float, time_sig: int, allow_triplet: bool = True) -> tuple[int, bool]:
        rel_pos_2_idx = {0: 0, 1/4: 2, 1/2: 4, 3/4: 6, 1: 8} # quantized pos -> pos idx

        if allow_triplet:
            rel_pos_2_idx.update({1/3: 3, 2/3: 5})

            if measure_end - measure_start >= 1.6:
                rel_pos_2_idx.update({1/6: 1, 5/6: 7})

        m_rel_time = max(0.0, min(1.0, (note_onset - measure_start) / (measure_end - measure_start)))
        b_idx = int(m_rel_time / (1 / time_sig))
        b_rel_time = (m_rel_time % (1 / time_sig)) / (1 / time_sig)
        b_rel_pos = rel_pos_2_idx[min(rel_pos_2_idx.keys(), key=lambda x: abs(x - b_rel_time))]

        pos_idx = b_idx * 8 + b_rel_pos
        is_last = pos_idx >= (8 * time_sig)

        return pos_idx, is_last
    
    def _parse_pos_idx(self, pos_idx: int) -> tuple[int, int]:
        """
        Parse the position index into bar index and relative position.
        Args:
            pos_idx (int): The position index.
        Returns:
            tuple[int, int]: The bar index and relative position.
        """
        b_idx, b_rel_idx = divmod(pos_idx, 8)
        b_rel_pos = IDX_2_POS[b_rel_idx]
        return b_idx, b_rel_pos
    
    def _create_measures(self) -> None:
        self.global_measures = []

        num_regions = len(self.tempo_data)

        for region_idx, region in enumerate(self.tempo_data):
            downbeats = region.get("downbeats", [])
            if not downbeats: continue

            bpm = region["bpm"]
            time_sig = region["time_sig"]
            beats_per_bar = time_sig
            seconds_per_beat = 60 / bpm
            bar_duration = beats_per_bar * seconds_per_beat

            next_region_start = None
            if region_idx < num_regions - 1:
                next_region_start = self.tempo_data[region_idx + 1]["start"]

            measures = []
            for i in range(len(downbeats)):
                start = downbeats[i]

                if i < len(downbeats) - 1:
                    end = downbeats[i + 1]
                elif next_region_start is not None:
                    end = next_region_start
                else:
                    end = start + bar_duration  # fallback only for final region

                measure = {
                    "bpm": bpm,
                    "start": start,
                    "end": end,
                    "notes": [],
                    "chords": defaultdict(list),
                    "time_sig": time_sig
                }
                self.global_measures.append(measure)
                measures.append(measure)

        first_region = self.tempo_data[0]
        first_downbeat = first_region["downbeats"][0]
        first_bar_duration = (60 / first_region["bpm"]) * first_region["time_sig"]
        self.global_measures.insert(0, {
            "bpm": first_region["bpm"],
            "start": first_downbeat - first_bar_duration,
            "end": first_downbeat,
            "notes": [],
            "chords": defaultdict(list),
            "time_sig": first_region["time_sig"]
        })

        last_region = self.tempo_data[-1]
        last_downbeat = last_region["downbeats"][-1]
        last_bar_duration = (60 / last_region["bpm"]) * last_region["time_sig"]
        self.global_measures.append({
            "bpm": last_region["bpm"],
            "start": last_downbeat + last_bar_duration,
            "end": last_downbeat + 2 * last_bar_duration,
            "notes": [],
            "chords": defaultdict(list),
            "time_sig": last_region["time_sig"]
        })

    def _assign_notes(self, midi_data: list[dict]) -> None:
        for note in midi_data:
            note_onset = note["onset"]
            for m_idx, m in enumerate(self.global_measures):
                if m["start"] <= note_onset < m["end"]:
                    pos_idx, is_last = self._compute_rel_pos(note_onset, m["start"], m["end"], m["time_sig"], allow_triplet=False)
                    
                    # Calculate duration
                    duration_sec = note["offset"] - note["onset"]
                    duration = self._map_duration_to_token_value(duration_sec, m["bpm"])

                    note_info = {**note, "duration": duration}

                    if is_last and m_idx + 1 < len(self.global_measures):
                        target_measure = self.global_measures[m_idx + 1]
                        target_measure["notes"].append({**note_info, "pos_idx": 0})
                        target_measure["chords"][0].append(note_info)
                    elif not is_last:
                        m["notes"].append({**note_info, "pos_idx": pos_idx})
                        m["chords"][pos_idx].append(note_info)
                    break


    def _add_bar_event(self, bos: bool = True, time_sig: int = 4) -> None:
        if bos:
            self.all_events.append(Event(type_="Bar", value="BOS"))
            # self.all_events.append(Event(type_="TimeSig", value=time_sig))
        else:
            self.all_events.append(Event(type_="Bar", value="EOS"))
    

    def _add_pos_event(self, pos_idx: int) -> None:
        self.all_events.append(Event(type_="Pos", value=pos_idx))


    def encode(self, midi_path: str, with_grace_note: bool = False) -> list[Event]:
        with open(midi_path, 'r') as f:
            midi_data = json.load(f)

        # Pre-process to find grace notes
        if with_grace_note:
            processed_midi_data = self._detect_and_link_grace_notes(midi_data)
            self._assign_notes(processed_midi_data)
        else:
            self._assign_notes(midi_data)

        for m_idx, m in enumerate(self.global_measures):
            m["chords"] = {k: v for k, v in sorted(m["chords"].items())}
            self._add_bar_event(bos=True)

            for pos_idx, note_list in m["chords"].items():
                note_list.sort(key=lambda x: -x["pitch"])
                unique_notes = [n for i, n in enumerate(note_list) if n["pitch"] not in {p["pitch"] for p in note_list[:i]}]
                m["chords"][pos_idx] = unique_notes

            for pos_idx, notes in m["chords"].items():
                self._add_pos_event(pos_idx)
                for note in notes:
                    # If the note has grace info, add the Grace token first
                    if 'grace_info' in note:
                        self.all_events.append(Event(type_="Grace", value=note['grace_info']))
                    
                    self.all_events.append(Event(type_="Note", value=note["pitch"]))
                    self.all_events.append(Event(type_="Duration", value=note["duration"]))

            self._add_bar_event(bos=False)
        
        return self.all_events
    

    def _parse_chords(self, events: list[Event]) -> list[dict]:
        n, m = len(events), len(self.global_measures)
        if m != events.count(Event(type_="Bar", value="BOS")):
            raise ValueError("Number of measures in events does not match the number of measures in the Tempo data.")
        
        chords = []
        e_idx = m_idx = 0
        while e_idx < n:
            event = events[e_idx]
            if event.type_ == "Bar" and event.value == "BOS":
                measure = self.global_measures[m_idx]
                m_start = measure["start"]
                m_end = measure["end"]
                m_duration = m_end - m_start
                m_b_duration = m_duration / measure["time_sig"]

                e_idx += 1
                while e_idx < n and events[e_idx].type_ != "Bar":
                    if events[e_idx].type_ == "TimeSig": 
                        e_idx += 1
                        continue

                    if events[e_idx].type_ == "Pos":
                        pos_idx = events[e_idx].value
                        # b_idx, b_rel_idx = divmod(pos_idx, 8)
                        # b_rel_pos = IDX_2_POS[b_rel_idx]
                        b_idx, b_rel_pos = self._parse_pos_idx(pos_idx)
                        onset = m_start + (b_idx + b_rel_pos) * m_b_duration
                        e_idx += 1
                        continue
                    
                    chord = {
                        "onset": onset,
                        "rel_pos": (m_idx, pos_idx),
                        "pitches": []
                    }
                    while e_idx < n and events[e_idx].type_ == "Note":
                        chord["pitches"].append(events[e_idx].value)
                        e_idx += 1
                    
                    chord["pitches"] = sorted(list(set(chord["pitches"])))

                    chords.append(chord)
            else:
                m_idx += 1
                e_idx += 1

        return chords
    
    def _split_hands(self, chords: list[dict]) -> tuple[list[dict], list[dict]]:
        prev_left, prev_right = (float("-inf"), []), (float("-inf"), []) # (onset, chord)
        r_chord_list, l_chord_list = [], []

        for chord in chords:
            onset = chord["onset"]
            rel_pos = chord["rel_pos"]
            pitches = chord["pitches"]
            l_chord = {"onset": onset, "rel_pos": rel_pos, "pitches": []}
            r_chord = {"onset": onset, "rel_pos": rel_pos, "pitches": []}

            for p in pitches:
                if p >= 60:
                    r_chord["pitches"].append(p)
                else:
                    l_chord["pitches"].append(p)
            
            if r_chord["pitches"]:
                r_chord["pitches"] = sorted(list(set(r_chord["pitches"])))
                r_chord_list.append(r_chord)
            if l_chord["pitches"]:
                l_chord["pitches"] = sorted(list(set(l_chord["pitches"])))
                l_chord_list.append(l_chord)

        return r_chord_list, l_chord_list
    
    def _calc_pos_diff(self, rel_pos_1: tuple[int, int], rel_pos_2: tuple[int, int]) -> int:
        m_idx_1, pos_idx_1 = rel_pos_1
        m_idx_2, pos_idx_2 = rel_pos_2

        b_idx_1, b_rel_pos_1 = self._parse_pos_idx(pos_idx_1)
        time_sig_1 = self.global_measures[m_idx_1]["time_sig"]

        b_idx_2, b_rel_pos_2 = self._parse_pos_idx(pos_idx_2)

        if m_idx_1 == m_idx_2:
            diff = (b_idx_2 + b_rel_pos_2) - (b_idx_1 + b_rel_pos_1)
        elif m_idx_1 + 1 == m_idx_2:
            diff = (b_idx_2 + b_rel_pos_2) + (time_sig_1 - (b_idx_1 + b_rel_pos_1))
        else:
            diff = (time_sig_1 - (b_idx_1 + b_rel_pos_1))

        closest = min(ALLOWED_DURATION, key=lambda x: abs(x - diff))

        if abs(closest - diff) > 0.01:
            idx = ALLOWED_DURATION.index(closest)
            if diff < closest and idx > 0:
                duration = ALLOWED_DURATION[idx - 1]
            else:
                duration = closest
        else:
            duration = closest

        return duration

    def _calc_next_pos(self, measure_info: list[dict], chords: list[dict]) -> int:
        n = len(chords)

        for i, chord in enumerate(chords):
            rel_pos = chord["rel_pos"]
            start_beat = measure_info[rel_pos[0]]["start_beat"]
            b_idx, b_rel_pos = self._parse_pos_idx(rel_pos[1])

            next_rel_pos = (float("inf"), 0) if i == n - 1 else chords[i + 1]["rel_pos"]
            duration = self._calc_pos_diff(rel_pos, next_rel_pos)

            measure_info[rel_pos[0]]["chords"].append({
                "start": start_beat + b_idx + b_rel_pos,
                "pitches": chord["pitches"],
                "duration": duration
            })

    def _insert_note(self, part: stream.Part, pitch: int, onset: float, duration: float) -> None:
        if pitch == 0:
            n = note.Rest()
        else:
            n = note.Note(midi=pitch)
        n.quarterLength = duration
        part.insert(onset, n)

    def _append_note(self, part: stream.Part, pitch: int, duration: float) -> None:
        if pitch == 0:
            n = note.Rest()
        else:
            n = note.Note(midi=pitch)
        n.quarterLength = duration
        part.append(n)

    def _insert_triplet(self, part: stream.Part, note_list: list[dict], type: str, onset: float):
        valid_types = {"eighth", "quarter", "half"}
        if type not in valid_types:
            raise ValueError(f"Triplet type must be one of {valid_types}, got '{type}'")
        
        if type == "eighth":
            total_duration = 1.0
        elif type == "quarter":
            total_duration = 2.0
        elif type == "half":
            total_duration = 4.0

        total_weight = sum(item["weight"] for item in note_list)
        if total_weight != 3:
            raise ValueError("The sum of weights in note_list must be 3.")

        triplet = duration.Tuplet(3, 2, type)
        
        current_onset = onset
        for item in note_list:
            p = item["pitch"]
            weight = item["weight"]
            note_duration = (weight / 3) * total_duration
            
            if p == 0:
                n = note.Rest()
            else:
                n = note.Note(midi=p)

            n.quarterLength = note_duration
            n.duration.type = type
            n.duration.appendTuplet(copy.deepcopy(triplet))
            
            part.insert(current_onset, n)
            current_onset += note_duration

    def _append_chord(self, part: stream.Part, pitches: list[int], duration: float) -> None:
        c = chord.Chord([*pitches])
        c.quarterLength = duration

        part.append(c)

    def _insert_chord(self, part: stream.Part, pitches: list[int], onset: float, duration: float) -> None:
        c = chord.Chord([*pitches])
        c.quarterLength = duration
        part.insert(onset, c)
    
    def _create_score(self, measure_info: list[dict], part: stream.Part) -> None:
        prev_time_sig = -1
        prev_bpm = -1

        for m in measure_info:
            time_sig = m["time_sig"]
            start_beat = m["start_beat"]
            bpm = round(m["bpm"])

            if time_sig != prev_time_sig:
                part.append(meter.TimeSignature(f"{str(time_sig)}/4"))
                prev_time_sig = time_sig
            
            if bpm != prev_bpm:
                part.insert(start_beat, tempo.MetronomeMark(number=bpm))
                prev_bpm = bpm

            for chord in m["chords"]:
                pitches = chord["pitches"]
                duration = chord["duration"]
                start = chord["start"]

                if not pitches:
                    self._insert_note(part, 0, start_beat, time_sig)
                elif len(pitches) == 1:
                    self._insert_note(part, pitches[0], start, duration)
                else:
                    self._insert_chord(part, pitches, start, duration)

    def decode_to_score(
        self,
        events,
        title: str = "Piano Cover", 
        composer: str = "None", 
        path_out: str = "output.musicxml"
    ) -> stream.Score:
        r_measure_info = []
        l_measure_info = []
        accumulated_beats = 0

        for measure in self.global_measures:
            r_measure_info.append({
                "start_beat": accumulated_beats,
                "bpm": measure["bpm"],
                "time_sig": measure["time_sig"],
                "chords": []
            })
            l_measure_info.append({
                "start_beat": accumulated_beats,
                "bpm": measure["bpm"],
                "time_sig": measure["time_sig"],
                "chords": []
            })

            accumulated_beats += measure["time_sig"]

        chords = self._parse_chords(events)
        r_chord, l_chord = self._split_hands(chords)
        self._calc_next_pos(r_measure_info, r_chord)
        self._calc_next_pos(l_measure_info, l_chord)

        score = stream.Score()
        
        score.metadata = metadata.Metadata()
        score.metadata.title = title
        score.metadata.composer = composer

        r_part = stream.Part()
        r_part.id = 'R-Part'
        r_part.append(instrument.Piano())
        r_part.append(clef.TrebleClef())

        l_part = stream.Part()
        l_part.id = 'L-Part'
        l_part.append(instrument.Piano())
        l_part.append(clef.BassClef())

        self._create_score(r_measure_info, r_part)
        self._create_score(l_measure_info, l_part)

        score.insert(0, key.KeySignature(2))
        score.append(r_part)
        score.append(l_part)

        # right_hand.append(meter.TimeSignature('4/4'))

        # left_hand.append(meter.TimeSignature('4/4'))
        score.write('musicxml', fp=path_out)

    def decode_to_notes(self, events: list[Event]) -> list[dict]:
        decoded_notes = []
        event_idx, measure_idx = 0, 0
        num_events = len(events)
        
        current_onset_sec = 0.0
        current_measure = None
        pending_grace_value = None

        while event_idx < num_events:
            event = events[event_idx]

            if event.type_ == "Bar" and event.value == "BOS":
                current_measure = self.global_measures[measure_idx] if measure_idx < len(self.global_measures) else None
                measure_idx += 1
                event_idx += 1
                continue

            if current_measure is None:
                event_idx += 1; continue

            if event.type_ == "Pos":
                b_idx, b_rel_pos = self._parse_pos_idx(event.value)
                seconds_per_beat = 60.0 / current_measure["bpm"]
                current_onset_sec = current_measure["start"] + ((b_idx + b_rel_pos) * seconds_per_beat)
                event_idx += 1
                continue
            
            # [NEW] Handle Grace token: store it and wait for the main note
            if event.type_ == "Grace":
                pending_grace_value = event.value
                event_idx += 1
                continue

            if event.type_ == "Note":
                main_note_pitch = event.value
                
                if event_idx + 1 < num_events and events[event_idx + 1].type_ == "Duration":
                    duration_event = events[event_idx + 1]
                    duration_token = duration_event.value
                    
                    seconds_per_beat = 60.0 / current_measure["bpm"]
                    seconds_per_16th = seconds_per_beat / 4.0
                    duration_sec = duration_token * seconds_per_16th
                    main_note_offset = current_onset_sec + duration_sec

                    # Add the main note
                    decoded_notes.append({
                        "pitch": main_note_pitch,
                        "onset": current_onset_sec,
                        "offset": main_note_offset,
                        "velocity": 80
                    })

                    # [NEW] If a grace note was pending, create it now
                    if pending_grace_value is not None:
                        grace_pitch = main_note_pitch + pending_grace_value
                        grace_onset = current_onset_sec - 0.05
                        grace_offset = current_onset_sec
                        decoded_notes.append({
                            "pitch": grace_pitch,
                            "onset": grace_onset,
                            "offset": grace_offset,
                            "velocity": 65 # Typically softer
                        })
                        pending_grace_value = None # Reset pending grace

                    event_idx += 2 # Consume Note and Duration
                else:
                    print(f"Warning: Note event at index {event_idx} not followed by Duration. Skipping.")
                    event_idx += 1
                continue
            
            event_idx += 1
            
        decoded_notes.sort(key=lambda x: (x["onset"], x["pitch"]))
        return decoded_notes

    
    def restore(self) -> list[dict]:
        """
        Restores notes from the internal global_measures structure, using the new duration_token info.
        """
        restored_notes = []
        idx_2_rel_pos = {v: k for k, v in IDX_2_POS.items()} # Invert for lookup

        for m in self.global_measures:
            m_start = m["start"]
            m_bpm = m["bpm"]
            seconds_per_beat = 60.0 / m_bpm
            
            for note in m["notes"]:
                # Calculate onset
                pos_idx = note["pos_idx"]
                b_idx, b_rel_idx = divmod(pos_idx, 8)
                b_rel_pos = idx_2_rel_pos.get(b_rel_idx, 0.0)
                beats_into_measure = b_idx + b_rel_pos
                onset_sec = m_start + (beats_into_measure * seconds_per_beat)

                # Calculate offset from duration_token
                duration_token = note.get("duration_token")
                if duration_token is not None:
                    seconds_per_16th = seconds_per_beat / 4.0
                    duration_sec = duration_token * seconds_per_16th
                    offset_sec = onset_sec + duration_sec
                else:
                    # Fallback if duration_token is missing for some reason
                    offset_sec = onset_sec + 0.5

                new_note = {
                    "pitch": note["pitch"],
                    "onset": onset_sec,
                    "offset": offset_sec,
                    "velocity": note["velocity"]
                }
                restored_notes.append(new_note)

        restored_notes.sort(key=lambda x: x["onset"])
        return restored_notes