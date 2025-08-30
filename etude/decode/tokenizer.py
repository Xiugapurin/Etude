# src/etude/data/tokenizer.py

import sys
import copy
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import pretty_midi
from music21 import stream, note, meter, chord, clef, instrument, metadata, duration, tempo, key

from .vocab import Event

# --- Tokenization Constants ---
PAD_CLASS_ID = 0
COND_CLASS_ID = 1
TGT_CLASS_ID = 2
IDX_2_POS = {0: 0.0, 1: 1/6, 2: 1/4, 3: 1/3, 4: 1/2, 5: 2/3, 6: 3/4, 7: 5/6}
ALLOWED_DURATION = [0.0, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 1.0, 1.5, 2.0, 3.0, 4.0]
ALLOWED_DURATIONS_IN_16THS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


class MidiTokenizer:
    def __init__(self, tempo_path: str):
        """
        Initializes the tokenizer.

        Args:
            tempo_path (Optional[Union[str, Path]]): Path to a JSON file containing tempo,
                                                     downbeats, and time signature data.
        """
        self.all_events = []
        if tempo_path and Path(tempo_path).exists():
            with open(tempo_path, 'r') as f:
                self.tempo_data = json.load(f)
            self._create_measures()
        else:
            self.tempo_data = []
            self.global_measures = []
        
        self.time_resolution_for_map = 20

    def split_sequence_into_bars(self, id_sequence: list, bar_bos_id: int, bar_eos_id: int) -> list[list[int]]:
        """
        Splits a token ID sequence into a list of bars based on BOS/EOS tokens.
        This is a utility function for parsing tokenized sequences.
        """
        bars = []
        current_bar = []
        in_bar = False
        if bar_bos_id < 0 or bar_eos_id < 0:
            print("[Warning] Invalid Bar BOS/EOS IDs.", file=sys.stderr)
            return [id_sequence] if id_sequence else []

        for token_id in id_sequence:
            if token_id == bar_bos_id:
                if in_bar and current_bar:
                    bars.append(current_bar)
                current_bar = [token_id]
                in_bar = True
            elif token_id == bar_eos_id:
                if in_bar:
                    current_bar.append(token_id)
                    bars.append(current_bar)
                    current_bar = []
                    in_bar = False
            elif in_bar:
                current_bar.append(token_id)

        # Handle the last bar if the sequence doesn't end cleanly with EOS
        if in_bar and current_bar:
            if current_bar[-1] != bar_eos_id:
                current_bar.append(bar_eos_id)
            bars.append(current_bar)

        return [b for b in bars if len(b) > 1 and b[0] == bar_bos_id and b[-1] == bar_eos_id]

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
    

    def _process_glissandos(self, notes: list[dict]) -> list[dict]:
        if len(notes) < 3:
            return notes

        notes_to_add = []
        indices_to_remove = set()
        
        grace_note_map = {i: note for i, note in enumerate(notes) if note.get("is_grace_note", False)}
        sorted_grace_indices = sorted(grace_note_map.keys())

        i = 0
        while i < len(sorted_grace_indices):
            start_grace_idx_in_notes = sorted_grace_indices[i]
            
            if start_grace_idx_in_notes in indices_to_remove:
                i += 1
                continue

            window_grace_indices = [start_grace_idx_in_notes]
            first_grace_note = notes[start_grace_idx_in_notes]
            window_direction = first_grace_note.get('grace_info')

            k = i + 1
            while k < len(sorted_grace_indices):
                next_grace_idx_in_notes = sorted_grace_indices[k]
                next_grace_note = notes[next_grace_idx_in_notes]

                time_span = next_grace_note['onset'] - first_grace_note['onset']
                if time_span > 1.0:
                    break

                if next_grace_note.get('grace_info') != window_direction:
                    break
                
                window_grace_indices.append(next_grace_idx_in_notes)
                k += 1

            if len(window_grace_indices) >= 3:
                indices_to_remove.update(window_grace_indices)
                main_note_onsets = {notes[idx]['offset'] for idx in window_grace_indices}
                for idx, note in enumerate(notes):
                    if not note.get("is_grace_note") and note['onset'] in main_note_onsets:
                        indices_to_remove.add(idx)
                
                start_note = notes[window_grace_indices[0]]
                end_note = notes[window_grace_indices[-1]]
                start_time = start_note['onset']
                end_time = end_note.get('main_note_offset', end_note['offset'])
                start_pitch = start_note['main_note_pitch']
                end_pitch = end_note['main_note_pitch']

                white_keys_mod = {0, 2, 4, 5, 7, 9, 11}
                pitches_in_run = [notes[idx]['main_note_pitch'] for idx in window_grace_indices]
                white_key_count = sum(1 for p in pitches_in_run if p % 12 in white_keys_mod)
                use_white_keys = white_key_count >= (len(pitches_in_run) - white_key_count)
                is_upward = (window_direction == -1)

                min_p, max_p = min(start_pitch, end_pitch), max(start_pitch, end_pitch)
                gliss_pitches = [p for p in range(min_p, max_p + 1) if (p % 12 in white_keys_mod) == use_white_keys]
                
                if not is_upward: gliss_pitches.reverse()
                
                if len(gliss_pitches) > 1:
                    note_duration = (end_time - start_time) / len(gliss_pitches)
                    for idx_p, pitch in enumerate(gliss_pitches):
                        note_onset = start_time + idx_p * note_duration
                        notes_to_add.append({"pitch": pitch, "onset": note_onset, "offset": note_onset + 0.1, "velocity": 80})
                
                i = k
            else:
                i += 1
                
        final_notes = [note for idx, note in enumerate(notes) if idx not in indices_to_remove]
        final_notes.extend(notes_to_add)

        return final_notes
    

    def _assign_velocity(self, notes: list[dict], volume_contour: Optional[np.ndarray] = None, gamma: int = 0.5) -> list[dict]:
        """
        Applies velocities to a list of decoded notes, using either a volume map or a rule-based fallback.
        """
        if not notes:
            return []
        
        notes_in_measures = [[] for _ in self.global_measures]
        for note in notes:
            for i, measure in enumerate(self.global_measures):
                if measure["start"] <= note["onset"] < measure["end"]:
                    notes_in_measures[i].append(note)
                    note['measure_idx'] = i
                    break

        for i, measure_notes in enumerate(notes_in_measures):
            if not measure_notes:
                continue

            velocity_base = 75
            if volume_contour is not None:
                measure_info = self.global_measures[i]
                start_step = int(measure_info['start'] * self.time_resolution_for_map)
                end_step = int(measure_info['end'] * self.time_resolution_for_map)
                
                if end_step > start_step and end_step <= len(volume_contour):
                    volume_slice = volume_contour[start_step:end_step]
                    if volume_slice.size > 0:
                        avg_loudness = np.mean(volume_slice)
                        perceived_loudness = avg_loudness ** gamma
                        velocity_base = 60 + perceived_loudness * 40
                    else:
                        velocity_base = 75 
                else:
                    velocity_base = 75
            else:
                note_count = len(measure_notes)
                if note_count < 20: velocity_base = 70
                elif note_count < 30: velocity_base = 80
                else: velocity_base = 90
            
            notes_by_onset = defaultdict(list)
            for note in measure_notes:
                notes_by_onset[round(note['onset'], 4)].append(note)

            for notes_at_onset in notes_by_onset.values():
                sorted_notes = sorted(notes_at_onset, key=lambda x: x['pitch'], reverse=True)
                for j, note_to_update in enumerate(sorted_notes):
                    vel = max(velocity_base - 10, velocity_base - (j * 2))
                    if note_to_update['pitch'] > 90: 
                        vel -= 10
                    note_to_update['velocity'] = int(max(0, min(127, vel)))

        for note in notes:
            if note.get("is_grace_note", False):
                main_note = next((n for n in notes if abs(n['onset'] - note['offset']) < 1e-4 and n['pitch'] == note.get('main_note_pitch')), None)
                if main_note and 'velocity' in main_note:
                     grace_vel = main_note['velocity'] - 15
                else:
                     grace_vel = 65
                
                if note['pitch'] > 90:
                    grace_vel -= 10

                note['velocity'] = int(max(0, min(127, grace_vel)))
                    
        return notes
    

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

    def decode_to_notes(self, events: list[Event], volume_map_path: Optional[str] = None) -> list[dict]:
        volume_contour = None
        if volume_map_path:
            try:
                with open(volume_map_path, 'r') as f:
                    volume_contour = np.array(json.load(f))
                print(f"Successfully loaded volume map from {volume_map_path}")
            except Exception as e:
                print(f"Warning: Could not load or parse volume map at {volume_map_path}. Error: {e}")
        
        raw_decoded_notes = []
        event_idx, measure_idx, current_onset_sec, pending_grace_value = 0, 0, 0.0, None
        current_measure = None
        
        while event_idx < len(events):
            event = events[event_idx]
            if event.type_ == "Bar" and event.value == "BOS":
                current_measure = self.global_measures[measure_idx] if measure_idx < len(self.global_measures) else None
                measure_idx += 1; event_idx += 1; continue
            if not current_measure: event_idx += 1; continue
            
            measure_duration_sec = self.global_measures[measure_idx]["start"] - current_measure["start"] if measure_idx < len(self.global_measures) else 0
            seconds_per_beat = (measure_duration_sec / current_measure.get("time_sig", 4)) if measure_duration_sec > 1e-6 else (60.0 / current_measure.get("bpm", 120.0))
            
            if event.type_ == "Pos":
                b_idx, b_rel_pos = self._parse_pos_idx(event.value)
                current_onset_sec = current_measure["start"] + ((b_idx + b_rel_pos) * seconds_per_beat)
                event_idx += 1; continue
            if event.type_ == "Grace":
                pending_grace_value = event.value; event_idx += 1; continue
            if event.type_ == "Note":
                main_note_pitch = event.value
                if event_idx + 1 < len(events) and events[event_idx + 1].type_ == "Duration":
                    duration_token = events[event_idx + 1].value
                    duration_sec = duration_token * (seconds_per_beat / 4.0)
                    if current_measure["start"] <= current_onset_sec < current_measure["end"]:
                        raw_decoded_notes.append({"pitch": main_note_pitch, "onset": current_onset_sec, "offset": current_onset_sec + duration_sec, "velocity": 80, "is_grace_note": False, "rel_pos": event.value})
                    if pending_grace_value is not None:
                        grace_onset = current_onset_sec - 0.05
                        if current_measure["start"] <= grace_onset:
                             raw_decoded_notes.append({"pitch": main_note_pitch + pending_grace_value, "onset": grace_onset, "offset": current_onset_sec, "velocity": 65, "is_grace_note": True, "main_note_pitch": main_note_pitch})
                        pending_grace_value = None
                    event_idx += 2
                else: event_idx += 1
                continue
            event_idx += 1
        
        processed_notes = self._process_glissandos(raw_decoded_notes)
        final_notes_with_velocity = self._assign_velocity(processed_notes, volume_contour)
        
        final_notes_with_velocity.sort(key=lambda x: (x["onset"], x["pitch"]))
        return final_notes_with_velocity
    
    @staticmethod
    def note_to_midi(note_list: list, output_path: Union[str, Path]):
        """
        Converts a list of note dictionaries to a MIDI file and saves it.

        Args:
            note_list (list): A list of note dictionaries, each containing
                              'pitch', 'onset', 'offset', and 'velocity'.
            output_path (Union[str, Path]): The path to save the output .mid file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0) # 0: Acoustic Grand Piano

        for note_data in note_list:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(note_data["velocity"]),
                    pitch=int(note_data["pitch"]),
                    start=note_data["onset"],
                    end=note_data["offset"],
                )
            )

        midi.instruments.append(instrument)
        midi.write(str(output_path))