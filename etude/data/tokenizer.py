# etude/data/tokenizer.py

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Union, Optional, List, Tuple
from fractions import Fraction

import numpy as np
import pretty_midi

from .vocab import Event

# --- Tokenization Constants ---
PAD_CLASS_ID = 0
SRC_CLASS_ID = 1
TGT_CLASS_ID = 2
IDX_2_POS = {0: 0.0, 1: 1/6, 2: 1/4, 3: 1/3, 4: 1/2, 5: 2/3, 6: 3/4, 7: 5/6}
ALLOWED_DURATIONS_IN_16THS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


class TinyREMITokenizer:
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
            
            grace_candidate = notes[i]
            
            for j in range(i + 1, len(notes)):
                main_candidate = notes[j]

                onset_diff = main_candidate['onset'] - grace_candidate['onset']
                if onset_diff >= 0.1:
                    break

                # Check conditions
                pitch_diff = main_candidate['pitch'] - grace_candidate['pitch']
                
                if 1e-6 < onset_diff < 0.1 and abs(pitch_diff) == 1:
                    grace_value = 1 if grace_candidate['pitch'] > main_candidate['pitch'] else -1
                    main_candidate['grace_info'] = grace_value

                    # Mark the grace note for removal
                    notes_to_keep[i] = False
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


    def _add_bar_event(self, bos: bool = True) -> None:
        if bos:
            self.all_events.append(Event(type_="Bar", value="BOS"))
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

    def decode_to_notes(self, events: list[Event], volume_map_path: Optional[str] = None) -> list[dict]:
        volume_contour = None
        if volume_map_path:
            try:
                with open(volume_map_path, 'r') as f:
                    volume_contour = np.array(json.load(f))
                print(f"[INFO] Successfully loaded volume map from {volume_map_path}")
            except Exception as e:
                print(f"[ERROR] Could not load or parse volume map at {volume_map_path}. Error: {e}")
        
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
        instrument = pretty_midi.Instrument(program=0)

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

    def decode_to_score(
        self,
        events: List[Event],
        output_path: Union[str, Path],
        title: str = "Piano Cover"
    ) -> None:
        """
        Converts decoded events to a MusicXML score using music21.

        This method:
        1. Separates notes into left/right hand parts based on pitch
        2. Groups simultaneous notes (same pos) into chords
        3. Calculates note durations based on gap to next chord
        4. Fills gaps with appropriate rests using greedy algorithm
        5. Handles time signature and tempo changes per measure

        Args:
            events: List of Event objects from decode
            output_path: Path to save the MusicXML file
            title: Title for the score (default: "Piano Cover")
        """
        try:
            from music21 import stream, note, chord, meter, tempo, clef, instrument, metadata, duration
        except ImportError:
            print("[ERROR] music21 is required for decode_to_score. Install with: pip install music21", file=sys.stderr)
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Allowed note durations in quarter notes
        ALLOWED_DURATIONS = [Fraction(8), Fraction(4), Fraction(3), Fraction(2), Fraction(3, 2),
                            Fraction(1), Fraction(3, 4), Fraction(1, 2), Fraction(1, 4)]

        def _get_valid_duration(target_duration: Fraction) -> Fraction:
            """Find the largest allowed duration that doesn't exceed target."""
            for d in ALLOWED_DURATIONS:
                if d <= target_duration:
                    return d
            return Fraction(1, 4)  # minimum duration

        def _fill_with_rests(gap: Fraction) -> List[Fraction]:
            """Fill a gap with rests using greedy algorithm."""
            rests = []
            remaining = gap
            while remaining > 0:
                rest_dur = _get_valid_duration(remaining)
                rests.append(rest_dur)
                remaining -= rest_dur
                if remaining < Fraction(1, 16):  # Prevent floating point issues
                    break
            return rests

        def _separate_hands(notes_with_pos: List[dict]) -> Tuple[List[dict], List[dict]]:
            """
            Separate notes into right hand (treble) and left hand (bass) parts.
            Uses pitch-based split: notes >= 60 (middle C) go to right hand.
            """
            right_hand = []
            left_hand = []

            for note_data in notes_with_pos:
                if note_data['pitch'] >= 60:
                    right_hand.append(note_data)
                else:
                    left_hand.append(note_data)

            return right_hand, left_hand

        def _group_by_measure_and_pos(notes: List[dict]) -> dict:
            """Group notes by measure index and position."""
            grouped = defaultdict(lambda: defaultdict(list))
            for n in notes:
                grouped[n['measure_idx']][n['pos_idx']].append(n)
            return grouped

        def _create_part_for_hand(
            hand_notes: List[dict],
            hand_name: str,
            clef_type: str
        ) -> stream.Part:
            """Create a music21 Part for one hand."""
            part = stream.Part()
            part.partName = hand_name

            # Set instrument
            piano = instrument.Piano()
            part.insert(0, piano)

            # Group notes by measure and position
            grouped = _group_by_measure_and_pos(hand_notes)

            # Process each measure
            for m_idx, measure_info in enumerate(self.global_measures):
                m = stream.Measure(number=m_idx + 1)

                # Add clef at first measure
                if m_idx == 0:
                    if clef_type == 'treble':
                        m.append(clef.TrebleClef())
                    else:
                        m.append(clef.BassClef())

                # Add time signature if it changes or at first measure
                time_sig = measure_info.get('time_sig', 4)
                if m_idx == 0 or (m_idx > 0 and self.global_measures[m_idx - 1].get('time_sig', 4) != time_sig):
                    m.append(meter.TimeSignature(f'{time_sig}/4'))

                # Add tempo if it changes or at first measure
                bpm = round(measure_info.get('bpm', 120))
                if m_idx == 0 or (m_idx > 0 and round(self.global_measures[m_idx - 1].get('bpm', 120)) != bpm):
                    m.append(tempo.MetronomeMark(number=bpm))

                # Get chords for this measure (sorted by position)
                measure_chords = grouped.get(m_idx, {})
                sorted_positions = sorted(measure_chords.keys())

                beats_per_measure = Fraction(time_sig)  # in quarter notes
                current_offset = Fraction(0)

                for i, pos_idx in enumerate(sorted_positions):
                    chord_notes = measure_chords[pos_idx]

                    # Calculate position in quarter notes
                    b_idx, b_rel_pos = self._parse_pos_idx(pos_idx)
                    pos_in_quarters = Fraction(b_idx) + Fraction(b_rel_pos).limit_denominator(12)

                    # Fill gap before this chord with rests
                    if pos_in_quarters > current_offset:
                        gap = pos_in_quarters - current_offset
                        for rest_dur in _fill_with_rests(gap):
                            r = note.Rest()
                            r.duration = duration.Duration(float(rest_dur))
                            m.append(r)
                        current_offset = pos_in_quarters

                    # Calculate duration: gap to next chord or end of measure
                    if i + 1 < len(sorted_positions):
                        next_pos_idx = sorted_positions[i + 1]
                        next_b_idx, next_b_rel_pos = self._parse_pos_idx(next_pos_idx)
                        next_pos_in_quarters = Fraction(next_b_idx) + Fraction(next_b_rel_pos).limit_denominator(12)
                        gap_to_next = next_pos_in_quarters - pos_in_quarters
                    else:
                        gap_to_next = beats_per_measure - pos_in_quarters

                    chord_duration = _get_valid_duration(gap_to_next)

                    # Create chord or single note
                    pitches = [n['pitch'] for n in chord_notes]
                    if len(pitches) == 1:
                        n = note.Note(pitches[0])
                        n.duration = duration.Duration(float(chord_duration))
                        m.append(n)
                    else:
                        c = chord.Chord(pitches)
                        c.duration = duration.Duration(float(chord_duration))
                        m.append(c)

                    current_offset = pos_in_quarters + chord_duration

                    # Fill gap between chord end and actual next position
                    if i + 1 < len(sorted_positions):
                        if current_offset < next_pos_in_quarters:
                            gap = next_pos_in_quarters - current_offset
                            for rest_dur in _fill_with_rests(gap):
                                r = note.Rest()
                                r.duration = duration.Duration(float(rest_dur))
                                m.append(r)
                            current_offset = next_pos_in_quarters

                # Fill remaining space with rests (always fill to complete the measure)
                if current_offset < beats_per_measure:
                    gap = beats_per_measure - current_offset
                    for rest_dur in _fill_with_rests(gap):
                        r = note.Rest()
                        r.duration = duration.Duration(float(rest_dur))
                        m.append(r)

                # Always add the measure to ensure left/right hand alignment
                part.append(m)

            return part

        # Parse events to get notes with position and measure info
        notes_with_info = []
        event_idx, measure_idx, current_pos_idx = 0, 0, 0
        current_measure = None

        while event_idx < len(events):
            event = events[event_idx]

            if event.type_ == "Bar" and event.value == "BOS":
                if measure_idx < len(self.global_measures):
                    current_measure = self.global_measures[measure_idx]
                measure_idx += 1
                event_idx += 1
                continue

            if event.type_ == "Bar" and event.value == "EOS":
                event_idx += 1
                continue

            if not current_measure:
                event_idx += 1
                continue

            if event.type_ == "Pos":
                current_pos_idx = event.value
                event_idx += 1
                continue

            if event.type_ == "Grace":
                # Skip grace notes for score (they complicate notation)
                event_idx += 1
                continue

            if event.type_ == "Note":
                pitch = event.value
                # Get duration token if available
                dur_token = 4  # default quarter note
                if event_idx + 1 < len(events) and events[event_idx + 1].type_ == "Duration":
                    dur_token = events[event_idx + 1].value
                    event_idx += 2
                else:
                    event_idx += 1

                notes_with_info.append({
                    'pitch': pitch,
                    'measure_idx': measure_idx - 1,  # -1 because we incremented after BOS
                    'pos_idx': current_pos_idx,
                    'duration_token': dur_token
                })
                continue

            event_idx += 1

        # Separate into hands
        right_hand_notes, left_hand_notes = _separate_hands(notes_with_info)

        # Create score
        score = stream.Score()

        # Add metadata
        score.insert(0, metadata.Metadata())
        score.metadata.title = title
        score.metadata.composer = "Generated by Etude"

        # Create parts for each hand
        if right_hand_notes:
            right_part = _create_part_for_hand(right_hand_notes, "Right Hand", "treble")
            score.insert(0, right_part)

        if left_hand_notes:
            left_part = _create_part_for_hand(left_hand_notes, "Left Hand", "bass")
            score.insert(0, left_part)

        # Write to file
        score.write('musicxml', fp=str(output_path))
        print(f"[INFO] Score saved to: {output_path}")