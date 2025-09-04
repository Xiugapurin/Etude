# etude/evaluation/metrics/base_metric.py

import numpy as np
import pretty_midi
import json
from pathlib import Path

def get_onsets_from_file(file_path: Path) -> np.ndarray:
    """Shared utility to extract a unique, sorted list of note onsets from a MIDI or JSON file."""
    onsets = []
    if not file_path.exists():
        return np.array([])
    
    try:
        if file_path.suffix.lower() == '.mid':
            midi_data = pretty_midi.PrettyMIDI(str(file_path))
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    onsets.extend([note.start for note in instrument.notes])
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                notes = json.load(f)
            if notes:
                onsets = [note['onset'] for note in notes]
        
        if len(onsets) < 2:
            return np.array([])
        
        return np.unique(onsets)
    except Exception:
        return np.array([])