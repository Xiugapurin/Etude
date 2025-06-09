import os
import json
import tempfile
import numpy as np
import pretty_midi
import soundfile as sf
from pydub import AudioSegment
import IPython.display as ipd
from scipy.io.wavfile import write
from midi_player import MIDIPlayer
from midi_player.stylers import basic

def json_to_midi(json_path, output_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    if not os.path.exists(json_path):
        print(f"Missing note info JSON file: {json_path}")
        return

    with open(json_path, "r") as f:
        notes = json.load(f)

    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=note["velocity"],
                pitch=note["pitch"],
                start=note["onset"],
                end=note["offset"],
            )
        )
    midi.instruments.append(instrument)
    midi.write(output_path)

def json_to_midi_with_beat(json_path, output_path, beat_json_path):
    midi = pretty_midi.PrettyMIDI()
    
    if not os.path.exists(json_path):
        print(f"Missing note info JSON file: {json_path}")
        return
    
    with open(json_path, "r") as f:
        notes = json.load(f)
    
    instrument = pretty_midi.Instrument(program=0)
    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=note["velocity"],
                pitch=note["pitch"],
                start=note["onset"],
                end=note["offset"],
            )
        )
    midi.instruments.append(instrument)
    
    if not os.path.exists(beat_json_path):
        print(f"Missing beat info JSON file: {beat_json_path}")
        return
    with open(beat_json_path, "r") as f:
        beat_info = json.load(f)
    
    downbeat_instrument = pretty_midi.Instrument(program=0, is_drum=True)
    beat_instrument = pretty_midi.Instrument(program=0, is_drum=True)
    
    beat_duration = 0.1 
    
    for downbeat in beat_info.get("downbeat_pred", []):
        note_downbeat = pretty_midi.Note(
            velocity=100,
            pitch=36,
            start=downbeat,
            end=downbeat + beat_duration,
        )
        downbeat_instrument.notes.append(note_downbeat)
    
    for beat in beat_info.get("beat_pred", []):
        note_beat = pretty_midi.Note(
            velocity=70,
            pitch=38,
            start=beat,
            end=beat + beat_duration,
        )
        beat_instrument.notes.append(note_beat)
    
    midi.instruments.append(downbeat_instrument)
    midi.instruments.append(beat_instrument)
    
    midi.write(output_path)

def get_midi_player(midi_file_path):
    return MIDIPlayer(url_or_file=midi_file_path, height=600, styler=basic, title='My Player')

def get_midi_audio(midi_file_path, output_wav_file, fs=44100):
    midi_data = pretty_midi.PrettyMIDI(midi_file=midi_file_path)
    midi_synth = midi_data.fluidsynth(fs)
    sf.write(output_wav_file, midi_synth, fs)

def align_and_play_stereo(wav_path1, wav_path2, output_path):
    audio1 = AudioSegment.from_file(wav_path1).set_channels(1)
    audio2 = AudioSegment.from_file(wav_path2).set_channels(1)

    sample_rate = audio1.frame_rate
    if audio2.frame_rate != sample_rate:
        audio2 = audio2.set_frame_rate(sample_rate)

    array1 = np.array(audio1.get_array_of_samples())
    array2 = np.array(audio2.get_array_of_samples())

    max_length = max(len(array1), len(array2))
    if len(array1) < max_length:
        array1 = np.pad(array1, (0, max_length - len(array1)), 'constant')
    if len(array2) < max_length:
        array2 = np.pad(array2, (0, max_length - len(array2)), 'constant')

    stereo_array = np.stack((array1, array2), axis=1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        sf.write(temp_path, stereo_array, sample_rate)

    ipd.display(ipd.Audio(temp_path))
    sf.write(output_path, stereo_array, 22050)
