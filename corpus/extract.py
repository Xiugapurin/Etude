"""
Source: https://github.com/misya11p/amt-apc  
This code was originally taken from the above GitHub repository.
Original file: /infer/__main__.py

Changes: style vector (sv) have been disabled as per project requirements.
"""

from pathlib import Path
import sys
from utils import download

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch
from models import Pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract(path_input: str, path_output_json: str, path_output_midi: str, path_model: str):
    extractor = Pipeline(path_model, DEVICE)

    if path_input.startswith("https://"):
        path_input = download(url=path_input)

    extractor.wav2midi(path_input, path_output_json, path_output_midi)

