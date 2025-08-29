#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import torch
from DilatedTransformer import Demixed_DilatedTransformerModel
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
import argparse
import sys
import traceback


def beat_detection_process(input_dir, output_file_name='beat_pred.json', checkpoint_base_dir='checkpoint'):
    """
    Performs beat and downbeat detection on a .npy file.

    Args:
        input_dir (str): Directory containing the 'sep.npy' file.
        output_file_name (str): Name for the output JSON file.
        checkpoint_base_dir (str): Base directory for model checkpoints, relative to this script's location.
    """
    script_location_dir = os.path.dirname(os.path.abspath(__file__))
    PARAM_PATH = {
        i: os.path.join(script_location_dir, checkpoint_base_dir, f"fold_{i}_trf_param.pt")
        for i in range(8)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = Demixed_DilatedTransformerModel(
            attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8, d_hid=1024, nlayers=9, norm_first=True
        )
        checkpoint_data = torch.load(PARAM_PATH[4], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_data['state_dict'])
        model.to(device)
        model.eval()

        FPS = 44100 / 1024
        MIN_BPM = 70.0
        MAX_BPM = 250.0
        THRESHOLD = 0.2

        beat_tracker = DBNBeatTrackingProcessor(min_bpm=MIN_BPM, max_bpm=MAX_BPM, fps=FPS, threshold=THRESHOLD)
        downbeat_tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], min_bpm=MIN_BPM, max_bpm=MAX_BPM, fps=FPS, threshold=THRESHOLD
        )

        npy_path = os.path.join(input_dir, 'sep.npy')
        output_path = os.path.join(input_dir, output_file_name)


        x = np.load(npy_path)

        with torch.no_grad():
            model_input = torch.from_numpy(x).unsqueeze(0).float().to(device)
            activation, _ = model(model_input)

        beat_activation = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
        downbeat_activation = torch.sigmoid(activation[0, :, 1]).detach().cpu().numpy()

        dbn_beat_pred = beat_tracker(beat_activation)
        combined_act = np.concatenate(
            (
                np.maximum(beat_activation - downbeat_activation, np.zeros(beat_activation.shape))[:, np.newaxis],
                downbeat_activation[:, np.newaxis],
            ),
            axis=-1,
        )

        dbn_downbeat_pred = downbeat_tracker(combined_act)
        dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1] == 1][:, 0]

        results = {
            'beat_pred': dbn_beat_pred.tolist(),
            'downbeat_pred': dbn_downbeat_pred.tolist(),
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        os.remove(npy_path)

    except Exception as e:
        print(f"\nAn error occurred during beat detection for {input_dir}:", file=sys.stderr)
        print(f"Error type: {type(e).__name__}", file=sys.stderr)
        print(f"Error message: {e}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform beat and downbeat detection on a separated audio NPY file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Directory containing the 'sep.npy' file and where 'beat_pred.json' will be saved."
    )
    parser.add_argument(
        '--output_file_name',
        type=str,
        default='beat_pred.json',
        help="Name for the output JSON file within the input_dir."
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoint',
        help="Base directory where the 'beat_transformer' checkpoint subfolder is located."
    )

    args = parser.parse_args()

    print("--- Starting Beat Detection Process ---")
    print(f"Input Directory:      {args.input_dir}")
    print(f"Output File Name:     {args.output_file_name}")
    print(f"Checkpoint Base Dir:  {args.checkpoint_dir}")
    print("---------------------------------------")

    beat_detection_process(
        input_dir=args.input_dir,
        output_file_name=args.output_file_name,
        checkpoint_base_dir=args.checkpoint_dir
    )

    print("---------------------------------------")
    print("--- Beat Detection Script Finished Successfully ---")
    sys.exit(0)
