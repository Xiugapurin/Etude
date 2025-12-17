# etude/utils/download.py

import sys
import traceback
from pathlib import Path
from typing import Union
import argparse

from yt_dlp import YoutubeDL

from .logger import logger

def download_audio_from_url(url: str, output_path: Union[str, Path], verbose: bool = False) -> bool:
    """
    Downloads the best audio from a given URL, converts it to WAV,
    and saves it to the specified output path.

    Args:
        url (str): The URL of the video or audio to download.
        output_path (Union[str, Path]): The full path (including filename and extension)
                                        where the final .wav file will be saved.

    Returns:
        bool: True if the download and conversion were successful, False otherwise.
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_stem = output_path.stem

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warn(f"Could not create directory {output_dir}: {e}")
        return False

    output_template = output_dir / output_stem
    ydl_opts = {
        "outtmpl": str(output_template),
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "ignoreerrors": True,
        "overwrites": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        if verbose:
            logger.substep(f"Downloading from {url}...")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.warn(f"Download exception: {e}")
        return False

    final_output_path = output_path

    if final_output_path.exists() and final_output_path.stat().st_size > 0:
        if verbose:
            logger.info(f"Audio saved to: {final_output_path}")
        return True
    else:
        logger.warn("Download failed: Output file not created or is empty.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to download audio from a URL as a WAV file."
    )
    parser.add_argument("url", type=str, help="URL of the video/audio.")
    parser.add_argument("output_path", type=str, help="Full path for the output .wav file.")
    args = parser.parse_args()

    success = download_audio_from_url(args.url, args.output_path, verbose=True)

    if success:
        logger.success("Download complete.")
        sys.exit(0)
    else:
        # Error already logged by download_audio_from_url
        sys.exit(1)