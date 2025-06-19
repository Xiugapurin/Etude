import os
import sys  # For sys.exit and error printing
from yt_dlp import YoutubeDL
import argparse # For command-line interface
import traceback

def download(url: str, dir_out: str = "") -> str:
    """
    Downloads the best audio from the given URL, converts it to WAV,
    and saves it as 'origin.wav' in the specified output directory.

    Args:
        url (str): The URL of the video/audio to download.
        dir_out (str): The directory to save the 'origin.wav' file.
                       If empty or None, defaults to the current directory.

    Returns:
        str: The full path to the downloaded 'origin.wav' file if successful,
             otherwise an empty string.
    """
    resolved_dir_out = dir_out if (dir_out and dir_out.strip()) else "."
    
    try:
        os.makedirs(resolved_dir_out, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {resolved_dir_out}: {e}", file=sys.stderr)
        return ""
    
    output_template = os.path.join(resolved_dir_out, 'origin')

    ydl_opts = {
        "outtmpl": output_template,
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "ignoreerrors": True,
        "overwrites": True,
    }

    download_reported_success = False
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            download_reported_success = True

    except Exception as e:
        print(f"Exception during yt-dlp download process: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    final_output_path = os.path.join(resolved_dir_out, "origin.wav")

    if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
        return final_output_path
    else:
        if not download_reported_success:
             print(f"Download process for {url} did not complete as expected (e.g. due to an exception).", file=sys.stderr)
        if not os.path.exists(final_output_path):
            print(f"Error: Output file '{final_output_path}' not found after download attempt.", file=sys.stderr)
        elif os.path.exists(final_output_path) and os.path.getsize(final_output_path) == 0:
            print(f"Error: Output file '{final_output_path}' is empty after download attempt.", file=sys.stderr)
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download audio from a URL as 'origin.wav' into a specified output directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Provides better help messages
    )
    parser.add_argument(
        "url",
        type=str,
        help="URL of the video or audio to download."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where 'origin.wav' will be saved. This directory will be created if it doesn't exist."
    )

    args = parser.parse_args() # Expects two arguments from the command line

    try:
        print(f"--- Download Script (_download.py) ---")
        print(f"Attempting to download audio from URL: {args.url}")
        print(f"Target output directory: {args.output_dir}")
        
        downloaded_file_path = download(args.url, args.output_dir)
        
        if downloaded_file_path:
            print(f"Download successful. Audio saved to: {downloaded_file_path}")
            sys.exit(0) # Exit with 0 on success
        else:
            print(f"Download failed or the output file '{os.path.join(args.output_dir, 'origin.wav')}' was not created or is empty.", file=sys.stderr)
            sys.exit(1) # Exit with a non-zero status code to indicate failure
            
    except Exception as e:
        print(f"An unexpected error occurred in the download script: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) # Exit with a non-zero status code