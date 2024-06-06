import numpy as np
import re
import unicodedata
from fairseq import checkpoint_utils

import logging

logging.getLogger("fairseq").setLevel(logging.WARNING)
import sys
import os
import numpy as np
import asyncio
from ffmpeg.asyncio import FFmpeg
from pathlib import Path
import subprocess

# Get the parent directory of the current file
this_dir = Path(__file__).resolve().parent.parent.parent.parent.parent

def load_audio(file, sampling_rate):
    try:
        # Get the parent directory of the current file
        this_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")

        # Set the ffmpeg_path variable based on the operating system
        if sys.platform == "win32":
            ffmpeg_path = os.path.join(this_dir, "system", "win_ffmpeg", "ffmpeg.exe")
        else:
            ffmpeg_path = "ffmpeg"  # Default path for Linux and macOS

        # Initialize the process variable
        process = subprocess.Popen(
            [ffmpeg_path, "-y", "-i", file, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sampling_rate), "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        out, err = process.communicate()
        if process.returncode != 0:
            print(f"FFmpeg error: {err.decode('utf-8')}")  # Debug statement
            raise RuntimeError(f"FFmpeg error: {err.decode('utf-8')}")
        #print("Audio loaded successfully")  # Debug statement
    except Exception as error:
        print(f"Error loading audio: {error}")  # Debug statement
        raise RuntimeError(f"Failed to load audio: {error}")

    return np.frombuffer(out, np.float32).flatten()



def format_title(title):
    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model):
    #print("EMBEDDER MODEL IS", embedder_model)
    embedding_list = {
        "contentvec": "contentvec_base.pt",
        "hubert": "hubert_base.pt",
    }
    
    try:
        this_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        model_path = this_dir / "models" / "rvc_base" / embedding_list[embedder_model]
        #print("MODEL PATH IS", model_path)
        
        # Load model ensemble and task
        models = checkpoint_utils.load_model_ensemble_and_task(
            [f"{model_path}"],
            suffix="",
        )
        
        #print(f"Embedding model {embedder_model} loaded successfully.")
        return models
    except KeyError as e:
        logging.error(f"Invalid embedder model name: {embedder_model}")
        raise ValueError(f"Invalid embedder model name: {embedder_model}") from e
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        raise RuntimeError(f"Error loading embedding model: {e}") from e
