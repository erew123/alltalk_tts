import numpy as np
import re
import unicodedata
from fairseq import checkpoint_utils
from fairseq.data import Dictionary
import torch

import logging

logging.getLogger("fairseq").setLevel(logging.WARNING)
import sys
import os
import numpy as np
import asyncio
import ffmpeg
from pathlib import Path
import subprocess

# Get the parent directory of the current file
this_dir = Path(__file__).resolve().parent.parent.parent.parent.parent

def load_audio(file, sampling_rate):
    try:
        file = str(file).strip(" ").strip('"').strip("\n").strip('"').strip(" ")

        try:
            # Use ffmpeg-python
            stream = (
                ffmpeg
                .input(file)
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=str(sampling_rate))
                .run(capture_stdout=True, capture_stderr=True)
            )
            out = stream[0]  # Get stdout data
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode('utf-8')}")
            raise RuntimeError(f"FFmpeg error: {e.stderr.decode('utf-8')}") from e

        return np.frombuffer(out, np.float32).flatten()

    except Exception as error:
        print(f"Error loading audio: {error}")
        raise RuntimeError(f"Failed to load audio: {error}") from error



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
        torch.serialization.add_safe_globals([Dictionary])
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
