import os
import sys
import time
import torch
import logging
import faiss
import numpy as np
import soundfile as sf
import librosa
from functools import lru_cache

now_dir = os.getcwd()
sys.path.append(now_dir)

from ..infer.pipeline import VC
from ..lib.utils import load_audio
from ..lib.tools.split_audio import process_audio, merge_audio
from ..lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from ..configs.config import Config
from ..lib.utils import load_embedding

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

config = Config()

@lru_cache
def load_hubert(embedder_model):
    models, _, _ = load_embedding(embedder_model)
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()
    return hubert_model

def voice_conversion(
    vc,
    sid=0,
    input_audio_path=None,
    f0_up_key=None,
    f0_file=None,
    f0_method=None,
    file_index=None,
    index_rate=None,
    resample_sr=0,
    rms_mix_rate=None,
    protect=None,
    hop_length=None,
    output_path=None,
    split_audio=False,
    f0autotune=False,
    filter_radius=None,
    embedder_model=None,
    debug_rvc=False,
):
    tgt_sr = vc.tgt_sr

    f0_up_key = int(f0_up_key)
    try:
        print(f"Loading audio from {input_audio_path}") if debug_rvc else None
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95

        if audio_max > 1:
            audio /= audio_max

        print(f"Loading hubert model with {embedder_model}") if debug_rvc else None
        hubert_model = load_hubert(embedder_model)

        file_index = (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        if split_audio == "True":
            print("Splitting audio") if debug_rvc else None
            result, new_dir_path = process_audio(input_audio_path)
            if result == "Error":
                return "Error with Split Audio", None
            dir_path = (
                new_dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )
            if dir_path != "":
                paths = [
                    os.path.join(root, name)
                    for root, _, files in os.walk(dir_path, topdown=False)
                    for name in files
                    if name.endswith(".wav") and root == dir_path
                ]
            try:
                for path in paths:
                    voice_conversion(
                        vc,
                        sid,
                        path,
                        f0_up_key,
                        None,
                        f0_method,
                        file_index,
                        index_rate,
                        resample_sr,
                        rms_mix_rate,
                        protect,
                        hop_length,
                        path,
                        False,
                        f0autotune,
                        embedder_model,
                        debug_rvc,
                    )
            except Exception as error:
                print(f"Error processing segmented audio: {error}")
                return f"Error {error}"
            print("Finished processing segmented audio, now merging audio...") if debug_rvc else None
            merge_timestamps_file = os.path.join(
                os.path.dirname(new_dir_path),
                f"{os.path.basename(input_audio_path).split('.')[0]}_timestamps.txt",
            )
            tgt_sr, audio_opt = merge_audio(merge_timestamps_file)
            os.remove(merge_timestamps_file)

        else:
            print("Processing audio with VC pipeline") if debug_rvc else None
            audio_opt = vc.pipeline(
                hubert_model,
                sid,
                audio,
                input_audio_path,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                filter_radius,
                tgt_sr,
                resample_sr,
                rms_mix_rate,
                protect,
                hop_length,
                f0autotune,
                f0_file=f0_file,
            )
            
        # Resample the audio to the target sample rate before saving
        if tgt_sr != resample_sr and resample_sr >= 16000:
            print(f"Resampling audio from {tgt_sr} to {resample_sr}") if debug_rvc else None
            audio_opt = librosa.resample(audio_opt, tgt_sr, resample_sr)
            tgt_sr = resample_sr
            
        if output_path is not None:
            print(f"Saving file to {output_path}") if debug_rvc else None
            sf.write(output_path, audio_opt, tgt_sr, format="WAV")
            print(f"File saved to {output_path}") if debug_rvc else None

        return (tgt_sr, audio_opt)

    except Exception as error:
        print(f"Error during voice conversion: {error}")
        return None, None


@lru_cache
def get_vc(weight_root, sid, file_index=None, training_data_size=10000, debug_rvc=False):
    net_g = None
    branding="AllTalk "
    
    if debug_rvc:
        print(f"[{branding}Debug] Entering get_vc function")
        print(f"[{branding}Debug] weight_root: {weight_root}")
        print(f"[{branding}Debug] sid: {sid}")
        print(f"[{branding}Debug] file_index: {file_index}")
        print(f"[{branding}Debug] training_data_size: {training_data_size}")

    if debug_rvc:
        print(f"[{branding}Debug] Loading model checkpoint")
        
    person = weight_root
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    
    if debug_rvc:
        print(f"[{branding}Debug] Model version: {version}")
    
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    
    if file_index is not None:
        if debug_rvc:
            print(f"[{branding}Debug] Loading index file: {file_index}")
        index_file = file_index
        index = faiss.read_index(index_file)
        if debug_rvc:
            print(f"[{branding}Debug] Extracting data from index")
        data = index.reconstruct_n(0, index.ntotal)
        d = index.d
        nlist = index.nlist
        
        training_data_size = min(training_data_size, len(data))
        train_data = data[:training_data_size]
        
    else:
        if debug_rvc:
            print(f"[{branding}Debug] No index file provided")
        data = None
        d = None
        nlist = None
        train_data = None
    
    if debug_rvc:
        print(f"[{branding}Debug] Creating VC instance")
        
    if data is not None and d is not None and nlist is not None and train_data is not None:
        if debug_rvc:
            print(f"[{branding}Debug] Data: {data}")
            print(f"[{branding}Debug] Dimensionality: {d}")
            print(f"[{branding}Debug] Number of centroids: {nlist}")
            print(f"[{branding}Debug] Training data size: {len(train_data)}")
        vc = VC(tgt_sr, config, version, if_f0, net_g, data, d, train_data, debug_rvc)
    else:
        vc = VC(tgt_sr, config, version, if_f0, net_g)

    if debug_rvc:
        print(f"[{branding}Debug] Leaving get_vc function")
    return vc


def infer_pipeline(
    f0up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0method,
    audio_input_path,
    audio_output_path,
    model_path,
    index_path,
    split_audio,
    f0autotune,
    embedder_model,
    training_data_size,
    debug_rvc,
):
    if index_path is None or index_path.strip() == "":
        file_index = None
    else:
        file_index = index_path
    vc = get_vc(model_path, 0, file_index, training_data_size, debug_rvc)
    
    try:
        start_time = time.time()
        voice_conversion(
            vc,
            sid=0,
            input_audio_path=audio_input_path,
            f0_up_key=f0up_key,
            f0_file=None,
            f0_method=f0method,
            file_index=index_path,
            index_rate=float(index_rate),
            rms_mix_rate=float(rms_mix_rate),
            protect=float(protect),
            hop_length=hop_length,
            output_path=audio_output_path,
            split_audio=split_audio,
            f0autotune=f0autotune,
            filter_radius=filter_radius,
            embedder_model=embedder_model,
            debug_rvc=debug_rvc
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Conversion completed. Output file: '{audio_output_path}' in {elapsed_time:.2f} seconds.") if debug_rvc else None

    except Exception as error:
        print(f"Voice conversion failed: {error}")

