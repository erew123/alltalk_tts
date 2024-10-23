import numpy as np
import torch
import sys
import os
import pdb
import parselmouth
from time import time as ttime
import torch.nn.functional as F
import torchcrepe
from torch import Tensor
import scipy.signal as signal
import pyworld, os, faiss, librosa, torchcrepe
from scipy import signal
from functools import lru_cache
import random
import gc
import re
import traceback
from pathlib import Path
# Get the current working directory
script_dir = Path(__file__).resolve().parent
now_dir = script_dir.parents[3]

sys.path.append(now_dir)
from ..lib.FCPEF0Predictor import FCPEF0Predictor

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)

    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()

    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)

    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)

class VC(object):
    def __init__(self, tgt_sr, config, version, if_f0, net_g, data=None, d=None, train_data=None, debug_rvc=False):
        branding = "AllTalk "
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000
        self.tgt_sr = tgt_sr
        self.__version = version
        self.__if_f0 = if_f0
        self.__net_g = net_g
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device
        self.ref_freqs = [
            65.41,
            82.41,
            110.00,
            146.83,
            196.00,
            246.94,
            329.63,
            440.00,
            587.33,
            783.99,
            1046.50,
        ]
        self.note_dict = self.generate_interpolated_frequencies()

        if data is not None and d is not None and train_data is not None:
            if debug_rvc:
                print(f"[{branding}Debug] Initializing FAISS quantizer with dimensionality: {d}")
            self.quantizer = faiss.IndexFlatL2(d)
            if debug_rvc:
                print(f"[{branding}Debug] FAISS quantizer initialized: {self.quantizer}")

            # Dynamic adjustment of new_nlist
            new_nlist = max(1, len(train_data) // 39)  # Ensure at least 1
            if debug_rvc:
                print(f"[{branding}Debug] Using nlist: {new_nlist}")
            self.index = faiss.IndexIVFFlat(self.quantizer, d, new_nlist)
            if debug_rvc:
                print(f"[{branding}Debug] FAISS IndexIVFFlat initialized: {self.index}")

            if debug_rvc:
                print(f"[{branding}Debug] Training FAISS index with {len(train_data)} training points and {new_nlist} centroids")
            self.index.train(train_data)
            if debug_rvc:
                print(f"[{branding}Debug] FAISS index trained")

            if debug_rvc:
                print(f"[{branding}Debug] Adding {len(data)} data points to FAISS index")
            self.index.add(data)
            if debug_rvc:
                print(f"[{branding}Debug] Data added to FAISS index")

        else:
            self.quantizer = None
            self.index = None
            if debug_rvc:
                print(f"[{branding}Debug] FAISS index not initialized due to missing data.")


    def generate_interpolated_frequencies(self):
        note_dict = []
        for i in range(len(self.ref_freqs) - 1):
            freq_low = self.ref_freqs[i]
            freq_high = self.ref_freqs[i + 1]
            interpolated_freqs = np.linspace(freq_low, freq_high, num=10, endpoint=False)
            note_dict.extend(interpolated_freqs)
        note_dict.append(self.ref_freqs[-1])
        return note_dict


    def autotune_f0(self, f0):
        # Autotunes the given fundamental frequency (f0) to the nearest musical note.
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            # Find the closest note
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = closest_note
        return autotuned_f0

    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        #print("GETTING OPTIMAL DEVICE")
        if torch.cuda.is_available():
            #print("GETTING OPTIMAL DEVICE CUDA")
            return torch.device(f"cuda:{index % torch.cuda.device_count()}")
        elif torch.backends.mps.is_available():
            #print("GETTING OPTIMAL DEVICE MPS")
            return torch.device("mps")
        #print("GETTING OPTIMAL DEVICE CPU")
        return torch.device("cpu")

    def get_f0_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length,
        model="full",
    ):
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        torch_device = self.get_optimal_torch_device()
        audio = torch.from_numpy(x).to(torch_device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        pitch: Tensor = torchcrepe.predict(
            audio,
            self.sr,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=torch_device,
            pad=True,
        )
        p_len = p_len or x.shape[0] // hop_length
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        #print("LEAVING get_f0_crepe_computation")
        return f0

    def get_f0_official_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        model="full",
    ):
        batch_size = 512
        audio = torch.tensor(np.copy(x))[None].float()
        f0, pd = torchcrepe.predict(
            audio,
            self.sr,
            self.window,
            f0_min,
            f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        #print("LEAVING get_f0_official_crepe_computation")
        return f0

    def get_f0_hybrid_computation(
        self,
        methods_str,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length,
    ):
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str:
            methods = [method.strip() for method in methods_str.group(1).split("+")]
        f0_computation_stack = []
        #print(f"Calculating f0 pitch estimations for methods {str(methods)}")
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        for method in methods:
            f0 = None
            if method == "crepe":
                f0 = self.get_f0_crepe_computation(
                    x, f0_min, f0_max, p_len, int(hop_length)
                )
            elif method == "rmvpe":
                if hasattr(self, "model_rmvpe") == False:
                    from ..lib.rmvpe import RMVPE
                    model_path = os.path.join(now_dir, "models", "rvc_base", "rmvpe.pt")
                    self.model_rmvpe = RMVPE(
                        model_path, is_half=self.is_half, device=self.device
                    )
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
                f0 = f0[1:]
            elif method == "fcpe":
                model_path = os.path.join(now_dir, "models", "rvc_base", "fcpe.pt")
                self.model_fcpe = FCPEF0Predictor(
                    model_path,
                    f0_min=int(f0_min),
                    f0_max=int(f0_max),
                    dtype=torch.float32,
                    device=self.device,
                    sampling_rate=self.sr,
                    threshold=0.03,
                )
                f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
                del self.model_fcpe
                gc.collect()
            f0_computation_stack.append(f0)

        #print(f"Calculating hybrid median f0 from the stack of {str(methods)}")
        f0_computation_stack = [fc for fc in f0_computation_stack if fc is not None]
        f0_median_hybrid = None
        if len(f0_computation_stack) == 1:
            f0_median_hybrid = f0_computation_stack[0]
        else:
            f0_median_hybrid = np.nanmedian(f0_computation_stack, axis=0)
        #print("LEAVING get_f0_hybrid_computation")
        return f0_median_hybrid

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        hop_length,
        f0autotune,
        inp_f0=None,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if int(filter_radius) > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            f0 = self.get_f0_crepe_computation(
                x, f0_min, f0_max, p_len, int(hop_length)
            )
        elif f0_method == "crepe-tiny":
            f0 = self.get_f0_crepe_computation(
                x, f0_min, f0_max, p_len, int(hop_length), "tiny"
            )
        elif f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from ..lib.rmvpe import RMVPE
                model_path = os.path.join(now_dir, "models", "rvc_base", "rmvpe.pt")
                self.model_rmvpe = RMVPE(
                    model_path, is_half=self.is_half, device=self.device
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        elif f0_method == "fcpe":
            model_path = os.path.join(now_dir, "models", "rvc_base", "fcpe.pt")
            self.model_fcpe = FCPEF0Predictor(
                model_path,
                f0_min=int(f0_min),
                f0_max=int(f0_max),
                dtype=torch.float32,
                device=self.device,
                sampling_rate=self.sr,
                threshold=0.03,
            )
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
            del self.model_fcpe
            gc.collect()
        elif "hybrid" in f0_method:
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = self.get_f0_hybrid_computation(
                f0_method,
                x,
                f0_min,
                f0_max,
                p_len,
                hop_length,
            )

        if f0autotune == "True":
            f0 = self.autotune_f0(f0)

        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)
        #print("LEAVING get_f0")
        return f0_coarse, f0bak

    def vc(
        self,
        model,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        protect,
    ):
        #print("ENTERING pipeline vc")
        #print(f"audio0 shape: {audio0.shape}")
        feats = torch.from_numpy(audio0)
        #print(f"feats initial dtype: {feats.dtype}")
        if self.is_half:
            feats = feats.half()
            #print("Converting feats to half precision")
        else:
            feats = feats.float()
            #print("Converting feats to float precision")
        #print(f"feats dtype after conversion: {feats.dtype}")
        if feats.dim() == 2:
            #print("feats has 2 dimensions, taking mean along last dimension")
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        #print(f"feats shape after mean: {feats.shape}")
        feats = feats.view(1, -1)
        #print(f"feats shape after view: {feats.shape}")
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        #print(f"padding_mask shape: {padding_mask.shape}")
        #print(f"padding_mask device: {padding_mask.device}")

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if self.__version == "v1" else 12,
        }
        #print(f"inputs source shape: {inputs['source'].shape}")
        #print(f"inputs source device: {inputs['source'].device}")
        #print(f"output_layer: {inputs['output_layer']}")
        t0 = ttime()
        #print("Pipeline vc STEP 1:")
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            #print(f"logits type: {type(logits)}")
            #print(f"logits length: {len(logits)}")
            #print(f"logits[0] shape: {logits[0].shape}")
            feats = model.final_proj(logits[0]) if self.__version == "v1" else logits[0]
            #print(f"feats shape after final_proj or logits[0]: {feats.shape}")
        #print("Pipeline vc STEP 1a:")
        if protect < 0.5 and pitch is not None and pitchf is not None:
            #print("Cloning feats to feats0")
            feats0 = feats.clone()
            #print(f"feats0 shape: {feats0.shape}")
        #print("Pipeline vc STEP 1b:")
        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            #print("Pipeline vc STEP 1c:")
            #print(f"feats shape: {feats.shape}")
            npy = feats[0, :feats.shape[1], :].cpu().numpy()
            #print(f"npy shape: {npy.shape}")
            #print("Pipeline vc STEP 1c1:")
            if self.is_half:
                #print("Pipeline vc STEP 1c2:")
                npy = npy.astype("float32")
                #print(f"Converted npy to float32. npy dtype: {npy.dtype}")
            #print("Pipeline vc STEP 1c3:")
            #print(f"npy size: {npy.size}")
            
            score, ix = self.index.search(npy, k=4)
            
            #print(f"score shape: {score.shape}")
            #print(f"ix shape: {ix.shape}")
            #print("Pipeline vc STEP 1c4:")
            weight = np.square(1 / score)
            #print(f"weight shape: {weight.shape}")
            #print("Pipeline vc STEP 1c5:")
            weight /= weight.sum(axis=1, keepdims=True)
            #print(f"weight shape after normalization: {weight.shape}")
            #print("Pipeline vc STEP 1d:")
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            #print(f"npy shape after weighted sum: {npy.shape}")
            #print("Pipeline vc STEP 1e:")
            if self.is_half:
                npy = npy.astype("float16")
                #print(f"Converted npy to float16. npy dtype: {npy.dtype}")
                #print("Pipeline vc STEP 1f:")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )
            #print(f"feats shape after updating with npy: {feats.shape}")
        #print("Pipeline vc STEP 2:")
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        #print(f"feats shape after interpolation: {feats.shape}")
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            #print(f"feats0 shape after interpolation: {feats0.shape}")
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        #print(f"Initial p_len: {p_len}")
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            #print(f"Updated p_len to match feats shape: {p_len}")
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
                #print(f"Updated pitch shape: {pitch.shape}")
                #print(f"Updated pitchf shape: {pitchf.shape}")
        #print("Pipeline vc STEP 3:")
        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            #print(f"pitchff shape: {pitchff.shape}")
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            #print(f"pitchff shape after unsqueeze: {pitchff.shape}")
            feats = feats * pitchff + feats0 * (1 - pitchff)
            #print(f"feats shape after updating with pitchff: {feats.shape}")
            feats = feats.to(feats0.dtype)
            #print(f"feats dtype after conversion: {feats.dtype}")
        p_len = torch.tensor([p_len], device=self.device).long()
        #print(f"p_len tensor: {p_len}")
        #print(f"p_len device: {p_len.device}")
        #print("Pipeline vc p_len is:", p_len)
        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                audio1 = (
                    (self.__net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (self.__net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        #print(f"audio1 shape: {audio1.shape}")
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        #print(f"Pipeline vc time breakdown:")
        #print(f"  Step 1: {t1 - t0:.3f}s")
        #print(f"  Step 2: {t2 - t1:.3f}s")
        #print(f"  Total: {t2 - t0:.3f}s")
        #print("Leaving pipeline vc")
        return audio1

    def pipeline(
        self,
        model,
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
        f0_file=None,
    ):
        if file_index != "" and os.path.exists(file_index) == True and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as error:
                print(error)
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name") == True:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except Exception as error:
                print(error)
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if self.__if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                hop_length,
                f0autotune,
                inp_f0,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        for t in opt_ts:
            t = t // self.window * self.window
            if self.__if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        index,
                        big_npy,
                        index_rate,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        index,
                        big_npy,
                        index_rate,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if self.__if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    index,
                    big_npy,
                    index_rate,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    index,
                    big_npy,
                    index_rate,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt