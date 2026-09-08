import numpy as np
import re
import unicodedata
import pickle
import sys
import torch
import torchaudio
import logging
import ffmpeg
from pathlib import Path

# Get the parent directory of the current file
this_dir = Path(__file__).resolve().parent.parent.parent.parent.parent


class _DummyObject:
    """Stand-in for fairseq/omegaconf objects when unpickling checkpoints."""
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that stubs out fairseq/omegaconf/hydra classes so we can
    load HuBERT/ContentVec checkpoints without importing fairseq at all."""
    def find_class(self, module, name):
        if module.startswith(("fairseq", "omegaconf", "hydra")):
            return _DummyObject
        return super().find_class(module, name)


# Build a pickle-module substitute for torch.load(pickle_module=...)
import types as _types
_safe_pickle_module = _types.ModuleType("_safe_pickle")
_safe_pickle_module.Unpickler = _SafeUnpickler
_safe_pickle_module.load = pickle.load


class _TorchaudioHuBERTWrapper(torch.nn.Module):
    """Wraps torchaudio's Wav2Vec2Model to match the fairseq HuBERT API
    that RVC's pipeline expects (extract_features with source/padding_mask/output_layer)."""

    def __init__(self, model):
        super().__init__()
        self.hubert = model  # registered as submodule so .to()/.half()/.eval() propagate

    def extract_features(self, source, padding_mask=None, output_layer=None):
        features_list, _ = self.hubert.extract_features(source, num_layers=output_layer)
        return (features_list[-1], padding_mask)


def _remap_fairseq_keys(state_dict):
    """Remap fairseq HuBERT checkpoint keys to torchaudio Wav2Vec2Model names."""
    mapped = {}
    for k, v in state_dict.items():
        new_key = k
        # Feature extractor conv layers
        if k.startswith("feature_extractor.conv_layers."):
            parts = k.split(".")
            layer_idx = parts[2]
            sub_idx = parts[3]   # "0" = conv, "2" = group norm (layer 0 only)
            rest = ".".join(parts[4:])
            if sub_idx == "0":
                new_key = f"feature_extractor.conv_layers.{layer_idx}.conv.{rest}"
            elif sub_idx == "2":
                new_key = f"feature_extractor.conv_layers.{layer_idx}.layer_norm.{rest}"
        # Post-extract projection
        elif k.startswith("post_extract_proj."):
            new_key = k.replace("post_extract_proj.", "encoder.feature_projection.projection.")
        # Feature projection layer norm (top-level layer_norm in fairseq)
        elif k in ("layer_norm.weight", "layer_norm.bias"):
            new_key = k.replace("layer_norm.", "encoder.feature_projection.layer_norm.")
        # Positional conv (weight norm: weight_g/weight_v → parametrizations)
        elif k == "encoder.pos_conv.0.bias":
            new_key = "encoder.transformer.pos_conv_embed.conv.bias"
        elif k == "encoder.pos_conv.0.weight_g":
            new_key = "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0"
        elif k == "encoder.pos_conv.0.weight_v":
            new_key = "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1"
        # Encoder layer norm
        elif k.startswith("encoder.layer_norm."):
            new_key = k.replace("encoder.layer_norm.", "encoder.transformer.layer_norm.")
        # Transformer layers
        elif k.startswith("encoder.layers."):
            new_key = k.replace("encoder.layers.", "encoder.transformer.layers.")
            new_key = new_key.replace(".self_attn.", ".attention.")
            new_key = new_key.replace(".self_attn_layer_norm.", ".layer_norm.")
            new_key = new_key.replace(".fc1.", ".feed_forward.intermediate_dense.")
            new_key = new_key.replace(".fc2.", ".feed_forward.output_dense.")
        # Skip training-only keys
        elif k in ("mask_emb", "label_embs_concat") or k.startswith("final_proj."):
            continue
        mapped[new_key] = v
    return mapped


def _load_embedding(model_path):
    """Load a HuBERT/ContentVec checkpoint into a torchaudio model.
    Uses a safe unpickler to avoid importing fairseq entirely."""
    checkpoint = torch.load(
        str(model_path), map_location="cpu", weights_only=False,
        pickle_module=_safe_pickle_module,
    )
    state_dict = checkpoint.get("model", checkpoint)

    # Remap fairseq key names to torchaudio names
    remapped = _remap_fairseq_keys(state_dict)

    model = torchaudio.models.hubert_base()
    model_state = model.state_dict()
    matched = {k: v for k, v in remapped.items()
               if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(matched, strict=False)
    logging.info(f"Loaded {len(matched)}/{len(model_state)} weights into torchaudio HuBERT")

    wrapped = _TorchaudioHuBERTWrapper(model)

    # Add final_proj to the wrapper (pipeline accesses model.final_proj directly)
    fp_weight = state_dict.get("final_proj.weight")
    fp_bias = state_dict.get("final_proj.bias")
    if fp_weight is not None:
        out_dim, in_dim = fp_weight.shape
        wrapped.final_proj = torch.nn.Linear(in_dim, out_dim)
        wrapped.final_proj.weight = torch.nn.Parameter(fp_weight)
        if fp_bias is not None:
            wrapped.final_proj.bias = torch.nn.Parameter(fp_bias)

    return [wrapped], {}, None


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
    embedding_list = {
        "contentvec": "contentvec_base.pt",
        "hubert": "hubert_base.pt",
    }

    try:
        this_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        model_path = this_dir / "models" / "rvc_base" / embedding_list[embedder_model]

        return _load_embedding(model_path)

    except KeyError as e:
        logging.error(f"Invalid embedder model name: {embedder_model}")
        raise ValueError(f"Invalid embedder model name: {embedder_model}") from e
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        raise RuntimeError(f"Error loading embedding model: {e}") from e
