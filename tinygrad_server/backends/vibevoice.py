"""
VibeVoice backend for the tinygrad TTS server.

Contract (matches system/tts_engines/vibevoice/vibevoice_settings_page.py "Remote server contract"):

    POST /v1/audio/speech
    {
      "model": "microsoft/VibeVoice-1.5b",
      "text": "Speaker 1: Welcome back.\nSpeaker 2: Thanks for having me.",
      "voices": {"Speaker 1": "<base64 wav>", "Speaker 2": "<base64 wav>"},
      "cfg_scale": 1.5,
      "ddpm_inference_steps": 10,
      "response_format": "wav"
    }

Not implemented yet. The real generation path needs, in order:
  1. Qwen2.5 backbone forward pass (architecture matches tinygrad's existing
     Qwen2/Qwen3 support in tinygrad/llm/model.py, but weights ship as HF
     safetensors, not GGUF, so loading needs adapting).
  2. A from-scratch autoregressive generation loop with classifier-free
     guidance: each step runs a conditional AND an unconditional/negative
     forward pass and blends the logits. This is not the plain next-token
     loop tinygrad/llm/serve.py implements.
  3. A DiT-style diffusion head (~286 lines, RMSNorm/SwiGLU/adaLN — see
     vibevoice/modular/modular_vibevoice_diffusion_head.py in the installed
     package) that denoises a continuous acoustic latent over
     ddpm_inference_steps whenever the LM emits a "speech" token, which then
     gets spliced back into the input embedding stream.
  4. A causal conv1d/conv-transpose VAE codec (~1194 lines, streaming-cache
     aware — see vibevoice/modular/modular_vibevoice_tokenizer.py) to turn
     reference audio into conditioning latents and the generated latent
     stream back into a waveform.

See MEMORY/task tracking from the AllTalk session that built this skeleton
for the full architecture notes.
"""

MODEL_IDS = [
    "microsoft/VibeVoice-1.5b",
    "microsoft/VibeVoice-Realtime-0.5B",
    "vibevoice-community/VibeVoice-Large-pt",
]


def generate(payload: dict) -> tuple[bytes, str]:
    from .common import require, content_type_for

    require(payload, "text", "voices")
    raise NotImplementedError(
        "VibeVoice generation is not implemented in the tinygrad server yet. "
        "The HTTP contract and request validation work; the model forward pass "
        "(Qwen2.5 backbone + CFG generation loop + diffusion head + tokenizer "
        "codec) still needs to be ported. See backends/vibevoice.py for the "
        "breakdown of what's missing."
    )
