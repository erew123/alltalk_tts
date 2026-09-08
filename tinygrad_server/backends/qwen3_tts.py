"""
Qwen3-TTS backend for the tinygrad TTS server.

Contract (matches system/tts_engines/qwen3tts/qwen3tts_settings_page.py "Remote server contract"):
`mode` selects which other fields are present.

    POST /v1/audio/speech
    // mode: "customvoice"
    {"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "mode": "customvoice", "text": "...",
     "language": "English", "speaker": "Vivian", "instruct": "", "response_format": "wav"}
    // mode: "voicedesign"
    {"model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "mode": "voicedesign", "text": "...",
     "language": "English", "instruct": "A deep, calm male narrator voice...", "response_format": "wav"}
    // mode: "base" (voice cloning)
    {"model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "mode": "base", "text": "...", "language": "English",
     "ref_audio": "<base64 wav>", "ref_text": "transcript of ref_audio", "response_format": "wav"}

Not implemented yet. Grounded against the real QwenLM/Qwen3-TTS source
(qwen_tts/core/models/modeling_qwen3_tts.py) rather than guessed -- this is
NOT a simple "load a GGUF Qwen3 and decode tokens" system, despite community
GGUF exports existing for the backbone. The real pipeline has:
  1. A custom "Talker" LM (Qwen3TTSTalkerForConditionalGeneration) with its
     own multimodal RoPE (mrope sections) and custom attention -- not vanilla
     Qwen3, so a GGUF conversion of just the backbone doesn't cover this.
  2. A second autoregressive sub-model, the "code predictor"
     (Qwen3TTSTalkerCodePredictorModel), that generates residual RVQ codebook
     tokens conditioned on the Talker's hidden states -- effectively a second
     transformer decode loop per generation step.
  3. A full ECAPA-TDNN-style speaker encoder (Res2Net blocks, squeeze-
     excitation, attentive statistics pooling) for x-vector voice cloning in
     Base-model mode.
  4. Two versions of the codec/tokenizer (25Hz V1, 12Hz V2 -- see
     qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py) to
     encode reference audio and decode generated codes back to a waveform.
  5. A generation loop that fuses text embeddings, speaker embeddings,
     language/dialect routing, and codec-prefill sequences before the two-
     stage talker -> code-predictor decode even starts.

See MEMORY/task tracking from the AllTalk session that built this skeleton
for the full architecture notes.
"""

MODEL_IDS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]


def generate(payload: dict) -> tuple[bytes, str]:
    from .common import require

    mode = payload.get("mode", "base")
    require(payload, "text")
    if mode == "customvoice":
        require(payload, "speaker")
    elif mode == "voicedesign":
        require(payload, "instruct")
    else:
        require(payload, "ref_audio", "ref_text")

    raise NotImplementedError(
        f"Qwen3-TTS generation (mode={mode}) is not implemented in the tinygrad "
        "server yet. The HTTP contract and request validation work; the model "
        "forward pass (Talker LM + code-predictor sub-model + speaker encoder + "
        "codec) still needs to be ported. See backends/qwen3_tts.py for the "
        "breakdown of what's missing."
    )
