# help_content.py
# pylint: disable=no-member

class AllTalkHelpContent:
    """CSS and help content for help_content.py"""
    custom_css = """
    /* Add this to your existing CSS */
    .gradio-container .prose {
        max-width: none !important;
        padding: 0.5rem !important; /* Reduced padding */
        margin: 0 !important;
    }

    .custom-markdown div {
    border: none !important; /* Remove the inner border */
    margin-top: 0 !important; /* Remove top margin */
    margin-bottom: 0 !important; /* Remove bottom margin */
    padding-top: 0 !important; /* Remove top padding */
    padding-bottom: 0 !important; /* Remove bottom padding */
    }

    /* Update the existing custom-markdown class */
    .custom-markdown {
        font-size: 15px !important;
        line-height: 1.6 !important;
        color: var(--body-text-color) !important;
        background-color: var(--background-fill-primary) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        margin: 0 !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        color: rgba(156, 163, 175, 1) !important;
    }

    .custom-markdown h1,
    .custom-markdown h2,
    .custom-markdown h3,
    .custom-markdown h4,
    .custom-markdown h5,
    .custom-markdown h6 {
        color: var(--heading-text-color, var(--body-text-color)) !important;
        font-weight: 600 !important;
    }

    .custom-markdown p,
    .custom-markdown li,
    .custom-markdown ul,
    .custom-markdown ol {
        color: rgba(156, 163, 175, 1) !important;
    }

    .gradio-container .prose > * {
        margin: 0 !important;
        padding: 0 !important;
    }

    .custom-markdown li {
        font-size: 0.95rem !important;
        margin-bottom: 0.2rem !important;
    }

    .custom-markdown li p {
        margin: 0 !important;
    }

    .custom-markdown li + li {
        margin-top: 0.2rem !important;
    }

    .custom-markdown li > ul,
    .custom-markdown li > ol {
        margin: 0.2rem 0 0.2rem 1rem !important;
    }

    .custom-markdown h2 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 1rem !important;
        color: var(--heading-text-color, var(--body-text-color)) !important;
        border-bottom: 1px solid var(--border-color-primary) !important;
        padding-bottom: 0.5rem !important;
    }

    .custom-markdown p:first-of-type {
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        margin-bottom: 1rem !important;
    }

    .custom-markdown p {
        font-size: 0.95rem !important;
        margin: 0.8rem 0 !important;
    }

    .custom-markdown ul,
    .custom-markdown ol {
        margin: 0.8rem 0 !important;
        padding-left: 1.5rem !important;
    }

    .custom-markdown code {
        font-family: ui-monospace, monospace !important;
        background-color: var(--background-fill-secondary) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-size: 0.8rem !important;
        color: var(--body-text-color) !important;
    }

    .custom-markdown strong {
        font-weight: 600 !important;
        color: var(--body-text-color) !important;
    }

    .custom-markdown em {
        font-style: italic !important;
    }

    .gradio-row > div {
        flex: 1 1 50% !important;
        min-width: 0 !important;
    }
    """

    ENGINE_INFORMATION = """
    ## TTS Engine Capabilities Help

    This guide explains the various capabilities that **may** be available in different TTS engines and models. Each capability affects how the TTS engine processes and generates speech output.
    """

    ENGINE_INFORMATION1 = """
    ## Performance Features

    - **DeepSpeed Capable**: Enables GPU acceleration using the DeepSpeed optimization library.
        - Not supported by Qwen3-TTS.

    - **Low VRAM Capable**: Optimized for systems with limited GPU memory.
        - Not exposed by Qwen3-TTS. Unload the model between uses to free VRAM instead.

    - **Streaming Capable**: Enables real-time speech generation.
        - Not implemented in this engine. Qwen3-TTS supports true token-level streaming
          in its reference server, but this integration generates the full clip and then
          chunks the resulting WAV for playback.

    ## Voice Control Features

    - **Pitch Capable**: Allows adjustment of voice pitch.
        - Not exposed as a slider. For CustomVoice/VoiceDesign models, describe pitch in
          the preset's **instruct** text instead (e.g. "a higher-pitched voice").

    - **Generation Speed Capable**: Controls speech rate.
        - Not exposed by the qwen-tts package's generation API.

    - **Temperature Capable**: Controls output randomness.
        - Not exposed by the qwen-tts package's generation API.
    """

    ENGINE_INFORMATION2 = """
    ## Quality Enhancement Features

    - **Repetition Penalty Capable**: Prevents unnatural speech patterns.
        - Not exposed by the qwen-tts package's generation API.

    ## Multi-Feature Support

    - **Multi-Languages Capable Models**: Each model supports 10 languages (Chinese, English,
      Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian) plus "Auto" detection.

    - **Multi-Voice Capable**: Supports multiple speaking voices via Custom Voice Presets
      (CustomVoice models), Voice Design Presets (VoiceDesign model), or reference-audio
      voice cloning (Base models).

    - **Multi-Model Capable Engine**: Supports both the 0.6B and 1.7B checkpoints, across
      Base, CustomVoice and VoiceDesign variants.

    ## Technical Features

    - **Default Audio Output Format**: mono WAV.

    - **Platform Support**: Windows, Linux and macOS (CUDA, MPS or CPU).
    """

    DEFAULT_SETTINGS = """
    ## TTS Engine Settings Help

    This guide explains the settings and configuration options available for individual Text-to-Speech engines within AllTalk.
    """

    DEFAULT_SETTINGS1 = """
    ## Engine Capabilities & Controls

    - **Low VRAM Mode**
        - Not applicable for Qwen3-TTS (model stays resident; unload/reload to free VRAM)

    - **DeepSpeed Capability**
        - Not supported by Qwen3-TTS

    - **Temperature Control**
        - Not exposed by the qwen-tts package

    - **Repetition Penalty**
        - Not exposed by the qwen-tts package

    - **Pitch Adjustment**
        - Not applicable as a slider. Describe pitch/tone inside a preset's **instruct** text
          on the Custom Voice Presets / Voice Design Presets tab instead.

    - **Generation Speed**
        - Not exposed by the qwen-tts package
    """

    DEFAULT_SETTINGS2 = """

    ## Voice Configuration

    ### OpenAI Voice Mappings
    - Only relevant when using the OpenAI-compatible API endpoint
    - Maps OpenAI's six standard voices to equivalent voices in the current engine:
        - `alloy`
        - `echo`
        - `fable`
        - `nova`
        - `onyx`
        - `shimmer`
    - Essential for maintaining compatibility with OpenAI API calls
    - Each mapping can be customized to any available voice in the current engine

    Information on the OpenAI Endpoint is available in the [Github Wiki](https://github.com/erew123/alltalk_tts/wiki)

    ### Default Voices
    - **Default/Character Voice**: Used when no specific voice is specified in API calls
    - **Narrator Voice**: Separate default for narrator-specific text
    - These defaults are engine-specific and won't affect other TTS engines
    - Can be overridden by explicitly specifying voices in API calls

    ## Important Notes

    - Settings availability is determined by engine capabilities
    - Grayed-out options indicate features not supported by the current engine
    - Changes only affect the currently selected TTS engine
    - Settings here act as defaults but can be overridden via API parameters
    - All changes require clicking "Update Settings" to take effect
    - Some settings require an engine reload to take effect
    """

    HELP_PAGE = """
    ## Qwen3-TTS Engine Help

    Qwen3-TTS is Alibaba Cloud's open-source text-to-speech series, released in 0.6B and 1.7B
    "12Hz" checkpoints. Three model flavours are supported:

    - **CustomVoice** — a fixed roster of built-in speakers, each steerable with a free-form
      natural-language **instruct** property for emotion/style. This is the "custom voice
      properties" system: manage named presets (speaker + instruct + language) on the
      **Custom Voice Presets** tab.
    - **VoiceDesign** — no fixed speakers at all; describe the voice you want entirely in
      natural language via the **instruct** property. Manage presets on the **Voice Design
      Presets** tab. 1.7B only.
    - **Base** — 3-second reference-audio voice cloning. Drop a `.wav` + matching
      `.reference.txt` pair into this engine's `voices/` folder.
    """

    HELP_PAGE1 = """
    ## Installation

    The engine auto-installs the `qwen-tts` package on first model load. You can also install it manually:

    ```bash
    pip install -U qwen-tts
    ```

    Model weights are pulled from HuggingFace automatically when a model is first loaded. To pre-download instead:

    ```bash
    pip install "huggingface_hub[cli]"
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --local-dir models/qwen3tts/Qwen3-TTS-12Hz-1.7B-CustomVoice
    ```

    Any of the five checkpoints can be pre-downloaded this way, into a matching
    `models/qwen3tts/<checkpoint-name>/` folder:

    - `Qwen3-TTS-12Hz-0.6B-Base`
    - `Qwen3-TTS-12Hz-0.6B-CustomVoice`
    - `Qwen3-TTS-12Hz-1.7B-Base`
    - `Qwen3-TTS-12Hz-1.7B-CustomVoice`
    - `Qwen3-TTS-12Hz-1.7B-VoiceDesign`

    ## Hardware

    | Device | Setup | Notes |
    |---|---|---|
    | **NVIDIA CUDA** | bf16 + flash_attention_2 | Recommended. Falls back to sdpa if flash-attn isn't installed. |
    | **Apple Silicon (MPS)** | fp32 + sdpa | Falls back to CPU if MPS loading fails. |
    | **CPU** | fp32 + sdpa | Functional but slow. |

    ## Custom Voice Properties (CustomVoice models)

    Each preset on the **Custom Voice Presets** tab has four fields:

    - **preset_name** — what shows up in AllTalk's voice list/dropdowns
    - **speaker** — one of the built-in Qwen3-TTS speakers (Vivian, Serena, Uncle_Fu, Dylan,
      Eric, Ryan, Aiden, Ono_Anna, Sohee)
    - **instruct** — a free-form emotion/style instruction, e.g. "Speak with sharp, furious
      anger." or "" for the speaker's neutral default delivery
    - **language** — a fixed language for this preset, or "Auto" to use whatever language
      AllTalk passes in for that generation

    Add as many presets per speaker as you like (e.g. `Ryan`, `Ryan_Angry`, `Ryan_Sad`) —
    each is just a different instruct on top of the same underlying speaker.

    ## Voice Design (VoiceDesign model, 1.7B only)

    Voice Design presets have no `speaker` field — the **instruct** text alone describes the
    entire voice from scratch (pitch, tone, pacing, character), e.g. "A deep, calm male
    narrator voice, slow-paced and authoritative."

    ## Voice Cloning (Base models)

    Drop a matching pair into the engine's `voices/` folder:

    ```
    system/tts_engines/qwen3tts/voices/my_voice.wav
    system/tts_engines/qwen3tts/voices/my_voice.reference.txt
    ```

    `my_voice.reference.txt` must contain the exact transcript of what is spoken in the WAV.
    The WAV's basename becomes the voice name in AllTalk.
    """

    HELP_PAGE2 = """
    ## Languages

    Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian,
    plus "Auto" for automatic detection. Set per-preset, or leave a preset on "Auto" to use
    the language AllTalk passes in for that call.

    ## Audio Output

    - **Streaming**: Not natively streamed by this integration — audio is generated in full,
      then chunked for playback if a streaming request is made.
    - **Format**: mono WAV.

    ## Troubleshooting

    | Error | Solution |
    |-------|----------|
    | `qwen-tts install failed` | Manually run `pip install -U qwen-tts` and check the error |
    | `Voice design preset 'X' not found` | Add a preset named `X` on the Voice Design Presets tab |
    | `Voice 'X' needs a matching '.wav' + '.reference.txt' pair` | Drop both files into the engine's `voices/` folder |
    | `flash_attention_2 failed` | Normal on Mac/CPU; engine auto-falls back to sdpa. On CUDA: `pip install flash-attn --no-build-isolation` |
    | mps load failed | Engine auto-falls back to CPU; or set Device to `cpu` manually on the Qwen3-TTS tab |
    | Very poor / wrong-language audio | Set an explicit `language` on the preset instead of "Auto" |
    """
