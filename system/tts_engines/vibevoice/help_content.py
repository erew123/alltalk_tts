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
        margin: 0 !important; /* Changed from 1rem 0 to 0 */
        max-width: 100% !important;
        box-sizing: border-box !important;
        /* Default text color for all content (grey) */
        color: rgba(156, 163, 175, 1) !important; /* Adjust this grey value to match your interface */
    }

    /* Make headings white */
    .custom-markdown h1,
    .custom-markdown h2,
    .custom-markdown h3,
    .custom-markdown h4,
    .custom-markdown h5,
    .custom-markdown h6 {
        color: var(--heading-text-color, var(--body-text-color)) !important;
        font-weight: 600 !important;
    }

    /* Keep all other elements in the grey color */
    .custom-markdown p,
    .custom-markdown li,
    .custom-markdown ul,
    .custom-markdown ol {
        color: rgba(156, 163, 175, 1) !important; /* Same grey as the base text */
    }

    /* Additional targeting for any wrapping elements */
    .gradio-container .prose > * {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Update the list item spacing in your existing CSS */
    .custom-markdown li {
        font-size: 0.95rem !important;
        margin-bottom: 0.2rem !important; /* Reduced from 0.5rem to 0.3rem */
    }

    /* Add specific styling for definition-style lists (like your Default, Recommendation, Tip items) */
    .custom-markdown li p {
        margin: 0 !important; /* Remove paragraph margins within list items */
    }

    /* If you need even tighter spacing for specific types of lists */
    .custom-markdown li + li {
        margin-top: 0.2rem !important; /* Space between consecutive list items */
    }

    /* Ensure nested lists maintain proper spacing */
    .custom-markdown li > ul,
    .custom-markdown li > ol {
        margin: 0.2rem 0 0.2rem 1rem !important; /* Reduced from 0.5rem */
    }

    /* Consistent heading styles */
    .custom-markdown h2 {
        font-size: 1.2rem !important; /* Fixed size relative to root */
        font-weight: 600 !important;
        margin: 1.5rem 0 1rem !important;
        color: var(--heading-text-color, var(--body-text-color)) !important;
        border-bottom: 1px solid var(--border-color-primary) !important;
        padding-bottom: 0.5rem !important;
    }

    /* First paragraph styling */
    .custom-markdown p:first-of-type {
        font-size: 0.95rem !important; /* Match base size */
        font-weight: 400 !important;
        margin-bottom: 1rem !important;
    }

    /* Regular paragraphs */
    .custom-markdown p {
        font-size: 0.95rem !important;
        margin: 0.8rem 0 !important;
    }

    /* List styling */
    .custom-markdown ul,
    .custom-markdown ol {
        margin: 0.8rem 0 !important;
        padding-left: 1.5rem !important;
    }

    /* Code styling */
    .custom-markdown code {
        font-family: ui-monospace, monospace !important;
        background-color: var(--background-fill-secondary) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-size: 0.8rem !important; /* Slightly smaller than regular text */
        color: var(--body-text-color) !important;
    }

    /* Add styles for bold and emphasis */
    .custom-markdown strong {
        font-weight: 600 !important;
        color: var(--body-text-color) !important;
    }

    .custom-markdown em {
        font-style: italic !important;
    }

    /* Ensure both columns take equal width */
    .gradio-row > div {
        flex: 1 1 50% !important;
        min-width: 0 !important; /* Prevents flex items from overflowing */
    }
    """

    ENGINE_INFORMATION = """
    ## TTS Engine Capabilities Help

    This guide explains the various capabilities that **may** be available in different TTS engines and models. Each capability affects how the TTS engine processes and generates speech output.
    """

    ENGINE_INFORMATION1 = """
    ## Performance Features

    - **DeepSpeed Capable**: Enables GPU acceleration using the DeepSpeed optimization library.
        - **Requires**: NVIDIA GPU with CUDA support
        - **Benefit**: Significantly faster text-to-speech generation
        - **Note**: Model must specifically support DeepSpeed inference

    - **Low VRAM Capable**: Optimized for systems with limited GPU memory.
        - **Benefit**: Efficient memory management between CPU and GPU
        - **Use Case**: Ideal when running alongside other GPU-intensive applications like LLMs
        - **Note**: May trade speed for memory efficiency

    - **Streaming Capable**: Enables real-time speech generation.
        - **Benefit**: Immediate playback without generating entire audio first
        - **Use Case**: Interactive applications and real-time responses
        - **Note**: Not all output formats support streaming

    ## Voice Control Features

    - **Pitch Capable**: Allows adjustment of voice pitch.
        - **Benefit**: Creates more expressive and varied speech output
        - **Use Case**: Customizing voice characteristics
        - **Note**: Adjustment range varies by model

    - **Generation Speed Capable**: Controls speech rate.
        - **Benefit**: Adjustable playback speed during generation
        - **Use Case**: Creating faster or slower speech output
        - **Note**: May affect audio quality at extreme settings

    - **Temperature Capable**: Controls output randomness.
        - **Benefit**: Adjustable speech variation and creativity
        - **Use Case**: Balancing consistency vs. naturality
        - **Note**: Higher values increase variation but may reduce quality
    """

    ENGINE_INFORMATION2 = """
    ## Quality Enhancement Features

    - **Repetition Penalty Capable**: Prevents unnatural speech patterns.
        - **Benefit**: Reduces repeated sounds and phrases
        - **Use Case**: Improving natural flow of longer texts
        - **Note**: Strength of penalty may be adjustable

    ## Multi-Feature Support

    - **Multi-Languages Capable Models**: Each model Supports multiple languages.
        - **Benefit**: Generate speech in different languages
        - **Note**: Quality may vary between languages
        - **Tip**: Check model-specific language support

    - **Multi-Voice Capable**: Supports multiple speaking voices.
        - **Benefit**: Different voices or speaking styles
        - **Types**: Pre-trained voices or voice cloning
        - **Note**: Voice quality may vary by model

    - **Multi-Model Capable Engine**: Supports multiple TTS models.
        - **Benefit**: Flexibility in model selection
        - **Use Case**: Switching between models for different needs
        - **Note**: Each model may have different capabilities

    ## Technical Features

    - **Default Audio Output Format**: Specifies output file format.
        - **Common Formats**: WAV, MP3, FLAC, Opus, AAC, PCM
        - **Note**: Transcoding to different formats adds processing time
        - **Warning**: Not all formats support streaming

    - **Platform Support**: Operating system compatibility.
        - **Platforms**: Windows, Linux, macOS
        - **Note**: Additional setup may be required
        - **Warning**: Support level may vary by platform
    """

    DEFAULT_SETTINGS = """
    ## TTS Engine Settings Help

    This guide explains the settings and configuration options available for individual Text-to-Speech engines within AllTalk.
    """

    DEFAULT_SETTINGS1 = """
    ## Engine Capabilities & Controls

    - **Low VRAM Mode**
        - Not applicable for VibeVoice (model stays resident; reload to free VRAM)

    - **DeepSpeed Capability**
        - Not supported by VibeVoice

    - **Temperature Control**
        - Not exposed. VibeVoice uses **CFG Scale** (classifier-free guidance) instead.
          Configure CFG Scale on the VibeVoice tab.

    - **Repetition Penalty**
        - Not applicable for VibeVoice

    - **Pitch Adjustment**
        - Not applicable for VibeVoice (timbre is set by the reference voice WAV)

    - **Generation Speed**
        - Not exposed by VibeVoice's public inference API
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
    ## VibeVoice TTS Engine Help

    VibeVoice is Microsoft Research's long-form, multi-speaker text-to-speech model. The 1.5B variant can synthesise up to ~90 minutes of conversational speech in a single pass with **up to 4 distinct speakers**, each driven by a reference WAV. Microsoft removed the TTS inference code from their official repository in Sept 2025; this engine uses the maintained community fork at **vibevoice-community/VibeVoice** (MIT). Output is 24 kHz mono WAV.
    """

    HELP_PAGE1 = """
    ## Installation

    The engine auto-installs VibeVoice from the community fork on first model load. You can also install it manually:

    ```bash
    pip install git+https://github.com/vibevoice-community/VibeVoice.git
    ```

    Model weights are pulled from HuggingFace automatically when the model is first loaded. To pre-download:

    ```bash
    pip install "huggingface_hub[cli]"
    hf download microsoft/VibeVoice-1.5b
    ```

    ## Hardware

    | Device | Setup | Notes |
    |---|---|---|
    | **NVIDIA CUDA** | bf16 + flash_attention_2 | Recommended. Falls back to sdpa if flash-attn isn't installed. ~5 GB VRAM for 1.5B, ~18 GB for Large. |
    | **Apple Silicon (MPS)** | fp32 + sdpa | ~12 GB unified memory for 1.5B. Large is **not recommended**. |
    | **CPU** | fp32 + sdpa | Functional but very slow. |

    ## Voice System — Reference WAVs

    VibeVoice clones a voice from a single reference WAV per speaker. Drop `.wav` files into the engine's `voices/` directory and they'll be auto-discovered. The community fork ships these starter voices under `demo/voices/` — copy them over:

    ```
    en-Carter_man.wav      en-Alice_woman.wav     en-Frank_man.wav
    en-Maya_woman.wav      en-Mary_woman_bgm.wav  in-Samuel_man.wav
    zh-Bowen_man.wav       zh-Xinran_woman.wav    zh-Anchen_man_bgm.wav
    ```

    Voices ending in `_bgm` carry musical/expressive characteristics.

    ## Multi-Speaker Scripts

    To use multiple speakers in a single generation, format your text like this:

    ```
    Speaker 1: Welcome back to the show.
    Speaker 2: Thanks for having me.
    Speaker 1: Let's start with the obvious question.
    Speaker 3: Wait, can I jump in here?
    ```

    Each `Speaker N` resolves to a reference WAV via the **Speaker → Voice Mapping**
    on the VibeVoice tab. Up to 4 distinct speakers per generation.

    If your text has no `Speaker N:` prefix, it's wrapped as `Speaker 1:` and uses
    whatever voice was passed in the API call (or the default character voice).
    """

    HELP_PAGE2 = """
    ## CFG Scale

    VibeVoice uses **Classifier-Free Guidance** (CFG) instead of temperature. Higher
    values make the output adhere more closely to the reference voice and script.

    - **1.3** (default) — balanced, recommended
    - **1.0** — most natural prosody, less faithful to reference
    - **2.0+** — strong adherence, can sound rigid

    ## DDPM Inference Steps

    The diffusion head runs N denoising steps per audio token. More steps = better
    fidelity at the cost of speed.

    - **10** (default) — good speed/quality balance
    - **5** — faster, slightly lower fidelity
    - **20+** — diminishing returns

    ## Audio Output

    - **Sample rate**: 24 kHz mono WAV
    - **Streaming**: Not supported (VibeVoice is non-streaming). The engine fakes
      streaming by chunking the final WAV.
    - **Max length**: ~90 minutes per call for the 1.5B model

    ## Languages

    Primary: **English** and **Mandarin Chinese**. The community fork's bundled
    voices include `in-Samuel` for Indian-accented English. The model has been
    observed to handle other languages with reduced quality.

    ## Troubleshooting

    | Error | Solution |
    |-------|----------|
    | `Could not install VibeVoice` | Manually run `pip install git+https://github.com/vibevoice-community/VibeVoice.git` and check the error |
    | `Voice 'X' not found in .../voices` | Drop a `X.wav` file into the engine's `voices/` folder |
    | `flash_attention_2 failed` | Normal on Mac/CPU; engine auto-falls back to sdpa. On CUDA, install: `pip install flash-attn --no-build-isolation` |
    | `RuntimeError: MPS does not support ...` | Switch device to `cpu` in the VibeVoice tab, or update PyTorch |
    | Very poor audio quality | Try a longer/cleaner reference WAV (10–30s), or bump CFG Scale toward 1.5 |
    | Generation extremely slow | Drop DDPM steps to 5; on CPU expect minutes per minute of audio |
    """
