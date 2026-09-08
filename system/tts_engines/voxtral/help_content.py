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
        - Not applicable for Voxtral (HTTP-client engine, no local model)

    - **DeepSpeed Capability**
        - Not applicable for Voxtral (HTTP-client engine)

    - **Temperature Control**
        - Not applicable for Voxtral

    - **Repetition Penalty**
        - Not applicable for Voxtral

    - **Pitch Adjustment**
        - Not applicable for Voxtral

    - **Generation Speed**
        - Controls the pace of generated speech
        - **Range**: 0.5 to 2.0
        - 1.0 represents normal speed
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
    ## Voxtral TTS Engine Help

    Voxtral is Mistral AI's 4B-parameter text-to-speech model. It supports 9 languages with 20 preset voices, zero-shot voice cloning from ~3 seconds of reference audio, and streaming with ~100ms time-to-first-audio. Voxtral is an HTTP-client engine — AllTalk sends requests to an external server rather than loading a model into memory.
    """

    HELP_PAGE1 = """
    ## Three Operating Modes

    ### Mode 1: mlx-audio (macOS / Apple Silicon)
    - Run Voxtral locally on Mac using the mlx-audio framework
    - Set **Backend** to `mlx-audio`
    - Leave **API Key** empty
    - Requires Apple Silicon (M1/M2/M3/M4) Mac

    **Setup:**
    ```bash
    pip install mlx-audio
    ```

    **Auto-Start (recommended):** Enable **Auto-Start Server** in the Voxtral API settings tab. AllTalk will automatically launch the mlx-audio server on the configured port (default 7852) when the engine loads, and stop it when the engine unloads. If a server is already running on that port, AllTalk will reuse it instead of starting a duplicate.

    **Note:** The first launch may take several minutes as the model (~3GB) is downloaded from HuggingFace. Check the `mlx_server.log` file in the voxtral engine directory for progress.

    **Manual start:** If you prefer to manage the server yourself, disable Auto-Start and run:
    ```bash
    python -m mlx_audio.server --model mlx-community/Voxtral-4B-TTS-2603-mlx-bf16 --port 7852
    ```
    Then set **API URL** to `http://localhost:7852`.

    mlx-audio natively reimplements the Voxtral architecture in Apple's MLX framework for optimized inference on Apple Silicon.

    ### Mode 2: vllm-omni (Linux / NVIDIA GPU)
    - Run the Voxtral model locally using vllm-omni
    - Set **Backend** to `vllm-omni`
    - Set **API URL** to `http://localhost:8000` (default)
    - Leave **API Key** empty
    - Requires an NVIDIA GPU with CUDA support and sufficient VRAM (~8GB+)

    **Setup:**
    ```bash
    pip install vllm-omni
    vllm serve mistralai/Voxtral-4B-TTS-2603 --omni
    ```

    The vllm-omni server also exposes an OpenAI-compatible chat completions endpoint, which can double as the LLM backend for AllTalk's narrative emotion detection (instead of LM Studio or Ollama).

    ### Mode 3: Mistral Cloud API
    - Uses Mistral's hosted inference API
    - Set **Backend** to `mistral-cloud`
    - Set **API URL** to `https://api.mistral.ai`
    - Set **API Key** to your Mistral API key
    - No GPU required locally
    - Usage is billed per request

    **Get an API key at:** https://console.mistral.ai

    ## Voice System

    ### Preset Voices (20 total)
    - **English**: casual_female, casual_male, cheerful_female, neutral_female, neutral_male
    - **French**: fr_female, fr_male
    - **German**: de_female, de_male
    - **Spanish**: es_female, es_male
    - **Italian**: it_female, it_male
    - **Portuguese**: pt_female, pt_male
    - **Dutch**: nl_female, nl_male
    - **Arabic**: ar_male
    - **Hindi**: hi_female, hi_male

    ### Voice Cloning
    Voxtral supports zero-shot voice cloning from ~3 seconds of reference audio. To use voice cloning, pass the path to a WAV file as the voice parameter in your API call. AllTalk will encode it and send it to the Voxtral API.
    """

    HELP_PAGE2 = """
    ## Emotion Handling

    Voxtral handles emotion through **reference audio steering** rather than explicit emotion tags. The model interprets the emotional tone from the voice prompt/reference audio. When using preset voices, the emotional quality comes from the voice style (e.g., `cheerful_female` vs `neutral_male`).

    For narrative emotion integration, Voxtral works with AllTalk's narrative emotion detector. When emotions are detected from narrative text, the system can select an appropriate voice style automatically.

    ## Using vllm-omni as LLM for Emotion Detection

    Since vllm-omni exposes an OpenAI-compatible chat completions endpoint alongside the TTS endpoint, you can use the same server for both TTS generation and LLM-based narrative emotion detection. In AllTalk's narrative emotion settings, set the LLM API URL to your vllm-omni server address (e.g., `http://localhost:8000`).

    ## Response Formats

    Voxtral supports multiple audio output formats:
    - **WAV** (default): Uncompressed, highest quality
    - **MP3**: Compressed, smaller file size
    - **FLAC**: Lossless compression
    - **Opus**: Efficient compression for streaming
    - **AAC**: Widely compatible compressed format
    - **PCM**: Raw audio data

    ## Audio Output

    - **Sample Rate**: 24kHz
    - **Streaming**: ~100ms time-to-first-audio
    - **Quality**: State-of-the-art for the model size

    ## Troubleshooting

    | Error | Solution |
    |-------|----------|
    | `Cannot connect to Voxtral API` | Ensure your local server (mlx-audio or vllm-omni) is running, or check your API key for Mistral Cloud |
    | `API returned 401` | Check your Mistral API key |
    | `API returned 429` | Rate limited — wait and retry |
    | `Request timed out` | Server may be overloaded; increase timeout or check server status |
    | `No voices found` | Check that `voxtral_voices.json` exists in the engine directory |
    | `mlx-audio won't start` | Ensure you have Apple Silicon (M1+) and mlx-audio is installed: `pip install mlx-audio`. Check `mlx_server.log` in the voxtral engine directory for details. |
    | `mlx-audio server did not become ready` | First run downloads ~3GB model; try increasing patience or start manually. Check `mlx_server.log` for progress. |
    | `vllm-omni won't install` | Requires NVIDIA GPU with CUDA; Flash Attention 3 (`fa3-fwd`) requires Hopper architecture or newer |
    """
