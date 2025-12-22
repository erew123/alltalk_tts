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
        - Optimizes memory usage for systems with limited GPU memory
        - Efficiently manages resources by moving data between CPU and GPU
        - Recommended for systems with less than 4GB VRAM or when running alongside other GPU-intensive applications

    - **DeepSpeed Capability**
        - Accelerates TTS generation using optimized inference
        - Only available for engines and models that support DeepSpeed
        - Requires NVIDIA GPU with CUDA support

    - **Temperature Control**
        - Adjusts the variability in speech generation
        - **Range**: 0.0 to 1.0
        - **Lower values** (0.1-0.5): More consistent, stable output
        - **Higher values** (0.6-1.0): More variable, potentially more natural-sounding output

    - **Repetition Penalty**
        - Helps prevent repetitive speech patterns
        - **Range**: 1.0 to 15.0
        - Higher values more strongly discourage repetition
        - Typically most effective between 1.0-3.0

    - **Pitch Adjustment**
        - Modifies the voice pitch when supported
        - **Range**: -20 to +20
        - Use subtle adjustments for most natural results

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
    ## Orpheus TTS Engine Help

    Orpheus is a state-of-the-art LLM-based text-to-speech system built on a Llama 3B backbone. It produces natural-sounding speech with human-like intonation and emotional expression, featuring unique emotion tags for expressive control.
    """

    HELP_PAGE1 = """
    ## Installation & Setup

    Orpheus uses a gated model on HuggingFace that requires authentication. Follow these steps carefully.

    ### Step 1: HuggingFace Account Setup (Required)

    **This step is mandatory** - the model will not download without it.

    1. **Create a HuggingFace account**: https://huggingface.co/join

    2. **Accept the model license**:
       - Visit: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
       - Click "Agree and access repository"

    3. **Create an access token**:
       - Visit: https://huggingface.co/settings/tokens
       - Click "New token"
       - Name it (e.g., "alltalk")
       - Select "Read" permission
       - Copy the token

    4. **Login via CLI**:
    ```bash
    pip install huggingface_hub
    huggingface-cli login
    # Paste your token when prompted
    ```

    ### Step 2: Platform-Specific Packages (Auto-Install)

    Python packages will be installed automatically when you first use Orpheus. Or install manually:

    **macOS (Apple Silicon - Metal)**:
    ```bash
    pip install orpheus-cpp
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
    ```

    **Windows/Linux (NVIDIA CUDA)**:
    ```bash
    pip install orpheus-speech
    pip install vllm==0.7.3
    ```

    ### Step 3: Verification

    After setup, AllTalk will show:
    - `[Orpheus] HuggingFace authenticated as: YourUsername`
    - `[Orpheus] orpheus-cpp backend available (Metal)` or `orpheus-speech backend available (vLLM/CUDA)`

    ### Troubleshooting

    | Error | Solution |
    |-------|----------|
    | `401 Unauthorized` | Run `huggingface-cli login` with your token |
    | `403 Forbidden` | Accept license at https://huggingface.co/canopylabs/orpheus-3b-0.1-ft |
    | `Not logged into HuggingFace` | Run `huggingface-cli login` |
    | `orpheus-cpp not installed` | Wait for auto-install or run `pip install orpheus-cpp` |
    | `vLLM errors` | Pin version: `pip install vllm==0.7.3` |

    ## Available Models

    - **orpheus-3b-0.1-ft**: Production-optimized finetuned model (recommended)
    - **orpheus-3b-0.1-pretrained**: Base model trained on 100k+ hours of English speech

    ## Available Voices

    Voices ranked by conversational realism:
    - **tara**: Most conversational and natural (recommended)
    - **leah**: Clear and articulate female voice
    - **jess**: Professional female voice
    - **mia**: Warm and friendly female voice
    - **zoe**: Youthful female voice
    - **leo**: Natural male voice
    - **dan**: Deep and authoritative male voice
    - **zac**: Youthful male voice

    ## Emotion Tags

    Add emotion tags inline within your text to control expression:
    - `<laugh>` - Laughter
    - `<chuckle>` - Light chuckle
    - `<sigh>` - Sighing sound
    - `<cough>` - Coughing
    - `<sniffle>` - Sniffling
    - `<groan>` - Groaning sound
    - `<yawn>` - Yawning
    - `<gasp>` - Gasping

    **Example**: "I can't believe it worked! <laugh> This is amazing."
    """

    HELP_PAGE2 = """
    ## Engine Capabilities

    - **Temperature Control**
        - Adjusts generation randomness
        - Range: 0.0 to 1.0
        - Default: 0.7
        - Lower = more consistent, Higher = more varied

    - **Repetition Penalty**
        - Prevents repeated speech patterns
        - Must be >= 1.1 for stable output
        - Higher values increase speech speed slightly

    - **Streaming Support**
        - Real-time audio generation (~200ms latency)
        - Can be reduced to ~100ms with input streaming

    ## Best Practices

    - **Use emotion tags sparingly** for natural results
    - **Temperature 0.6-0.8** works well for most use cases
    - **Repetition penalty 1.1-1.3** is recommended
    - **tara voice** is generally the most natural sounding

    ## System Requirements

    ### macOS
    - Apple Silicon (M1/M2/M3) with Metal support
    - Minimum 8GB RAM (16GB recommended)
    - `orpheus-cpp` + `llama-cpp-python` with Metal wheel

    ### Windows/Linux
    - NVIDIA GPU with CUDA support
    - Minimum 8GB VRAM recommended
    - `orpheus-speech` + `vllm` packages

    ## Audio Output

    - **Format**: WAV
    - **Sample Rate**: 24kHz
    - **Channels**: Mono

    ## Limitations

    - No multi-language support (English only)
    - No pitch adjustment
    - No speed adjustment

    ## Troubleshooting

    - **vLLM bugs**: Pin to version 0.7.3 (`pip install vllm==0.7.3`)
    - **Memory issues**: Try reducing `max_model_len` parameter
    - **Slow generation**: Ensure GPU acceleration is active

    ## Zero-Shot Voice Cloning

    Orpheus supports zero-shot voice cloning using the **pretrained** model (not the finetuned model).
    This feature requires additional setup beyond the standard AllTalk integration.

    ### How It Works

    Voice cloning uses the SNAC (Speech Neural Audio Codec) model to:
    1. Encode reference audio into discrete tokens
    2. Include text-speech pairs in the prompt as examples
    3. Generate new speech that mimics the reference voice

    The more text-speech pairs you provide, the more reliably it generates in the target voice.

    ### Requirements

    - **Model**: `orpheus-3b-0.1-pretrained` (not the finetuned version)
    - **SNAC Model**: `hubertsiuzdak/snac_24khz` for audio encoding/decoding
    - **Reference Audio**: 5-30 seconds of clean speech at 24kHz
    - **Transcript**: Accurate text of what's spoken in the reference

    ### Basic Process

    ```python
    # 1. Load SNAC model for audio tokenization
    from snac import SNAC
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

    # 2. Tokenize reference audio
    audio_codes = snac.encode(reference_audio)

    # 3. Structure prompt with special tokens:
    #    [START] + transcript_tokens + [END] + audio_codes + [SEP]
    #    + new_text_tokens + [GEN]

    # 4. Run inference with the pretrained model

    # 5. Decode output tokens back to audio
    output_audio = snac.decode(generated_codes)
    ```

    ### Tips for Best Results

    - Use **clean, noise-free** reference audio
    - Provide **accurate transcripts** of reference speech
    - Include **multiple text-speech pairs** (3-5) for better voice matching
    - Use reference audio with **varied intonation** for expressive cloning
    - Keep reference clips **5-30 seconds** each

    ### Resources

    - [Orpheus Voice Cloning Issue #6](https://github.com/canopyai/Orpheus-TTS/issues/6)
    - [OrpheusTTS-WebUI](https://github.com/Saganaki22/OrpheusTTS-WebUI) - Community fork with voice cloning UI
    - [Pretrained Model Colab](https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS) - Conditioned generation examples

    ### Current Status in AllTalk

    The current AllTalk integration uses the **finetuned** model with 8 built-in voices.
    Full voice cloning support would require:
    - Switching to the pretrained model
    - Adding SNAC model integration
    - Implementing audio tokenization pipeline
    - UI for uploading reference audio and transcripts

    This is planned for a future update.
    """
