# help_content.py
# pylint: disable=no-member

class AllTalkHelpContent:
    """CSS and help content for Fish Speech TTS engine"""
    custom_css = """
    /* Add this to your existing CSS */
    .gradio-container .prose {
        max-width: none !important;
        padding: 0.5rem !important;
        margin: 0 !important;
    }

    .custom-markdown div {
    border: none !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    }

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
    ## Fish Speech (OpenAudio S1)

    Fish Speech is a leading open-source text-to-speech system featuring zero-shot and few-shot voice cloning capabilities. It uses RLHF (Reinforcement Learning from Human Feedback) training for high-quality natural speech synthesis.

    **Key Features:**
    - Zero-shot and few-shot voice cloning
    - Multi-language support (8+ languages)
    - High-quality natural speech
    - RLHF-trained for improved output
    """

    ENGINE_INFORMATION1 = """
    ## Available Models

    **OpenAudio S1 (4B parameters)**
    - Full flagship model with highest quality
    - Requires 12GB+ VRAM
    - Best for production use

    **OpenAudio S1-mini (0.5B parameters)**
    - Distilled lighter model
    - Requires ~6GB VRAM
    - Good balance of quality and speed
    - Recommended for most users

    ## Supported Languages

    Fish Speech supports the following languages:
    - English (EN)
    - Japanese (JA)
    - Korean (KO)
    - Chinese (ZH)
    - French (FR)
    - German (DE)
    - Arabic (AR)
    - Spanish (ES)
    """

    ENGINE_INFORMATION2 = """
    ## Voice Cloning Setup

    Fish Speech uses reference audio files for voice cloning. To set up a new voice:

    1. **Prepare Reference Audio**
       - Record or obtain 10-30 seconds of clear speech
       - Save as a WAV file (e.g., `my_voice.wav`)
       - Place in the `voices/` folder

    2. **Create Reference Text**
       - Create a text file with the same name (e.g., `my_voice.reference.txt`)
       - Add the exact transcription of the audio
       - Place in the same folder as the WAV file

    **Tips for Best Results:**
    - Use high-quality, noise-free audio
    - Ensure clear pronunciation in reference
    - Match reference language to target language
    - Longer references (20-30s) generally work better

    ## Generation Parameters

    - **Temperature**: Controls randomness (0.5-1.0 recommended)
    - **Repetition Penalty**: Prevents repetitive patterns (1.0-1.3)
    - **Speed**: Adjusts playback speed post-generation

    ## Emotion & Speech Control

    OpenAudio S1 supports emotion and speech style markers. Use `:marker:` syntax
    (colons survive AllTalk's text filtering, which strips parentheses):

    **Emotions:** `:angry:` `:sad:` `:excited:` `:surprised:` `:happy:` `:nervous:`
    `:scared:` `:disgusted:` `:confused:` `:proud:` `:embarrassed:` `:grateful:`

    **Tones:** `:whispering:` `:shouting:` `:soft tone:` `:in a hurry tone:`

    **Effects:** `:laughing:` `:sighing:` `:sobbing:` `:crying loudly:` `:panting:`

    **Example:** `":excited: Wow, this is amazing! :whispering: But don't tell anyone."`
    """

    MAIN_PAGE_HELP = """
    ## Quick Start

    1. **Download a Model**: Use the Model Download section to get OpenAudio S1-mini
    2. **Set Up Voices**: Add WAV files with matching .reference.txt files to `voices/`
    3. **Generate Speech**: Select a voice and enter your text

    ## Model Download

    Models are downloaded from HuggingFace. The S1-mini model (~3.6GB) is recommended for most users.
    The full S1 model (~16GB) provides higher quality but requires more VRAM.
    """

    MODEL_DOWNLOAD_HELP = """
    ## Downloading Models

    Click the download button for your preferred model. Downloads may take several minutes depending on your connection speed.

    **Note**: These are gated models on HuggingFace. You may need to:
    1. Create a HuggingFace account
    2. Accept the model license terms
    3. Generate an access token if required

    Models are saved to: `models/fishspeech/[model-name]/`
    """

    VOICE_SETUP_HELP = """
    ## Voice File Requirements

    For each voice, you need two files:

    1. **Audio File** (`voice_name.wav`)
       - Format: WAV (mono or stereo)
       - Length: 10-30 seconds
       - Quality: Clear speech, no background noise

    2. **Reference Text** (`voice_name.reference.txt`)
       - Contains exact transcription of the audio
       - UTF-8 encoded text file
       - Should match the spoken content precisely

    **Example:**
    ```
    voices/
      narrator.wav
      narrator.reference.txt
      character1.wav
      character1.reference.txt
    ```
    """

    LOW_VRAM_HELP = """
    ## Low VRAM Mode

    When enabled, models are moved between GPU and CPU memory as needed.

    **Benefits:**
    - Allows running on GPUs with limited VRAM
    - Can run alongside other GPU applications

    **Trade-offs:**
    - Slightly slower generation
    - Memory transfers add latency

    **Recommended for:**
    - GPUs with less than 8GB VRAM
    - Running alongside LLMs or other models
    """

    TEMPERATURE_HELP = """
    ## Temperature Setting

    Controls the randomness of speech generation.

    - **Lower (0.5-0.7)**: More consistent, predictable output
    - **Default (0.7-0.8)**: Balanced natural variation
    - **Higher (0.8-1.0)**: More expressive, varied output

    **Tip**: Start with 0.7 and adjust based on results.
    """

    REPETITION_PENALTY_HELP = """
    ## Repetition Penalty

    Prevents the model from generating repetitive patterns.

    - **1.0**: No penalty (may cause loops)
    - **1.1-1.2**: Light penalty (recommended)
    - **1.3+**: Strong penalty (may affect fluency)

    **Tip**: Increase if you notice repeated sounds or phrases.
    """
