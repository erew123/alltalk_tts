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
    ## 🔊 TTS Engine Capabilities Help

    This guide explains the various capabilities that **may** be available in different TTS engines and models. Each capability affects how the TTS engine processes and generates speech output.
    """

    ENGINE_INFORMATION1 = """
    ## 🚀 Performance Features

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

    ## 🎵 Voice Control Features

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
    ## 🎯 Quality Enhancement Features

    - **Repetition Penalty Capable**: Prevents unnatural speech patterns.
        - **Benefit**: Reduces repeated sounds and phrases
        - **Use Case**: Improving natural flow of longer texts
        - **Note**: Strength of penalty may be adjustable

    ## 🌐 Multi-Feature Support

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

    ## 📁 Technical Features

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
    ## 🎯 TTS Engine Settings Help

    This guide explains the settings and configuration options available for individual Text-to-Speech engines within AllTalk.
    """

    DEFAULT_SETTINGS1 = """
    ## 🎚️ Engine Capabilities & Controls

    - **Low VRAM Mode**
        - Optimizes memory usage for systems with limited GPU memory
        - Efficiently manages resources by moving data between CPU and GPU
        - Recommended for systems with less than 4GB VRAM or when running alongside other GPU-intensive applications

    - **DeepSpeed Capability**
        - Accelerates TTS generation using optimized inference
        - Only available for engines and models that support DeepSpeed
        - Requires NVIDIA GPU with CUDA support
        
    - **Stream Response Capability**
        - Enables real-time streaming of generated speech output
        - Reduces latency for faster feedback during synthesis
        - Only available for engines and models that support Streaming

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

    ## 🗣️ Voice Configuration

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

    ## ⚙️ Important Notes

    - Settings availability is determined by engine capabilities
    - Grayed-out options indicate features not supported by the current engine
    - Changes only affect the currently selected TTS engine
    - Settings here act as defaults but can be overridden via API parameters
    - All changes require clicking "Update Settings" to take effect    
    - Some settings require an engine reload to take effect
    """

    HELP_PAGE = """
    ## 🎯 Parler TTS Engine Settings Help

    Parler TTS is a unique text-to-speech system that generates voices based on descriptive prompts, similar to how image generation AI works with text descriptions.
    """

    HELP_PAGE1 = """
    ## 📁 File Locations & Management

    - **Model Storage**
        - Models are stored in `/alltalk_tts/models/parler/{model name}`

    ## 🎚️ Engine Capabilities & Controls

    - **Low VRAM Mode**
        - Enabled by default for optimized memory management
        - Helps balance resource usage when running alongside other applications
        - Recommended for most setups

    - **Multi-Model Support**
        - Supports multiple internal models for voice generation
        - Allows for diverse voice characteristics and styles
        - Enables both native and custom voice descriptions

    ## 🗣️ Voice System

    ### Native Voices
    - 34 built-in consistent voices including:
        - Aaron, Alisa, Anna, Barbara, Bill, etc.
        - Each native voice provides more consistent results
        - Use these when you need reproducible voice characteristics

    ### Custom Voice Descriptions
    - Create voices by describing desired characteristics:
        - Gender and age
        - Speaking style and pace
        - Accent and tone
        - Audio quality characteristics
    - **Example**: "A female speaker with an enthusiastic and lively voice. Her tone is bright and energetic, with a fast pace and a lot of inflection."

    ### Voice Editor
    - Add and manage custom voice descriptions
    - Save frequently used voice configurations
    - Edit existing voice descriptions
    - Voice settings are stored in `system/tts_engines/parler/parler_voices.json`
    """
        
    HELP_PAGE2 = """
    ## 📝 Important Notes

    - **Voice Consistency**
        - Unlike traditional TTS engines, each generation may sound slightly different
        - Native voices provide more consistent results
        - Custom voice descriptions may vary between generations

    - **Audio Quality Control**
        - Include "very clear audio" in description for highest quality
        - Use "very noisy audio" for intentional background noise
        - Control reverb and recording proximity through descriptions

    - **Best Practices**
        - Use punctuation to control speech rhythm
        - Be specific in voice descriptions
        - Test different description combinations for desired results
        - Consider using native voices for consistent output

    ## ⚠️ Known Limitations

    - No support for:
        - DeepSpeed acceleration
        - Pitch adjustment
        - Generation speed control
        - Repetition penalty
        - Temperature adjustment
        - Streaming generation
        - Multi-language capabilities

    ## 🔧 System Requirements

    - Runs on all major operating systems:
        - Windows
        - Linux
        - macOS
    - Outputs in WAV format
    - May show flash attention warnings on Windows (normal behavior)
    """

    VOICE_EDITOR = """
    ## 🎙️ Parler Voice Editor Help

    The Voice Editor allows you to create, manage, and customize voice descriptions for Parler TTS. Each voice consists of a name and a detailed description that defines its characteristics.
    """
    
    VOICE_EDITOR1 = """
    ## 📝 Creating Voice Descriptions

    ### Basic Structure
    - **Voice Name**: A unique identifier for your voice (e.g., "Soothing female", "Angry Tom")
    - **Voice Description**: Detailed description of voice characteristics

    ### Description Components
    Include these elements for best results:
    - Gender and age characteristics
    - Speaking style and pace
    - Tone and emotional quality
    - Audio environment/quality
    - Accent or language characteristics (if desired)

    ### Example Descriptions
    `A female speaker with a soft, soothing voice. Her tone is gentle and calming, with a slight reverb as if she is in a small, cozy room.`

    `A male speaker with a strong, authoritative voice. His tone is firm and commanding, with a slight bass boost that adds to the depth. He speaks at a moderate pace.`
    """

    VOICE_EDITOR2 = """
    ## 💡 Best Practices

    - **Be Specific**: The more detailed your description, the better the results
    - **Include Audio Quality**: Mention desired audio characteristics (clear, reverb, etc.)
    - **Consider Pacing**: Specify speaking speed (slow, moderate, fast)
    - **Define Personality**: Include emotional and stylistic elements

    ## 🔧 Managing Voices

    - Use **Add/Save Voice** to create new or update existing voices
    - **Delete Selected Voice** removes the currently selected voice
    - **Clear Text Boxes** resets the input fields
    - Select any voice from the dropdown to edit its properties

    ## ⚠️ Important Notes

    - Each generation may sound slightly different even with the same description
    - Native voices tend to be more consistent than custom voices
    - Changes are saved to the voice configuration file automatically
    """