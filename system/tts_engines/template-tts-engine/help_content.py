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
    ## 🎯 XTTS Voice Cloning Engine Help

    XTTS is a powerful multi-speaker, voice cloning model that can generate speech in multiple languages using short audio samples for voice cloning.
    """

    HELP_PAGE1 = """
    ## 🎚️ Engine Capabilities & Controls

    - **DeepSpeed Support**
        - Accelerates generation using optimized inference
        - Requires NVIDIA GPU with CUDA support
        - 2-3x speed improvement in generation
        - Recommended when available

    - **Multi-Language Support**
        - Clone voices across multiple languages
        - No need for language-specific training
        - Full Hindi support in v2.0.3 (API Local method only)
        - Natural-sounding output in target language

    - **Generation Methods**
        - API Local: Standard generation method
        - XTTSv2 Local: Enhanced method with DeepSpeed support
        - Each method may produce slightly different results
        - Try both methods if voice quality isn't optimal

    ## 🎤 Voice Sample Creation Guide

    ### Audio Requirements
    - Format: WAV or MP3
    - Sample Rate: 22050Hz
    - Channels: Mono
    - Bit Depth: 16-bit
    - Duration: 6-30 seconds (optimal: 8-10 seconds)

    ### Recording Tips
    1. **Audio Quality**
        - Choose clear, clean recordings
        - Avoid background noise, music, or echoes
        - Watch for audio artifacts like hiss or hum
        - Ensure consistent volume levels

    2. **Speech Content**
        - Use flowing, natural speech
        - Avoid long pauses or gaps
        - Include some emotional range
        - Avoid breathing sounds at start/end
        - Choose samples with good vocal variety

    ### Using Audacity
    1. **Basic Editing**
        - Open your audio file in Audacity
        - Select the best 6-30 second segment
        - Remove unwanted sounds or silence

    2. **Processing Steps**
        ```
        1. Clean up audio if needed (noise reduction, normalization)
        2. Tracks > Resample to 22050Hz
        3. Tracks > Mix > Stereo to Mono
        4. File > Export Audio as WAV
        ```

    ## 🗣️ Voice Management

    ### Single Voice Setup
    - Place WAV/MP3 files in `/alltalk_tts/voices/`
    - Use descriptive filenames (e.g., `broadcaster_male.wav`)
    - Files appear as individual voices in interface
    - Best for quick testing or simple use cases

    ### Multi-Voice Sets
    - Create folders in `/alltalk_tts/voices/xtts_multi_voice_sets/`
    - Add multiple samples per voice
    - System randomly selects up to 5 samples
    - Better for consistent voice reproduction
    - Example: `/voices/xtts_multi_voice_sets/broadcaster_voice/`

    ### Pre-computed Latents
    - Stored in `/voices/xtts_latents/`
    - Faster generation times
    - Reduced memory usage
    - Recommended for frequently used voices
    """
        
    HELP_PAGE2 = """
    ## 📝 Model Versions

    ### v2.0.3 (Recommended)
    - Latest stable release
    - Full language support including Hindi
    - Best overall performance
    - Complete feature set

    ### v2.0.2 (Alternative)
    - Stable fallback option
    - Good performance
    - Use if v2.0.3 has issues

    ### Required Files
    - `config.json`: Model configuration
    - `model.pth`: Core model weights
    - `mel_stats.pth`: Audio statistics
    - `speakers_xtts.pth`: Speaker embeddings
    - `vocab.json`: Tokenizer vocabulary
    - `dvae.pth`: Discrete VAE weights

    ## 🔍 Troubleshooting Voice Quality

    If voice cloning results aren't satisfactory:

    1. **Audio Quality Issues**
        - Verify sample meets technical requirements
        - Check for background noise
        - Ensure proper resampling
        - Try processing audio to improve clarity

    2. **Generation Options**
        - Try both API Local and XTTSv2 Local methods
        - Adjust temperature settings
        - Use multiple voice samples
        - Consider creating latents

    3. **Advanced Solutions**
        - Try model finetuning (see Github Wiki)
        - Use RVC pipeline for additional processing
        - Test different voice samples
        - Remember: 100% match isn't possible

    ## 📁 File Management

    - **Models**: `/alltalk_tts/models/xtts/`
    - **Voice Samples**: `/alltalk_tts/voices/`
    - **Multi-Voice Sets**: `/alltalk_tts/voices/xtts_multi_voice_sets/`
    - **Latents**: `/alltalk_tts/voices/xtts_latents/`
    - **Outputs**: `/alltalk_tts/outputs/`
        - Automatic cleanup available
        - Configurable retention period
        - Set to 'Disabled' to keep all files

    ## 💡 Best Practices

    1. **Sample Selection**
        - Use high-quality recordings
        - Choose emotionally varied speech
        - Avoid background noise
        - Test multiple samples

    2. **Performance**
        - Enable DeepSpeed when available
        - Use pre-computed latents
        - Consider multi-voice sets
        - Monitor system resources

    3. **Workflow**
        - Process audio before use
        - Test both generation methods
        - Keep original samples organized
        - Regular cleanup of output files
    """