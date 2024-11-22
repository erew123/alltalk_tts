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
    ## 🎯 F5-TTS Voice Cloning Engine Help

    F5-TTS is a zero-shot voice cloning system that specializes in high-fidelity voice reproduction with precise reference text matching.
    """

    HELP_PAGE1 = """
    ## 🎚️ Model Variants

    ### F5-TTS (High Fidelity)
    - Optimized for voice accuracy and naturalness
    - Superior voice characteristic capture
    - Higher computational requirements
    - Best for quality-focused applications
    - Longer generation times

    ### E2-TTS (Efficient)
    - Focused on speed and efficiency
    - Lower resource requirements
    - Faster generation times
    - Good for longer texts
    - Suitable for resource-constrained systems

    ## 🗣️ Voice Cloning System

    ### Reference Text Requirements
    - Each voice sample needs a matching `.reference.txt` file
    - Text must exactly match spoken audio
    - Punctuation significantly affects output quality
    - Use the Reference Text/Sample Manager for setup
    - Supports English and Chinese languages only

    ### Audio Sample Guidelines
    - Format: WAV/MP3/Flac files
    - Length: Maximum 15 seconds recommended
    - Quality: Clear audio, minimal background noise
    - Location: `/alltalk_tts/voices/` directory
    - Names: Should match reference text filename

    ### Punctuation Impact
    - Precise punctuation affects voice reproduction
    - Examples of differences:
        - `Im` vs `I'm` vs `I am`
        - Commas affect pacing
        - Periods affect sentence flow
        - Capital letters may be spelled out
    - Use appropriate punctuation marks:
        - Periods (.)
        - Commas (,)
        - Semi-colons (;)
        - Apostrophes (')

    ## ⚙️ Performance Settings

    ### Speed Adjustment
    - Default speed setting: 0.9
    - 1.0 seems faster than reference audio
    - Compensates for faster audio generation
    - Adjustable in Default Settings tab
    - Affected by:
        - Sample length
        - Text punctuation
        - Reference text accuracy
    """
        
    HELP_PAGE2 = """
    ### Best Practices
    1. **Voice Samples**
        - Keep under 15 seconds
        - Use clear recordings
        - Ensure consistent audio quality
        - Avoid background noise

    2. **Reference Text**
        - Match spoken content exactly
        - Include proper punctuation
        - Verify text accuracy
        - Consider natural speech patterns

    3. **Model Selection**
        - F5-TTS: When quality is priority
        - E2-TTS: When speed is priority
        - Consider system resources
        - Test both for optimal results

    ## 📁 File Management

    - **Models**: `/alltalk_tts/models/f5tts/`
    - **Voice Samples**: `/alltalk_tts/voices/`
    - **Reference Texts**: `.reference.txt` files alongside WAVs
    - **Outputs**: `/alltalk_tts/outputs/`
        - Automatic cleanup available
        - Configurable retention period
        - Set to 'Disabled' to keep all files

    ## 🔄 Reference Text Management

    ### Using the Manager
    - Access via Reference Text/Sample Manager tab
    - Color-coded file status:
        - 🔴 Red: Needs reference text
        - 🟢 Green: Has reference text
    - Built-in audio playback
    - Whisper transcription available

    ### Whisper Transcription Options
    - **base** (200MB download): Fast, basic quality
    - **medium** (1.5GB download): Better quality
    - **large-v3** (3GB download): Best quality
    - Automatic model download
    - Helps create initial reference text

    ## 🔧 System Integration

    - Click "Refresh Server Settings" after adding new voices
    - Monitor system resources for optimal performance
    - Consider storage space for models and samples
    - Regular maintenance of output files recommended

    ## ⚠️ Important Notes

    - Language Support: English and Chinese only
    - Reference text accuracy is crucial
    - Punctuation affects voice reproduction
    - Sample length impacts playback speed
    - Model selection based on use case
    - System resources affect performance
    - Regular updates may improve functionality
    """
    REFERENCE_MANAGER = """
    ## 🎯 Reference Text/Sample Manager Help

    The Reference Text/Sample Manager is a crucial tool for F5-TTS voice cloning, helping you manage and create accurate reference text files for your voice samples.
    """
    REFERENCE_MANAGER1 = """
    ## 📝 Core Functionality

    ### File Status Indicators
    - 🔴 Red: Files needing a reference text file
    - 🟢 Green: Files with existing reference text file
    - Visual system helps track incomplete samples

    ### Basic Operations
    - **Preview Voice Sample**: Listen to selected audio
    - **Refresh List**: Update file listing
    - **Save Reference Text**: Create/update reference file
    - **Delete Reference Text**: Remove existing reference
    - **Auto-Transcribe**: Use Whisper for automatic transcription

    ## 📒 Creating Reference Text Files

    ### Manual Process
    1. Select voice file from dropdown
    2. Play audio using Preview button
    3. Type exact spoken text in editor
    4. Add appropriate punctuation
    5. Save reference text
    6. Click "Refresh Server Settings" on main page

    ### Punctuation Guidelines
    - **Period (.)**: End of sentences
    - **Comma (,)**: Natural pauses
    - **Semi-colon (;)**: Longer pauses
    - **Apostrophe (')**: Contractions like:
        - "I'm"
        - "don't"
        - "can't"
        - "we're"
    - Where necessary use phonetics
        - This is a test of TTS (vs)
        - This is a test of tea tea ess

    ## 🎤 Voice Sample Requirements

    ### Audio Guidelines
    - Maximum length: 15 seconds
    - Format: WAV files
    - Quality: Clear, minimal noise
    - Location: `/alltalk_tts/voices/`

    ### Important Notes
    - Longer samples may affect playback speed
    - Only English and Chinese languages supported
    - Exact text matching is crucial
    - Include natural speech punctuation
    """

    REFERENCE_MANAGER2 = """
    ## 🤖 Whisper Transcription

    ### Available Models
    1. **base (1GB VRAM)**
        - Fast processing
        - Basic quality
        - Good for most uses
        - Smallest disk space
        
    2. **medium (4GB VRAM)**
        - Better quality
        - Slower processing
        - Moderate disk space

    3. **large-v3 (10GB VRAM)**
        - Best quality
        - Slowest processing
        - Largest disk space    

    ### Using Whisper
    - Select desired model from dropdown
    - Click "Auto-Transcribe Missing Files"
    - Review and edit transcriptions
    - Add punctuation as needed
    - Save corrected text

    ### Storage Notes
    - Models download automatically
    - Verify available disk space
    - One-time download per model
    - Models persist between sessions

    ## 💡 Best Practices

    1. **Text Accuracy**
        - Listen to audio multiple times
        - Match text exactly
        - Include all spoken words
        - Maintain speech patterns

    2. **Punctuation Usage**
        - Mark natural pauses
        - Use appropriate marks
        - Consider speech flow
        - Be consistent

    3. **Workflow Tips**
        - Process files in batches
        - Verify saved references
        - Use Whisper for initial draft
        - Double-check automated transcriptions

    ## ⚠️ Important Reminders

    - Always refresh server settings after changes
    - Keep voice samples under 15 seconds
    - Back up reference text files
    - Review automated transcriptions
    - Consider system storage for Whisper models
    - Language support limited to English and Chinese
    """