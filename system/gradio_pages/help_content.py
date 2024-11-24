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

    .small-file-upload { 
    min-height: 200px;
    max-height: 200px;
    }

    .small-file-upload2 { 
    min-height: 90px;
    max-height: 90px;
    }    

    """

    WELCOME = """
    # Welcome to AllTalk
    Thanks for trying out AllTalk! Below you'll find information about getting help, reporting issues, and contributing to the project. If you're just getting started, you might want to check out the [Quick Start Guide](https://github.com/erew123/alltalk_tts/wiki/AllTalk-V2-QuickStart-Guide).
    """

    WELCOME1 = """
    ## 📚 Documentation and Features
    The Gradio interface includes expandable accordions throughout, providing help for each page and its settings/features. For more detailed information about features, settings, setup, and third-party add-ons, check out the [GitHub Wiki](https://github.com/erew123/alltalk_tts/wiki).

    💡 Tip: Use the Light/Dark Mode button in the top right corner for easier reading.

    ## ⚠️ Errors and Issues
    You can find solutions to all known errors and issues in the [Error Messages List](https://github.com/erew123/alltalk_tts/wiki/Error-Messages-List) on the GitHub Wiki. Also keep an eye on the terminal/console where AllTalk is running, as it provides detailed output for troubleshooting.

    ## 💡Support and Help
    If you need help, here's where to look:
        - The help sections within the interface
        - The [GitHub Wiki](https://github.com/erew123/alltalk_tts/wiki)
        - The [Discussion forums](https://github.com/erew123/alltalk_tts/discussions)
        - Past solutions in [GitHub Issues](https://github.com/erew123/alltalk_tts/issues)

    For detailed troubleshooting guidance and debugging tools, check out the [Support Section](https://github.com/erew123/alltalk_tts/wiki#-support--help).
    """

    WELCOME2 = """
    ## 🚀 Feature Requests
    I track feature requests in a dedicated thread in the [GitHub Discussions area](https://github.com/erew123/alltalk_tts/discussions/74). While I'm open to suggestions, I can't guarantee I'll implement every request. If you have a specific feature in mind, please add it to the discussion.

    ## 🤝 Contributing to AllTalk
    I welcome contributions to AllTalk and am grateful to anyone willing to contribute. While I don't have specific guidelines, I'd ask that you familiarize yourself with the [GitHub Wiki](https://github.com/erew123/alltalk_tts/wiki) to understand how AllTalk works. If you'd like to contribute, please drop me a message in the discussions area describing what you'd like to work on - this helps avoid duplicate efforts and lets me share any relevant development plans.

    ## 💖 Sponsor AllTalk
    I've developed AllTalk in my free time, handling everything from coding and documentation to testing and community support. It's grown far beyond what I initially imagined, requiring significant time and effort to maintain and improve. If you'd like to support my work, you can make a one-time or ongoing contribution through my [Ko-Fi page](https://ko-fi.com/erew123). Your support means a lot to me!
    """

    DEBUG_HELP1 = """
    ## 🔍 Function and Component-Specific Debug Options

    - **debug_func**
        - Tracks function entry points throughout the application
        - Shows the execution flow and function call hierarchy
        - Essential for understanding the program's execution path
        - Example: `[AllTalk TTS] Debug debug_func Function entry: generate_audio`

    **Note:** When debugging with `debug_func`, remember that functions often call other functions. The debug output shows the most recently entered function, but the actual issue might originate in a parent function. For effective debugging, review the sequence of function entries leading up to an error. For example:

    ```
    [AllTalk TTS] Debug debug_func Function entry: process_text
    [AllTalk TTS] Debug debug_func Function entry: clean_text
    [AllTalk TTS] Debug debug_func Function entry: generate_audio
    [AllTalk TTS] Error in generate_audio
    ```

    Here, while the error appears in `generate_audio`, the root cause might be in how `process_text` or `clean_text` prepared the data.

    - **debug_api**
        - Logs API endpoint interactions and requests
        - Shows incoming request data and response statuses
        - Useful for troubleshooting API integrations & calls
        - Example: `[AllTalk API] Debug debug_api Successfully retrieved 19 available voices`

    - **debug_tts**
        - Monitors core TTS generation processes
        - Shows generation progress, timing, and status updates
        - Helps diagnose basic TTS generation issues
        - Use `debug_openai` for OpenAI TTS generation
        - Example: `[AllTalk GEN] Debug debug_tts Starting TTS generation for file: output_12345.wav`

    - **debug_tts_variables**
        - Displays detailed TTS generation parameters
        - Shows voice settings, language options, and generation parameters etc
        - Useful for verifying correct parameter passing
        - Example: `[AllTalk GEN] Debug debug_tts_variables temperature: 0.75, pitch: 1.0`

    - **debug_narrator**
        - Tracks narrator mode text processing
        - Shows how text is split between narrator and character voices
        - Helps debug multi-voice generation issues
        - Example: `[AllTalk GEN] Debug debug_narrator Narrator voice: voice_A`
    """
    
    DEBUG_HELP2 = """
    - **debug_fullttstext**
        - Shows complete text being processed for TTS
        - Without this, text is truncated to 90 characters in logs
        - Useful when troubleshooting long text processing
        - Example: `[AllTalk GEN] Debug debug_fullttstext Full text: [entire text content]`

    - **debug_gradio_IP**
        - Monitors URL/IP configuration used by Gradio and environment status
        - Shows runtime environment (Docker/Colab) and active AllTalk API endpoint set
        - Helps diagnose connection issues between Gradio UI and AllTalk's API server
        - Example: `[AllTalk TTS] Debug debug_gradio_IP Base URL is set as : http://127.0.0.1:7851/api/voices`        

    ## 🔍 Audio Processing Debug Options

    - **debug_transcode**
        - Monitors audio format conversion processes
        - Shows input/output formats and transcoding steps
        - Useful for troubleshooting audio format issues
        - Use `debug_openai` for OpenAI endpoint transcoding
        - Example: `[AllTalk TTS] Debug debug_transcode Converting wav to mp3`

    - **debug_concat**
        - Tracks audio file concatenation operations within the Narrator
        - Shows files being combined and resulting output
        - Helps debug multi-part audio assembly
        - Example: `[AllTalk TTS] Debug debug_concat Combining files: [file1.wav, file2.wav]`

    - **debug_rvc**
        - Monitors RVC (Retrieval-based Voice Conversion) processes
        - Shows model loading, voice conversion steps
        - Essential for troubleshooting voice conversion issues
        - Example: `[AllTalk GEN] Debug debug_rvc Processing with model: character_voice.pth`

    - **debug_openai**
        - Tracks OpenAI API integration
        - Shows OpenAI-specific request parameters and responses
        - Useful for troubleshooting OpenAI voice generation
        - Example: `[AllTalk TTS] Debug debug_openai Using voice: alloy`

    ## 🔍 Usage Tips
    - Enable multiple debug options together for comprehensive logging
    - Use `debug_func` with other options to track execution flow
    - Enable `debug_fullttstext` only when necessary to avoid log clutter
    - Combine related options (e.g., `debug_tts` + `debug_tts_variables`) for detailed troubleshooting
    """
    
    ALLTALK_SETTINGS_PAGE1 = """
    ## ⚙️ Global Settings Help

    The **Global Settings** page allows you to configure essential system-wide parameters and enable various debugging options for detailed troubleshooting. This section provides an overview of each setting, its purpose, and guidelines for usage.

    ## ⚙️ File Management
    
    - **Del WAV's Older Than**: On start-up automatically deletes WAV files older than the specified duration to save disk space.  
        - **Default**: `Disabled`     
        - **Options**: Disabled or specify a duration (e.g., 7 days, 30 days).
        - **Recommendation**: Set at least 30 days to keep your outputs folder from growing too large.

    - **Output Folder Name (sub AllTalk)**: Defines the subfolder where all generated files will be stored.  
        - **Default**: `outputs`
        - **Recommendation**: Preferably do not change this.
        - **Tip**: Ensure the folder name is unique for clarity when managing multiple projects.

    ## ⚙️ API and Gradio Interface Settings
    
    - **API Port Number**: The port through which the AllTalk API is accessible.  
        - **Default**: `7851`
        - **Note**: As standard AllTalk binds to 0.0.0.0 meaning all available IP addresses on your machine.
        - **Note**: Ensure this port is not blocked by firewalls or used by other applications.

    - **Gradio Port Number**: The port number for the Gradio interface used for the frontend.  
        - **Default**: `7852`
        - **Note**: As standard AllTalk binds to 0.0.0.0 meaning all available IP addresses on your machine.
        - **Note**: Ensure this port is not blocked by firewalls or used by other applications

    - **Gradio Theme Selection**: Customize the appearance of the Gradio interface by choosing a theme.  
        - **Options**: `gradio/base` (default) or other themes supported by Gradio.
        - **Note**: A restart is required for the change to take effect.

    - **Gradio Interface**: Enables or disables the Gradio interface. 
        - **Default**: `Enabled` 
        - **Warning**: Disabling this option will disable Gradio from loading.
        - **Tip**: You can re-enable the gradio interface by using the API web page (e.g., `http://127.0.0.1:7851/`).
    """
    
    ALLTALK_SETTINGS_PAGE2 = """
   ## ⚙️ Transcoding
   
   - **Audio Transcoding**
        - **Default**: `Disabled`
        - Enables or disables automatic audio format conversion for compatibility with the system.
        - **Recommendation**: Enable only if your workflow requires consistent audio formats (e.g., converting WAV to MP3).
        - **Note**: Ensure this port is not blocked by firewalls or used by other applications

    ## 🔍 Debugging Options

    #### Purpose
    The debugging options provide granular insights into various processes, such as TTS generation, API interactions, and audio processing. Enabling these options can help identify and resolve issues during system usage.

    - **Available Debug Options**
        - `debug_func`, `debug_tts`, `debug_rvc`, etc. (See the Debugging Help Section for detailed descriptions).

    #### Usage Tips**
    
    - Enable relevant debugging categories based on the issue being investigated.
    - Combine `debug_func` with other options for tracing execution paths.
    - Avoid enabling all options simultaneously to prevent log clutter.

    ## ⚙️ Interface Tab Management

    #### Disable Gradio Interface Tabs
    Use these checkboxes to hide specific tabs in the Gradio interface, simplifying the user interface for specific workflows.
     
    - **Tabs Available**
        - **Generate Help**: Displays help documentation.
        - **Voice2RVC**: Access the Voice-to-RVC tools.
        - **TTS Generator**: Manage TTS generation tasks.
        - **TTS Engines Settings**: Configure and manage TTS engines.
        - **AllTalk Documentation**: View detailed documentation and usage instructions.
        - **API Documentation**: Reference API endpoints and integration details.
    """    
    
    RVC_PAGE1 = """
    The **RVC Settings** page provides configuration options for Real-Time Voice Cloning (RVC), a feature that enhances TTS by replicating voice characteristics for characters or narrators. This section provides detailed descriptions of the available settings and recommendations for their usage.

    ## 🗣️ Setup and Model Management

    - **Enable RVC**: 
        - **Purpose**: Toggles the Real-Time Voice Cloning feature & downloads the base RVC model required.
        - **Default**: Disabled
        - **Recommendation**: Enable this only if you plan to use RVC-enhanced TTS.
        - **Tip**: Enable this to download and setup your base RVC models and folders.

    - **Refresh Model Choices**: 
        - **Purpose**: Refreshes the list of available voice models from the `/models/rvc_voices` directory.
        - **Tip**: Use this after adding new voice models to ensure they are recognized by the system.

    ## 🗣️ Voice Model Configuration

    - **Default Character Voice Model**
        - **Purpose**: Selects the voice model used for character conversion if none other specifed in the API request.
        - **Default**: Disabled
        - **Recommendation**: Assign a model here if you regularly use RVC for character voices and no model is specified in API requests.

    - **Default Narrator Voice Model**
        - **Purpose**: Selects the voice model used for narrator conversion if none other specifed in the API request.
        - **Default**: Disabled
        - **Recommendation**: Assign a model here if you regularly use RVC for narrator voices and no model is specified in API requests.

    ## 🗣️ Conversion Parameters

    - **Pitch**
        - **Purpose**: Adjusts the pitch of the audio output.
        - **Default**: `0` (no pitch adjustment)
        - **Range**: Any positive or negative value.
        - **Recommendation**: Use small adjustments to fine-tune the output pitch. Larger values can distort the audio.

    - **Volume Envelope**
        - **Purpose**: Determines how the output volume envelope blends with the original.
        - **Default**: `1` (fully blends with the output envelope)
        - **Range**: `0.1` to `1`
        - **Tip**: Use higher values for natural blending, especially for speech-focused TTS.

    - **Protect Voiceless Consonants/Breath Sounds**
        - **Purpose**: Reduces artifacts in voiceless consonants and breath sounds.
        - **Default**: `0.5`
        - **Range**: `0.1` to `0.5`
        - **Recommendation**: Use the default value for most cases to preserve sound quality without overloading the index.
    
    - **Filter Radius**
        - **Purpose**: Applies median filtering to reduce respiration artifacts.
        - **Default**: `3`
        - **Range**: `1` to `5`
        - **Recommendation**: Higher values are recommended for smoother audio but may slightly reduce naturalness.
    """
    
    RVC_PAGE2 = """
    ## 🗣️ Advanced Settings

    - **Index Influence Ratio**
        - **Purpose**: Adjusts the weight of the index file in shaping the output.
        - **Default**: `0.75`
        - **Range**: `0.1` to `1`
        - **Recommendation**: Increase for highly detailed voices, but keep lower for natural-sounding output.

    - **Training Data Size**
        - **Purpose**: Sets the number of data points used for training the FAISS index.
        - **Default**: `45000`
        - **Range**: Based on the index file size.
        - **Tip**: Increase for complex voices, but monitor system performance.

    ## 🗣️ Embedder and Audio Handling

    - **Embedder Model**
        - **Purpose**: Selects the model used for learning speaker embedding.
        - **Options**:
            - **hubert**: Focuses on phonetic accuracy.
            - **contentvec**: Captures subtle vocal nuances.
        - **Default**: `hubert`
        - **Recommendation**: Use `contentvec` for more natural and expressive voice cloning.

    - **Split Audio**
        - **Purpose**: Splits audio into smaller chunks for better inference.
        - **Default**: Enabled
        - **Recommendation**: Leave enabled for longer audio inputs.

    ## 🗣️ Pitch Extraction

    - **Pitch Extraction Algorithm**
        - **Purpose**: Determines the algorithm used for pitch extraction during audio conversion.
        - **Options**:
            - **crepe**: High accuracy, robust against noise.
            - **crepe-tiny**: Faster but slightly less accurate.
            - **dio**: Efficient, suitable for real-time usage.
            - **fcpe**: Precise pitch extraction.
            - **harvest**: Smooth and natural pitch contours.
            - **hybrid[rmvpe+fcpe]**: Combines strengths of `rmvpe` and `fcpe`.
            - **pm**: Balanced speed and accuracy.
            - **rmvpe**: Recommended for most cases.
        - **Default**: `rmvpe`
        - **Recommendation**: Use `rmvpe` or `hybrid` for general cases; experiment with others for specific needs.

    ## 🗣️ Usage Tips

    1. **Optimize Index Influence**: Adjust the ratio for better clarity and detail without introducing artifacts.
    2. **Refine Pitch and Volume**: Use small, incremental changes to maintain naturalness.
    3. **Choose the Right Embedder**: Experiment with `contentvec` for expressive voices or `hubert` for precise replication.
    4. **Test Pitch Algorithms**: The default (`rmvpe`) works well for most scenarios, but alternative options can enhance specific audio characteristics.
    """    

    API_DEFAULTS1 = """
    ## 🎯 Quick Start Guide
    The **API Settings** page allows you to configure global defaults and behaviors for all AllTalk API calls. These settings serve as fallbacks when specific parameters aren't provided in individual API requests.

    ### 🎯 Key Concepts:
    1. **Default Values**: These settings act as fallbacks when parameters aren't specified in API calls
    2. **Global Filters**: Control what text/characters can be processed system-wide
    3. **Version Compatibility**: Manage how API responses are formatted
    4. **Playback Behavior**: Configure how generated audio is handled

    ### 🎯 Common Use Cases:
    - Setting system-wide language preferences
    - Configuring output file naming conventions
    - Managing text filtering and character support
    - Controlling audio playback behavior

    ## 🎯 Understanding API Defaults

    API defaults determine how AllTalk behaves when specific parameters are omitted from API requests. Think of them as your system's "automatic" settings.

    ### How Defaults Work:
    1. When an API request includes a specific parameter, it overrides the default
    2. When a parameter is omitted, the system uses the value configured here
    3. Defaults apply to all API calls unless explicitly overridden

    ## 🎯 Example Scenarios:

    **Language Selection:**
    ```plaintext
    # With language in request:
    POST /tts {"text": "Hello", "language": "fr"}
    → Uses French, ignoring default

    # Without language:
    POST /tts {"text": "Hello"}
    → Uses default language set in API Settings
    ```

    **Text Filtering:**
    ```plaintext
    # With filter specified:
    POST /tts {"text": "<text>", "filter": "html"}
    → Uses HTML filtering

    # Without filter:
    POST /tts {"text": "<text>"}
    → Uses default filter setting
    ```

    ## 🎯 Important Notes:
    - Default settings affect ALL API calls
    - Changes require server restart to take effect
    - Monitor logs when changing defaults to ensure desired behavior
    - Test API calls after changing defaults to verify results
    """
    
    API_DEFAULTS2 = """
    ## 🔄 API Version Selection
    Choose which API response format your system should use:

    - **AllTalk v2 API (Recommended):**
        - Returns relative file paths only
        - More flexible for different deployments
        - Example response: 
            ```json
            {
                "output_file": "/outputs/tts_output.wav",
                "status": "success"
            }
            ```
        - Best for:
            - New integrations
            - Modern web applications
            - Flexible deployment environments
            - Container-based systems

    - **AllTalk v1 API (Legacy):**
        - Returns complete URLs with protocol and IP
        - Maintains older integration support
        - Example response:
            ```json
            {
                "output_file": "http://127.0.0.1:7851/outputs/tts_output.wav",
                "status": "success"
            }
            ```
        - Best for:
            - Existing integrations
            - Systems requiring full URLs
            - Direct file access needs
            - Backward compatibility

    ## 🔄 Legacy API Configuration
    Settings specific to v1 API functionality:

    - **IP Address Setting:**
        - Only applies when using v1 API mode
        - Default value: `127.0.0.1`
        - Configuration options:
            - Local testing: `127.0.0.1`
            - Network access: Your server's IP
            - Domain usage: Your domain name
        - Important considerations:
            - Must be accessible to client systems
            - Affects all v1 API responses
            - Impacts Gradio interface functionality
            - Required for legacy client compatibility

    ## 🔄 Integration Notes

    - **API Version Choice:**
        - New projects should use v2 API
        - Only use v1 if specifically needed
        - Can't mix versions in same client
        - Version affects all API endpoints

    - **URL Construction:**
        - V2: Client builds full URLs
        - V1: Server provides full URLs
        - Consider security implications
        - Check firewall/network access

    - **Compatibility Checking:**
        - Test API responses after changes
        - Verify client handling
        - Check file access methods
        - Confirm URL resolution
   """
   
    API_DEFAULTS3 = """
    ## 🎛️ Character Limits
    Control text processing boundaries and system load:

    - **Minimum Length Filter:**
        - Strips sentences shorter than specified length
        - Acts as first-pass text cleanup
        - Examples:
            ```
            Setting: 3 characters
            "Hi" → filtered out
            "Hey" → processed
            "Hello" → processed
            ```
        - Considerations:
            - Lower values allow shorter utterances
            - Higher values reduce processing overhead
            - Recommended: 3 characters
            - Affects all API requests

    - **Maximum Request Size:**
        - Controls total characters per API call
        - Prevents system overload
        - Default: 2000 characters
        - Important factors:
            - Server resources
            - TTS engine limits
            - Processing time
            - Memory usage
        - Example:
            ```
            2000 chars ≈ 300-400 words
            Longer texts should be split into multiple requests
            ```

    ## 🎛️ Output File Management
    Configure how the system handles generated audio files:

    - **Base Filename:**
        - Sets default output name pattern
        - Used when no filename provided
        - Example settings:
            ```
            Basic: myoutputfile
            With prefix: user1_output
            With category: audiobook_chapter1
            ```
        - Tips:
            - Use descriptive names
            - Avoid spaces
            - Consider adding prefixes
            - Keep names consistent

    - **Timestamp Options:**
        - **Enabled (Recommended):**
            - Adds unique timestamp to each file
            - Format: `filename_YYYYMMDD_HHMMSS.wav`
            - Prevents file overwrites
            - Maintains history
            - Example: `myoutputfile_20240320_143022.wav`

        - **Disabled:**
            - Uses exact filename specified
            - Overwrites existing files
            - Example: `myoutputfile.wav`
            - Use with caution

    ## 🎛️ Language Settings
    Define default language behavior:

    - **Default Language:**
        - Used when no language specified in request
        - Affects:
            - Text processing
            - Pronunciation
            - Character handling
            - Voice selection
        - Available codes:
            ```
            en: English    de: German    fr: French
            es: Spanish    it: Italian   ja: Japanese
            zh: Chinese    ru: Russian   ko: Korean
            ```
        - Notes:
            - Must match TTS engine capabilities
            - Can be overridden per request
            - Affects all unspecified requests
            - Consider your primary use case

    ## Request Processing Tips

    - **Optimal Request Size:**
        - Keep requests under 1500 characters for best performance
        - Split long texts at natural breaks
        - Consider paragraph or sentence boundaries
        - Balance between size and processing speed

    - **File Organization:**
        - Use consistent naming conventions
        - Consider adding date-based folders
        - Implement regular cleanup
        - Monitor disk usage
   """

    API_DEFAULTS4 = """
    ## 📝 Text Processing & Filtering

    ## 📝 Text Filtering Modes
    Three levels of processing to handle different input types:

    - **none:**
        - Raw, unfiltered text processing
        - No character filtering applied
        - Maximum flexibility
        - Use cases:
            - Pre-filtered content
            - Known clean input
            - Testing purposes
        - Warning: May expose TTS to problematic characters

    - **standard (Recommended):**
        - Balanced filtering approach
        - Removes problematic characters
        - Maintains readability
        - Handles:
            - Common punctuation
            - Multiple languages
            - Basic formatting
            - Special characters
        - Best for most use cases

    - **html:**
        - Specialized HTML content handling
        - Strips HTML tags
        - Preserves content
        - Example transformations:
            - `<p>Hello</p>` becomes `Hello`
            - `<br>` becomes a line break
            - `&quot;` becomes `"`
            - `&amp;` becomes `&`
            - `&lt;` becomes `<`
        - Perfect for:
            - Web content
            - Rich text
            - Formatted documents
    """
   
    API_DEFAULTS5 = """
    ## 🗣️ Text Types & Formatting
    The narrator function recognizes three distinct types of text:

    - **Narrated Text:**
        - Enclosed in asterisks: `*like this*`
        - Used for story narration or scene description
        - Example: `*She walked into the room*`
        - Can span multiple sentences

    - **Character Text:**
        - Enclosed in double quotes: `"like this"`
        - Represents direct dialogue
        - Example: `"Hello! How are you today?"`
        - Always spoken in character voice

    - **Text-Not-Inside:**
        - Any text not in asterisks or quotes
        - Handling configurable
        - Example: `She thought to herself.`
        - Can be assigned to either voice

    ## 🗣️ Narrator Configuration
    Control how text processing and voice assignment works:

    - **Disabled:**
        - Turns off narrator processing
        - All text uses character voice
        - Fastest processing mode
        - Example:
            ```
            Input: *She smiled.* "Hello!" She waved.
            Output: Everything in character voice
            ```

    - **Enabled:**
        - Full narrator processing active
        - Respects all text type rules
        - Best for story narration
        - Example:
            ```
            Input: *She smiled.* "Hello!" She waved.
            Output: 
            - "She smiled" → Narrator voice
            - "Hello!" → Character voice
            - "She waved" → Based on Text-Not-Inside setting
            ```

    - **Enabled (Silent):**
        - Processes but doesn't speak narrator text
        - Character dialogue still spoken
        - Perfect for action descriptions
        - Example:
            ```
            Input: *She smiled.* "Hello!" She waved.
            Output:
            - "She smiled" → Not spoken
            - "Hello!" → Character voice
            - "She waved" → Based on Text-Not-Inside setting
            ```

    ## 🗣️ Text-Not-Inside Handling
    Configure how untagged text is processed:

    - **Character Mode:**
        - Treats untagged text as dialogue
        - Best for:
            - Dialogue-heavy content
            - Direct thought narration
            - Single-voice stories
        - Example:
            ```
            Text: "Hi there!" She smiled warmly.
            Result: Both spoken in character voice
            ```

    - **Narrator Mode:**
        - Treats untagged text as narration
        - Best for:
            - Story narration
            - Scene description
            - Third-person perspective
        - Example:
            ```
            Text: "Hi there!" She smiled warmly.
            Result: "Hi there" in character voice
                "She smiled warmly" in narrator voice
            ```

    - **Silent Mode:**
        - Untagged text not spoken
        - Useful for:
            - Stage directions
            - Scene markers
            - Internal notes
        - Example:
            ```
            Text: "Hi there!" She smiled warmly.
            Result: Only "Hi there" is spoken
            ```

    ## 🗣️ Story Example
    How different settings affect a story passage:

    ```
    *It was a bright morning when Sarah entered the cafe.*
    "I'd love a coffee, please!"
    She approached the counter eagerly.
    *The barista nodded with a smile.*
    ```

    - **With Standard Settings (Enabled, Narrator mode):**
        - Line 1: Narrator voice
        - Line 2: Character voice
        - Line 3: Narrator voice
        - Line 4: Narrator voice

    - **With Silent Narrator:**
        - Line 1: Not spoken
        - Line 2: Character voice
        - Line 3: Based on Text-Not-Inside
        - Line 4: Not spoken

    ## Playback Configuration
    Control audio output handling:

    - **Playback Location:**
        - **Local:**
            - Browser-based playback
            - Client-side control
            - Lower server load
            - Best for web apps

        - **Remote:**
            - Server-side playback
            - Console output
            - Testing purposes
            - Development use

    - **Remote Volume:**
        - Range: 0.1 to 0.9
        - Default: 0.9
        - Server playback only
        - Adjust for testing needs
   """
   
    API_DEFAULTS6 = """
    ## 📚 API Allowed Text Filtering/Passthrough
    The global character filter that defines what can pass through to the TTS engine:

    - **Default Filter:**
        ```
        [a-zA-Z0-9.,;:!?'"\\s\\-$\\u00C0-\\u00FF\\u0400-\\u04FF\\u0900-\\u097F\\u4E00-\\u9FFF\\u3400-\\u4DBF\\uF900-\\uFAFF\\u0600-\\u06FF\\u0750-\\u077F\\uFB50-\\uFDFF\\uFE70-\\uFEFF\\u3040-\\u309F\\u30A0-\\u30FF\\uAC00-\\uD7A3\\u1100-\\u11FF\\u3130-\\u318F\\u0150\\u0151\\u0170\\u0171\\u2018\\u2019\\u201C\\u201D\\u3001\\u3002\\uFF01\\uFF0C\\uFF1A\\uFF1B\\uFF1F]
        ```

    - **Filter Breakdown:**
        - **Basic Characters:**
            - `a-zA-Z`: English alphabet
            - `0-9`: Numeric digits
            - `.,;:!?`: Basic punctuation
            - `'"`: Quotes
            - `\\s`: Whitespace
            - `\\-`: Hyphen/dash
            - `$`: Dollar sign

        - **Extended Latin:**
            - `\\u00C0-\\u00FF`: Latin-1 characters (À-ÿ)
            - `\\u0150\\u0151\\u0170\\u0171`: Hungarian characters

        - **Asian Scripts:**
            - `\\u4E00-\\u9FFF`: CJK Unified Ideographs
            - `\\u3400-\\u4DBF`: CJK Extension A
            - `\\uF900-\\uFAFF`: CJK Compatibility
            - `\\u3040-\\u309F`: Hiragana
            - `\\u30A0-\\u30FF`: Katakana
            - `\\uAC00-\\uD7A3`: Hangul Syllables

        - **Other Scripts:**
            - `\\u0400-\\u04FF`: Cyrillic
            - `\\u0900-\\u097F`: Devanagari
            - `\\u0600-\\u06FF`: Arabic
            - `\\u0750-\\u077F`: Arabic Supplement
            - `\\uFB50-\\uFDFF`: Arabic Presentation Forms-A
            - `\\uFE70-\\uFEFF`: Arabic Presentation Forms-B

        - **Special Punctuation:**
            - `\\u2018\\u2019`: Smart single quotes
            - `\\u201C\\u201D`: Smart double quotes
            - `\\u3001\\u3002`: CJK punctuation
            - `\\uFF01\\uFF0C\\uFF1A\\uFF1B\\uFF1F`: Fullwidth punctuation

    - **Usage Notes:**
        - Characters not in this list are stripped before processing
        - Modify to support additional scripts/characters
        - Test thoroughly after modifications
        - Consider TTS engine limitations

    - **Common Modifications:**
        - **Add Mathematical Symbols:**
            ```
            \\u2200-\\u22FF
            ```
        - **Add Currency Symbols:**
            ```
            \\u20A0-\\u20CF
            ```
        - **Add Musical Notation:**
            ```
            \\u2669-\\u266F
            ```
    ## 📚 Character Passthrough Configuration
    Global character filtering system:

    - **Purpose:**
        - First-line text defense
        - Prevents TTS engine issues
        - Ensures consistent processing
        - Maintains output quality

    - **Supported Character Sets:**
        - **Basic Characters:**
            - A-Z, a-z: English alphabet
            - 0-9: Numeric digits
            - Space, tab: Whitespace
            - Basic punctuation: .,!?

        - **Extended Latin:**
            - Accented characters: é, ñ, ü
            - Special symbols: €, £, ©
            - Extra punctuation: «, », „

        - **CJK Support:**
            - Chinese characters
            - Japanese kanji/kana
            - Korean hangul
            - Related punctuation: 。、

        - **Other Scripts:**
            - Cyrillic (Russian, etc.)
            - Arabic and Hebrew
            - Devanagari (Hindi, etc.)
            - Thai and Vietnamese

    ## 📚 Unicode Ranges
    Specific character ranges allowed:

    - **Common Ranges:**
        ```
        Basic Latin: \u0000-\u007F
        Latin-1 Supplement: \u0080-\u00FF
        Latin Extended-A: \u0100-\u017F
        CJK Unified: \u4E00-\u9FFF
        ```

    - **Specialized Ranges:**
        ```
        Currency Symbols: \u20A0-\u20CF
        Arrows: \u2190-\u21FF
        Mathematical Operators: \u2200-\u22FF
        ```

    ## 📚 Best Practices

    - **Character Filtering:**
        - Start with standard filtering
        - Test with target languages
        - Monitor TTS output quality
        - Add ranges as needed

    - **Common Issues:**
        - Missing characters in output
        - Unexpected symbols
        - Broken punctuation
        - Encoding problems

    - **Solutions:**
        - Verify character support
        - Check input encoding
        - Update filter settings
        - Test with sample texts

    ## 📚 Optimization Tips

    - **Performance:**
        - Use minimal character ranges
        - Enable appropriate filtering
        - Pre-process when possible
        - Cache filtered results

    - **Quality:**
        - Balance filtering vs needs
        - Test with real content
        - Monitor error logs
        - Regular expression updates

    - **Testing Changes:**
        - Always test with sample text
        - Verify TTS engine compatibility
        - Check for unexpected interactions
        - Monitor processing performance
        """
        
    API_DEFAULTS7 = """
    ## ❗ Troubleshooting & Best Practices

    - **Missing or Silent Audio:**
        - **Symptoms:**
            - No audio output
            - Some text not spoken
            - Unexpected silence
        - **Checks:**
            - Verify narrator isn't in silent mode
            - Check Text-Not-Inside settings
            - Confirm playback location setting
            - Validate file permissions
        - **Solutions:**
            - Switch narrator to "Enabled"
            - Change Text-Not-Inside to "Character" or "Narrator"
            - Test both local and remote playback
            - Check output directory permissions

    - **Character Filtering Issues:**
        - **Symptoms:**
            - Missing text in output
            - Unexpected characters
            - Broken formatting
        - **Checks:**
            - Review allowed character ranges
            - Check input text encoding
            - Verify language settings
        - **Solutions:**
            - Adjust character filter settings
            - Convert text to proper encoding
            - Match language to content

    - **API Response Problems:**
        - **Symptoms:**
            - Incorrect file paths
            - Missing URLs
            - Access issues
        - **Checks:**
            - Confirm API version setting
            - Verify legacy IP configuration
            - Check client compatibility
        - **Solutions:**
            - Match API version to client needs
            - Update legacy IP settings
            - Adjust client URL handling

    ## ❗ Best Practices

    - **Configuration Management:**
        - Document your settings
        - Test changes incrementally
        - Maintain backup configurations
        - Regular setting reviews
        
    - **Text Processing:**
        - Keep requests under limits
        - Use appropriate filtering
        - Test with sample content
        - Validate formatting

    - **File Management:**
        - Regular cleanup of old files
        - Consistent naming patterns
        - Monitor disk usage
        - Backup important outputs

    ## ❗ Performance Optimization

    - **Server-side:**
        - Limit maximum request size
        - Use appropriate character filtering
        - Enable timestamps for tracking
        - Monitor resource usage

    - **Client-side:**
        - Use local playback when possible
        - Split large requests
        - Cache common responses
        - Handle responses efficiently

    ## ❗ Maintenance Checklist

    - **Daily:**
        - Monitor disk space
        - Check error logs
        - Verify API responses
        - Test basic functionality

    - **Weekly:**
        - Clean old output files
        - Review unusual errors
        - Test all playback modes
        - Validate complex requests

    - **Monthly:**
        - Full system testing
        - Configuration review
        - Update documentation
        - Performance evaluation

    ## ❗ Debug Mode Tips

    - **When to Enable:**
        - New integrations
        - Performance issues
        - Unexpected behavior
        - Configuration changes

    - **What to Monitor:**
        - Server logs
        - API responses
        - File generation
        - Processing times

    ## ❗ Security Considerations

    - **API Access:**
        - Control access points
        - Monitor usage patterns
        - Rate limit requests
        - Validate inputs

    - **File Security:**
        - Set proper permissions
        - Clean sensitive data
        - Secure output directory
        - Monitor access logs
   """
   
    GENERATE_SCREEN1 = """
    ## 🎯 Basic Operation
    1. Select your TTS engine and model/voice
    2. Enter text to convert to speech
    3. Adjust available settings
    4. Click "Generate TTS"

    ## 🎯 Key Controls
    - **Swap TTS Engine**: Change between different TTS systems
    - **Load Different Model**: Select different model files (if supported)
    - **Character/Narrator Voice**: Choose voices for speech generation
    - **Refresh Server Settings**: Update interface with latest configuration
    - **Generate TTS**: Create speech from input text

    ## 🎯 TTS Engine & Model Selection

    - **TTS Engine:**
        - Different engines have different capabilities
        - Some use AI models, others use pre-trained voices
        - Settings update automatically when swapped
        - Requires downloaded models/voices to function

    - **Models & Voices:**
        - AI Model Based:
            - Uses voice samples for cloning
            - More flexible but requires setup
        - Pre-trained Voices:
            - Ready-to-use voice sets
            - Limited to available voices

    ## 🎯 Voice Configuration

    - **Character Voice:**
        - Main voice for general text
        - Options depend on loaded engine/model
        - May require voice samples or use pre-trained voices

    - **Narrator Voice:**
        - Secondary voice for narration
        - Used with narrator features enabled
        - Can be same or different from character voice

    - **RVC Voices:**
        - Additional voice modification layer
        - Shows as "Disabled" until RVC activated
        - Requires RVC set to `Enabled` in Global Settings   
    """
   
    GENERATE_SCREEN2 = """
    ## 🎯 Generation Settings

    - **Basic Controls:**
        - Generation Mode: Standard (Streaming planned)
        - Language: If supported by model
        - Text Filtering: None/Standard/HTML
        - Output Filename & Timestamping

    - **Advanced Settings:**
        - Speed: Adjust speech rate
        - Pitch: Modify voice pitch
        - Temperature: Control variation
        - Repetition Penalty: Adjust uniqueness

    - **Playback Options:**
        - Local: Browser playback
        - Remote: Server-side playback
        - Volume: For remote playback

    ## 🎯 Narrator System

    - **Modes:**
        - Enabled: Full narration support
        - Disabled: Single voice only
        - Silent: Process but don't speak narrator text

    - **Text Formatting:**
        - Text in `"quotes"`: Character voice
        - Text in `*asterisks*`: Narrator voice
        - Other text: Based on settings

    ## 🎯 Important Notes

    - **Availability:**
        - Features vary by TTS engine
        - Unsupported options are grayed out
        - Check engine documentation for specifics

    - **Performance:**
        - Generation can be interrupted if supported
        - Refresh settings after major changes
        - Some engines require more resources

    - **Compatibility:**
        - Not all engines support all features
        - Streaming currently unavailable
        - RVC requires separate activation
   
   """
   
    GENERATE_SCREEN3 = """
    ## 🔄 Engine & Model Management

    - **Swap TTS Engine:**
        - Changes the active TTS system
        - Automatically updates available features
        - Updates available voices and models
        - Requires:
            - Engine properly installed
            - At least one model/voice downloaded
            - Sufficient system resources

    - **Load Different Model:**
        - Changes the active model file
        - May reset voice selections
        - Types vary by engine:
            - Base models with voice sampling
            - Pre-trained voice models
            - Language-specific models
            - Multi-voice collections

    ## 🔄 Voice Configuration

    - **Character (Main) Voice:**
        - Primary voice for generation
        - Sources vary by engine:
            - Pre-trained voices
            - Voice cloning results
            - Custom trained models
        - Selected voice persists until changed

    - **Narrator Voice:**
        - Secondary voice for narration
        - Can match or differ from character voice
        - Used when narrator system enabled
        - Perfect for:
            - Story narration
            - Mixed voice content
            - Audio book creation

    - **RVC Voice System:**
        - Optional voice conversion layer
        - Appears as "Disabled" until activated
        - Requires:
            - RVC `Enabled` in Global Settings
            - RVC models downloaded
            - Voice models in correct folder
        - Can be applied to:
            - Character voice
            - Narrator voice
            - Both independently

    ## 🔄 Generation Controls

    - **Generation Mode:**
        - Standard: Complete file generation
        - Streaming: (Gradio interface doesnt support!)
            - Real-time audio generation
            - Lower latency when available

    - **Language Settings:**
        - Multi-language capable models:
            - Language selection available
            - Supports various languages
        - Single-language models:
            - Shows "Model not multi-language"
            - Fixed to model's language
            - Selection disabled

    - **Text Processing:**
        - None:
            - Raw text processing
            - No character filtering
            - May cause issues with special characters
        - Standard:
            - Basic text cleanup
            - Removes problematic characters
            - Best for general use
        - HTML:
            - Strips HTML tags
            - Converts HTML entities
            - Ideal for web content
    """
    
    GENERATE_SCREEN4 = """
    ## 🔄 Advanced Settings

    - **Audio Parameters:**
        - Speed:
            - Adjusts speech rate
            - Range varies by engine
            - May affect quality
        - Pitch:
            - Changes voice pitch
            - Maintains speech rate
            - Engine dependent
        - Temperature:
            - Controls variation
            - Higher = more random
            - Lower = more consistent
        - Repetition Penalty:
            - Reduces repeated sounds
            - Affects naturalness
            - Engine specific

    - **Output Settings:**
        - Filename:
            - Base name for files
            - Timestamp optional
        - Playback:
            - Local: Browser audio
            - Remote: Server playback
            - Volume control for remote

    ## 🔄 Narrator System Settings

    - **Operation Modes:**
        - Enabled:
            - Full narration processing
            - Dual voice support
            - Tag interpretation
        - Disabled:
            - Single voice only
            - No tag processing
            - Faster generation
        - Silent:
            - Processes tags
            - Skips narrator audio
            - Keeps character voice

    - **Text Processing:**
        - Character Text:
            - Enclosed in quotes
            - Uses character voice
            - Always processed
        - Narrator Text:
            - Enclosed in asterisks
            - Uses narrator voice
        - Untagged Text:
            - Configurable handling
            - Can be silenced

    ## 🔄 System Controls

    - **Refresh Settings:**
        - Updates all dropdowns
        - Reloads voice lists
        - Updates engine status
        - When to use:
            - After adding voices
            - Changed engine files
            - System changes

    - **Generation Control:**
        - Start generation
        - Interrupt if supported

    ## 🔄 Tips & Best Practices

    - **Voice Selection:**
        - Test voices before long generation
        - Match voice to content type
        - Consider using narrator for variety

    - **Text Preparation:**
        - Format text appropriately
        - Use correct tags if narrating
        - Check language compatibility
    """

    DOCKER_EXPLAINER = """
    ## 🐳 Docker IP/URL Configuration

    When running AllTalk in a Docker environment, you need to set the correct API address where the AllTalk API/TTS engine can be reached. This is necessary because:

    1. AllTalk runs as two separate components:
    - The TTS API server (handles text-to-speech generation)
    - The Gradio web interface (handles user interaction)

    2. For these components to communicate, the Gradio interface needs to know where to find the AllTalk API server.

    ## 🐳 What to Enter
    - Provide the complete URL including protocol and port: 
    - Example: `http://127.0.0.1:7851` or `https://myserver.ontheinternet.com:7851`
    - This should be the address where AllTalk's API server is accessible via LAN/Internet
    - The address may change each time you restart your Docker environment

    ## 🐳 When to Update
    - After starting AllTalk in Docker
    - If you're using a tunneling service
    - If your server's IP or domain changes
    - If you modify AllTalk's API port or port forwarding settings

    ## 🐳 Troubleshooting
    Enable `debug_gradio_IP` in your settings to see detailed connection information:

    ```
    [AllTalk TTS] Debug debug_gradio_IP Running on Docker is: True
    [AllTalk TTS] Debug debug_gradio_IP Running on Google is: False
    [AllTalk TTS] Debug debug_gradio_IP Base URL is set as  : http://127.0.0.1:7851/api/stop-generation
    ```

    This shows your current environment and the exact URLs being used for API communication.
    """

    VOICE2RVC = """    
    ## 🎯 Voice2RVC Help

    Voice2RVC is a tool that converts spoken audio into different voices using RVC (Retrieval-based Voice Conversion) models.
    """

    VOICE2RVC1 = """  
    ## 🎤 Input Methods

    ### Microphone Recording
    - **Browser Compatibility**
        - Chrome: Best compatibility
        - Firefox: Good support
        - Safari: Limited support
        - Edge: Good support

    - **Microphone Tips**
        - Use external mic for better quality
        - Check browser permissions
        - Monitor input levels
        - Test recording before long sessions

    ### File Upload
    - Supported formats: WAV, MP3
    - Recommended length: 5-30 seconds
    - Quality: Clear audio, minimal background noise
    - Avoid clipping or distortion

    ### Audio Editor Features
    - Trim unwanted sections
    - Adjust audio boundaries
    - Preview before processing
    - Undo/redo capabilities
    - Save edited version

    ## 🎚️ Voice Conversion Settings

    ### RVC Voice Selection
    - Choose from available RVC models
    - Each model has unique characteristics
    - Some models may require pitch adjustment
    - Test different models for best results

    ### Pitch Control
    - **Purpose**: Adjusts output voice pitch
    - **Default**: 0 (neutral)
    - **Adjustment Range**: Positive or negative values
    - **Tips**:
        - Start with small adjustments (±2-3)
        - Match input voice pitch
        - Avoid extreme values
        - Different for each RVC model

    ### Pitch Extraction Algorithms

    1. **rmvpe** (Recommended)
        - Best overall performance
        - Balanced accuracy/speed
        - Default choice
        - Suitable for most uses

    2. **hybrid[rmvpe+fcpe]**
        - Combines two algorithms
        - High accuracy
        - Slower processing
        - Good for complex audio

    3. **crepe**
        - High accuracy
        - CPU intensive
        - Good noise handling
        - Best for clean audio

    4. **crepe-tiny**
        - Faster than full crepe
        - Lower resource usage
        - Slightly less accurate
        - Good for quick tests    
    """

    VOICE2RVC2 = """
    ### Pitch Extraction Algorithms cont...

    5. **dio**
        - Fast processing
        - Real-time capable
        - Lower accuracy
        - Good for live use

    6. **fcpe**
        - Precise extraction
        - Good overall accuracy
        - Moderate speed
        - Reliable results

    7. **harvest**
        - Smooth pitch curves
        - Natural sound
        - Slower processing
        - Good for music

    8. **pm**
        - Balanced algorithm
        - Decent accuracy
        - Fast processing
        - Good alternative to rmvpe

    ## 💡 Best Practices

    1. **Recording Quality**
        - Use good microphone
        - Quiet environment
        - Consistent volume
        - Clear enunciation

    2. **Processing Tips**
        - Trim silence
        - Remove background noise
        - Test different algorithms
        - Start with default settings

    3. **Model Selection**
        - Test multiple models
        - Note pitch requirements
        - Consider voice similarity
        - Check model recommendations

    ## ⚠️ Troubleshooting

    - **No Microphone Found**
        - Check browser permissions
        - Verify microphone connection
        - Try different browser
        - Restart browser

    - **Poor Conversion Quality**
        - Check input audio quality
        - Adjust pitch settings
        - Try different algorithms
        - Consider model compatibility

    - **Browser Issues**
        - Clear browser cache
        - Update browser
        - Check WebRTC support
        - Disable interfering extensions

    ## 🔧 System Requirements

    - Modern web browser
    - Active microphone (for recording)
    """

    TTS_GENERATOR = """
    ## 🎯 TTS Generator Help

    The TTS Generator is a powerful tool for converting large volumes of text to speech, ideal for audiobooks, voice content, and text narration.
    """

    TTS_GENERATOR1 = """
    ## 🚀 Getting Started

    ### Access
    - Default URL: `http://127.0.0.1:7851/static/tts_generator/tts_generator.html`
    - Available via HTML interface
    - Standalone browser interface

    ### Basic Controls
    - **Text Input**: Paste or type your text
    - **Generate TTS**: Start conversion
    - **Pause/Resume**: Control playback
    - **Stop**: Halt current audio (generation continues)
    - **Dark/Light Mode**: Visual preference

    ## ⚙️ Generation Options

    ### Voice Settings
    - **Character Voice**: Primary TTS voice
    - **RVC Voice**: Optional voice modification
    - **RVC Pitch**: Fine-tune voice (-24 to +24)
    - **Language**: Select text language
    - **Chunk Size**: Text segment length
    - **Custom Filename**: Output file naming

    ### Generation Modes

    1. **WAV Chunks**
        - Best for audiobooks
        - Creates separate WAV files
        - Supports editing/regeneration
        - Three playback options:
            - In Browser (populates list)
            - On Server
            - No Playback (memory efficient)

    2. **Streaming**
        - Immediate playback
        - No file saving
        - Cannot be interrupted
        - Browser playback only

    ## 💾 Memory Management

    ### System Recommendations
    - 16GB+ RAM for large projects
    - GPU with adequate VRAM
    - Modern web browser

    ### Large Project Tips
    - Use "No Playback" for 20,000+ words
    - Break text into 5,000-10,000 word blocks
    - Generate blocks separately
    - Combine using external software (e.g., Audacity)
    - Export in smaller batches (500 or less)

    ## ⚡ Performance Optimization

    ### Best Settings
    - Enable DeepSpeed (if supported)
    - Disable Low VRAM mode
    - Unload LLM models from GPU
    - Use "No Playback" for large texts
    - Split exports into smaller groups

    ### Performance Example
    With RTX 4070:
    - 58,000 words
    - DeepSpeed enabled
    - ~1,000 words/minute
    - 2-3 minutes for WAV export
    """

    TTS_GENERATOR2 = """
    ## 📝 Text Processing

    ### Pronunciation Tips
    - Use punctuation for pauses:
        - Semi-colons (;)
        - Colons (:)
        - Periods (.)

    ### Acronym Handling
    Multiple approaches:
    ```
    Chat G P T
    Chat G,P,T
    Chat G.P.T
    Chat G-P-T
    Chat gee pee tea
    ```

    ## 💡 Export Options

    ### WAV Export
    - Combines TTS into single file
    - 1GB file size limit
    - Adjustable batch size
    - Merge large files externally

    ### JSON Export/Import
    - Saves generation catalog
    - Includes text and references
    - Excludes audio files
    - Project backup/transfer

    ### SRT Export
    - Creates synchronized subtitles
    - Matches WAV timestamps
    - Useful for video production

    ## 🔍 Analysis Tools

    ### TTS Analyzer
    - Uses Whisper
    - 2.5GB initial download
    - 96-98% recommended accuracy
    - Flags inconsistencies
    - CPU/GPU compatible
    - **TIP** Small/Meduim Whisper models are much faster

    ## ⭐ Best Practices

    1. **Text Preparation**
        - Keep chunks under 250 characters
        - Test with small samples first
        - Use proper punctuation
        - Consider phonetic spelling

    2. **System Management**
        - Regular JSON exports
        - Monitor memory usage
        - Clean output directory
        - Use appropriate chunk sizes

    3. **Large Projects**
        - Break into sections
        - Use external audio editor
        - Regular backups
        - Monitor system resources

    4. **Quality Control**
        - Test voice combinations
        - Verify pronunciations
        - Check generated files
        - Use analyzer for verification
    """

    NARRATOR = """
    ## Understanding the Narrator Function

    ### Please note detailed explnations can be found on the [Github Wiki](https://github.com/erew123/alltalk_tts/wiki)

    The Narrator function is a powerful feature in AllTalk that enables dynamic voice switching between character dialogue and narration. This creates a more immersive experience, similar to an audiobook where different voices are used for narration and character speech.
    """

    NARRATOR1 = """
    ## How It Works

    When enabled, the Narrator function analyzes your text and identifies three distinct types of content:

    1. **Narrated Text**
    Text enclosed in asterisks is treated as narration. For example:
    `*The wind howled through the empty streets*`
    This would be spoken using your selected narrator voice.

    2. **Character Text**
    Text enclosed in quotation marks is treated as character dialogue. For example:
    `"I've been waiting for you to arrive"`
    This would be spoken using your selected character voice.

    3. **Unenclosed Text**
    Any text not enclosed in either asterisks or quotation marks is handled according to your Text-Not-Inside settings.

    ## Narrator Mode Settings

    The Narrator function offers three operational modes:

    ### Enabled
    - Narrated text (in asterisks) uses the narrator voice
    - Character text (in quotes) uses the character voice
    - Unenclosed text follows Text-Not-Inside settings

    ### Enabled (Silent)
    - Narrated text is recognized but not spoken
    - Character text is spoken normally
    - Useful for including stage directions or context
    - Perfect for scripts or roleplaying scenarios

    ### Disabled
    - All text uses the character voice
    - No special handling of asterisks
    - Simplest processing mode

    ## Managing Unenclosed Text

    The Text-Not-Inside setting determines how text without markers is handled:

    ### As Character
    - Treats unmarked text as character dialogue
    - Useful for casual conversation formats
    - Spoken in the character voice

    ### As Narrator
    - Treats unmarked text as narration
    - Good for story-focused content
    - Spoken in the narrator voice

    ### Silent
    - Unmarked text is not spoken
    - Useful for formatting or instructions
    - Allows for hidden context or notes
    """

    NARRATOR2 = """
    ## Practical Examples

    Here's how different combinations work:

    ### Story Format
    `*Sarah entered the coffee shop, the aroma of fresh coffee filling the air* "I'd love a cappuccino, please" *she said softly*`

    - Narration describes the scene and actions
    - Dialogue presents the character's words
    - Clear separation of voice roles

    ### Script Format
    `*[Scene: A dimly lit room]* "Who's there?" *[Sound of footsteps]*`

    Using Enabled (Silent) for narration:
    - Stage directions remain silent
    - Only dialogue is spoken
    - Maintains script structure

    ## Tips for Best Results

    1. **Consistent Formatting**
    - Use clear markers for different text types
    - Maintain formatting throughout your text
    - Check for matching asterisks and quotes

    2. **Voice Selection**
    - Choose distinct voices for narrator and character
    - Consider the tone of your content
    - Test combinations before long generations

    3. **Text Structure**
    - Keep narration concise
    - Break up long passages
    - Mix narration and dialogue naturally

    4. **Common Patterns**
    - Use narration for scene-setting
    - Include character actions in narration
    - Keep dialogue natural and flowing

    ## Advanced Usage

    For more complex projects:

    1. **Silent Annotations**
    - Use Enabled (Silent) for stage directions
    - Include timing or performance notes
    - Add context without affecting audio

    2. **Mixed Content**
    - Combine different text types effectively
    - Use narration for transitions
    - Balance dialogue and description

    3. **API Integration**
    - Set defaults in the API
    - Override settings as needed
    - Maintain consistent formatting

    Remember that the Narrator function is designed to be flexible. Experiment with different combinations of settings to find what works best for your specific needs.
    """

    API_STANDARD = """
    ## Please note detailed explnations, along with code samples can be found on the [Github Wiki](https://github.com/erew123/alltalk_tts/wiki)

    This endpoint allows you to generate Text-to-Speech (TTS) audio based on text input. It supports both character and narrator speech generation.

    To understand how tts requests to this endpoint flow through AllTalk V2, please see the [flowchart here](https://github.com/erew123/alltalk_tts/wiki/API-%E2%80%90-TTS-Request-Flowchart)

    ## Endpoint Details

    - **URL**: `http://{ipaddress}:{port}/api/tts-generate`
    - **Method**: `POST`
    - **Content-Type**: `application/x-www-form-urlencoded`

    ## Request Parameters

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `text_input` | string | The text you want the TTS engine to produce. |
    | `text_filtering` | string | Filter for text. Options: `none`, `standard`, `html` |
    | `character_voice_gen` | string | The name of the character's voice file (WAV format). |
    | `rvccharacter_voice_gen` | string | The name of the RVC voice file for the character. Format: `folder\file.pth` or `Disabled` |
    | `rvccharacter_pitch` | integer | The pitch for the RVC voice for the character. Range: -24 to 24 |
    | `narrator_enabled` | boolean | Enable or disable the narrator function. |
    | `narrator_voice_gen` | string | The name of the narrator's voice file (WAV format). |
    | `rvcnarrator_voice_gen` | string | The name of the RVC voice file for the narrator. Format: `folder\file.pth` or `Disabled` |
    | `rvcnarrator_pitch` | integer | The pitch for the RVC voice for the narrator. Range: -24 to 24 |
    | `text_not_inside` | string | Specify handling of lines not inside quotes or asterisks. Options: `character`, `narrator`, `silent` |
    | `language` | string | Choose the language for TTS. (See supported languages below) |
    | `output_file_name` | string | The name of the output file (excluding the .wav extension). |
    | `output_file_timestamp` | boolean | Add a timestamp to the output file name. |
    | `autoplay` | boolean | Enable or disable playing the generated TTS to your standard sound output device at the Terminal/Command prompt window. |
    | `autoplay_volume` | float | Set the autoplay volume. Range: 0.1 to 1.0 |
    | `speed` | float | Set the speed of the generated audio. Range: 0.25 to 2.0 |
    | `pitch` | integer | Set the pitch of the generated audio. Range: -10 to 10 |
    | `temperature` | float | Set the temperature for the TTS engine. Range: 0.1 to 1.0 |
    | `repetition_penalty` | float | Set the repetition penalty for the TTS engine. Range: 1.0 to 20.0 |

    ### Supported Languages

    | Code | Language |
    |------|----------|
    | `ar` | Arabic |
    | `zh-cn` | Chinese (Simplified) |
    | `cs` | Czech |
    | `nl` | Dutch |
    | `en` | English |
    | `fr` | French |
    | `de` | German |
    | `hi` | Hindi (limited support) |
    | `hu` | Hungarian |
    | `it` | Italian |
    | `ja` | Japanese |
    | `ko` | Korean |
    | `pl` | Polish |
    | `pt` | Portuguese |
    | `ru` | Russian |
    | `es` | Spanish |
    | `tr` | Turkish |

    ## Example Requests

    ### Standard TTS Speech Example

    Generate a time-stamped file for standard text and play the audio at the command prompt/terminal:

    ```bash
    curl -X POST "http://127.0.0.1:7851/api/tts-generate" \
        -d "text_input=All of this is text spoken by the character. This is text not inside quotes, though that doesnt matter in the slightest" \
        -d "text_filtering=standard" \
        -d "character_voice_gen=female_01.wav" \
        -d "narrator_enabled=false" \
        -d "narrator_voice_gen=male_01.wav" \
        -d "text_not_inside=character" \
        -d "language=en" \
        -d "output_file_name=myoutputfile" \
        -d "output_file_timestamp=true" \
        -d "autoplay=false" \
        -d "autoplay_volume=0.8"
    ```

    ### Narrator Example

    Generate a time-stamped file for text with narrator and character speech and play the audio at the command prompt/terminal:

    ```bash
    curl -X POST "http://127.0.0.1:7851/api/tts-generate" \
        -d "text_input=*This is text spoken by the narrator* \"This is text spoken by the character\". This is text not inside quotes." \
        -d "text_filtering=standard" \
        -d "character_voice_gen=female_01.wav" \
        -d "narrator_enabled=true" \
        -d "narrator_voice_gen=male_01.wav" \
        -d "text_not_inside=character" \
        -d "language=en" \
        -d "output_file_name=myoutputfile" \
        -d "output_file_timestamp=true" \
        -d "autoplay=false" \
        -d "autoplay_volume=0.8"
    ```

    Note: If your text contains double quotes, escape them with \\" (see the narrator example).

    ## Minimal Request Example

    You can send a request with any mix of settings you wish. Missing fields will be populated using default API Global settings and default TTS engine settings:

    ```bash
    curl -X POST "http://127.0.0.1:7851/api/tts-generate" \
        -d "text_input=All of this is text spoken by the character. This is text not inside quotes, though that doesnt matter in the slightest"
    ```

    ## Response

    The API returns a JSON object with the following properties:

    | Property | Description |
    |----------|-------------|
    | `status` | Indicates whether the generation was successful (`generate-success`) or failed (`generate-failure`). |
    | `output_file_path` | The on-disk location of the generated WAV file. |
    | `output_file_url` | The HTTP location for accessing the generated WAV file for browser playback. |
    | `output_cache_url` | The HTTP location for accessing the generated WAV file as a pushed download. |

    Example response:

    ```json
    {
        "status": "generate-success",
        "output_file_path": "C:\\text-generation-webui\\extensions\\alltalk_tts\\outputs\\myoutputfile_1704141936.wav",
        "output_file_url": "/audio/myoutputfile_1704141936.wav",
        "output_cache_url": "/audiocache/myoutputfile_1704141936.wav"
    }
    ```

    Note: The response no longer includes the IP address and port number. You will need to add these in your own software/extension.

    ## Additional Notes

    - All global settings for the API endpoint can be configured within the AllTalk interface under Global Settings > AllTalk API Defaults.
    - TTS engine-specific settings, such as voices to use or engine parameters, can be set on an engine-by-engine basis in TTS Engine Settings > TTS Engine of your choice.
    - Although you can send all variables/settings, the loaded TTS engine will only support them if it is capable. For example, you can request a TTS generation in Russian, but if the TTS model that is loaded only supports English, it will only generate English-sounding text-to-speech.
    - Voices sent in the request have to match the voices available within the TTS engine loaded. Generation requests where the voices don't match will result in nothing being generated and possibly an error message.
    """

    API_STREAMING = """
    ## Please note detailed explnations, along with code samples can be found on the [Github Wiki](https://github.com/erew123/alltalk_tts/wiki)

    This endpoint allows you to generate and stream TTS audio directly for real-time playback. It does not support narration and will generate an audio stream, not a file. It also does not support the RVC pipeline.

    Only TTS engines that can support streaming can stream audio e.g. Coqui XTTS supports streaming.

    ## Endpoint Details

    - **URL**: `http://{ipaddress}:{port}/api/tts-generate-streaming`
    - **Method**: `POST`
    - **Content-Type**: `application/x-www-form-urlencoded`

    ## Request Parameters

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `text` | string | The text to convert to speech. |
    | `voice` | string | The voice type to use. |
    | `language` | string | The language for the TTS. |
    | `output_file` | string | The name of the output file. |

    ## Example Request

    ```bash
    curl -X POST "http://127.0.0.1:7851/api/tts-generate-streaming" \
        -d "text=Here is some text" \
        -d "voice=female_01.wav" \
        -d "language=en" \
        -d "output_file=stream_output.wav"
    ```

    ## Response

    The endpoint returns a StreamingResponse for the audio stream.

    The API also returns a JSON object with the following property:

    | Property | Description |
    |----------|-------------|
    | `output_file_path` | The name of the output file. |

    Example response:

    ```json
    {
        "output_file_path": "stream_output.wav"
    }
    ```

    ## JavaScript Example for Streaming Playback

    Here's an example of how to use the streaming endpoint in JavaScript for real-time audio playback:

    ```javascript
    const text = "Here is some text";
    const voice = "female_01.wav";
    const language = "en";
    const outputFile = "stream_output.wav";
    const encodedText = encodeURIComponent(text);
    const streamingUrl = `http://localhost:7851/api/tts-generate-streaming?text=${encodedText}&voice=${voice}&language=${language}&output_file=${outputFile}`;
    const audioElement = new Audio(streamingUrl);
    audioElement.play();
    ```

    ## Additional Notes

    1. **No Narration Support**: This endpoint does not support the narrator function available in the standard TTS generation endpoint.

    2. **No RVC Pipeline**: The streaming endpoint does not support the RVC (Real-time Voice Conversion) pipeline.

    3. **Real-time Playback**: This endpoint is designed for scenarios where you need immediate audio output, such as interactive applications or real-time text-to-speech conversions.

    4. **Browser Compatibility**: The streaming functionality works well with modern web browsers that support audio streaming. Make sure to test compatibility with your target browsers. Firefox may NOT work.

    5. **Error Handling**: Implement proper error handling in your client-side code to manage potential issues with the audio stream.

    6. **Bandwidth Considerations**: Streaming audio requires a stable network connection. Consider the bandwidth requirements when implementing this in your application, especially for longer text inputs.

    7. **File Output**: Although the API returns an `output_file_path`, the primary purpose of this endpoint is streaming. The file output is a side effect and might not be necessary for all use cases.

    8. **Voice Selection**: Ensure that the voice you specify in the request is available in your AllTalk configuration. Using an unavailable voice may result in an error or default voice selection.

    9. **Language Support**: The language support for streaming TTS generation is the same as the standard TTS generation. Refer to the supported languages list in the standard TTS generation documentation.
    """

    API_CONTROL = """
    ## Please note detailed explnations, along with code samples can be found on the [Github Wiki](https://github.com/erew123/alltalk_tts/wiki)

    This set of endpoints allows you to control various aspects of the AllTalk server, including stopping generation, reloading configurations, switching models, and adjusting performance settings.

    ## 1. Stop Generation Endpoint

    Interrupt the current TTS generation process, if the currently loaded TTS engine & method supports it. Currently only XTTS Streaming supports this.

    - **URL**: `http://{ipaddress}:{port}/api/stop-generation`
    - **Method**: `PUT`

    ### Response

    ```json
    {
        "message": "Cancelling current TTS generation"
    }
    ```

    ### Example Request

    ```bash
    curl -X PUT "http://127.0.0.1:7851/api/stop-generation"
    ```

    ### Notes
    - This sets `tts_stop_generation` in the model_engine to True.
    - Stop requests will only be honored if the current TTS engine is capable of handling and can honor a stop request partway through generation.

    ## 2. Reload Configuration Endpoint

    Reload the TTS engine's configuration and scan for new voices and models.

    - **URL**: `http://{ipaddress}:{port}/api/reload_config`
    - **Method**: `GET`

    ### Response

    ```
    Config file reloaded successfully
    ```

    ### Example Request

    ```bash
    curl -X GET "http://127.0.0.1:7851/api/reload_config"
    ```

    ### Notes
    - This ensures that subsequent calls to `/api/currentsettings`, `/api/voices`, and `/api/rvcvoices` return up-to-date information.

    ## 3. Reload/Swap Model Endpoint

    Load or swap to one of the models presented by `/api/currentsettings` in the `models_available` list.

    - **URL**: `http://{ipaddress}:{port}/api/reload`
    - **Method**: `POST`

    ### Parameters

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `tts_method` | string | The name of the model to load |

    ### Response

    ```json
    {"status": "model-success"}
    ```
    or
    ```json
    {"status": "model-failure"}
    ```

    ### Example Request

    ```bash
    curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=xtts%20-%20xttsv2_2.0.2"
    ```

    ## 4. Switch DeepSpeed Endpoint

    Enable or disable DeepSpeed mode, is the currently loaded TTS engine supports it.

    - **URL**: `http://{ipaddress}:{port}/api/deepspeed`
    - **Method**: `POST`

    ### Parameters

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `new_deepspeed_value` | boolean | `True` to enable DeepSpeed, `False` to disable |

    ### Response

    ```json
    {
        "status": "deepspeed-success"
    }
    ```

    ### Example Request

    ```bash
    curl -X POST "http://127.0.0.1:7851/api/deepspeed?new_deepspeed_value=True"
    ```

    ## 5. Switch Low VRAM Endpoint

    Enable or disable Low VRAM mode. Will only benefit TTS Engines that support CUDA.

    - **URL**: `http://{ipaddress}:{port}/api/lowvramsetting`
    - **Method**: `POST`

    ### Parameters

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `new_low_vram_value` | boolean | `True` to enable Low VRAM mode, `False` to disable |

    ### Response

    ```json
    {
        "status": "lowvram-success"
    }
    ```

    ### Example Request

    ```bash
    curl -X POST "http://127.0.0.1:7851/api/lowvramsetting?new_low_vram_value=True"
    ```

    ## Usage Tips

    1. Use the Stop Generation endpoint when you need to cancel an ongoing TTS generation process.
    2. The Reload Configuration endpoint is useful after making changes to the AllTalk configuration or adding new voice models.
    3. Use the Reload/Swap Model endpoint to change the active TTS model dynamically.
    4. The DeepSpeed and Low VRAM endpoints allow you to adjust performance settings based on your system's capabilities and current needs.
    5. Always check the response status to ensure your configuration changes were applied successfully.
    6. After making configuration changes, it's a good practice to use the status endpoints (from the AllTalk Server and TTS Engine Status API) to verify the new state of the server.
    """

    API_STATUS = """
    ## Please note detailed explnations, along with code samples can be found on the [Github Wiki](https://github.com/erew123/alltalk_tts/wiki)

    This set of endpoints allows you to retrieve information about the current state of the AllTalk server and its loaded TTS engine, including available voices, settings, and server readiness.

    ## 1. Server Ready Status Endpoint

    Check if the Text-to-Speech (TTS) engine has started and is ready to accept requests.

    - **URL**: `http://{ipaddress}:{port}/api/ready`
    - **Method**: `GET`

    ### Response

    | Status | Description |
    |--------|-------------|
    | `Ready` | The TTS engine is ready to process requests. |
    | `Unloaded` | The TTS engine is restarting or not ready. |

    ### Example Request

    ```bash
    curl -X GET "http://127.0.0.1:7851/api/ready"
    ```

    ### Notes
    - Useful when performing reload model/engine requests.
    - You can poll this endpoint to confirm a "Ready" status.
    - A "Ready" status only indicates that the engine has loaded, not that all models are correctly loaded.

    ## 2. Standard Voices List Endpoint

    Retrieve a list of available voices for the currently loaded TTS engine and model.

    - **URL**: `http://{ipaddress}:{port}/api/voices`
    - **Method**: `GET`

    ### Response

    ```json
    {
        "status": "success",
        "voices": ["voice1", "voice2", "voice3"]
    }
    ```

    ### Example Request

    ```bash
    curl -X GET "http://127.0.0.1:7851/api/voices"
    ```

    ### Notes
    - If the currently loaded TTS engine does not load a model directly (e.g., Piper), the models themselves will be displayed as voices.

    ## 3. RVC Voices List Endpoint

    Retrieve a list of available RVC voices for further processing your TTS with the RVC pipeline. If RVC is disabled on the server or there are no available voices, the voice returned will just be "Disabled". "Disabled" will always be added as a voice to the list, even if RVC is enabled on the server. This way Disabled can be chosen as a voice to bypass RVC processing, even when RVC is enabled server side.

    - **URL**: `http://{ipaddress}:{port}/api/rvcvoices`
    - **Method**: `GET`

    ### Response

    ```json
    {
        "status": "success",
        "voices": ["Disabled", "folder1\\voice1.pth", "folder2\\voice2.pth", "folder3\\voice3.pth"]
    }
    ```

    ### Example Request

    ```bash
    curl -X GET "http://127.0.0.1:7851/api/rvcvoices"
    ```

    ### Notes
    - `Disabled` will always be included in the list.
    - If the RVC pipeline is globally disabled in AllTalk, `Disabled` will be the only item in the list.
    - Index files matching their RVC voices will not be displayed; if an index file exists for an RVC voice, it will be automatically selected during generation.

    ## 4. Current Settings Endpoint

    Retrieve the current settings of the currently loaded TTS engine.

    - **URL**: `http://{ipaddress}:{port}/api/currentsettings`
    - **Method**: `GET`

    ### Response

    The response is a JSON object containing various settings. Here are some key fields:

    | Field | Description |
    |-------|-------------|
    | `engines_available` | List of available TTS engines that can be loaded |
    | `current_engine_loaded` | The currently loaded TTS engine |
    | `models_available` | List of available models for the current TTS engine |
    | `current_model_loaded` | The currently loaded model |
    | `manufacturer_name` | The manufacturer of the current TTS engine |
    | `audio_format` | The primary format in which the current engine produces audio output |
    | `deepspeed_capable` | Whether the current TTS engine supports DeepSpeed |
    | `generationspeed_capable` | Whether the current TTS engine supports generating TTS at different speeds |
    | `pitch_capable` | Whether the current TTS engine supports generating TTS at different pitches |
    | `temperature_capable` | Whether the current TTS engine supports different temperature settings |
    | `languages_capable` | Whether the models within the current TTS engine support multiple languages |
    | `multivoice_capable` | Whether the current model supports multiple voices |
    | `multimodel_capable` | Whether the current TTS engine uses models as voices |

    ### Example Request

    ```bash
    curl -X GET "http://127.0.0.1:7851/api/currentsettings"
    ```

    ```
    {
    "engines_available": ["parler", "piper", "vits", "xtts"],
    "current_engine_loaded": "xtts",
    "models_available": [
        {"name": "xtts - xttsv2_2.0.2"},
        {"name": "apitts - xttsv2_2.0.2"},
        {"name": "xtts - xttsv2_2.0.3"},
        {"name": "apitts - xttsv2_2.0.3"}
    ],
    "current_model_loaded": "xtts - xttsv2_2.0.3",
    "manufacturer_name": "Coqui",
    "audio_format": "wav",
    "deepspeed_capable": true,
    "deepspeed_available": true,
    "deepspeed_enabled": true,
    "generationspeed_capable": true,
    "generationspeed_set": 1,
    "lowvram_capable": true,
    "lowvram_enabled": false,
    "pitch_capable": false,
    "pitch_set": 0,
    "repetitionpenalty_capable": true,
    "repetitionpenalty_set": 10,
    "streaming_capable": true,
    "temperature_capable": true,
    "temperature_set": 0.75,
    "ttsengines_installed": true,
    "languages_capable": true,
    "multivoice_capable": true,
    "multimodel_capable": true
    }
    ```



    ### Notes
    - The response includes many more fields detailing the capabilities and current settings of the loaded TTS engine.
    - Use this endpoint to dynamically adjust your application's UI based on the current capabilities of the loaded TTS engine.

    ## Usage Tips

    1. Check the server readiness before making other API calls.
    2. Use the voices and RVC voices endpoints to populate selection menus in your application.
    3. Use the current settings endpoint to adjust your application's features based on the capabilities of the loaded TTS engine.
    4. Regularly poll these endpoints if you need to maintain an up-to-date status of the AllTalk server in a long-running application.
    """

    API_OPENAPI = """
    ## Please note detailed explnations, along with code samples can be found on the [Github Wiki](https://github.com/erew123/alltalk_tts/wiki)

    AllTalk provides an endpoint compatible with the OpenAI Speech v1 API. This allows for easy integration with existing systems designed to work with OpenAI's text-to-speech service.

    ## Endpoint Details

    - **URL**: `http://{ipaddress}:{port}/v1/audio/speech`
    - **Method**: `POST`
    - **Content-Type**: `application/json`

    ## Request Format

    The request body must be a JSON object with the following fields:

    | Field | Type | Description |
    |-------|------|-------------|
    | `model` | string | The TTS model to use. Currently ignored, but required in the request. |
    | `input` | string | The text to generate audio for. Maximum length is 4096 characters. |
    | `voice` | string | The voice to use when generating the audio. |
    | `response_format` | string | (Optional) The format of the audio. Audio will be transcoded to the requested format. |
    | `speed` | float | (Optional) The speed of the generated audio. Must be between 0.25 and 4.0. Default is 1.0. |

    ### Supported Voices

    The `voice` parameter supports the following values:

    - `alloy`
    - `echo`
    - `fable`
    - `nova`
    - `onyx`
    - `shimmer`

    These voices are mapped to AllTalk voices on a one-to-one basis within the AllTalk Gradio interface, on a per-TTS engine basis.

    ## Example cURL Request

    ```bash
    curl -X POST "http://127.0.0.1:7851/v1/audio/speech" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "any_model_name",
            "input": "Hello, this is a test.",
            "voice": "nova",
            "response_format": "wav",
            "speed": 1.0
            }'
    ```

    ## Response

    The endpoint returns the generated audio data directly in the response body.

    ## Additional Notes

    - There is no capability within this API to specify a language. The response will be in whatever language the currently loaded TTS engine and model support.
    - If RVC is globally enabled in AllTalk settings and a voice other than "Disabled" is selected for the character voice, the chosen RVC voice will be applied after the TTS is generated and before the audio is transcoded and sent back out.

    ## Voice Remapping

    You can remap the 6 OpenAI voices to any voices supported by the currently loaded TTS engine using the following endpoint:

    - **URL**: `http://{ipaddress}:{port}/api/openai-voicemap`
    - **Method**: `POST`
    - **Content-Type**: `application/json`

    ### Example Voice Remapping Request

    ```bash
    curl -X POST "http://localhost:7851/api/openai-voicemap" \
        -H "Content-Type: application/json" \
        -d '{
            "alloy": "female_01.wav",
            "echo": "female_01.wav",
            "fable": "female_01.wav",
            "nova": "female_01.wav",
            "onyx": "male_01.wav",
            "shimmer": "male_02.wav"
            }'
    ```

    Note: The Gradio interface will not reflect these changes until AllTalk is reloaded, as Gradio caches the list.
    """

    TRANSCRIBE = """
    ## 🎯 Transcribe Help

    Transcribe is a tool that converts spoken audio into text using OpenAI's Whisper speech recognition models. It supports multiple output formats and batch processing capabilities.
    """

    TRANSCRIBE1 = """
    ## 🎤 Input Methods

    ### File Upload
    - Supported formats: MP3, WAV, M4A, OGG, FLAC, AAC, WMA, AIFF, ALAC, OPUS
    - Recommended max size: 50MB per file (larger allowed)
    - Multiple files supported
    - Quality tips:
        - Clear audio with minimal background noise
        - Consistent volume levels
        - Good microphone quality
        - Minimal echo or reverb

    ## 📝 Output Options

    ### Format Selection
    1. **TXT Format**
        - Plain text transcription
        - Simple and clean output
        - Easy to edit and process
        - No timing information

    2. **JSON Format**
        - Complete transcription data
        - Includes timing information
        - Speaker segments
        - Confidence scores
        - Additional metadata

    3. **SRT Format**
        - Standard subtitle format
        - Includes timestamps
        - Compatible with video players
        - Useful for captioning

    ### Organization Features
    - Optional prefix naming
        - Add custom prefix to files
        - Helps organize batches
        - Useful for project management
        - Examples: "meeting_", "interview_"

    - Automatic timestamping
        - Each batch uniquely identified
        - Prevents file overwrites
        - Easy chronological sorting
        - Format: YYYYMMDD_HHMMSS

    ## 🎚️ Model Selection

    ### Available Models
    1. **Tiny**
        - Fastest processing
        - Lowest resource usage
        - Basic accuracy
        - Good for quick tests

    2. **Base**
        - Balanced performance
        - Good for most uses
        - Moderate accuracy
        - Efficient processing
        
    3. **Small**
        - Better accuracy
        - Slightly slower
        - Good all-rounder
        - Recommended default

    4. **Medium**
        - High accuracy
        - Slower processing
        - Better with accents
        - Good for complex audio
    """
        
    TRANSCRIBE2 = """
    ## 🎚️ Model Selection cont...

    5. **Large**
        - Best accuracy
        - Slowest processing
        - Most resource intensive
        - Best for critical transcription

    ## 💡 Best Practices

    1. **File Preparation**
        - Remove unnecessary silence
        - Ensure clear audio quality
        - Split very long recordings
        - Check file format compatibility

    2. **Model Selection Tips**
        - Start with 'base' model
        - Use 'tiny' for quick tests
        - 'medium'/'large' for accuracy
        - Consider processing time needs

    3. **Batch Processing**
        - Group related files
        - Use meaningful prefixes
        - Monitor available space
        - Check progress indicators

    4. **File Management**
        - Use descriptive prefixes
        - Clean up temporary files
        - Download results promptly
        - Organize by project/date

    ## ⚠️ Troubleshooting

    - **File Upload Issues**
        - Check file format
        - Verify file size
        - Try splitting large files
        - Ensure stable connection

    - **Processing Errors**
        - Check audio quality
        - Try different model size
        - Ensure sufficient space
        - Monitor system resources

    - **Output Quality Issues**
        - Try larger model
        - Check input audio quality
        - Consider background noise
        - Verify speech clarity

    ## 🔧 System Requirements

    - Modern web browser
    - Sufficient storage space
    - Stable internet connection
    - Adequate processing power
        - More important for larger models
        - GPU recommended for large batches

    ## 🗂️ Output Directory Structure

    ```
    transcriptions/
    ├── uploads/
    │   └── (temporary audio files)
    └── output/
        ├── project1_20240122_123456/
        │   ├── project1_audio1.txt
        │   ├── project1_audio1.srt
        │   └── project1_summary.json
        └── meeting_20240122_234567/
            ├── meeting_recording1.txt
            ├── meeting_recording1.srt
            └── meeting_summary.json
    """

    WHISPER_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}
    WHISPER_HELP = """
    ## 🎤 Dictate Help

    The Dictate feature enables real-time speech-to-text conversion using OpenAI's Whisper models. This guide explains all available settings and provides recommendations for optimal use.
    """

    WHISPER_HELP1 = """
    ## 📝 Basic Settings

    **Known Bug** After unloading the model you have to **refresh the page** to start trancription again. Believed to be a Gradio bug/issue.

    ### Whisper Model Selection
    - **Purpose**: Determines the model size and accuracy used for transcription
    - **Options**:
        * `tiny` (0.5GB VRAM): Fastest, basic accuracy, good for testing
        * `base` (1GB VRAM): Good balance of speed and accuracy
        * `small` (2GB VRAM): Better accuracy, suitable for clear speech
        * `medium` (4GB VRAM): High accuracy, handles accents better
        * `large-v3` (10GB VRAM): Best accuracy, handles complex audio
        * `large-v3-turbo` (10GB VRAM): Fast high-quality processing
        * `turbo` (10GB VRAM): Fastest high-quality option

    **Recommendation**: Start with a `turbo` model for speed, or `base` if VRAM is very limited and `large-v3`for best quality<br>
    **Note**: Ensure your GPU has sufficient VRAM. Running out of VRAM will cause crashes

    ### Language Settings
    - **Language**: Primary language for transcription
    - **Source Language**: Original speech language if translating
    - **Translate to English**: Converts source language to English
    - **Recommendation**: Set both to match your speech language unless translation needed

    1. **Language**:
        - Sets the language model that Whisper uses to decode speech into text
        - Used as the OUTPUT language for transcription
        - Controls what language the text will be written in
        - Example: Set to English to get English text output

    2. **Source Language**:
        - Tells Whisper what language to expect in the input audio
        - Helps the model better recognize speech patterns and phonemes
        - Improves accuracy when processing non-English speech
        - Example: Set Source Language to French when speaker is speaking French

    ### Language Use Case Examples:
    **Same Language Transcription**: Set both to same language (e.g., English/English for English speech to English text)<br>
    **Translation**: Set Source Language to speaker's language (e.g., French) and Language to desired output language (e.g., English)<br>
    **Best Accuracy**: Always set Source Language to match the actual spoken language, even when translating

    **Note**: If "Translate to English" is enabled, it overrides the Language setting and forces English output.

    ## Output Format
        - **Types**:
            * `txt`: Simple text file
            * `srt`: Subtitle format with timestamps
            * `json`: Detailed format with metadata
        - **Location**: Files saved to `transcriptions/live_dictation/`
        - **Naming**: `[prefix]_dictation_[timestamp].[format]`

    ## 🎚️ Advanced Settings

    ### Audio Processing
    - **Enable ALL Audio Enhancements**:
        * Activates **ALL** audio processing features
        * Use when recording quality needs improvement
        * The checkboxes for other settings will not show checked
        
    - **Apply Noise Reduction**:
        * Reduces background noise
        * **Recommendation**: Enable for noisy environments

    - **Apply Audio Compression**:
        * Evens out volume levels
        * **Use When**: Speaking volume varies significantly

    - **Apply Bandpass Filter**:
    The Bandpass Filter isolates frequencies within the typical range of human speech, filtering out low-frequency rumbles and high-frequency hiss or noise. This improves transcription quality in challenging environments.<br><br>
    - **∿ Low Frequency (Hz)** *(Range: 50 - 120 Hz)*  
        - **Default Value:** 85 Hz  
        - **Lower Values (e.g., 50–70 Hz):**
            - Recommended for deeper male voices or audio with a broader vocal range.
            - Useful when preserving low vocal harmonics in clean recordings.
        - **Higher Values (e.g., 100–120 Hz):**
            - Filters out low-frequency background hums (e.g., AC noise, mic handling noise).
            - Ideal for recordings with rumble or low-frequency distortion.<br><br>

    - **∿ High Frequency (Hz)** *(Range: 3000 - 5000 Hz)*  
        - **Default Value:** 3800 Hz  
        - **Lower Values (e.g., 3000–3500 Hz):**
            - Reduces high-pitched noise, hissing, or squeals in noisy environments.
            - Useful for telephone-quality recordings or audio with limited frequency range.
        - **Higher Values (e.g., 4000–5000 Hz):**
            - Captures more speech detail and overtones, particularly for tonal languages or professionally recorded audio.
            - Suitable for higher-quality recordings where clarity is paramount.

    **Use Bandpass Filter in Noisy Environments:** Helps focus on human speech frequencies while ignoring irrelevant noise.

    **Skip Bandpass for Pristine Audio:** For high-quality recordings, skipping this filter ensures no speech frequencies are lost.    
    """

    WHISPER_HELP2 = """
    ### Timing and Organization
    - **Add Timestamps**:
        * Adds time markers to transcription
        * Useful for long recordings
        * Format: `[HH:MM:SS] Text`

    - **Enable Speaker Diarization**:
        * Attempts to identify different speakers
        * Basic implementation - best for clear speaker transitions
        * Format: `Speaker 1: Text`

    - **Silence Threshold** *(Range: 0.001 - 0.02)*  
    This setting determines the amplitude below which audio is treated as silence. Adjusting this is essential for balancing noise filtering and capturing soft speech.  
        - **Default Value:** 0.008  
        - **Use Higher Values (e.g., 0.01–0.02):**
            - Reduces false triggers from faint background noise (e.g., air conditioners, hums).
            - Suitable for noisier environments where background noise needs to be suppressed.
        - **Use Lower Values (e.g., 0.001–0.007):**
            - Helps capture quieter or softer speech.
            - Ideal for high-quality recordings with minimal background noise.

    **Tip:** Monitor the **Audio Levels graph** during preprocessing or testing to ensure speech is detected while avoiding unnecessary noise capture. Fine-tune based on the quietest speaker in your audio.

    ### **Additional Notes for Optimization:**
    - **Experiment with Combined Settings:**
        - Pair a **higher Silence Threshold** (e.g., 0.01) with a **narrower Bandpass range** (e.g., 100–3500 Hz) in noisy environments.
        - Use a **lower Silence Threshold** (e.g., 0.005) and a **wider Bandpass range** (e.g., 50–4000 Hz) for high-quality or wide-spectrum recordings.
    - **Adapt Based on Speaker Profiles:**
        - Adjust **Low Frequency** for male vs. female speakers.
        - Raise **High Frequency** for tonal clarity in languages like Mandarin or musical recordings.
        
    ## 📊 Real-Time Feedback

    ### Audio Levels Graph
    - Shows audio input strength in real-time
    - **How to Read**:
        * X-axis: Time
        * Y-axis: Audio level
        * **Good levels**: Consistent peaks without flatlines or clipping
        * **Too low**: Flat or minimal movement
        * **Too high**: Constant maximum peaks

    ## 🔄 Workflow

    1. **Setup**:
        - Select Whisper model based on your GPU's VRAM
        - Choose output format and optional file name prefix 
        - Configure advanced settings if needed

    2. **Recording**:
        - Click "Load Model" to initialize
        - Press "Record" to start transcription
        - Speak clearly at a consistent volume
        - Watch Audio Levels for feedback
        - Press "Stop" to pause/resume
        - Use "Finish & Unload" when completely done

    3. **File Management**:
        - Transcripts auto-save to `transcriptions/live_dictation/`
        - Each session creates new timestamped files
        - Multiple formats saved if selected

    ## ⚠️ Important Notes

    - **VRAM Usage**:
        * Monitor GPU memory
        * Close other GPU applications
        * Restart application if performance degrades

    - **Best Practices**:
        * Speak clearly and at consistent volume
        * Position microphone correctly
        * Use Audio Levels graph for feedback

    - **Performance Tips**:
        * Use `large-v3-turbo` for best speed/quality balance
        * Enable audio processing only when needed
        * Refresh page between sessions if issues occur
    """