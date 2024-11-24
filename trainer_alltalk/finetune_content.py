# finetune_content.py
# pylint: disable=no-member

class FinetuneContent:
    """CSS and help content for finetune.py"""
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
    
    STEP1_QUICKSTART = """
    1. Place audio files (MP3, WAV, or FLAC) in folder `/alltalk_tts/finetune/put-voice-samples-in-here` or upload through the interface
    2. Provide 2+ minutes of clear audio (more is better and 5+ recommended). Variation in cadence and speech patterns recommended
    3. Set your project name and adjust settings as needed
    4. Generate the dataset
    5. When your dataset it generated, you can either validate the dataset to confirm transcriptions or move to Step 2

    **Minimum Requirements:**
    - 2 minutes of clear audio
    - Consistent audio quality
    - Minimal background noise
    """
    
    STEP1_DETAILED_INSTRUCTIONS = """
    #### Step-by-Step Process
    1. Review [Coqui's guide on creating a good dataset](https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html)
    2. Place your audio files in `{str(audio_folder)}` or upload through interface
    3. Ensure at least 2 minutes of audio (5+ minutes recommended)
    4. Set Maximum Audio Length (controls segment size per epoch)
    5. Enter unique Project Name for dedicated training folder
    6. Select appropriate Whisper model (see Model Selection guide below)
    7. Choose Dataset Language (en Wisper models only support English)
    8. Set Evaluation Split percentage (default 15% recommended)
    9. Consider enabling BPE Tokenizer for unique speech patterns
    """
    STEP1_PROCESS_OVERVIEW = """
    #### What Happens During Generation
    1. Whisper transcribes and segments your audio
    2. System creates smaller clips with transcriptions
    3. Generates training and evaluation datasets
    4. Organizes files in project directory

    #### Generated Files:
    - `/finetune/[project_name]/lang.txt`: Language code
    - `/finetune/[project_name]/metadata_train.csv`: Training data
    - `/finetune/[project_name]/metadata_eval.csv`: Evaluation data
    - `/finetune/[project_name]/wavs/`: Processed audio clips
    """

    STEP1_WHISPER_MODEL_SELECTION = """
    ### Choose Your Model

    #### Quick Selection Guide:
    - **English Only, Limited Resources:** `tiny.en` or `base.en`
    - **Multilingual, Limited Resources:** `tiny` or `base`
    - **Balanced Performance:** `small` or `medium`
    - **Highest Quality:** `large-v3`
    - **Speed Priority:** `large-v3-turbo` or `turbo`

    ### Detailed Model Comparison
    | Model            | Language | Speed         | VRAM  | Best For                   |
    |------------------|----------|---------------|-------|-----------------------------|
    | `tiny.en`        | English  | ⚡️⚡️⚡️⚡️⚡️     | 0.5GB | Quick English tests         |
    | `tiny`           | All      | ⚡️⚡️⚡️⚡️⚡️     | 0.5GB | Quick multilingual tests    |
    | `base.en`        | English  | ⚡️⚡️⚡️⚡️       | 1GB   | Basic English use           |
    | `base`           | All      | ⚡️⚡️⚡️⚡️       | 1GB   | Basic multilingual use      |
    | `small.en`       | English  | ⚡️⚡️⚡️         | 2GB   | Daily English use           |
    | `small`          | All      | ⚡️⚡️⚡️         | 2GB   | Daily multilingual use      |
    | `medium.en`      | English  | ⚡️⚡️           | 4GB   | Better English quality      |
    | `medium`         | All      | ⚡️⚡️           | 4GB   | Better multilingual quality |
    | `large-v1`       | All      | ⚡️             | 8GB   | Legacy support              |
    | `large-v2`       | All      | ⚡️             | 8GB   | Improved legacy             |
    | `large-v3`       | All      | ⚡️             | 10GB  | Best overall quality        |
    | `large`          | All      | ⚡️             | 10GB  | Standard high quality       |
    | `large-v3-turbo` | All      | ⚡️⚡️⚡️         | 10GB  | Fast high quality           |
    | `turbo`          | All      | ⚡️⚡️⚡️⚡️        | 10GB  | Fastest high quality        |
    """
    
    STEP1_ADVANCED_SETTINGS = """
    ### Voice Activity Detection (VAD)
    - Recommended: Enabled
    - Detects and isolates speech segments
    - Removes silence and non-speech audio
    - Improves accuracy and processing speed

    ### Model Precision
    #### Mixed Precision (Recommended for Modern GPUs)
    - Balances speed and accuracy
    - Best for NVIDIA GPUs Pascal architecture (GTX 10 series) and newer
    - Automatically switches between FP16 and FP32 as needed
    - Can significantly reduce VRAM usage

    #### FP16 (Half Precision)
    - Fastest processing and lowest memory usage
    - Only supported on NVIDIA GPUs Pascal architecture (GTX 10 series) and newer
    - May have slightly lower accuracy
    - Not recommended for older GPUs as they lack hardware support

    #### FP32 (Full Precision)
    - Universal compatibility - works on all GPU models
    - Required for older NVIDIA GPUs (pre-GTX 10 series)
    - Default option for CPU processing
    - Highest accuracy but uses more VRAM
    - Slower processing speed compared to FP16/Mixed
    - Use if experiencing stability issues with other precision modes

    ### GPU Compatibility Guide

    | Precision Mode   | Description                                   | Supported GPU Architectures                      | Examples of Compatible GPUs                        |
    |------------------|-----------------------------------------------|--------------------------------------------------|----------------------------------------------------|
    | **FP32 (Full Precision)** | Universal compatibility, highest accuracy  | All GPUs, including older models               | GTX 900 series and older, all AMD GPUs, all laptop GPUs before 2016 |
    | **Mixed Precision (Recommended)** | Balances speed and accuracy, auto-switches between FP16 and FP32 | NVIDIA Turing (RTX 20 series) and newer       | RTX 20 series, RTX 30 series, RTX 40 series, Tesla V100, A100        |
    | **FP16 (Half Precision)** | Fastest processing, lowest memory usage     | NVIDIA Turing and newer (with Tensor Cores)    | RTX 20 series, RTX 30 series, RTX 40 series, Tesla V100, A100        |
    """
    
    STEP1_DEBUG_SETTINGS = """
    Each debug category serves a specific purpose in helping you understand what's happening during the dataset creation process:

    #### GPU Memory Monitoring
    - **What it shows:** Detailed GPU VRAM usage and allocation
    - **When it's useful:**
        * Troubleshooting out-of-memory errors
        * Monitoring VRAM consumption during processing
        * Understanding memory spikes during model loading
        * Tracking memory cleanup effectiveness
    - **Example messages:**
        * "GPU Memory Status: Total: 8192 MB, Used: 3584 MB, Free: 4608 MB"
        * "Low GPU memory available!"

    #### Model Operations
    - **What it shows:** Model loading, inference, and cleanup steps
    - **When it's useful:**
        * Tracking model initialization progress
        * Monitoring transcription operations
        * Debugging model-related errors
        * Verifying proper model cleanup
    - **Example messages:**
        * "Loading Whisper model: large-v3"
        * "Using FP16 precision"
        * "Model successfully unloaded from GPU"

    #### Data Processing
    - **What it shows:** Text and dataset handling operations
    - **When it's useful:**
        * Understanding how text is being processed
        * Tracking dataset creation progress
        * Monitoring CSV file operations
        * Debugging transcription issues
    - **Example messages:**
        * "Loaded existing training metadata"
        * "Processing metadata and handling duplicates"
        * "Created train/eval split with ratio 0.15"

    #### General Flow
    - **What it shows:** Overall process status and file operations
    - **When it's useful:**
        * Following the general progress
        * Tracking file system operations
        * Understanding process flow
        * Identifying process bottlenecks
    - **Example messages:**
        * "Initializing output directory"
        * "Dataset Generated. Move to Dataset Validation"
        * "Successfully wrote metadata files"

    #### Audio Processing
    - **What it shows:** Audio file statistics and processing details
    - **When it's useful:**
        * Understanding audio segmentation
        * Tracking audio duration statistics
        * Identifying problematic audio files
        * Monitoring audio quality issues
    - **Example messages:**
        * "Total segments: 157"
        * "Average duration: 8.3s"
        * "Found 3 segments under minimum duration"

    #### Segments
    - **What it shows:** Detailed information about audio segmentation
    - **When it's useful:**
        * Understanding how audio is being split
        * Debugging segment length issues
        * Tracking segment merging operations
        * Identifying problematic splits
    - **Example messages:**
        * "Extended segment from 4.2s to 6.0s"
        * "Merged 5 segments into 2 longer segments"
        * "Segment too short (2.1s)"

    #### Duplicates
    - **What it shows:** Information about duplicate handling
    - **When it's useful:**
        * Tracking duplicate detection
        * Understanding how duplicates are resolved
        * Monitoring transcription quality
        * Verifying duplicate cleanup
    - **Example messages:**
        * "Found 2 files with multiple transcriptions"
        * "Re-transcribing duplicate files"
        * "Updated transcription for file.wav"

    ### Usage Tips
    - Enable specific categories based on what you're trying to troubleshoot
    - Use GPU Memory when dealing with VRAM issues
    - Enable Segments when audio splitting seems incorrect
    - Turn on Data Process when checking transcription quality
    - Enable Duplicates when investigating repeated content
    - Use General for overall process monitoring
    - Enable Audio when investigating quality issues
    - Turn on Model Ops when dealing with model-related problems
    """
    
    STEP1_TROUBLESHOOTING = """
    ### Common Issues

    #### Windows Users
    - If you see `UserWarning: huggingface_hub cache-system uses symlinks` you can safely ignore it or alternatively:
    1. Close the application
    2. Restart command prompt as Administrator
    3. Relaunch the finetuning process

    ### Managing Whisper Model Files

    #### Why Delete Model Files?
    - Free up disk space (each model is ~3GB)
    - Troubleshoot model-related issues
    - Switch to different model versions
    - Clear corrupted downloads

    #### How to Find and Delete Whisper Models

    Each Whisper model is downloaded automatically when first used and stored in your system's cache. Here's how to manage them:

    **Windows:**
    1. Open File Explorer
    2. Copy and paste this path into the address bar:
    ```
    C:\\Users\\<YourUsername>\\.cache\\whisper
    ```
    (Replace `<YourUsername>` with your actual Windows username)
    3. You'll see folders named after each model you've used (e.g., "base", "large-v3")

    **Linux:**
    1. Open your terminal or file manager
    2. Navigate to:
    ```
    /home/<YourUsername>/.cache/whisper
    ```
    (Replace `<YourUsername>` with your Linux username)
    3. Model files will be in this directory

    **macOS:**
    1. Open Finder
    2. Press `Cmd+Shift+G` to open "Go to Folder"
    3. Enter this path:
    ```
    /Users/<YourUsername>/.cache/whisper
    ```
    (Replace `<YourUsername>` with your macOS username)

    #### To Delete Models:
    1. **Individual Models:** Delete specific model folders to free up space
    2. **All Models:** Delete the entire `whisper` folder to remove all cached models
    3. Don't worry - models will automatically redownload when needed

    #### Important Notes:
    - Deleting models won't affect your training data or results
    - New model downloads will occur next time you use a model
    - Consider keeping frequently used models to avoid redownloading
    """
    
    DATASET_VALIDATION_1 = """
    # Dataset Validation and Correction Tool

    This tool helps you ensure the quality of your training data by comparing:
    - Original transcriptions from your CSV dataset files
    - Running new Whisper transcriptions of the same audio files

    Only mismatched transcriptions are shown for review, helping you focus on potential problems.

    ### Understanding the Display

    #### Table Contents:
    - **Original Text**: Current transcription from metadata files
    - **Whisper Text**: New transcription from Whisper
    - **Filename**: The audio file being transcribed

    ### Common Mismatch Causes:
    - Background noise interference
    - Multiple speakers in audio
    - Unclear pronunciation
    - Audio quality issues
    - Whisper confidence variations
    - Punctuation differences

    ### Interactive Audio Editor

    The built-in audio editor provides several powerful features:

    #### Playback Controls:
    - Play/Pause: Standard audio playback
    - Seek Bar: Navigate through audio
    - Volume Control: Adjust playback volume
    - Speed Control: Adjust playback speed

    #### Audio Editing Features:
    - **Trim Start/End**: Remove silence or unwanted audio
        * Drag handles on waveform edges
        * Fine-tune start/end points
        * Preview trimmed audio
    """

    DATASET_VALIDATION_2 = """
    ...continued

    #### Tips for Audio Editing:
    - Trim silence from start/end
    - Remove background noises
    - Keep only clear speech
    - Maintain natural pauses
    - Ensure clean cut points

    ### Correction Workflow:

    1. **Select an Entry**:
        * Click any row in the mismatch table
        * Audio loads automatically
        * Both transcriptions display

    2. **Review Audio**:
        * Listen to original
        * Edit if necessary
        * Preview changes

    3. **Choose Text Correction Method**:
        * **Use Original**: Keep current transcription
        * **Use Whisper**: Accept new transcription
        * **Edit Manually**: Write custom transcription

    4. **Save Changes**:
        * Click "Save Audio and Correction"
        * Both audio and text will be saved
        * The table will reflect the change

    ### Best Practices:
    - Listen before editing
    - Make minimal necessary cuts
    - Verify transcription accuracy
    - Check edited audio playback

    Changes automatically update the appropriate CSV file and refresh the display.
    """
    
    STEP2_QUICKSTART = """
    Before diving into specific settings, it's important to understand the basic workflow of training your model.
    This section provides an overview of the essential parameters you'll need to get started with training.

    The training process involves feeding your prepared dataset through the model multiple times (epochs) while adjusting
    the model's parameters using specific optimization strategies. The goal is to teach the model to replicate your voice
    accurately while maintaining natural speech patterns.

    #### Required Settings:
    - **Project Name:** Name for your training session
    - **Train/Eval CSV:** Should be auto-populated from Step 1
    - **Model Selection:** Choose your base model
    - **Basic Parameters:**
        * Epochs: How many training cycles (default: 10)
        * Batch Size: Samples per training step (default: 4)
        * Grad Accumulation: Steps between updates (default: 1)
        * Learning Rate: Training speed (default: 5e-6)

    #### Learning Settings
    - **Learning Rate:** 5e-6 (recommended start)
    - **Optimizer:** AdamW (default)
    - **Scheduler:** CosineAnnealingWarmRestarts (default)

    These default settings are carefully chosen to provide a good starting point for most training scenarios. They
    balance training speed with stability and are suitable for most consumer-grade GPUs.
    """

    STEP2_YOUR_OWN_DATASET = """
    If you are **not** using a dataset built on Step 1 and wish to use your own custom pre-built dataset, you can do so with the following instructions.

    To train the model with your custom dataset, follow these steps carefully to ensure your files and folder structure are set up correctly.

    #### Folder and File Structure:
    1. **Create a project folder:** Navigate to `/alltalk_tts/finetune/` and create a new folder with the name of your project (e.g., `myproject`).
    2. **Prepare Required Files:**
        - Inside your project folder, add a file named `lang.txt` containing the two-letter language code for your dataset (e.g. `en` for English, `de` for German).
        - Place your `metadata_train.csv` and `metadata_eval.csv` files in the project folder. These files must follow the dataset formatting guidelines provided by [Coqui AI documentation](https://docs.coqui.ai/en/latest/formatting_your_dataset.html).
    3. **Add Your Audio Files:**
        - Within your project folder, create a subfolder named `wavs` and add your audio files here. The audio files should be in `.wav` format.

    #### CSV Files: Training vs. Evaluation
    - **Training Data (`metadata_train.csv`):** This CSV file contains the majority of your dataset. The model uses this data to learn and adapt to the patterns in your audio. It’s recommended to have about 85% of your audio files in the training dataset, though this ratio can be adjusted based on your needs.
    - **Evaluation Data (`metadata_eval.csv`):** This CSV file contains a smaller subset of your audio (typically around 15%). During training, the model periodically tests itself on this evaluation data to gauge how well it is learning. This evaluation step helps prevent overfitting, where the model might learn training data too closely and struggle with new data.

    **Guideline:** The `metadata_train.csv` and `metadata_eval.csv` files must follow the formatting guidelines provided by [Coqui AI documentation](https://docs.coqui.ai/en/latest/formatting_your_dataset.html).

    #### Updating the Gradio Interface:
    - In the `Project Name` field, enter the name of your project folder (e.g., `myproject`).
    - Populate the `Train CSV file path` and `Eval CSV file path` fields with the full paths to your `metadata_train.csv` and `metadata_eval.csv` files, respectively.
    - Once set up, you are ready to begin training with your custom dataset.

    #### Directory Structure Example:
    ```plaintext
    finetune/
    └── {projectfolder}/
        ├── metadata_train.csv
        ├── metadata_eval.csv
        ├── lang.txt
        └── wavs/
            ├── audio1.wav
            ├── audio2.wav
            └── ...
    ```
    """

    STEP2_TRAINING_METRICS = """
    ### Understanding Training Metrics

    The training progress is visualized through six different graphs, each showing different aspects of the training process:

    #### 1. Epoch Metrics (Avg Losses)
    - Shows average losses across full epochs
    - Lower values indicate better model performance
    - Three metrics tracked:
        * Avg Loss (Green): Overall model loss (may be hidden behind the blue line)
        * Avg Loss Text CE (Red): Text cross-entropy loss
        * Avg Loss MEL CE (Blue): Audio MEL spectrogram loss

    #### 2. Step-wise Loss Metrics
    - Shows individual training step losses
    - More granular view than epoch metrics
    - Same color coding as epoch metrics
    - Helps identify specific problematic training steps

    #### 3. Learning Rate Schedule
    - Shows learning rate changes over training steps
    - Yellow line indicates current learning rate
    - Useful for verifying scheduler behavior
    - Cosine annealing shows cyclical pattern

    #### 4. Gradient Norm over Steps
    - Indicates how dramatically weights are changing
    - Helps monitor training stability
    - Sudden spikes may indicate training issues
    - Should generally remain stable

    #### 5. Step and Loader Times
    - Orange: Time taken for each training step
    - Cyan: Time taken to load data
    - Helps identify performance bottlenecks
    - Higher loader times may indicate I/O issues

    #### 6. Training vs Validation Loss
    - Compares training and validation performance
    - Green: Training loss
    - Red: Validation loss
    - Growing gap may indicate overfitting

    ### Detailed Training Logs

    A complete record of your training run is stored in `trainer_0_log.txt`, located in your training folder:
    ```
    finetune/[project_name]/training/XTTS_FT-[timestamp]/trainer_0_log.txt
    ```

    This log contains:
    - All training metrics
    - Detailed step information
    - Model configuration
    - Performance statistics
    - Error messages (if any)

    **Note:** This log file will be deleted when you use the "Delete Training Data" option in the Model Export step. If you want to keep a record of your training run, make sure to save a copy of this file before deleting the training data.

    ### Reading the Graphs

    - **Downward Trends** in loss values indicate improving model performance
    - **Smooth Lines** suggest stable training
    - **Jagged Lines** might indicate instability or need for hyperparameter adjustment
    - **Plateaus** in loss curves might indicate learning has stalled
    - **Diverging** validation and training losses might indicate overfitting

    ### Common Patterns

    - **Good Training Run:**
        * Steadily decreasing losses
        * Stable gradient norms
        * Consistent step times
        * Close training/validation losses

    - **Potential Issues:**
        * Rapidly increasing losses
        * Highly erratic gradient norms
        * Growing gap between training/validation loss
        * Inconsistent step times
    """

    STEP2_MEMORY_MANAGEMENT = """
    Memory management is crucial for successful training. Understanding how different parameters affect memory usage
    can help you optimize your training process and avoid out-of-memory errors. The two main types of memory to
    consider are GPU VRAM (Video RAM) and system RAM.

    Your GPU's VRAM is the primary resource that will limit what you can do during training. The more VRAM you have,
    the larger batches you can process and the faster your training can progress. However, even with limited VRAM,
    there are several strategies you can use to successfully train your model.

    #### VRAM Requirements:
    - **12GB+ VRAM (Recommended)**
        * Optimal performance
        * Fewer memory constraints
        * Better batch size options
        * Suitable for larger models
    - **8GB VRAM (Minimum)**
        * Windows: Requires 24GB+ System RAM
        * Linux: May encounter limitations
        * Requires careful parameter tuning
        * May need smaller batch sizes

    #### Memory Optimization Strategies:
    - **Batch Size Reduction**
        * Smaller batches use less VRAM
        * May increase training time
        * Start small, increase if stable
        * Monitor memory usage while adjusting

    - **Gradient Accumulation**
        * Simulates larger batches
        * Uses less peak memory
        * Helps with limited VRAM
        * Maintains training stability

    - **Precision Settings**
        * Mixed precision for modern GPUs
        * FP32 for older GPUs
        * FP16 for maximum efficiency
        * Balance accuracy vs. memory

    - **Worker Threads**
        * Fewer workers = less memory
        * Balance between speed and RAM usage
        * Adjust based on system resources
        * Start with 4-8 workers

    #### Shared Memory Option:
    Controls GPU memory allocation during training:

    **Disable Shared Memory Use (Enabled/Checked):**
    * Reserves 5% of GPU VRAM, limiting maximum usage to 95%
    * Can help prevent some out-of-memory crashes
    * May be useful on systems experiencing memory issues

    * **Disable Shared Memory Use (Default/Unchecked):**
    * Allows PyTorch to use full GPU VRAM
    * Maximum training performance
    * May use more VRAM
    """

    STEP2_BATCH_SIZE = """
    Understanding batch size and gradient accumulation is crucial for successful training. These two parameters work
    together to determine how your model processes data and updates its weights during training.

    #### Batch Size Explained
    Batch size determines how many audio samples are processed simultaneously during training. Think of it like
    cooking multiple meals at once - while it's more efficient to cook several dishes simultaneously, your kitchen
    (GPU memory) needs to be large enough to handle it.

    - **What It Controls:**
        * Number of samples processed together
        * Memory usage per step
        * Training stability
        * Learning effectiveness

    - **Guidelines:**
        * Larger = faster but more VRAM
        * Smaller = stable but slower
        * Find balance for your GPU
        * Monitor training stability

    - **Recommended Ranges:**
        * 12GB+ VRAM: 4-8 batch size
        * 8GB VRAM: 2-4 batch size
        * Adjust based on performance
        * Start small and increase if stable

    #### Gradient Accumulation Deep Dive
    Gradient accumulation is like keeping a running tally of changes before updating your recipe. Instead of
    adjusting your cooking technique after each meal, you gather feedback from several attempts before making
    changes.

    - **Purpose:**
        * Simulates larger batch sizes
        * Reduces memory usage
        * Maintains training stability
        * Improves learning quality

    - **How It Works:**
        * Accumulates gradients over steps
        * Updates model less frequently
        * Allows larger effective batches
        * Maintains learning quality

    - **Setting Guidelines:**
        * Start with 1 (default)
        * Increase if OOM errors occur
        * Multiply with batch size for true batch
        * Monitor training progress

    - **Common Values:**
        * 1-4: Standard usage
        * 4-8: Memory constrained
        * 8+: Severe memory limitations
        * Adjust based on stability

    By carefully balancing batch size and gradient accumulation, you can achieve good training results even
    with limited GPU resources. Start conservative and adjust based on your system's performance.
    """

    STEP2_LEARNING_RATE = """
    The learning rate and its scheduling are perhaps the most critical aspects of training. Think of the learning
    rate as the size of the steps your model takes when learning - too large and it might overshoot the best
    solution, too small and it might take too long to get there.

    #### Understanding Learning Rates
    The learning rate determines how much the model changes in response to each batch of training data. Finding
    the right learning rate is crucial for successful training.

    #### Learning Rate Options:
    | Learning Rate Range        | Description                               | Best Use Cases                | Additional Notes                  |
    |----------------------------|-------------------------------------------|-------------------------------|------------------------------------|
    | **1e-6 (Very Conservative)** | Extremely stable, very slow progress      | Fine adjustments              | Use when stability is crucial      |
    | **5e-6 (Recommended Start)** | Good balance of speed and stability       | Most training scenarios       | Best starting point, stable        |
    | **1e-5 to 5e-5 (Moderate)**  | Faster learning, can be unstable          | Experienced users             | Requires monitoring                |
    | **1e-4 to 1e-3 (Aggressive)**| Very fast, high risk of instability       | Advanced users                | Caution required, may diverge      |
    | **5e-4 to 1e-3 (Very Aggressive)** | Rapid rate, often too large            | Experimental only             | High risk of divergence, high regularization needed |

    #### Learning Rate Schedulers Explained
    Schedulers automatically adjust the learning rate during training. This can help fine-tune the learning
    process and avoid getting stuck in suboptimal solutions.

    #### Scheduler Types:
    | Scheduler                       | Description                                    | Best Use Cases                             | Additional Notes              |
    |---------------------------------|------------------------------------------------|--------------------------------------------|--------------------------------|
    | **None**                        | Fixed learning rate throughout                 | Testing                                    | Simple, predictable behavior; may miss optimal points |
    | **CosineAnnealingWarmRestarts** | Cyclical learning rate changes                 | General training, exploration             | Requires 4+ epochs; best all-around choice |
    | **CosineAnnealingLR**           | Smooth cosine decay, no restarts               | Continuous decay needs, stable convergence | Suitable for gradual reduction |
    | **StepLR**                      | Step-wise reduction                            | Planned schedules, predictable decay       | Allows manual step control |
    | **MultiStepLR**                 | Step reductions at multiple intervals          | Customized training schedules              | Requires precise tuning |
    | **ReduceLROnPlateau**           | Adapts to training performance                 | Avoiding plateaus, unknown scenarios       | Reduces rate when improvement slows |
    | **ExponentialLR**               | Exponential decay by a fixed factor            | Rapid convergence                          | Requires monitoring |
    | **CyclicLR**                    | Cycles between min and max rates               | Early-stage convergence, high variability  | Advanced usage |
    | **OneCycleLR**                  | Peaks and returns over single cycle            | Large datasets, fast single-cycle training | Efficient but requires setup |

    The choice of scheduler can significantly impact your training results. The recommended
    CosineAnnealingWarmRestarts provides a good balance of exploration and exploitation during training.

    #### Perform Warmup Learning
    The warmup learning phase is a useful feature for stabilizing early training. During warmup, the learning rate gradually increases from a low initial value to the target learning rate over a set number of steps or epochs. This helps the model adjust gradually, especially beneficial for high learning rates or large batch sizes.

    - **What Warmup Does:**
        * Prevents large early adjustments
        * Improves initial stability
        * Gradually increases learning rate
        * Helps avoid gradient explosion

    - **When to Use:**
        * Recommended with high learning rates
        * Useful for models that diverge early
        * Ideal for large batch sizes
        * Suitable for most cases

    - **Guidelines:**
        * Enable to improve training stability
        * Monitor initial learning curve
        * Adjust warmup steps as needed

    The warmup feature, labeled as **Perform Warmup Learning**, provides a smoother start to the training process, making it ideal for ensuring stability and allowing the model to settle into a balanced learning pattern.
    """
    
    STEP2_OPTIMIZERS = """
    Optimizers are the algorithms that actually update your model's weights during training. Different optimizers
    have different characteristics and can be better suited for certain types of training tasks.

    Think of optimizers as different strategies for adjusting your model's parameters - some are more aggressive,
    some more conservative, and some try to be more adaptive to the specific characteristics of your training data.

    #### Available Optimizers:
    | Optimizer                 | Description                                     | Best Use Cases                      | Additional Notes                             |
    |---------------------------|-------------------------------------------------|-------------------------------------|----------------------------------------------|
    | **AdamW (Recommended)**   | Best all-around performance with weight decay   | General training, modern standard  | Stable convergence, handles most scenarios well |
    | **Adam**                  | Similar to AdamW, less regularization           | Small datasets, testing            | Reliable, well-established optimizer        |
    | **SGD with Momentum**     | Traditional approach with momentum              | Learning optimization              | Requires more tuning, good for manual control |
    | **RMSprop**               | Adaptive learning rates, good for RNNs          | Specialized use cases              | Handles varying gradients, alternative to Adam |
    | **RAdam**                 | Rectified Adam with stable start                | Large batches, advanced setups     | Improved convergence, newer variant         |
    | **Adagrad**               | Adjusts learning rates for sparse data          | Tasks with rare features           | Slower convergence, handles infrequent updates well |
    | **Step Wise (StepLR)**    | Reduces learning rate in predefined steps       | Long training schedules            | Requires careful setup, effective for staged training |
    | **Noam (NoamLR)**         | Uses warmup and decay, designed for NLP         | Transformer architectures          | Good for gradual adaptation, specialized for certain tasks |

    For most users, AdamW provides the best combination of performance and stability. It includes good defaults
    that work well across a wide range of scenarios and includes proper weight decay regularization.
    """

    STEP2_EPOCHS = """
    An epoch represents one complete pass through your entire training dataset. Understanding how many epochs to
    use and how to monitor training progress is crucial for achieving good results.

    The number of epochs needed can vary significantly depending on your dataset size, model complexity, and
    desired output quality. Too few epochs might result in underfitting, while too many could lead to overfitting.

    #### Understanding Epochs:
    - **What Is An Epoch**
        * Complete dataset pass
        * Training milestone
        * Progress measurement
        * Learning cycle
        * Quality checkpoint
        * Resource consideration

    - **How Many to Use:**
        * Minimum: 4-5 epochs
        * Standard: 10-15 epochs
        * Maximum: Based on monitoring
        * Dataset dependent
        * Quality dependent
        * Resource dependent

    - **Monitoring Factors:**
        * Loss curves
        * Evaluation metrics
        * Generated audio quality
        * Training stability
        * Resource usage
        * Time constraints

    - **Warning Signs:**
        * Increasing eval loss
        * Degrading quality
        * Plateauing metrics
        * Unstable metrics
        * Resource exhaustion
        * Diminishing returns

    Training for the right number of epochs is crucial. Start with the standard range of 10-15 epochs and adjust
    based on your results. Pay attention to both quantitative metrics and qualitative audio output quality.
    """

    STEP2_AUDIO_LENGTH = """
    The maximum audio length setting controls how long individual audio segments can be during training. This
    setting has important implications for both memory usage and training effectiveness.

    This parameter needs to be balanced carefully - longer segments can capture more context but require more
    memory and processing power. Shorter segments are more memory-efficient but might miss important long-term
    patterns in speech.

    #### Configuration Impact:
    - **Purpose:**
        * Controls maximum clip length
        * Affects memory usage
        * Influences training stability
        * Determines context window
        * Impacts training speed
        * Memory management tool

    - **Setting Guidelines:**
        * Default: 11 seconds
        * Minimum: 2 seconds
        * Maximum: 20 seconds
        * System dependent
        * Quality dependent
        * Resource balanced

    - **Considerations:**
        * Longer = more VRAM needed
        * Shorter = more segments
        * Balance quality vs resources
        * Context preservation
        * Memory limitations
        * Training stability

    #### Optimization Tips:
    - Start with default (11s)
    - Reduce if encountering OOM
    - Increase if audio quality suffers
    - Monitor system resources
    - Check segment distribution
    - Balance with batch size

    The default setting of 11 seconds works well for most cases, providing a good balance between context
    preservation and resource usage. Adjust based on your specific needs and system capabilities.
    """

    STEP3_TESTING_OVERVIEW = """
    The testing phase allows you to assess your model's TTS performance by loading a fine-tuned XTTS model, selecting a speaker reference audio, and generating sample speech. This helps verify that your model closely matches the desired speaker's voice, tone, and inflection.

    - **Select a Speaker Reference**: Reference audios 6 seconds or longer are available for testing to ensure a meaningful TTS quality check. Change with `Min Audio Length` and click refresh.
    - **Testing Multiple References**: Use various speaker reference audios to identify which one provides the most accurate TTS results for your intended use.
    - **Exporting Your Model**: Once satisfied with your model's performance, you can proceed to export it. All reference audio files will be packaged with your model export, along with a text document explaining the available audio files, their durations, and the relevance of each for TTS generation.

    After testing, you'll be ready to export your model and prepare it for integration into your TTS applications.
    """
 
    STEP3_IMPORTANT = """
    Before beginning the testing phase, please review these important considerations to ensure a smooth experience.

    - **Refreshing Dropdowns**: If dropdown lists do not seem fully populated or correct, use the **Refresh Dropdowns** button. This rescans the finetuning folder path and repopulates the lists with updated reference files. You can also manually edit paths if necessary.
    - **Compatible Configurations**: You may use any `config.json` or `vocab.json` files as long as they match the model version used during training. The key file is the XTTS checkpoint, which is essential for model loading.
    - **Reference File Length Requirement**: Any audio clips under 6 seconds are excluded from the reference list by default, as shorter clips generally do not provide sufficient information for quality TTS generation. Change with `Min Audio Length` and click refresh.
    - **Unexpected Reference Counts**: If the number of reference files is lower than anticipated, it may be due to Whisper's initial segmentation step, which sometimes results in shorter or longer-than-ideal clips. If needed, try manually segmenting your audio files to ensure optimal segments for Whisper processing and training.
    """
    
    STEP3_INSTRUCTIONS = """
    Follow these instructions to test and validate your fine-tuned model effectively.

    1. **Refresh Dropdowns**: Begin by clicking **Refresh Dropdowns** to ensure all speaker reference WAV files are listed correctly in the dropdown. This step ensures that any new files added are included.
    2. **Load the Fine-tuned Model**: Load the XTTS model from your training. The dropdowns should automatically populate with available reference options.
    3. **Select a Speaker Reference Audio**: Choose from the available speaker references in the dropdown list.
    - **Experiment with Multiple References**: Test with different reference audios to identify which best captures your target speaker's characteristics.
    - **Reference Length Considerations**: Reference audios longer than 6 seconds are prioritized here, as they generally provide more reliable TTS quality.
    4. **Troubleshooting Long Reference Files**: Some references may be too lengthy, potentially causing issues in generating TTS output. If this occurs, try a different reference sample from the list.
    5. **Evaluate Results**: Listen to the generated TTS outputs. Consider both the voice quality and consistency with your intended speaker's tone, as well as clarity and naturalness of speech.

    Once satisfied with the TTS performance, you can move to the next stage for exporting your model and managing your training data.
    """
    
    STEP3_WHAT_IT_DOES = """
    The testing step serves as the final stage for validating your fine-tuned model's performance.

    - **Load Model and Configuration**: This step loads the fine-tuned XTTS model into memory, along with any compatible configuration files specified.
    - **Generate TTS Output**: Using the selected speaker reference audio, the model generates TTS to assess its quality and voice accuracy. This allows you to evaluate how well the model aligns with the desired voice characteristics and whether further refinement is needed.
    - **Evaluate Model Effectiveness**: Listen carefully to the TTS samples generated. This step is essential to confirm that the model is effectively fine-tuned and ready for deployment.

    The testing stage allows you to confirm that your model accurately matches the intended voice characteristics before final export.
    """

    EXPORT_OVERVIEW = """
    The model export process compresses your trained model, organizes your voice samples, and prepares everything for production use. This phase helps you:

    - Package your fine-tuned model efficiently
    - Organize voice samples by quality
    - Clean up training data
    - Manage disk space effectively

    Before proceeding with export, ensure you've completed testing and are satisfied with your model's performance.

    Your model will be moved to `/alltalk_tts/models/xtts/{your-chosen-folder-name}`
    """
    
    EXPORT_VOICE_SAMPLES = """
    During export, your voice samples are automatically analyzed and organized into three categories:

    #### Directory Structure
    - **suitable/**: Contains files between 6-30 seconds - ideal for voice cloning
    - **too_short/**: Files under 6 seconds - may lack sufficient voice characteristics
    - **too_long/**: Files over 30 seconds - may cause processing issues

    #### Audio Report
    The system generates an `audio_report.txt` file in your export folder `/alltalk_tts/models/xtts/{your-chosen-folder-name}/wavs/` containing:
    - Complete list of processed audio files
    - Duration of each sample
    - Quality categorization
    - Recommendations for best samples to use
    - Usage guidelines for each category

    #### Using Voice Samples
    To use your exported voice samples:
    1. Review the `audio_report.txt` file in your export folder
    2. Select preferred samples from the 'suitable' directory
    3. Manually copy chosen samples to the AllTalk 'voices' folder
    4. Use these samples for TTS generation in the main interface
    """

    EXPORT_OPTIONS = """
    The system provides several export and cleanup options:

    #### Compact & Move Model
    - Compresses the fine-tuned model for efficient storage
    - Organizes voice samples into quality-based directories
    - Creates detailed audio report
    - Moves everything to your specified folder in the models directory
    - Preserves all necessary files for future use

    #### Cleanup Options
    1. **Delete Generated Training Data**
    - Removes intermediate training files
    - Clears temporary data
    - Frees up significant disk space
    - Should be done after successful model export

    2. **Delete Original Voice Samples**
    - Removes files from 'put-voice-samples-in-here' folder
    - Optional cleanup step
    - Consider keeping originals for future training
    """
    
    EXPORT_STORAGE = """
    After export, you can free up additional disk space:

    #### Whisper Model Cleanup
    If you don't plan to train more models, you can remove the Whisper model from your cache (approximately 3GB):

    **Linux Users:**
    ```
    ~/.cache/huggingface/hub/(whisper-folder)
    ```

    **Windows Users:**
    ```
    C:\\Users\\(your-username)\\.cache\\huggingface\\hub\\(whisper-folder)
    ```

    #### Storage Tips
    - Keep exported models and selected voice samples
    - Remove training data after successful export
    - Consider archiving original audio files
    - Maintain backup of successful exports

    Remember to test your exported model and voice samples in the main AllTalk interface before removing any files.
    """
    
    report_content = """WAV Files Processing Report
    ===========================

    This folder contains WAV files categorized by their duration for use with Coqui TTS.
    Suitable files for voice cloning should be between 6 and 30 seconds in length.

    Directory Structure:
    ------------------
    - 'suitable': Contains files between 6-30 seconds - ideal for voice cloning
    - 'too_short': Files under 6 seconds - may not contain enough voice characteristics
    - 'too_long': Files over 30 seconds - may cause processing issues

    Voice Sample Usage:
    -----------------
    1. The 'suitable' directory contains the ideal voice samples:
      - These files are ready to use for voice cloning
      - Copy your preferred samples to '/alltalk_tts/voices/' for use in the main interface
      - Clean, clear samples with minimal background noise work best
      - Consider using multiple samples to test which gives the best results

    2. Files in 'too_long':
      - Can be used but may cause issues or inconsistent results
      - Recommended: Use audio editing software (like Audacity) to:
        * Split these into 6-30 second segments
        * Remove any silence, background noise, or unwanted sounds
        * Save segments as individual WAV files
        * Consider overlap in sentences for more natural breaks

    3. Files in 'too_short':
      - Not recommended for voice cloning as they lack sufficient voice characteristics
      - If most/all files are here, consider:
        * Recording longer samples of continuous speech
        * Combining multiple short segments (if they flow naturally)
        * Using audio editing software to create longer cohesive samples
        * Aim for clear, natural speech between 6-30 seconds

    Best Practices:
    -------------
    - Choose samples with clear, consistent speech
    - Avoid background noise, music, or other speakers
    - Natural speaking pace and tone usually work best
    - Multiple samples of varying lengths (within 6-30s) can provide better results
    - Test different samples to find which produces the best voice cloning results

    Summary:
    --------
    """
    
    report_content2 = """
    Notes:
    ------
    - Files in 'suitable' are ready for use with Coqui TTS
    - Files in 'too_short' are under 6 seconds and may need to be checked or excluded
    - Files in 'too_long' are over 30 seconds and may need to be split or excluded

    Please review the files in 'too_short' and 'too_long' directories."""    

    