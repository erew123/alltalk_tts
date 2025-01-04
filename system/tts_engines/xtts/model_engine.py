"""
XTTS (Coqui TTS) Engine Implementation for AllTalk TTS
Version: 1.0
Last Updated: 2024

This implementation serves as both a functional XTTS engine and a reference example
for implementing new TTS engines in AllTalk. It demonstrates proper:
- Variable initialization and management
- Debug message handling
- Model loading/unloading patterns
- Resource management
- Generation handling with streaming support

Implementation Notes:
- All core system variables must be maintained
- All functions prefixed with "DONT CHANGE" must remain unmodified
- Code additions should be placed between the marked sections in each function
- Debug messages use self.print_message() with appropriate message types
- All functions must implement self.debug_func_entry() for trace logging

Note: Text between "↑↑↑ Keep everything above this line ↑↑↑" and "↓↓↓ Keep everything below this line ↓↓↓"
markers **TYPICALLY** must remain unchanged as it contains critical system integration code.

Note: You can add new functions, just DONT remove the functions that are already there, even if they 
are doing nothing as `tts_server.py` will still look for their existance and fail if they are missing.
"""

########################################
# Default imports # Do not change this #
########################################
import os
import gc
import sys
import glob
import json
import time
import inspect
import torch
import logging
from pathlib import Path
from fastapi import (HTTPException)
logging.disable(logging.WARNING)

# Confguration file management for confignew.json 
try:
    from .config import AlltalkConfig, AlltalkTTSEnginesConfig, AlltalkNewEnginesConfig # TGWUI import
except ImportError:
    from config import AlltalkConfig, AlltalkTTSEnginesConfig, AlltalkNewEnginesConfig # Standalone import

def initialize_configs():
    """Initialize all configuration instances"""
    config = AlltalkConfig.get_instance()
    tts_engines_config = AlltalkTTSEnginesConfig.get_instance()
    new_engines_config = AlltalkNewEnginesConfig.get_instance()
    return config, tts_engines_config, new_engines_config

# Load in the central config management
config, tts_engines_config, new_engines_config = initialize_configs()

######################################################
# Get Pytorch & Python versions # Do not change this #
######################################################
pytorch_version = torch.__version__
cuda_version = torch.version.cuda
major, minor, micro = sys.version_info[:3]
python_version = f"{major}.{minor}.{micro}"

############################################
# DeepSpeed imports # POSSIBLY change this #
############################################
"""
If the new TTS engine you are importing doesnt support DeepSpeed, then
you can simply change `model_supports_deepspeed_true_or_false` to `False`
This will be much faster when starting up the engine. This is seperate
from what is stored in your `model_settings.json` file.
"""
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ↓↓↓ MODIFY THIS LINE ↓↓↓
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

model_supports_deepspeed_true_or_false = True

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# ↑↑↑ MODIFY THIS LINE ↑↑↑
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

if model_supports_deepspeed_true_or_false:
    try:
        import deepspeed
        deepspeed_available = True
    except ImportError:
        deepspeed_available = False
        pass

#######################################################
# TTS Engine-Specific Imports and Setup # Change this #
#######################################################
"""
This section is for TTS engine specific imports and global variable setup.

Guidelines:
1. Import all required modules for your TTS engine
2. Handle import errors with appropriate error messages
3. Set up any global variables or configurations needed by your engine
4. Use try/except blocks to gracefully handle missing dependencies
5. Ensure all error messages follow the AllTalk format for print messages.
   At this stage the `def print_messages` is not available, so please use
   standard `print("[AllTalk ENG] some message here")` messages.

Example structure:

try:
    # Import your TTS engine's required modules
    from your_tts_engine import required_modules
except ModuleNotFoundError:
    # Handle missing dependencies with helpful error messages
    print("Missing required modules. Please install...")
    raise

For XTTS specifically, we need:
- torchaudio: For audio processing
- TTS modules: Core Coqui TTS functionality
- Support libraries: wave, io, random, numpy for audio handling
"""
try:
    import torchaudio
    import wave
    import io
    import random
    import numpy as np
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    brand = "[AllTalk ENG]"
    print(f"{brand} \033[91mError\033[0m Could not find the Coqui TTS modules.")
    print(f"{brand} \033[91mError\033[0m Please re-install the requirements.")
    raise


#########################################################
# Class setup # Change the relevant functions as needed #
#########################################################
class tts_class:
    """
    TTS Engine Implementation Class
    
    This class provides the interface between `tts_server.py` and whatever TTS engine you install.
    It handles model loading, voice management, and TTS generation in both streaming
    and non-streaming modes. Streaming will only be supported if the underlying TTS engine
    actually supports streaming.

    Key Responsibilities:
    1. Model Management:
       - Loading/unloading models
       - Managing model state between CPU and GPU
       - Handling DeepSpeed integration

    2. Voice Management:
       - Managing voice samples or model files

    3. TTS Generation:
       - Converting text to speech
       - Supporting streaming output (If the engine supports it)
       - Managing generation parameters

    4. System Integration:
       - Implementing standard AllTalk interfaces
       - Managing engine state and configuration
       - Handling resource allocation
    """

    ###############################################
    # Central print function # Do not change this #
    ###############################################
    def print_message(self, message, message_type="standard", component="ENG"):
        """
        Centralized print function for messages. Use this for print output to console.
        As this is the model Engine, all `component` printouts are set to ENG as default.
        
        Args:
            message (str): The message to print
            message_type (str): Type of message (standard/warning/error/debug_*/debug)
            component (str): Component identifier (TTS/ENG/GEN/API/etc.)

        Example Use:
            self.print_message("This is a standard print out mesage to a user)
            self.print_message("This is a debug_tts message, message_type="debug_tts")
            self.print_message("This is an error message to a user, message_type="error")
            self.print_message("This is an warning message to a user, message_type="warning")

        Debug Types:
            debug_func: Tracks function entry
            debug_tts: Enable TTS process debugging
            debug_tts_variables: Enable variable state debugging

        WARNING: This is a core system function. Do not modify its implementation
        as it provides standardized version reporting across all engines.
        """
        # ANSI color codes
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        YELLOW = "\033[93m"  
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m" 
        prefix = f"[{config.branding}{component}] "
        if message_type.startswith("debug_"):
            debug_flag = getattr(config.debugging, message_type, False)
            if not debug_flag:
                return
            if message_type == "debug_func" and "Function entry:" in message:
                message_parts = message.split("Function entry:", 1)
                print(f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} Function entry:{GREEN}{message_parts[1]}{RESET} in model_engine")
            else:
                print(f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} {message}")
        elif message_type == "debug":
            print(f"{prefix}{BLUE}Debug{RESET} {message}")  
        elif message_type == "warning":
            print(f"{prefix}{YELLOW}Warning{RESET} {message}")  
        elif message_type == "error":
            print(f"{prefix}{RED}Error{RESET} {message}")  
        else:
            print(f"{prefix}{message}")

    def debug_func_entry(self):
        """Log function entry if debug_func is enabled."""
        if config.debugging.debug_func:
            current_func = inspect.currentframe().f_back.f_code.co_name
            self.print_message(f"Function entry: {current_func}", "debug_func")

    #############################################
    # Script initalisation # Do not change this #
    #############################################
    def __init__(self):
        """
        Initialize the XTTS engine instance.
        
        WARNING: This class requires specific variables to interface with AllTalk's main system (tts_server.py).
        Do not remove or rename any of the predefined variables as they are required for proper system integration.
        
        Required System Interface Variables:

        1. Core System Variables (DO NOT MODIFY):
           Base Configuration:
           - self.this_dir: Engine directory path (where this script is located)
           - self.main_dir: AllTalk root directory
           - self.device: Processing device ("cuda" or "cpu")
           - self.cuda_is_available: Whether GPU/CUDA is available
           
           State Tracking:
           - self.tts_generating_lock: Prevents concurrent generation requests 
           - self.tts_stop_generation: Signals generation stop request
           - self.tts_narrator_generatingtts: Tracks narrator mode for optimization
           - self.model: Active TTS model instance
           - self.is_tts_model_loaded: Whether a model is currently loaded
           - self.current_model_loaded: Name of currently loaded model
           - self.available_models: List of models found by scan_models_folder
           - self.setup_has_run: Tracks if setup() has completed
        
        2. Engine Configuration Variables (DO NOT MODIFY):
           - self.engines_available: List of all available TTS engines
           - self.engine_loaded: Currently selected TTS engine
           - self.selected_model: Currently selected model name
        
        3. Model Settings (SET VIA model_settings.json):
           Capability Flags:
           - self.audio_format: Output audio format (wav, mp3, etc.)
           - self.deepspeed_capable: DeepSpeed acceleration support
           - self.generationspeed_capable: Speed adjustment support
           - self.languages_capable: Multi-language support
           - self.lowvram_capable: Low VRAM mode support
           - self.multimodel_capable: Multiple model support
           - self.repetitionpenalty_capable: Repetition penalty support
           - self.streaming_capable: Audio streaming support
           - self.temperature_capable: Temperature adjustment support
           - self.multivoice_capable: Multiple voice support
           - self.pitch_capable: Pitch adjustment support
           
           Engine Settings:
           - self.def_character_voice: Default character voice
           - self.def_narrator_voice: Default narrator voice
           - self.deepspeed_enabled: DeepSpeed status
           - self.engine_installed: Engine installation status
           - self.generationspeed_set: Current speed setting
           - self.lowvram_enabled: Low VRAM mode status
           - self.repetitionpenalty_set: Current repetition penalty
           - self.temperature_set: Current temperature setting
           - self.pitch_set: Current pitch setting
           
           OpenAI Voice Mappings:
           - self.openai_alloy: Alloy voice mapping
           - self.openai_echo: Echo voice mapping
           - self.openai_fable: Fable voice mapping
           - self.openai_nova: Nova voice mapping
           - self.openai_onyx: Onyx voice mapping
           - self.openai_shimmer: Shimmer voice mapping
        
        Integration Requirements:
        - All variables must be present even if unused by your engine
        - Capability flags should accurately reflect engine features
        - Settings should have sensible defaults even if not used
        - OpenAI mappings should be set even if not supporting OpenAI compatibility
        
        Note: Variables marked (DO NOT MODIFY) are critical system integration points.
        Other variables should be configured through their respective JSON files or
        you can add new central variables in the section provided down below.
        """
        # DO NOT MODIFY - Sets up the base variables required for any tts engine #
        self.this_dir = Path(__file__).parent.resolve()
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_is_available = torch.cuda.is_available()
        self.tts_generating_lock = False
        self.tts_stop_generation = False
        self.tts_narrator_generatingtts = False
        self.model = None
        self.is_tts_model_loaded = False
        self.current_model_loaded = None
        self.available_models = None
        self.setup_has_run = False
        self.engines_available = tts_engines_config.get_engine_names_available()
        self.engine_loaded = tts_engines_config.engine_loaded
        self.selected_model = tts_engines_config.selected_model

        # DO NOT MODIFY - Load in the current TTS Engines model_settings.json file
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            model_settings_file = json.load(f)
    
        # DO NOT MODIFY - Model details from model_settings.json
        self.manufacturer_name = model_settings_file["model_details"]["manufacturer_name"]
        self.manufacturer_website = model_settings_file["model_details"]["manufacturer_website"]
        
        # DO NOT MODIFY - Model capabilities from model_settings.json
        self.audio_format = model_settings_file["model_capabilties"]["audio_format"]
        self.deepspeed_capable = model_settings_file["model_capabilties"]["deepspeed_capable"]
        self.deepspeed_available = 'deepspeed' in globals()
        self.generationspeed_capable = model_settings_file["model_capabilties"]["generationspeed_capable"]
        self.languages_capable = model_settings_file["model_capabilties"]["languages_capable"]
        self.lowvram_capable = model_settings_file["model_capabilties"]["lowvram_capable"]
        self.multimodel_capable = model_settings_file["model_capabilties"]["multimodel_capable"]
        self.repetitionpenalty_capable = model_settings_file["model_capabilties"]["repetitionpenalty_capable"]
        self.streaming_capable = model_settings_file["model_capabilties"]["streaming_capable"]
        self.temperature_capable = model_settings_file["model_capabilties"]["temperature_capable"]
        self.multivoice_capable = model_settings_file["model_capabilties"]["multivoice_capable"]
        self.pitch_capable = model_settings_file["model_capabilties"]["pitch_capable"]
        
        # DO NOT MODIFY - Engine settings from model_settings.json
        self.def_character_voice = model_settings_file["settings"]["def_character_voice"]
        self.def_narrator_voice = model_settings_file["settings"]["def_narrator_voice"]
        self.deepspeed_enabled = model_settings_file["settings"]["deepspeed_enabled"]
        self.streaming_enabled = model_settings_file["settings"]["streaming_enabled"]
        self.engine_installed = model_settings_file["settings"]["engine_installed"]
        self.generationspeed_set = model_settings_file["settings"]["generationspeed_set"]
        self.lowvram_enabled = model_settings_file["settings"]["lowvram_enabled"]
        self.lowvram_enabled = False if not torch.cuda.is_available() else self.lowvram_enabled
        self.repetitionpenalty_set = model_settings_file["settings"]["repetitionpenalty_set"]
        self.temperature_set = model_settings_file["settings"]["temperature_set"]
        self.pitch_set = model_settings_file["settings"]["pitch_set"]
        
        # DO NOT MODIFY - OpenAI voice mappings from model_settings.json
        self.openai_alloy = model_settings_file["openai_voices"]["alloy"]
        self.openai_echo = model_settings_file["openai_voices"]["echo"]
        self.openai_fable = model_settings_file["openai_voices"]["fable"]
        self.openai_nova = model_settings_file["openai_voices"]["nova"]
        self.openai_onyx = model_settings_file["openai_voices"]["onyx"]
        self.openai_shimmer = model_settings_file["openai_voices"]["shimmer"]        

        """
        Below is the name of the folder that will be created-used under `/models/{folder}`
        And is further used by the base functions of this script. Change the name stored in
        `self.model_folder_name` to the model folder name you will be using. Ensure to use
        the correct CAPS/Non-CAPS spelling as Linux OS is CAPS specific on folder names.
        """
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ MODIFY THIS LINE ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        self.model_folder_name = "xtts"
         
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ MODIFY THIS LINE ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Add your own central `self.myvariable` variables in here if needed for your engine ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        """
        If you need globally accessable variables of your own for your own purposes, you can put
        them in here as self.myvariable = "whatever".

        """
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Add your own central `self.myvariable` variables in here if needed for your engine ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # DO NOT MODIFY - log the function call to this function
        self.debug_func_entry() 

    #####################################################
    # Printout engine loading bits # Do not change this #
    #####################################################
    def printout_versions(self):
        """
        Print Python, DeepSpeed, Pytorch and CUDA version on start-up.
        
        WARNING: This is a core system function. Do not modify its implementation
        as it provides standardized version reporting across all engines.
        """
        self.debug_func_entry()
        if not model_supports_deepspeed_true_or_false:
            self.print_message(f"\033[92mDeepSpeed version :\033[93m Not supported on {self.model_folder_name}\033[0m", message_type="standard")
        else:
            if deepspeed_available:
                self.print_message("\033[92mDeepSpeed version :\033[93m " + str(deepspeed.__version__) + "\033[0m", message_type="standard")
            else:
                self.print_message("\033[92mDeepSpeed version :\033[93m Not available\033[0m", message_type="standard")
        self.print_message(f"\033[92mPython Version    :\033[93m {python_version}\033[0m", message_type="standard")
        self.print_message(f"\033[92mPyTorch Version   :\033[93m {pytorch_version}\033[0m", message_type="standard")
        if cuda_version is None:
            self.print_message("\033[92mCUDA Version      :\033[91m Not available\033[0m", message_type="standard")
        else:
            self.print_message(f"\033[92mCUDA Version      :\033[93m {cuda_version}\033[0m", message_type="standard")
            
        self.print_message("", message_type="standard")
        return

    ################################################################
    # Handle low VRAM change between CUDA/CPU # Do not change this #
    ################################################################
    async def handle_lowvram_change(self):
        """
        Manage model location between CPU and GPU memory for low VRAM operation.
        
        This function handles the movement of models between CPU and GPU memory
        to support systems with limited VRAM. It's called automatically during
        generation when low VRAM mode is enabled.
        
        Operation:
        1. Checks CUDA availability
        2. Moves model between devices based on current location:
           - GPU (cuda) -> CPU
           - CPU -> GPU (cuda)
        3. Manages CUDA cache to optimize memory usage
        
        States Affected:
        - self.device: Updated to reflect current processing device
        - self.model.device: Model's current memory location
        
        Requirements:
        - CUDA must be available for GPU operations
        - Model must be loaded (self.model is not None)
        - lowvram_enabled must be `True` in the `model_settings.json` file
        
        Note: This function is only called when self.lowvram_enabled is True
        meaning the engine does or doesnt support the call, hence if its not
        True, then this function would never be called anyway, so doesnt need
        changing.
        """
        self.debug_func_entry()
        
        # Initial validation
        if not self.is_tts_model_loaded:
            self.print_message("No model is currently loaded. Please select a model to load.", message_type="error")
            raise HTTPException(status_code=400, detail="No model is currently loaded. Please select a model to load.")
                       
        if torch.cuda.is_available():
            if self.device == "cuda":
                self.print_message("Moving model to CPU", message_type="debug_tts")
                self.device = "cpu"
                self.model.to(self.device)
                torch.cuda.empty_cache()
                gc.collect()
            else:
                self.device = "cuda"
                self.print_message("Moving model to GPU", message_type="debug_tts")
                self.model.to(self.device)
                gc.collect()

    ################################################
    # Handle DeepSpeed change # Do not change this #
    ################################################
    async def handle_deepspeed_change(self, value):
        """
        Handle enabling/disabling of DeepSpeed acceleration.
        
        This function manages the process of reloading the model with or without 
        DeepSpeed acceleration. DeepSpeed can significantly improve performance on 
        supported hardware.
        
        Args:
            value (bool): True to enable DeepSpeed, False to disable
            
        Operation:
        1. Unloads current model
        2. Updates DeepSpeed settings
        3. Reloads model with new configuration
        
        States Affected:
        - self.deepspeed_enabled: Updated to reflect new state
        - self.model: Reloaded with new configuration
        
        Returns:
            bool: The new DeepSpeed state (same as input value)
        
        Note: DeepSpeed must be installed and available in the system for 
        this functionality to work. `deepspeed_capable` must be set `True`
        in the `model_settings.json` file
        """
        self.debug_func_entry()
        # Initial validation        
        if self.current_model_loaded.startswith("apitts"): # Specific only to the XTTS engine
            self.print_message("\033[93mDeepSpeed not supported in API mode\033[0m", message_type="error")
            self.deepspeed_enabled = False
            return False

        if not self.is_tts_model_loaded:
            self.print_message("No model is currently loaded. Please select a model to load.", message_type="error")
            raise HTTPException(status_code=400, detail="No model is currently loaded. Please select a model to load.")
        
        if value:
            self.print_message("\033[93mDeepSpeed Activating\033[0m", message_type="standard")
            await self.unload_model()
            self.deepspeed_enabled = True
            await self.setup()
        else:
            self.print_message("\033[93mDeepSpeed De-Activating\033[0m", message_type="standard")
            self.deepspeed_enabled = False
            await self.unload_model()
            await self.setup()
        return value


    ################################################################
    # Unload models from VRAM/RAM if possible # Do not change this #
    ################################################################
    async def unload_model(self):
        """
        Unload the current model and free associated resources.
        
        This function handles the cleanup of model resources, including:
        1. Setting model loaded flag to False
        2. Deleting the model instance
        3. Clearing CUDA cache if available
        
        Operation:
        1. Updates model loading status
        2. Logs unloading process if a model is loaded
        3. Removes model from memory
        4. Cleans up CUDA cache if using GPU
        
        States Affected:
        - self.is_tts_model_loaded: Set to False
        - self.model: Set to None after unloading
        """
        self.debug_func_entry()
        
        self.is_tts_model_loaded = False
        if not self.current_model_loaded == None:
            self.print_message("Unloading model", message_type="debug_tts")
        if hasattr(self, 'model'):
            del self.model            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    ############################################################
    # On start-up, perform these actions # Change as necessary #
    ############################################################
    async def setup(self):
        """
        Initialize the XTTS engine and load initial model configuration.
        
        This function is called during system startup and handles:
        1. Version information display
        2. Model scanning and availability check
        3. Initial model loading if specified
        
        The setup sequence ensures:
        - Proper version reporting
        - Model availability verification
        - Graceful handling of missing models
        - Correct initial model loading state
        
        States Set:
        - self.available_models: Updated with found models
        - self.current_model_loaded: Set to loaded model name or None
        - self.setup_has_run: Set True when complete
        
        Returns:
            None
            
        Note: Custom initialization code should be placed between the marked sections.
        """
        self.debug_func_entry()
        self.print_message("Initializing XTTS engine", message_type="debug_tts")
        self.printout_versions()
        self.available_models = self.scan_models_folder()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        if self.selected_model:
            tts_model = f"{self.selected_model}"
            if tts_model in self.available_models:
                self.print_message(f"Loading selected model: {tts_model}", message_type="debug_tts")
                await self.handle_tts_method_change(tts_model)
                self.current_model_loaded = tts_model
                
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓                  
            else:
                self.current_model_loaded = "No Models Available"
                self.print_message(f"Selected model '{self.selected_model}' not found in models folder.", message_type="error")
                self.print_message(f"Please download a model or select a different model file.", message_type="error")
        self.setup_has_run = True


    ###########################################################################
    # Scan your models folder for models OR voice files # Change as necessary #
    ###########################################################################
    def scan_models_folder(self):
        """
        Scan for available XTTS models in the models directory.
        
        This function searches the models directory for valid XTTS model installations.
        Each model must contain all required files to be considered valid.
        
        Required Files for Each Model:
        - config.json: Model configuration
        - model.pth: Model weights
        - mel_stats.pth: Mel spectrogram statistics
        - speakers_xtts.pth: Speaker embeddings
        - vocab.json: Tokenizer vocabulary
        - dvae.pth: Discrete VAE weights
        
        Operation:
        1. Scans the models/xtts directory
        2. Checks each subfolder for required files
        3. Registers valid models in two formats:
           - "xtts - {model_name}": For local inference
           - "apitts - {model_name}": For API-based inference
        
        States Affected:
        - self.available_models: Updated with found models
        
        Returns:
            dict: Dictionary of available models in format:
                 {model_identifier: engine_type}
        
        Note: If no valid models are found, returns {"No Models Available": "xtts"}
        """
        self.debug_func_entry()
        
        models_folder = self.main_dir / "models" / self.model_folder_name
        self.available_models = {}
        
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑  
            
        required_files = ["config.json", "model.pth", "mel_stats.pth", 
                         "speakers_xtts.pth", "vocab.json", "dvae.pth"]
        
        for subfolder in models_folder.iterdir():
            if subfolder.is_dir():
                model_name = subfolder.name
                self.print_message(f"Checking model folder: {model_name}", message_type="debug_tts")
                
                if all(subfolder.joinpath(file).exists() for file in required_files):
                    self.print_message(f"Found valid model: {model_name}", message_type="debug_tts")
                    self.available_models[f"xtts - {model_name}"] = "xtts"
                    self.available_models[f"apitts - {model_name}"] = "apitts"
                    
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  
                else:
                    self.available_models = {'No Models Available': self.model_folder_name}
                    self.print_message(f"Model folder '{model_name}' is missing required files", message_type="warning")
                    self.print_message("Required files or folder does not exist", message_type="warning")
                    self.print_message("Please download some models/voices for this engine", message_type="warning")                
        return self.available_models

    ################################################################
    # Scan your voice folder for voice files # Change as necessary #
    ################################################################
    def voices_file_list(self):
        """
        Scan and compile a list of available voice samples and latents.
        
        This function scans multiple directories to find voice samples in different formats:
        1. Individual WAV files in the main voices directory
        2. Collections of WAV files in the xtts_multi_voice_sets directory
        3. Pre-computed voice latents in the xtts_latents directory
        
        Directory Structure:
        - voices/: Individual WAV files
        - voices/xtts_multi_voice_sets/: Folders containing multiple WAV files
        - voices/xtts_latents/: JSON files containing pre-computed latents
        
        Voice Types:
        - Standard: Direct WAV files
        - voiceset: Multiple WAV files for one voice
        - latent: Pre-computed speaker embeddings
        
        Returns:
            list: Available voices with appropriate prefixes:
                 - Standard WAV: filename.wav
                 - Voice sets: "voiceset:foldername"
                 - Latents: "latent:filename.json"
        
        Note: Returns ["No Voices Found"] if no valid voices are detected
        """
        self.debug_func_entry()
        
        try:
            voices = [] # An empy variable for the list of voices to be put into.
            directory = self.main_dir / "voices" # Base directory that voices are stored in.
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            # ↑↑↑ Keep everything above this line ↑↑↑
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 

            json_latents_dir = directory / "xtts_latents" # XTTS specific
            multi_voice_dir = directory / "xtts_multi_voice_sets" # XTTS specific

            # Scan for individual WAV files
            self.print_message("Scanning for individual voice files", message_type="debug_tts")
            voices.extend([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) 
                        and f.endswith(".wav")])
            
            # Scan for voice sets
            if os.path.exists(multi_voice_dir):
                self.print_message("Found Multi_Voice_Sets directory", message_type="debug_tts")
                for voice_set in os.listdir(multi_voice_dir):
                    voice_set_path = multi_voice_dir / voice_set
                    if os.path.isdir(voice_set_path):
                        if any(f.endswith(".wav") for f in os.listdir(voice_set_path)):
                            voices.append(f"voiceset:{voice_set}")
                            self.print_message(f"Added voice set: {voice_set}", message_type="debug_tts_variables")
                
            # Scan for JSON latents
            if not self.current_model_loaded.startswith("apitts"): # APITTS doesnt support latents
                if os.path.exists(json_latents_dir):
                    self.print_message("Found JSON_Latents directory", message_type="debug_tts")
                    json_files = [f for f in os.listdir(json_latents_dir) if f.endswith('.json')]
                    for json_file in json_files:
                        voices.append(f"latent:{json_file}")
                    if json_files:
                        self.print_message(f"Added {len(json_files)} JSON latent files", message_type="debug_tts")
            
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  
            # Sort voices by type alphabetically
            voices.sort(key=lambda x: (x.startswith("voiceset:"), x.startswith("latent:"), x))
            if not voices:
                return ["No Voices Found"]
            return voices
            
        except Exception as e:
            self.print_message(f"Error scanning for voices: {str(e)}", message_type="error")
            return ["No Voices Found"]
        
    ############################################
    # Load in your model # Change as necessary #
    ############################################
    async def load_model(self, model_name):
        """
        Load a model using the Coqui TTS API interface.
        
        This is one of two model loading methods for XTTS. This method uses the high-level
        TTS API which provides a simpler interface but less control over model parameters.
        
        Args:
            model_name (str): Name of the model to load
            
        Operation:
        1. Validates model availability
        2. Constructs model and config paths
        3. Initializes model using TTS API
        4. Moves model to appropriate device (CPU/GPU)
        
        States Affected:
        - self.model: Updated with loaded model
        - self.is_tts_model_loaded: Set to True on success
        
        Returns:
            The loaded model instance
            
        Raises:
            HTTPException: If no models are available to load
        """
        self.debug_func_entry()
        if "No Models Available" in self.available_models:
            self.print_message("No models for this TTS engine were found to load", message_type="error")
            return
        model_path = self.main_dir / "models" / self.model_folder_name / model_name # You may need to edit/modfy depending on how your TTS engine works with models
        self.print_message(f"Loading model from: {model_path}", message_type="debug_tts")
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑        
        
        self.model = TTS(
            model_path=model_path,
            config_path=model_path / "config.json",
        ).to(self.device)
        
        self.print_message("\033[94mModel License : \033[93mhttps://coqui.ai/cpml.txt\033[0m")
        
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.is_tts_model_loaded = True
        return self.model

    ###################################################################################################################
    # Load in your model # XTTS specific as XTTS has 2x loader types so you wouldnt normally have 2x loader functions #
    ###################################################################################################################
    async def xtts_manual_load_model(self, model_name):
        """
        Load a model using the direct XTTS interface.
        
        This is the second model loading method for XTTS. It provides direct access
        to the XTTS model interface, allowing for more detailed control over model
        parameters and DeepSpeed integration.
        
        Args:
            model_name (str): Name of the model to load
            
        Operation:
        1. Validates model availability
        2. Initializes XttsConfig with model settings
        3. Loads model with appropriate configurations
        4. Handles DeepSpeed integration if enabled
        5. Moves model to appropriate device
        
        States Affected:
        - self.model: Updated with loaded model
        - self.is_tts_model_loaded: Set to True on success
        
        Returns:
            The loaded model instance
            
        Raises:
            HTTPException: If no models are available to load
        """
        self.debug_func_entry()
        
        if "No Models Available" in self.available_models:
            self.print_message("No models for this TTS engine were found to load", message_type="error")
            return

        # Enhanced debugging for paths and configuration
        config = XttsConfig()
        model_path = self.main_dir / "models" / self.model_folder_name / model_name
        config_path = model_path / "config.json"
        vocab_path_dir = model_path / "vocab.json"
        checkpoint_dir = model_path
        
        self.print_message(f"Model path: {model_path}", message_type="debug_tts_variables")
        self.print_message(f"Config path: {config_path}", message_type="debug_tts_variables")
        self.print_message(f"Vocab path: {vocab_path_dir}", message_type="debug_tts_variables")
        
        # Load and debug configuration
        config.load_json(str(config_path))
        self.print_message("Model configuration:", message_type="debug_tts_variables")
        self.print_message(f"├─ Model dimension: {config.model_args.gpt_n_model_channels}", message_type="debug_tts_variables")
        self.print_message(f"├─ Number of layers: {config.model_args.gpt_layers}", message_type="debug_tts_variables")
        self.print_message(f"├─ Number of heads: {config.model_args.gpt_n_heads}", message_type="debug_tts_variables")
        self.print_message(f"├─ Max audio tokens: {config.model_args.gpt_max_audio_tokens}", message_type="debug_tts_variables")
        self.print_message(f"├─ Max text tokens: {config.model_args.gpt_max_text_tokens}", message_type="debug_tts_variables")
        self.print_message(f"└─ Using DeepSpeed: {self.deepspeed_enabled}", message_type="debug_tts_variables")

        # Initialize model
        self.print_message("Initializing model from config...", message_type="debug_tts")
        self.model = Xtts.init_from_config(config)
        
        # Load checkpoint with detailed progress
        self.print_message("Loading model checkpoint...", message_type="debug_tts")
        self.model.load_checkpoint(
            config,
            checkpoint_dir=str(checkpoint_dir),
            vocab_path=str(vocab_path_dir),
            use_deepspeed=self.deepspeed_enabled,
        )
        
        # Device management debugging
        self.print_message(f"Moving model to device: {self.device}", message_type="debug_tts")
        self.model.to(self.device)
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            self.print_message("CUDA Memory Status:", message_type="debug_tts_variables")
            self.print_message(f"├─ Allocated: {memory_allocated:.2f} MB", message_type="debug_tts_variables")
            self.print_message(f"└─ Reserved: {memory_reserved:.2f} MB", message_type="debug_tts_variables")
        
        self.is_tts_model_loaded = True
        self.print_message("\033[94mModel License : \033[93mhttps://coqui.ai/cpml.txt\033[0m")
        
        return self.model


    async def handle_tts_method_change(self, tts_method):
        """
        Handle switching between different XTTS model types and loading methods.
        
        This function manages the process of changing between different model loading
        methods (XTTS local vs API) and handles the actual model loading process.
        
        Args:
            tts_method (str): Format "type - modelname" where type is either
                            "xtts" or "apitts"
        
        Operation:
        1. Validates model availability
        2. Unloads current model if any
        3. Parses method string to determine loader type
        4. Calls appropriate model loader
        5. Updates current model tracking
        
        States Affected:
        - self.current_model_loaded: Updated to new model identifier
        - self.model: Updated with newly loaded model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        
        Timing:
            Records and reports model loading time
        """
        self.debug_func_entry()
        
        # Track loading time
        generate_start_time = time.time()
        
        # Validate model availability
        if "No Models Available" in self.available_models:
            self.print_message("No models for this TTS engine were found to load", message_type="error")
            return False

        # Unload current model
        await self.unload_model()
        
        # Handle different loading methods
        if tts_method.startswith("xtts"):
            model_name = tts_method.split(" - ")[1]
            self.print_message(f"\033[94mLoading XTTS model \033[93m{model_name} \033[94mon \033[93m{self.device}\033[0m")
            self.model = await self.xtts_manual_load_model(model_name)
            self.current_model_loaded = f"xtts - {model_name}"
            
        elif tts_method.startswith("apitts"):
            model_name = tts_method.split(" - ")[1]
            self.print_message(f"\033[94mLoading API model \033[93m{model_name} \033[94mon \033[93m{self.device}\033[0m")
            self.model = await self.load_model(model_name)
            self.current_model_loaded = f"apitts - {model_name}"
            
        else:
            self.print_message(f"Unknown model type in: {tts_method}", message_type="error")
            self.current_model_loaded = None
            return False

        # Report loading time
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        self.print_message(f"\033[94mModel Loadtime: \033[93m{generate_elapsed_time:.2f}\033[94m seconds\033[0m")
        return True

    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        """
        Generate speech from text using the XTTS model.

        This core function handles all TTS generation, supporting both streaming and
        non-streaming output, multiple voice input types (WAV, voice sets, latents),
        and various generation parameters.

        Args:
            text (str): Text to convert to speech
            voice (str): Voice identifier (WAV file, voice set, or latent)
            language (str): Target language code
            temperature (float): Generation temperature (0.0-1.0)
            repetition_penalty (float): Penalty for repetitive generation
            speed (float): Speech speed multiplier
            pitch (float): Voice pitch adjustment (not used in XTTS)
            output_file (str): Path for output audio file
            streaming (bool): Whether to stream audio chunks

        Returns:
            For streaming=True: Generator yielding audio chunks
            For streaming=False: None (saves to output_file)

        States Used:
            - self.model: Active TTS model
            - self.device: Current processing device
            - self.lowvram_enabled: Low VRAM mode status
            - self.current_model_loaded: Current model type
        """        
        self.debug_func_entry()
        
        # Initial validation
        if not self.is_tts_model_loaded:
            self.print_message("No TTS model loaded", message_type="error")
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")
        
        # Lock generation and track start time
        self.tts_generating_lock = True
        self.print_message("Starting TTS generation process", message_type="debug_tts")
        self.print_message(f"Generation parameters: temperature={temperature}, speed={speed}, streaming={streaming}", 
                        message_type="debug_tts_variables")
        
        # Handle low VRAM mode if needed
        if self.lowvram_enabled and self.device == "cpu":
            self.print_message("Low VRAM mode: Moving model to GPU", message_type="debug_tts")
            await self.handle_lowvram_change()
        
        generate_start_time = time.time()

        try:
            # Voice input processing
            self.print_message(f"Processing voice input: {voice}", message_type="debug_tts")
            gpt_cond_latent = None
            speaker_embedding = None
            
            # Handle different voice types
            if voice.startswith('latent:'):
                if self.current_model_loaded.startswith("xtts"):
                    gpt_cond_latent, speaker_embedding = self._load_latents(voice)
                
            elif voice.startswith('voiceset:'):
                voice_set = voice.replace("voiceset:", "")
                voice_set_path = os.path.join(self.main_dir, "voices", "xtts_multi_voice_sets", voice_set)
                self.print_message(f"Processing voice set from: {voice_set_path}", message_type="debug_tts")
                
                wavs_files = glob.glob(os.path.join(voice_set_path, "*.wav"))
                if not wavs_files:
                    self.print_message(f"No WAV files found in voice set: {voice_set}", message_type="error")
                    raise HTTPException(status_code=400, detail=f"No WAV files found in voice set: {voice_set}")
                
                if len(wavs_files) > 5:
                    wavs_files = random.sample(wavs_files, 5)
                    self.print_message(f"Using 5 random samples from voice set", message_type="debug_tts")
                
                if self.current_model_loaded.startswith("xtts"):
                    self.print_message("Generating conditioning latents from voice set", message_type="debug_tts")
                    gpt_cond_latent, speaker_embedding = self._generate_conditioning_latents(wavs_files)
                
            else:
                normalized_path = os.path.normpath(os.path.join(self.main_dir, "voices", voice))
                wavs_files = [normalized_path]
                self.print_message(f"Using single voice sample: {normalized_path}", message_type="debug_tts")
                
                if self.current_model_loaded.startswith("xtts"):
                    self.print_message("Generating conditioning latents from single sample", message_type="debug_tts")
                    gpt_cond_latent, speaker_embedding = self._generate_conditioning_latents(wavs_files)

            # Generate speech
            if self.current_model_loaded.startswith("xtts"):
                self.print_message(f"Generating speech for text: {text}", message_type="debug_tts")
                
                common_args = {
                    "text": text,
                    "language": language,
                    "gpt_cond_latent": gpt_cond_latent,
                    "speaker_embedding": speaker_embedding,
                    "temperature": float(temperature),
                    "length_penalty": float(self.model.config.length_penalty),
                    "repetition_penalty": float(repetition_penalty),
                    "top_k": int(self.model.config.top_k),
                    "top_p": float(self.model.config.top_p),
                    "speed": float(speed),
                    "enable_text_splitting": True
                }
                
                self.print_message("Generation settings:", message_type="debug_tts_variables")
                self.print_message(f"├─ Temperature: {temperature}", message_type="debug_tts_variables")
                self.print_message(f"├─ Speed: {speed}", message_type="debug_tts_variables")
                self.print_message(f"├─ Language: {language}", message_type="debug_tts_variables")
                self.print_message(f"└─ Text length: {len(text)} characters", message_type="debug_tts_variables")

                # Handle streaming vs non-streaming
                if streaming:
                    self.print_message("Starting streaming generation", message_type="debug_tts")
                    self.print_message(f"Using streaming-based generation and files {wavs_files}")
                    output = self.model.inference_stream(**common_args, stream_chunk_size=20)

                    file_chunks = []
                    wav_buf = io.BytesIO()
                    with wave.open(wav_buf, "wb") as vfout:
                        vfout.setnchannels(1)
                        vfout.setsampwidth(2)
                        vfout.setframerate(24000)
                        vfout.writeframes(b"")
                    wav_buf.seek(0)
                    yield wav_buf.read()

                    for i, chunk in enumerate(output):
                        if self.tts_stop_generation:
                            self.print_message("Generation stopped by user", message_type="debug_tts")
                            self.tts_stop_generation = False
                            self.tts_generating_lock = False
                            break

                        self.print_message(f"Processing chunk {i+1}", message_type="debug_tts")
                        file_chunks.append(chunk)
                        if isinstance(chunk, list):
                            chunk = torch.cat(chunk, dim=0)
                        chunk = chunk.clone().detach().cpu().numpy()
                        chunk = chunk[None, : int(chunk.shape[0])]
                        chunk = np.clip(chunk, -1, 1)
                        chunk = (chunk * 32767).astype(np.int16)
                        yield chunk.tobytes()
                else:
                    self.print_message("Starting non-streaming generation", message_type="debug_tts")
                    output = self.model.inference(**common_args)
                    torchaudio.save(str(output_file), torch.tensor(output["wav"]).unsqueeze(0), 24000)
                    self.print_message(f"Saved audio to: {output_file}", message_type="debug_tts")

            elif self.current_model_loaded.startswith("apitts"):
                if streaming:
                    raise ValueError("Streaming is only supported in XTTSv2 local mode")
                # Common arguments for both error and normal cases
                common_args = {
                    "file_path": output_file,
                    "language": language,
                    "temperature": temperature,
                    "length_penalty": self.model.config.length_penalty,
                    "repetition_penalty": repetition_penalty,
                    "top_k": self.model.config.top_k,
                    "top_p": self.model.config.top_p,
                    "speed": speed
                }     
                if voice.startswith('latent:'):
                    self.print_message("API TTS method does not support latent files - Please use an audio reference file", message_type="error")
                    self.model.tts_to_file(
                        text="The API TTS method only supports audio files not latents. Please select an audio reference file instead.",
                        speaker="Ana Florence",
                        **common_args
                    )
                else:
                    self.print_message("Using API-based generation", message_type="debug_tts")
                    self.model.tts_to_file(
                        text=text,
                        speaker_wav=wavs_files,
                        **common_args
                    )
                
                self.print_message(f"API generation completed, saved to: {output_file}", message_type="debug_tts")

        finally:
            # Generation complete
            generate_end_time = time.time()
            generate_elapsed_time = generate_end_time - generate_start_time
            
            # Standard output message (not debug)
            self.print_message(
                f"\033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled} \033[94mStreaming: \033[33m{self.streaming_enabled}\033[0m",
                message_type="standard"
            )
            
            # Handle low VRAM cleanup
            if self.lowvram_enabled and self.device == "cuda" and not self.tts_narrator_generatingtts:
                self.print_message("Low VRAM mode: Moving model back to CPU", message_type="debug_tts")
                await self.handle_lowvram_change()
                
            self.tts_generating_lock = False

    ##############################################################################
    # Helper Functions that are specific to this script & not generically needed #
    ##############################################################################
    def _generate_conditioning_latents(self, audio_paths):
        """Generate conditioning latents from audio files."""
        self.debug_func_entry()
        self.print_message(f"Generating latents from {len(audio_paths)} audio files", message_type="debug_tts")
        return self.model.get_conditioning_latents(
            audio_path=audio_paths,
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )

    def _load_latents(self, voice):
        """Load speaker latents from JSON file."""
        self.debug_func_entry()
        try:
            json_file = voice.replace("latent:", "")
            json_path = os.path.join(self.main_dir, "voices", "xtts_latents", json_file)
            self.print_message(f"Loading latents from: {json_path}", message_type="debug_tts")
            
            with open(json_path) as f:
                latent_data = json.load(f)
                gpt_cond_latent = torch.tensor(latent_data['gpt_cond_latent'])
                speaker_embedding = torch.tensor(latent_data['speaker_embedding'])
            
            self.print_message("Successfully loaded speaker latents", message_type="debug_tts")
            return gpt_cond_latent, speaker_embedding
        except Exception as e:
            self.print_message(f"Failed to load speaker latents: {str(e)}", message_type="error")
            raise HTTPException(status_code=400, detail=f"Failed to load voice latents: {str(e)}")
