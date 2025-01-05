###############################################
# DONT CHANGE # These are base imports needed #
###############################################
import os
import sys
import json
import time
import torch
import logging
import inspect
from pathlib import Path
from fastapi import (HTTPException)
logging.disable(logging.WARNING)
###############################################
# DONT CHANGE # Get Pytorch & Python versions #
###############################################
pytorch_version = torch.__version__
cuda_version = torch.version.cuda
major, minor, micro = sys.version_info[:3]
python_version = f"{major}.{minor}.{micro}"
try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    deepspeed_available = False
    pass

#############################################################################################################
#############################################################################################################
# CHANGE ME # Run any specifc imports, requirements or setup any global vaiables needed for this TTS Engine #
#############################################################################################################
#############################################################################################################
# In this section you will import any imports that your specific TTS Engine will use. You will provide any
# start-up errors for those bits, as if you were starting up a normal Python script. Note the logging.disable
# a few lines up from here, you may want to # that out while debugging!
try:
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    print(
        f"[Startup] \033[91mWarning\033[0m Could not find the TTS module. Make sure to install the requirements for the alltalk_tts extension.",
        f"[Startup] \033[91mWarning\033[0m Linux / Mac:\npip install -r extensions/alltalk_tts/requirements.txt\n",
        f"[Startup] \033[91mWarning\033[0m Windows:\npip install -r extensions\\alltalk_tts\\requirements.txt\n",
        f"[Startup] \033[91mWarning\033[0m If you used the one-click installer, paste the command above in the terminal window launched after running the cmd_ script. On Windows, that's cmd_windows.bat."
    )
    raise

#############################################################
# DONT CHANGE # Do not change the Class name from tts_class #
#############################################################
class tts_class:
    def __init__(self):
        ########################################################################
        # DONT CHANGE # Sets up the base variables required for any tts engine #
        ########################################################################
        self.branding = None                                                    
        self.this_dir = Path(__file__).parent.resolve()                         # Sets up self.this_dir as a variable for the folder THIS script is running in.
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()    # Sets up self.main_dir as a variable for the folder AllTalk is running in
        self.device = "cuda" if torch.cuda.is_available() else "cpu"            # Sets up self.device to cuda if torch exists with Nvidia/CUDA, otherwise sets to cpu
        self.cuda_is_available = torch.cuda.is_available()                      # Sets up cuda_is_available as a True/False to track if Nvidia/CUDA was found on the system
        self.tts_generating_lock = False                                        # Used to lock and unlock the tts generation process at the start/end of tts generation. 
        self.tts_stop_generation = False                                        # Used in conjunction with tts_generating_lock to call for a stop to the current generation. If called (set True) it needs to be set back to False when generation has been stopped.
        self.tts_narrator_generatingtts = False                                 # Used to track if the current tts processes is narrator based. This can be used in conjunction with lowvram and device to avoid moving model between GPU(CUDA)<>RAM(CPU) each chunk of narrated text generated.
        self.model = None                                                       # If loading a model into CUDA/VRAM/RAM "model" is used as the variable name to load and interact with (see the XTTS model_engine script for examples.)
        self.is_tts_model_loaded = False                                        # Used to track if a model is actually loaded in and error/fail things like TTS generation if its False
        self.current_model_loaded = None                                        # Stores the name of the currenly loaded in model
        self.available_models = None                                            # List of available models found by "def scan_models_folder"
        self.setup_has_run = False                                              # Tracks if async def setup(self) has run, by setting to True, so that the /api/ready endpoint can provide a "Ready" status
        ##############################################################################################
        # DONT CHANGE # Load in a list of the available TTS engines and the currently set TTS engine #
        ##############################################################################################
        tts_engines_file = os.path.join(self.main_dir, "system", "tts_engines", "tts_engines.json")
        with open(tts_engines_file, "r") as f:
            tts_engines_data = json.load(f)
        self.engines_available = [engine["name"] for engine in tts_engines_data["engines_available"]]       # A list of ALL the TTS engines available to be loaded by AllTalk
        self.engine_loaded = tts_engines_data["engine_loaded"]                                              # In "tts_engines.json" what is the currently set TTS engine loading into AllTalk
        self.selected_model = tts_engines_data["selected_model"]                                            # In "tts_engines.json" what is the currently set TTS model loading into AllTalk
        ############################################################################
        # DONT CHANGE # Pull out all the settings for the currently set TTS engine #
        ############################################################################
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            tts_model_loaded = json.load(f)
        # Access the model details
        self.manufacturer_name = tts_model_loaded["model_details"]["manufacturer_name"]                     # The company/person/body that generated the TTS engine/models etc
        self.manufacturer_website = tts_model_loaded["model_details"]["manufacturer_website"]               # The website of the company/person/body where people can find more information
        # Access the features the model is capable of:
        self.audio_format = tts_model_loaded["model_capabilties"]["audio_format"]                           # This details the audio format your TTS engine is set to generate TTS in e.g. wav, mp3, flac, opus, acc, pcm. Please use only 1x format.
        self.deepspeed_capable = tts_model_loaded["model_capabilties"]["deepspeed_capable"]                 # Is your model capable of DeepSpeed
        self.deepspeed_available = 'deepspeed' in globals()                                                 # When we did the import earlier, at the top of this script, was DeepSpeed available for use
        self.generationspeed_capable = tts_model_loaded["model_capabilties"]["generationspeed_capable"]     # Does this TTS engine support changing the speed of the generated TTS
        self.languages_capable = tts_model_loaded["model_capabilties"]["languages_capable"]                 # Are the actual models themselves capable of generating in multiple languages OR is each model language specific
        self.lowvram_capable = tts_model_loaded["model_capabilties"]["lowvram_capable"]                     # Is this engine capable of using low VRAM (moving the model between CPU And GPU memory)
        self.multimodel_capable = tts_model_loaded["model_capabilties"]["multimodel_capable"]               # Is there just the one model or are there multiple models this engine supports.
        self.repetitionpenalty_capable = tts_model_loaded["model_capabilties"]["repetitionpenalty_capable"] # Is this TTS engine capable of changing the repititon penalty
        self.streaming_capable = tts_model_loaded["model_capabilties"]["streaming_capable"]                 # Is this TTS engine capabale of generating streaming audio
        self.temperature_capable = tts_model_loaded["model_capabilties"]["temperature_capable"]             # Is this TTS engine capable of changing the temperature of the models
        self.multivoice_capable = tts_model_loaded["model_capabilties"]["multivoice_capable"]               # Are the models multi-voice or single vocice models
        self.pitch_capable = tts_model_loaded["model_capabilties"]["pitch_capable"]                         # Is this TTS engine capable of changing the pitch of the genrated TTS
        # Access the current enginesettings
        self.def_character_voice = tts_model_loaded["settings"]["def_character_voice"]                      # What is the current default main/character voice that will be used if no voice specified.
        self.def_narrator_voice = tts_model_loaded["settings"]["def_narrator_voice"]                        # What is the current default narrator voice that will be used if no voice specified.
        self.deepspeed_enabled = tts_model_loaded["settings"]["deepspeed_enabled"]                          # If its available, is DeepSpeed enabled for the TTS engine
        self.engine_installed = tts_model_loaded["settings"]["engine_installed"]                            # Has the TTS engine been setup/installed (not curently used)
        self.generationspeed_set = tts_model_loaded["settings"]["generationspeed_set"]                      # What is the set/stored speed for generation.
        self.lowvram_enabled = tts_model_loaded["settings"]["lowvram_enabled"]                              # If its available, is LowVRAM enabled for the TTS engine
        # Check if someone has enabled lowvram on a system that's not CUDA enabled
        self.lowvram_enabled = False if not torch.cuda.is_available() else self.lowvram_enabled             # If LowVRAM is mistakenly set and CUDA is not available, this will force it back off
        self.repetitionpenalty_set = tts_model_loaded["settings"]["repetitionpenalty_set"]                  # What is the currenly set repitition policy of the model (If it support repetition)
        self.temperature_set = tts_model_loaded["settings"]["temperature_set"]                              # What is the currenly set temperature of the model (If it support temp)
        self.pitch_set = tts_model_loaded["settings"]["pitch_set"]                                          # What is the currenly set pitch of the model (If it support temp)
        # Gather the OpenAI API Voice Mappings
        self.openai_alloy = tts_model_loaded["openai_voices"]["alloy"]                                      # The TTS engine voice that will be mapped to Open AI Alloy voice
        self.openai_echo = tts_model_loaded["openai_voices"]["echo"]                                        # The TTS engine voice that will be mapped to Open AI Echo voice
        self.openai_fable = tts_model_loaded["openai_voices"]["fable"]                                      # The TTS engine voice that will be mapped to Open AI Fable voice
        self.openai_nova = tts_model_loaded["openai_voices"]["nova"]                                        # The TTS engine voice that will be mapped to Open AI Nova voice
        self.openai_onyx = tts_model_loaded["openai_voices"]["onyx"]                                        # The TTS engine voice that will be mapped to Open AI Onyx voice
        self.openai_shimmer = tts_model_loaded["openai_voices"]["shimmer"]                                  # The TTS engine voice that will be mapped to Open AI Shimmer voice
        ###################################################################
        # DONT CHANGE #  Load params and api_defaults from confignew.json #
        ###################################################################
        # Define the path to the confignew.json file
        configfile_path = self.main_dir / "confignew.json"
        # Load config file and get settings
        with open(configfile_path, "r") as configfile:
            configfile_data = json.load(configfile)
        self.branding = configfile_data.get("branding", "")                                                 # Sets up self.branding for outputting the name stored in the "confgnew.json" file, as used in print statements.
        self.params = configfile_data                                                                       # Loads in the curent "confgnew.json" file to self.params.
        self.debug_tts = configfile_data.get("debugging").get("debug_tts")                                  # Can be used within this script as a True/False flag for generally debugging the TTS generation process. 
        self.debug_tts_variables = configfile_data.get("debugging").get("debug_tts_variables")              # Can be used within this script as a True/False flag for generally debugging variables (if you wish).      
    ################################################################
    # DONT CHANGE #  Print out Python, CUDA, DeepSpeed versions ####
    ################################################################
    def printout_versions(self):
        if deepspeed_available:
            print(f"[{self.branding}ENG] \033[92mDeepSpeed version :\033[93m",deepspeed.__version__,"\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mDeepSpeed version :\033[93m Not available\033[0m")
        print(f"[{self.branding}ENG] \033[92mPython Version    :\033[93m {python_version}\033[0m")
        print(f"[{self.branding}ENG] \033[92mPyTorch Version   :\033[93m {pytorch_version}\033[0m")
        if cuda_version is None:
            print(f"[{self.branding}ENG] \033[92mCUDA Version      :\033[91m Not available\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mCUDA Version      :\033[93m {cuda_version}\033[0m")
        print(f"[{self.branding}ENG]")
        return


    ###############################################
    # Central print function # Do not change this #
    ###############################################
    def debug_func_entry(self):
        """Log function entry if debug_func is enabled."""
        quick_debug = False
        if quick_debug:
            current_func = inspect.currentframe().f_back.f_code.co_name
            print(f"Function entry: {current_func}", "debug_func")


    ###################################################################################      
    ###################################################################################
    # CHANGE ME # Inital setup of the model and engine. Called when the script starts #
    ###################################################################################
    ###################################################################################
    # In here you will add code to load in your model and do its inital setup. So you
    # may be calling your model loader via handle_tts_method_change or if your TTS
    # engine doesnt actually load a model into CUDA or System RAM, you may be doing 
    # Something to fake its start-up.
    async def setup(self):
        self.debug_func_entry()
        self.printout_versions()
        self.available_models = self.scan_models_folder()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        print("in setup - self.selected_model:", self.selected_model) if self.debug_tts else None
        if self.selected_model:
            tts_model = f"{self.selected_model}"
            print("in setup - tts_model:", tts_model) if self.debug_tts else None
            if tts_model in self.available_models:
                print("in setup - sending to handle_tts_method_change:") if self.debug_tts else None
                await self.handle_tts_method_change(tts_model)
                self.current_model_loaded = tts_model
                
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓                  
            else:
                self.current_model_loaded = "No Models Available"
                print(f"[{self.branding}ENG] \033[91mError\033[0m: Selected model '{self.selected_model}' not found in the models folder.")   
        self.setup_has_run = True  # Set to True, so that the /api/ready endpoint can provide a "Ready" status                  
       
    ##################################
    ##################################
    # CHANGE ME #  Low VRAM Swapping #
    ##################################
    ##################################
    # If your model does load into CUDA and you want to support LowVRAM, aka moving the model
    # on the fly between CUDA and System RAM on each generation request, you will be adding
    # The code in here to do it. See the XTTS tts engine for an example. Piper however, doesnt
    # Load into VRAM or System RAM, also low vram is set globally disabled by the model_settings.JSON
    # file, so this would never get called anywyay, however, we still need to keep an empty 
    # function in place. Piper TTS uses "pass" telling function to just exit out cleanly if called.
    # However, its quite a simple check along the lines of "if CUDA is available and model is
    # in X place, then send it to Y place (or Y to X).
    async def handle_lowvram_change(self):
        self.debug_func_entry()
        if torch.cuda.is_available():
            if self.device == "cuda":
                self.device = "cpu"
                self.model.to(self.device)
                torch.cuda.empty_cache()
            else:
                self.device == "cpu"
                self.device = "cuda"
                self.model.to(self.device)
                
    ########################################
    ########################################
    # CHANGE ME #  DeepSpeed model loading #
    ########################################
    ########################################
    # If the model supports CUDA and DeepSpeed, this is where you can handle re-loading
    # the model as/when DeepSpeed is enabled/selected in the user interface. If your 
    # TTS model doesnt support DeepSpeed, then it should be globally set in your
    # model_settings.JSON and this function will never get called, however it still needs
    # to exist as a function.
    async def handle_deepspeed_change(self, value):
        self.debug_func_entry()
        if value:
            # DeepSpeed enabled
            print(f"[{self.branding}ENG] \033[93mDeepSpeed Activating\033[0m")
            await self.unload_model()
            self.params["tts_method_api_local"] = False
            self.deepspeed_enabled = True
            await self.setup()
        else:
            # DeepSpeed disabled
            print(f"[{self.branding}ENG] \033[93mDeepSpeed De-Activating\033[0m")
            self.deepspeed_enabled = False
            await self.unload_model()
            await self.setup()
        return value  # Return new checkbox value    

    ##############################################################################################################################################
    ##############################################################################################################################################
    # CHANGE ME # scan for available models/voices that are relevant to this TTS engine # XTTS is very unusal as it has 2x model loading methods #
    ##############################################################################################################################################
    ##############################################################################################################################################
    # This function looks and reports back the list of possible models your TTS engine can
    # load in. Some TTS engines have multiple models they can load and you will want to use
    # code for checking if the models are in the correct location/placement within the disk
    # the correct files exst per model etc (check XTTS for an example of this). Some models
    # like Piper, the actual models are the voices, so in the Piper scan_models_folder
    # function, we fake the only model being available as {'piper': 'piper'} aka, model name
    # then engine name, then we use the voices_file_list to populate the models as available
    # voices that can be selected in the interface.    
    # If no models are found, we return "No Models Available" and continue on with the script.
    def scan_models_folder(self):
        self.debug_func_entry()
        models_folder = self.main_dir / "models" / "vits" # Edit to match the name of your folder where voices are stored
        self.available_models = {}
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑     
  
        # Define possible model file names
        model_files = ["model_file.pth", "model.pth"]
        
        found_any_models = False
        for subfolder in models_folder.iterdir():
            if subfolder.is_dir():
                model_name = subfolder.name
                
                # Check if config.json exists (this is the minimum requirement)
                if not (subfolder / "config.json").exists():
                    print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Model folder '{model_name}' missing config.json")
                    continue

                # Load config to check model type
                try:
                    with open(subfolder / "config.json", "r") as f:
                        config = json.load(f)
                    use_speaker_embedding = config["model_args"].get("use_speaker_embedding", False)
                    num_speakers = config["model_args"].get("num_speakers", 0)
                except Exception as e:
                    print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Couldn't read config.json for '{model_name}': {str(e)}")
                    continue

                # Check for model file
                has_model_file = any(subfolder.joinpath(model_file).exists() for model_file in model_files)
                has_model_file_tar = subfolder.joinpath("model_file.pth.tar").exists()
                
                if not (has_model_file or has_model_file_tar):
                    print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Model folder '{model_name}' missing model file")
                    continue

                # Determine model type based on files and config
                if (subfolder / "language_ids.json").exists() and (subfolder / "speaker_ids.json").exists():
                    self.available_models[f"vits - {model_name}"] = "vits_multi_language"
                    found_any_models = True
                    #print(f"[{self.branding}ENG] Found multi-language model: {model_name}")
                    
                elif (subfolder / "speaker_ids.json").exists() or (use_speaker_embedding or num_speakers > 0):
                    self.available_models[f"vits - {model_name}"] = "vits_multi_speaker"
                    found_any_models = True
                    #print(f"[{self.branding}ENG] Found multi-speaker model: {model_name}")
                    
                else:
                    self.available_models[f"vits - {model_name}"] = "vits_single_speaker"
                    found_any_models = True
                    #print(f"[{self.branding}ENG] Found single-speaker model: {model_name}")

                # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                # ↓↓↓ Keep everything below this line ↓↓↓
                # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  

        if not found_any_models:
            self.available_models = {'No Models Available': 'vits'} # Change the name in here to your TTS engine name                
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Model folder '{model_name}' is missing required")
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: files or the folder does not exist.")
        
        return self.available_models
    
    #############################################################
    #############################################################
    # CHANGE ME #  POPULATE FILES LIST FROM VOICES DIRECTORY ####
    #############################################################
    #############################################################
    # This function looks and reports back the list of possible voice your TTS engine can
    # load in. Some TTS engines the voices are wav file samples (XTTS), some are models 
    # (Piper) and some are text (Parler) thats stored in a JSON file. We just need to 
    # populate the "voices" variable somehow and if no voices are found, we return
    # "No Voices Found" back to the interface/api.    
    def voices_file_list(self):
        self.debug_func_entry()
        try:
            voices = []
            
            model_name = self.selected_model.replace("vits - ", "")
            print("model_name", model_name) if self.debug_tts else None
            model_type = self.available_models[self.selected_model]
            print("model_type", model_type) if self.debug_tts else None
            
            # Load config to check model type
            config_path = self.main_dir / "models" / "vits" / model_name / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                use_speaker_embedding = config["model_args"].get("use_speaker_embedding", False)
                num_speakers = config["model_args"].get("num_speakers", 0)
            else:
                print(f"[{self.branding}ENG] \033[91mWarning\033[0m: config.json not found for model '{model_name}'")
                return ["No Voices Found"]

            # Check for speakers file and model configuration
            if model_type in ["vits_multi_speaker", "vits_multi_language"]:
                speaker_ids_path = self.main_dir / "models" / "vits" / model_name / "speaker_ids.json"
                if speaker_ids_path.exists():
                    with open(speaker_ids_path, "r") as f:
                        speaker_ids = json.load(f)
                    voices = [speaker for speaker in speaker_ids.keys() if speaker != "ED\n"]
                    print(f"[{self.branding}ENG] Found {len(voices)} voices in speaker_ids.json") if self.debug_tts else None
                elif use_speaker_embedding or num_speakers > 0:
                    # If there's no speaker_ids.json but config indicates multiple speakers,
                    # create a default list of numbered speakers
                    voices = [f"Speaker{i}" for i in range(num_speakers)]
                    print(f"[{self.branding}ENG] Created {len(voices)} default voice names from config") if self.debug_tts else None
                else:
                    print(f"[{self.branding}ENG] No speaker information found for multi-speaker model '{model_name}'") if self.debug_tts else None
            
            elif model_type == "vits_single_speaker":
                voices = ["default"]
                print(f"[{self.branding}ENG] Using default voice for single-speaker model") if self.debug_tts else None
                        
            if not voices:  # If the voices list is empty
                voices = ["default"]  # Fallback to default if no voices were found
                print(f"[{self.branding}ENG] No voices found, using default") if self.debug_tts else None
                
            return voices  # Return the list of voices
            
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voices/Voice Models not found. Cannot load a list of voices.")
            print(f"[{self.branding}ENG] Exception: {str(e)}")
            return ["No Voices Found"]

    #############################
    #############################
    # CHANGE ME # Model loading #
    #############################
    #############################
    # This function will handle the loading of your model, into VRAM/CUDA, System RAM or whatever.
    # In XTTS, which has 2x model loader types, there are 2x loaders. They are called by "def handle_tts_method_change"
    # In Piper we fake a model loader as Piper doesnt actually load a model into CUDA/System RAM as such. So, in that
    # situation, api_manual_load_model is kind of a blank function. Though we do set self.is_tts_model_loaded = True
    # as this is used elsewhere in the scripts to confirm that a model is available to be used for TTS generation.
    # We always check for "No Models Available" being sent as that means we are trying to load in a model that 
    # doesnt exist/wasnt found on script start-up e.g. someone deleted the model from the folder or something.   
    async def api_manual_load_model(self, model_name):
        self.debug_func_entry()
        if "No Models Available" in self.available_models:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load. Please download a model.")
            return    
        model_path = self.main_dir / "models" / "vits" / model_name # change the name to the correct folder name
        print("model_path is:", model_path) if self.debug_tts_variables else None
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Load the config file
        config_file_path = model_path / "config.json"
        with open(config_file_path, "r") as f:
            config_dict = json.load(f)
        # Check if the model is single-speaker, multi-speaker, or multi-language
        model_type = self.available_models[f"vits - {model_name}"]
        use_speaker_embedding = config_dict["model_args"]["use_speaker_embedding"]
        num_speakers = config_dict["model_args"]["num_speakers"]
        if model_type == "vits_multi_speaker" or model_type == "vits_multi_language":
            # Update the speakers_file path in the config dictionary
            config_dict["model_args"]["speakers_file"] = str(model_path / "speaker_ids.json")
            speakers_file = model_path / "speaker_ids.json"
        else:
            # Set speakers_file to None for single-speaker models
            config_dict["model_args"]["speakers_file"] = None
            speakers_file = None
        # Save the updated config dictionary back to the config.json file
        with open(config_file_path, "w") as f:
            json.dump(config_dict, f, indent=4)

        def get_model_path(model_dir, model_type):
            """Helper function to find the correct model file"""
            print(f"Searching for model file in {model_dir} for type {model_type}") if self.debug_tts else None
            
            if model_type == "vits_multi_language":
                # Check for .pth.tar file first for multi-language models
                if (model_dir / "model_file.pth.tar").exists():
                    print(f"Found model_file.pth.tar") if self.debug_tts else None
                    return model_dir / "model_file.pth.tar"
            
            # Check for regular .pth files
            if (model_dir / "model_file.pth").exists():
                print(f"Found model_file.pth") if self.debug_tts else None
                return model_dir / "model_file.pth"
            elif (model_dir / "model.pth").exists():
                print(f"Found model.pth") if self.debug_tts else None
                return model_dir / "model.pth"
            
            print(f"No model file found in {model_dir}") if self.debug_tts else None
            raise FileNotFoundError(f"No model file found in {model_dir}")

        # Then in api_manual_load_model:
        model_type = self.available_models[f"vits - {model_name}"]
        model_file = get_model_path(model_path, model_type)

        if model_type == "vits_multi_language":
            # Load multi-language model
            self.model = Synthesizer(
                tts_checkpoint=model_file,  # Use the detected model file
                tts_config_path=config_file_path,
                tts_speakers_file=speakers_file,
                tts_languages_file=model_path / "language_ids.json",
                use_cuda=self.cuda_is_available,
            )
        elif not use_speaker_embedding and num_speakers == 0:
            # Single-speaker model without speaker embeddings
            self.model = Synthesizer(
                tts_checkpoint=model_file,
                tts_config_path=config_file_path,
                use_cuda=self.cuda_is_available,
            )
        else:
            # Multi-speaker model or single-speaker model with speaker embeddings
            self.model = Synthesizer(
                tts_checkpoint=model_file,
                tts_config_path=config_file_path,
                tts_speakers_file=speakers_file,
                use_cuda=self.cuda_is_available,
            )
            
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.is_tts_model_loaded = True
        return self.model

    ###############################
    ###############################
    # CHANGE ME # Model unloading #
    ###############################
    ###############################
    # This function will handle the UN-loading of your model, from VRAM/CUDA, System RAM or whatever.
    # In XTTS, that model loads into CUDA/System Ram, so when we swap models, we want to unload the current model
    # free up the memory and then load in the new model to VRAM/CUDA. On the flip side of that, Piper doesnt
    # doesnt load into memory, so we just need to put a fake function here that doesnt really do anything
    # other than set "self.is_tts_model_loaded = False", which would be set back to true by the model loader. 
    # So look at the Piper model_engine.py if you DONT need to unload models. 
    async def unload_model(self):
        self.debug_func_entry()
        self.is_tts_model_loaded = False
        if not self.current_model_loaded == None:
            print(f"[{self.branding}ENG] \033[94mUnloading model \033[0m") if self.debug_tts else None
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    ###################################################################################################################################    
    ###################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one # XTTS is very unusal as it has 2x model loading methods #
    ###################################################################################################################################
    ###################################################################################################################################
    # This function is your central model loading/unloading handler that deals with the above functions as necesary, to call loading, unloading,
    # swappng DeepSpeed, Low vram etc. This function gets called with a "engine name - model name" type call. In XTTS, because there are 2x
    # model loader types, (XTTS and APILocal), we take tts_method and split the "engine name - model name" into a loader type and the model
    # that it needs to load in and then we call the correct loader function. Whereas in Piper, which doesnt load models into memory at all, 
    # we just have a fake function that doesnt really do anything. We always check to see if the model name has "No Models Available" in the
    # name thats sent over, just to catch any potential errors. We display the start load time and end load time. Thats about it.
    async def handle_tts_method_change(self, tts_method):
        self.debug_func_entry()
        generate_start_time = time.time() # Record the start time of loading the model
        if "No Models Available" in self.available_models:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load. Please download a model.")
            return False
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        await self.unload_model()
        # Set other parameters to False
        if tts_method.startswith("vits"):
            model_name = tts_method.split(" - ")[1]
            print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m {model_name}\033[94m loading into\033[93m", self.device,"\033[0m")
            self.model = await self.api_manual_load_model(model_name)
            self.current_model_loaded = f"vits - {model_name}"
            self.selected_model = tts_method  # Update self.selected_model
        else:
            self.current_model_loaded = None
            return False
            
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 
        generate_end_time = time.time() # Create an end timer for calculating load times
        generate_elapsed_time = generate_end_time - generate_start_time  # Calculate start time minus end time
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m") # Print out the result of the load time
        return True

    ##########################################################################################################################################    
    ##########################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one # XTTS is very unusal as it has 2x model TTS generation methods #
    ##########################################################################################################################################
    ##########################################################################################################################################
    # In here all the possible options are sent over (text, voice to use, lanugage, speed etc etc) and its up to you how you use them, or not.
    # obviously if your TTS engine doesnt support speed for example, generationspeed_capable should be set False in your model_settings.JSON file
    # and a fake "generationspeed_set" value should be set. This allows a fake value to be sent over from the main script, even though it
    # wouldnt actually ever be used in the generation below. Nonethless all these values, real or just whats inside the configuration file
    # will be sent over for use. 
    # Setting the xxxxxxx_capabale in the model_settings.JSON file, will enable/disable them being selectable by the user. For example, if you
    # set "generationspeed_capable" as false in the model_settings.JSON file, a user will not be able to select OR set the setting for 
    # generation speed.
    # One thing to note is that we HAVE to have something in this generation request that is synchronous from the way its called, which means
    # we have to have an option for Streaming, even if our TTS engine doesnt support streaming. So in that case, we would set streaming_capable
    # as false in our model_settings.JSON file, meaning streaming will never be called. However, we have to put a fake streaming routine in our
    # function below (or a real function if it does support streaming of course). Parler has an example of a fake streaming function, which is
    # very clearly highlighted in its model_engine.py script.
    # Piper TTS, which uses command line based calls and therefore has different ones for Windows and Linux/Mac, has an example of doing this
    # within its model_engine.py file.   
    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        self.debug_func_entry()
        print(f"[{self.branding}Debug] Entered model_engine.py generate_tts function") if self.debug_tts else None
        if not self.is_tts_model_loaded: # Check if a model is loaded and error out if not.
            error_message = f"[{self.branding}ENG] \033[91mError\033[0m: You currently have no TTS model loaded." 
            print(error_message)
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")  # Raise an exception with a meaningful HTTP status code
        self.tts_generating_lock = True # Set the tts_generating lock to True, which stops other generation requests being sent into the pipeline
        print(f"[{self.branding}Debug] Checking low VRAM") if self.debug_tts else None
        if self.lowvram_enabled and self.device == "cpu": # If necessary, move the model out of System Ram to VRAM
            print(f"[{self.branding}Debug] Switching device") if self.debug_tts else None
            await self.handle_lowvram_change()
        print(f"[{self.branding}Debug] Setting a generate time") if self.debug_tts else None
        generate_start_time = time.time()  # Record the start time of generating TTS
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        if self.current_model_loaded.startswith("vits"):
            print(f"[{self.branding}Debug] generate_tts Text arriving at engine {text}") if self.debug_tts else None
            model_type = self.available_models[self.current_model_loaded]
            if model_type == "vits_multi_speaker":
                # Get the available speakers for the current model
                model_name = self.current_model_loaded.replace("vits - ", "")
                speaker_ids_path = self.main_dir / "models" / "vits" / model_name / "speaker_ids.json"
                if speaker_ids_path.exists():
                    with open(speaker_ids_path, "r") as f:
                        speaker_ids = json.load(f)
                    available_speakers = [speaker for speaker in speaker_ids.keys() if speaker != "ED\n"]
                else:
                    print(f"[{self.branding}ENG] \033[91mWarning\033[0m: speaker_ids.json not found for model '{model_name}'.")
                    available_speakers = []

                # Check if the specified voice matches any of the available speakers
                if voice in available_speakers:
                    speaker_idx = voice
                else:
                    # Use the first available speaker as the default
                    if available_speakers:
                        speaker_idx = available_speakers[0]
                        print(f"[{self.branding}ENG] Incorrect or no voice specified, using default voice: {speaker_idx}")
                    else:
                        speaker_idx = None
                        print(f"[{self.branding}ENG] No valid speakers found for the current model.")

                print(f"[{self.branding}Debug] generate_tts speaker_idx is:", speaker_idx) if self.debug_tts else None
                print(f"[{self.branding}Debug] generate_tts output_file is:", output_file) if self.debug_tts else None

                # Generate TTS with speaker_name argument
                if speaker_idx is not None:
                    wavs = self.model.tts(text, speaker_name=speaker_idx)
                else:
                    wavs = None
            else:
                # Generate TTS without speaker_name argument for single-speaker models
                wavs = self.model.tts(text)

            streaming = False
            if streaming and wavs is not None:
                # Streaming-specific operations
                for wav_chunk in wavs:
                    yield wav_chunk.numpy().tobytes()
            else:
                if wavs is not None:
                    # Save the generated audio to a file
                    self.model.save_wav(wavs, output_file)
                yield b""  # Yield an empty byte string to indicate completion

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 
        generate_end_time = time.time()  # Record the end time to generate TTS
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled} \033[94mStreaming: \033[33m{self.streaming_enabled}\033[0m")
        if self.lowvram_enabled and self.device == "cuda" and self.tts_narrator_generatingtts == False:
            await self.handle_lowvram_change()
        self.tts_generating_lock = False # Unlock the TTS generation queue to allow TTS generation requests to come in again.
        print(f"[{self.branding}Debug] generate_tts function completed") if self.debug_tts else None
        return


