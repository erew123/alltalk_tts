###############################################
# DONT CHANGE # These are base imports needed #
###############################################
import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from fastapi import (HTTPException)
logging.disable(logging.WARNING)
#################################################################
# DONT CHANGE # Get Pytorch & Python versions & setup DeepSpeed #
#################################################################
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
import asyncio


#################################################################################################################################
# DONT CHANGE # Do not change the Class name from tts_class as this is what will be imported into the main tts_server.py script #
#################################################################################################################################
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
        self.streaming_enabled = tts_model_loaded["settings"]["streaming_enabled"]                          # If its available, is Streaming enabled for the TTS engine
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
        self.printout_versions()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Set a timer to calculate load times
        generate_start_time = time.time() # Record the start time of loading the model - Piper doesnt load a model, so its a bit unique that we put a load time in here.
        self.available_models = self.scan_models_folder()
        print("in setup - self.selected_model:", self.selected_model) if self.debug_tts else None
        self.current_model_loaded = "piper"
        self.is_tts_model_loaded = True # Set self.is_tts_model_loaded to True so that other portions of AllTalk know that the script is ready to recieve generation requests, as piper doesnt load in models
        generate_end_time = time.time() # Create an end timer for calculating load times
        generate_elapsed_time = generate_end_time - generate_start_time # Calculate start time minus end time
        # Print out the result of the load time
        print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m Piper\033[94m Ready\033[0m")
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m")
         
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.setup_has_run = True # Flag that setup has run, so the /api/ready endpoint will send a "Ready" status and load the webui
    
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
    # function in place. The "pass" tells the function to just exit out cleanly if called.
    # However, its quite a simple check along the lines of "if CUDA is available and model is
    # in X place, then send it to Y place (or Y to X).
    async def handle_lowvram_change(self):
        pass # Piper does not stay in CUDA or support swapping of location
        
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
        if value:
            # DeepSpeed enabled
            print(f"[{self.branding}ENG] \033[93mDeepSpeed Activating\033[0m")
            await self.unload_model()
            self.params["tts_method_api_local"] = False
            self.params["tts_method_xtts_local"] = True
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
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        self.available_models = {'piper': 'piper'}
        
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 
        if not self.available_models:
            self.available_models["No Models Found"] = "No Models Found" # Return a list with {'No Models Found'} if there are no models found.
        return self.available_models # Return a list of models in the format {'engine name': 'model 1', 'engine name': 'model 2", etc}

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
        try:
            voices = []
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            # ↑↑↑ Keep everything above this line ↑↑↑
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 

            models_dir = self.main_dir / "models" / "piper"
            for root, dirs, files in os.walk(models_dir):
                onnx_files = [file for file in files if file.endswith(".onnx")]
                for onnx_file in onnx_files:
                    json_file = onnx_file + ".json"
                    if json_file in files:
                        relative_path = os.path.relpath(os.path.join(root, onnx_file), models_dir)
                        voices.append(relative_path)
                        
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓                     
            if not voices:
                return ["No Voices Found"] # Return a list with {'No Voices Found'} if there are no voices/voice models.
            return voices # Return a list of models in the format {'engine name': 'model 1', 'engine name': 'model 2", etc}
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voices/Voice Models not found. Cannot load a list of voices.")
            print(f"[{self.branding}ENG]")
            return ["No Voices Found"]

    #################################################################################
    #################################################################################
    # CHANGE ME # Model loading # Piper does not actually load/stay resident in RAM #
    #################################################################################
    #################################################################################
    # This function will handle the loading of your model, into VRAM/CUDA, System RAM or whatever.
    # In XTTS, which has 2x model loader types, there are 2x loaders. They are called by "def handle_tts_method_change"
    # In Piper we fake a model loader as Piper doesnt actually load a model into CUDA/System RAM as such. So, in that
    # situation, api_manual_load_model is kind of a blank function. Though we do set self.is_tts_model_loaded = True
    # as this is used elsewhere in the scripts to confirm that a model is available to be used for TTS generation.
    # We always check for "No Models Available" being sent as that means we are trying to load in a model that 
    # doesnt exist/wasnt found on script start-up e.g. someone deleted the model from the folder or something.
    async def api_manual_load_model(self, model_name):
        if model_name == "No Models Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load.")
            raise HTTPException(status_code=400, detail="No models for this TTS engine were found to load.")   
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓        
        self.is_tts_model_loaded = True
        return None

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
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 
        self.is_tts_model_loaded = False
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
        generate_start_time = time.time() # Record the start time of loading the model
        if "No Models Available" in self.available_models:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load. Please download a model.")
            return False
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
           
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
        if voice == "No Voices Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No voices found to generate TTS.")
            raise HTTPException(status_code=400, detail="No voices found to generate TTS.")    
        if not self.is_tts_model_loaded:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: You currently have no TTS model loaded.")
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")
        self.tts_generating_lock = True # Lock the process to say its currently in use
        generate_start_time = time.time()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑  

        model_dir = self.main_dir / "models" / "piper" / voice
        model_file = model_dir.with_suffix(".onnx")
        config_file = model_file.with_suffix(".onnx.json")
        inverted_speed = 1.0 / float(speed)

        if sys.platform == "win32":
            # Windows: Use piper.exe
            piper_exe = self.main_dir / "system" / "tts_engines" / "piper" / "engine" / "piper.exe"
            # Ensure all files exist
            if not piper_exe.exists():
                print(f"[{self.branding}Debug] Piper executable not found: {piper_exe}") if self.debug_tts else None
                raise HTTPException(status_code=500, detail="Piper executable not found.")
            if not model_file.exists():
                print(f"[{self.branding}Debug] Model file not found: {model_file}") if self.debug_tts else None
                raise HTTPException(status_code=500, detail="Model file not found.")
            if not config_file.exists():
                print(f"[{self.branding}Debug] Config file not found: {config_file}") if self.debug_tts else None
                raise HTTPException(status_code=500, detail="Config file not found.")
            command = [
                str(piper_exe),
                "--model", str(model_file),
                "--config", str(config_file),
                "--output_file", str(output_file),
                "--length_scale", str(inverted_speed),
            ]
            if streaming:
                print(f"[{self.branding}Debug] Streaming enabled") if self.debug_tts else None
                command.append("--output-raw")
        else:
            # Linux and macOS: Use piper command directly # In piper there are different call methods between Windows and Linux/Mac
            if not model_file.exists():
                print(f"[{self.branding}Debug] Model file not found: {model_file}") if self.debug_tts else None
                raise HTTPException(status_code=500, detail="Model file not found.")
            if not config_file.exists():
                print(f"[{self.branding}Debug] Config file not found: {config_file}") if self.debug_tts else None
                raise HTTPException(status_code=500, detail="Config file not found.")
            command = [
                "piper",
                "-m", str(model_file),
                "-c", str(config_file),
                "-f", str(output_file),
                "--length-scale", str(inverted_speed),
            ]
            if streaming:
                print(f"[{self.branding}Debug] Streaming enabled") if self.debug_tts else None
                command.append("--output-raw")

        try:
            print(f"[{self.branding}Debug] Creating subprocess") if self.debug_tts else None
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            print(f"[{self.branding}Debug] Communicating with subprocess") if self.debug_tts else None
            stdout, stderr = await process.communicate(text.encode())
            if stderr:
                print(f"[{self.branding}Debug] Subprocess stderr: {stderr.decode()}") if self.debug_tts else None
            return_code = process.returncode
            print(f"[{self.branding}Debug] Subprocess return code: {return_code}") if self.debug_tts else None
            if return_code != 0:
                print(f"[{self.branding}Debug] Subprocess failed with return code: {return_code}") if self.debug_tts else None
                raise HTTPException(status_code=500, detail=f"Subprocess failed with return code: {return_code}")
            if not os.path.exists(output_file):
                print(f"[{self.branding}Debug] Output file does not exist: {output_file}") if self.debug_tts else None
                if streaming:
                    print(f"[{self.branding}Debug] Yielding stdout") if self.debug_tts else None
                    yield stdout
                else:
                    print(f"[{self.branding}Debug] Yielding empty byte string") if self.debug_tts else None
                    yield b""

        except Exception as e:
            print(f"[{self.branding}Debug] Exception while running subprocess: {str(e)}") if self.debug_tts else None
            raise HTTPException(status_code=500, detail=f"Exception while running subprocess: {str(e)}")
        
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled} \033[94mStreaming: \033[33m{self.streaming_enabled}\033[0m")
        self.tts_generating_lock = False
