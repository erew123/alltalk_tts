import os
import sys
import json
import time
import argparse
import librosa
import logging
import importlib
import subprocess
from pathlib import Path
from config import AlltalkConfig, AlltalkTTSEnginesConfig

logging.disable(logging.WARNING)
#####################
# Webserver Imports #
#####################
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, Request, Response, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse, StreamingResponse
##########################
# Generation API Imports #
##########################
import re
import html
import uuid
import hashlib
import numpy as np
import soundfile as sf
from typing import Union, Dict, List, Optional
from pydantic import BaseModel, ValidationError, Field, field_validator
########################################################################################
# START-UP # Silence RVC warning about torch.nn.utils.weight_norm even though not used #
########################################################################################
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional", lineno=5476)
# Filter ComplexHalf support warning
warnings.filterwarnings("ignore", message="ComplexHalf support is experimental")
# Filter Flash Attention warning
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention")
####################
# Setup local path #
####################
this_dir = Path(__file__).parent.resolve()  # Set this_dir as the current alltalk_tts folder

infer_pipeline = None
config: AlltalkConfig | None = None
tts_engines_config: AlltalkTTSEnginesConfig | None = None

def load_config(force_reload = False):
    global config, tts_engines_config
    config = AlltalkConfig.get_instance(force_reload)
    tts_engines_config = AlltalkTTSEnginesConfig.get_instance(force_reload)
    after_config_load()

def after_config_load():
    global infer_pipeline
    if config.rvc_settings.rvc_enabled:
        from system.tts_engines.rvc.infer.infer import infer_pipeline
    else:
        infer_pipeline = None

load_config()

####################
# Check for FFMPEG #
####################
def check_ffmpeg(this_dir):
    if sys.platform == "win32":
        # Check for ffmpeg.exe in the specified directory on Windows
        ffmpeg_path = os.path.join(this_dir, "system", "win_ffmpeg", "ffmpeg.exe")
        if os.path.exists(ffmpeg_path):
            return True
    else:
        # Check for FFmpeg on Linux and macOS using the command line
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return False

# Check if FFmpeg is installed
ffmpeg_installed = check_ffmpeg(this_dir)

if not ffmpeg_installed:
    print(f"[{config.branding}ENG] \033[92mTranscoding       :\033[94m ffmpeg not found\033[0m")
    print(f"[{config.branding}ENG] FFmpeg is not installed. Transcoding will be disabled.")
    print(f"[{config.branding}ENG] Please install FFmpeg on your system.")
    
    if sys.platform == "win32":
        print(f"[{config.branding}ENG] \033[92mTranscoding       :\033[94m ffmpeg not found\033[0m")
        print(f"[{config.branding}ENG] Installation instructions for Windows:")
        print(f"[{config.branding}ENG] Copy the 'ffmpeg.exe' file to '{os.path.join(this_dir, 'system', 'win_ffmpeg')}'")
    else:
        print(f"[{config.branding}ENG] \033[92mTranscoding       :\033[94m ffmpeg not found\033[0m")
        print(f"[{config.branding}ENG] Installation instructions:")
        print(f"[{config.branding}ENG] Linux (Debian-based systems): Run 'sudo apt-get install ffmpeg' in the terminal.")
        print(f"[{config.branding}ENG] macOS: Run 'brew install ffmpeg' in the terminal (requires Homebrew).")
    # You can choose to exit the script or continue without FFmpeg
    # exit(1)

if ffmpeg_installed:
    from ffmpeg.asyncio import FFmpeg
    print(f"[{config.branding}ENG] \033[92mTranscoding       :\033[93m ffmpeg found\033[0m")
    
################################
# Check for portaudio on Linux #
################################
try:
    import sounddevice as sd
    sounddevice_installed=True
except OSError:
    print(f"[{config.branding}Startup] \033[91mInfo\033[0m PortAudio library not found. If you wish to play TTS in standalone mode through the API suite")
    print(f"[{config.branding}Startup] \033[91mInfo\033[0m please install PortAudio. This will not affect any other features or use of Alltalk.")
    print(f"[{config.branding}Startup] \033[91mInfo\033[0m If you don't know what the API suite is, then this message is nothing to worry about.")
    sounddevice_installed=False
    if sys.platform.startswith('linux'):
        print(f"[{config.branding}Startup] \033[91mInfo\033[0m On Linux, you can use the following command to install PortAudio:")
        print(f"[{config.branding}Startup] \033[91mInfo\033[0m sudo apt-get install portaudio19-dev")

#######################################################################
# Attempt to import the ModelLoader class for the selected TTS engine #
#######################################################################
if tts_engines_config.is_valid_engine(tts_engines_config.engine_loaded):
    loader_module = importlib.import_module(f"system.tts_engines.{tts_engines_config.engine_loaded}.model_engine")
    tts_class = getattr(loader_module, "tts_class")
    # Setup model_engine as the way to call the functions within the Class.
    model_engine = tts_class()
else:
    raise ValueError(f"Invalid TTS engine: {tts_engines_config.engine_loaded}")

##########################################
# Run setup function in the model_engine #
##########################################
@asynccontextmanager
async def startup_shutdown(no_actual_value_it_demanded_something_be_here):
    try:
        await model_engine.setup()
    except FileNotFoundError as e:
        print(f"Error during setup: {e}. Continuing without the TTS model.")
    yield
    # Shutdown logic

###############################
# Setup FastAPI with Lifespan #
###############################
app = FastAPI(lifespan=startup_shutdown)
# Allow all origins, and set other CORS options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############################
# API Endpoint - /api/reload #
##############################
model_currently_changing = False
@app.route("/api/reload", methods=["POST"])
async def apifunction_reload(request: Request):
    global model_currently_changing
    if model_currently_changing == True:
        return Response(content=json.dumps({"status": "model-is currently changing"}), media_type="application/json")
    model_currently_changing == True
    requested_model = request.query_params.get("tts_method")
    if requested_model not in model_engine.available_models:
        model_currently_changing == False
        return {"status": "error", "message": "Invalid TTS method specified"}
    # Call the handle_tts_method_change method from the imported class
    success = await model_engine.handle_tts_method_change(requested_model)
    if success:
        model_engine.current_model_loaded = requested_model
        model_currently_changing == False
        return Response(content=json.dumps({"status": "model-success"}), media_type="application/json")
    else:
        model_currently_changing == False
        return Response(content=json.dumps({"status": "model-failure"}), media_type="application/json")
    
####################################
# API Endpoint - /api/enginereload #
####################################
import asyncio
uvicorn_server = None
def restart_self():
    """Restart the current script."""
    #print(f"[{config.branding}ENG] Restarting subprocess...")
    os.execv(sys.executable, ['python'] + sys.argv)

async def handle_restart():
    global uvicorn_server
    if uvicorn_server:
        print(f"[{config.branding}ENG] Stopping uvicorn server...")
        uvicorn_server.should_exit = True
        uvicorn_server.force_exit = True
        # Check if the server has stopped
        while not uvicorn_server.is_stopped:
            await asyncio.sleep(0.1)  # Check every 100ms if server is stopped
    restart_self()

@app.post("/api/enginereload")
async def apifunction_reload(request: Request):
    requested_engine = request.query_params.get("engine")
    if not tts_engines_config.is_valid_engine(requested_engine):
        return {"status": "error", "message": "Invalid TTS engine specified"}
    
    print(f"[{config.branding}ENG]")
    print(f"[{config.branding}ENG] \033[94mChanging model loaded. Please wait.\033[00m")
    print(f"[{config.branding}ENG]")

    # Update the tts_engines_data with the requested engine:
    try:
        tts_engines_config.change_engine(requested_engine).save()
    finally:
        # Reload just to be sure the right config is used (e.g. in case of an exception):
        tts_engines_config.reload()

    # Start the restart process in the background
    asyncio.create_task(handle_restart())

    return Response(content=json.dumps({"status": "engine-success"}), media_type="application/json")


#######################################
# API Endpoint - /api/stop-generation #
#######################################
# When this endpoint it called it will set tts_stop_generation in the model_engine to True, which can be used to interrupt the current TTS generation.
@app.put("/api/stop-generation")
async def apifunction_stop_generation():
    if model_engine.tts_generating_lock and not model_engine.tts_stop_generation:
        model_engine.tts_stop_generation = True
    return {"message": "Cancelling current TTS generation"}

#############################
# API Endpoint - /api/audio #
#############################
# Gives web access to the output files as downloads
@app.get("/audio/{filename}")
async def apifunction_get_audio(filename: str):
    audio_path = this_dir / config.get_output_directory() / filename
    return FileResponse(audio_path)

##################################
# API Endpoint - /api/audiocache #
##################################
@app.get("/audiocache/{filename}")
async def apifunction_get_audio(filename: str):
    audio_path = Path({config.get_output_directory()}) / filename
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    response = FileResponse(path=audio_path, media_type='audio/wav', filename=filename)
    # Set caching headers
    response.headers["Cache-Control"] = "public, max-age=604800"  # Cache for one week
    response.headers["ETag"] = str(audio_path.stat().st_mtime)  # Use the file's last modified time as a simple ETag
    return response

##############################
# API Endpoint - /api/voices #
##############################
@app.get("/api/voices")
async def apifunction_get_voices():
    if not model_engine.multivoice_capable:
        return {"status": "error", "message": f"The currently loaded TTS engine '{model_engine.engine_loaded}' does not support multiple voices."}
    
    available_voices = model_engine.voices_file_list()
    return {"status": "success", "voices": available_voices}

#################################
# API Endpoint - /api/rvcvoices #
#################################
@app.get("/api/rvcvoices")
async def apifunction_get_voices():
    load_config()
    if not config.rvc_settings.rvc_enabled:
        return {"status": "success", "rvcvoices": ["Disabled"]}  
    # Define the directory path
    directory = os.path.join(this_dir, "models", "rvc_voices")
    # Check if the directory exists
    if not os.path.exists(directory):
        return {"status": "success", "rvcvoices": ["Disabled"]}
    # List all .pth files recursively in the directory
    pth_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pth"):
                pth_files.append(os.path.relpath(os.path.join(root, file), start=directory))
    # If no .pth files found, return "Disabled"
    if not pth_files:
        return {"status": "success", "rvcvoices": ["Disabled"]}
    # Sort the list alphabetically
    pth_files.sort()
    # Add "Disabled" at the beginning of the list
    pth_files.insert(0, "Disabled")
    return {"status": "success", "rvcvoices": pth_files}

#####################################
# API Endpoint - /api/reload_config #
#####################################
@app.get("/api/reload_config")
async def apifunction_reload_config():
    load_config(True)
    return Response("Config file reloaded successfully")

#############################
# API Endpoint - /api/ready #
#############################
@app.get("/api/ready")
async def apifunction_ready():
    if model_engine.setup_has_run:
        return Response("Ready")
    else:
        return Response("Unloaded")

#######################################
# API Endpoint - /api/currentsettings #
#######################################
@app.get('/api/currentsettings')
def apifunction_get_current_settings():
    settings = {
        "engines_available": tts_engines_config.get_engine_names_available(),                                 # Lists the currently available TTS engines
        "current_engine_loaded": model_engine.engine_loaded,                    # Lists the currently loaded TTS engine
        "models_available": [{"name": name} for name in model_engine.available_models.keys()],  # Lists the currently available models that can be loaded on this TTS engine.
        "current_model_loaded": model_engine.current_model_loaded,              # Lists the currently loaded model
        "manufacturer_name": model_engine.manufacturer_name,                    # The name of the person/company/group who made the current TTS engine
        "audio_format": model_engine.audio_format,                              # The default audio format the the currently loaded TTS engine generates in
        "deepspeed_capable": model_engine.deepspeed_capable,                    # Is the TTS engine capable of supporting DeepSpeed
        "deepspeed_available": model_engine.deepspeed_available,                # Was DeepSpeed detected as being available for use
        "deepspeed_enabled": model_engine.deepspeed_enabled,                    # Is DeepSpeed current enabled or disabled
        "generationspeed_capable": model_engine.generationspeed_capable,        # Does the TTS engine support changing speed of generated TTS
        "generationspeed_set": model_engine.generationspeed_set,                # What is the currently set TTS speed
        "lowvram_capable": model_engine.lowvram_capable,                        # Does the TTS engine support moving the model between VRAM and System Ram
        "lowvram_enabled": model_engine.lowvram_enabled,                        # What is the currently enabled/disabled state of low vram
        "pitch_capable": model_engine.pitch_capable,                            # Does the TTS engine support pitch on tts generation
        "pitch_set": model_engine.pitch_set,                                    # What is the currently set value for pitch
        "repetitionpenalty_capable": model_engine.repetitionpenalty_capable,    # Does the TTS engine support differnt repitition penalties
        "repetitionpenalty_set": model_engine.repetitionpenalty_set,            # What is the currently set repetition penalty
        "streaming_capable": model_engine.streaming_capable,                    # Does the model support streaming TTS generation
        "temperature_capable": model_engine.temperature_capable,                # Does the TTS engine support model temperature settings
        "temperature_set": model_engine.temperature_set,                        # What is the currently set temperature
        "ttsengines_installed": model_engine.engine_installed,                  # 
        "languages_capable": model_engine.languages_capable,                    # Does the TTS engine support multi languages
        "multivoice_capable": model_engine.multivoice_capable,                  # Does the TTS engine support multi voice models
        "multimodel_capable": model_engine.multimodel_capable,                  # Does the engine support loading many different models
    }
    return settings

######################################
# API Endpoint - /api/lowvramsetting #
######################################    
@app.post("/api/lowvramsetting")
async def apifunction_low_vram(request: Request, new_low_vram_value: bool):
    if not model_engine.lowvram_capable:
        return {"status": "error", "message": f"The currently loaded TTS engine '{model_engine.engine_loaded}' does not support lowvram."}
    try:
        if new_low_vram_value is None:
            raise ValueError("Missing 'low_vram' parameter")
        if model_engine.lowvram_enabled == new_low_vram_value:
            return Response(content=json.dumps({"status": "success", "message": f"[{config.branding}Model] LowVRAM is already {'enabled' if new_low_vram_value else 'disabled'}.",}))
        model_engine.lowvram_enabled = new_low_vram_value
        await model_engine.unload_model()
        
        if model_engine.cuda_is_available:
            if model_engine.lowvram_enabled:
                model_engine.device = "cpu"
                print(f"[{config.branding}Engine] \033[94mLowVRAM Enabled.\033[0m Model will move between \033[93mVRAM(cuda) <> System RAM(cpu)\033[0m")
            else:
                model_engine.device = "cuda"
                print(f"[{config.branding}Engine] \033[94mLowVRAM Disabled.\033[0m Model will stay in \033[93mVRAM(cuda)\033[0m")
            await model_engine.setup()
        else:
            # Handle the case where CUDA is not available
            print(f"[{config.branding}Engine] \033[91mError:\033[0m Nvidia CUDA is not available on this system. Unable to use LowVRAM mode.")
            model_engine.lowvram_enabled = False
        
        return Response(content=json.dumps({"status": "lowvram-success"}))
    except Exception as e:
        return Response(content=json.dumps({"status": "error", "message": str(e)}))

#################################
# API Endpoint - /api/deepspeed #
#################################
@app.post("/api/deepspeed")
async def deepspeed(request: Request, new_deepspeed_value: bool):
    if not model_engine.deepspeed_capable or not model_engine.deepspeed_available:
        return {"status": "error", "message": f"The currently loaded TTS engine '{model_engine.engine_loaded}' does not support DeepSpeed or DeepSpeed is not available on this system."}
    try:
        if new_deepspeed_value is None:
            raise ValueError("Missing 'deepspeed' parameter")
        if model_engine.deepspeed_enabled == new_deepspeed_value:
            return Response(
                content=json.dumps({"status": "success", "message": f"DeepSpeed is already {'enabled' if new_deepspeed_value else 'disabled'}.",}))
        model_engine.deepspeed_enabled = new_deepspeed_value
        await model_engine.handle_deepspeed_change(new_deepspeed_value)
        return Response(content=json.dumps({"status": "deepspeed-success"}))
    except Exception as e:
        return Response(content=json.dumps({"status": "error", "message": str(e)}))

#################################
# API Endpoint - /api/voice2rvc #
#################################
@app.post("/api/voice2rvc")
async def voice2rvc(input_tts_path: str = Form(...), output_rvc_path: str = Form(...), pth_name: str = Form(...), pitch: str = Form(...), method: str = Form(...)):
    try:
        # Handle "Disabled" case
        if pth_name.lower() in ["disabled", "disable"]:
            print(f"[{config.branding}TTS] \033[94mVoice2RVC Convert: No voice was specified or the name was Disabled\033[0m")
            return {"status": "error", "message": "No voice was specified or the name was Disabled"}
        input_tts_path = Path(input_tts_path)
        output_rvc_path = Path(output_rvc_path) 
        # Check if input file exists
        if not input_tts_path.is_file():
            raise HTTPException(status_code=400, detail=f"Input file {input_tts_path} does not exist.")
        # Define the model path based on the selected model name
        pth_path = this_dir / "models" / "rvc_voices" / pth_name
        # Check if model file exists
        if not pth_path.is_file():
            raise HTTPException(status_code=400, detail=f"Model file {pth_path} does not exist.")
        # Run the RVC conversion
        result_path = run_voice2rvc(input_tts_path, output_rvc_path, pth_path, pitch, method)
        if result_path:
            return {"status": "success", "output_path": str(result_path)}
        else:
            raise HTTPException(status_code=500, detail="RVC conversion failed.")
    except Exception as e:
        print(f"Error during Voice2RVC conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Define the run_rvc function
def run_voice2rvc(input_tts_path, output_rvc_path, pth_path, pitch, method):
    print(f"[{config.branding}GEN] \033[94mVoice2RVC Convert: Started\033[0m")
    generate_start_time = time.time()
    f0up_key = pitch
    filter_radius = config.rvc_settings.filter_radius
    index_rate = config.rvc_settings.index_rate
    rms_mix_rate = config.rvc_settings.rms_mix_rate
    protect = config.rvc_settings.protect
    hop_length = config.rvc_settings.hop_length
    f0method = method
    split_audio = config.rvc_settings.split_audio
    f0autotune = config.rvc_settings.autotune
    embedder_model = config.rvc_settings.embedder_model
    training_data_size = config.rvc_settings.training_data_size
    # Convert path variables to strings
    input_tts_path = str(input_tts_path)
    pth_path = str(pth_path)
    # Check if the model file exists
    if not os.path.isfile(pth_path):
        print(f"Model file {pth_path} does not exist. Exiting.")
        return
    # Get the directory of the model file
    model_dir = os.path.dirname(pth_path)
    # Find all .index files in the model directory
    index_files = [file for file in os.listdir(model_dir) if file.endswith(".index")]
    if len(index_files) == 1:
        index_path = str(os.path.join(model_dir, index_files[0]))
    else:
        index_path = ""
    # Call the infer_pipeline function
    infer_pipeline(f0up_key, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method,
                input_tts_path, output_rvc_path, pth_path, index_path, split_audio, f0autotune, embedder_model, training_data_size, config.debugging.debug_rvc)
    generate_end_time = time.time()
    generate_elapsed_time = generate_end_time - generate_start_time
    print(f"[{config.branding}GEN] \033[94mVoice2RVC Convert: \033[91m{generate_elapsed_time:.2f} seconds.\033[0m")
    return output_rvc_path

##################################
# Transcode between file formats #
##################################
async def transcode_audio(input_file, output_format):
    print(f"[{config.branding}Debug] *************************************************",) if config.debugging.debug_transcode else None
    print(f"[{config.branding}Debug] transcode_audio function called (debug_transcode)",) if config.debugging.debug_transcode else None
    print(f"[{config.branding}Debug] *************************************************",) if config.debugging.debug_transcode else None
    print(f"[{config.branding}Debug] Input file    : {input_file}") if config.debugging.debug_transcode else None
    print(f"[{config.branding}Debug] Output format : {output_format}") if config.debugging.debug_transcode else None
    if output_format == "Disabled":
        print(f"[{config.branding}Debug] Transcode format is set to Disabled so skipping transcode.") if config.debugging.debug_transcode else None
        return input_file
    if not ffmpeg_installed:
        print(f"[{config.branding}TTS] FFmpeg is not installed. Format conversion is not possible.")
        raise Exception("FFmpeg is not installed. Format conversion is not possible.")
    # Get the file extension of the input file
    input_extension = os.path.splitext(input_file)[1][1:].lower()
    print(f"[{config.branding}Debug] Input file extension: {input_extension}") if config.debugging.debug_transcode else None
    # Check if the input extension matches the requested output format
    if input_extension == output_format.lower():
        print(f"[{config.branding}Debug] Input file is already in the requested format: {output_format}") if config.debugging.debug_transcode else None
        return input_file
    output_file = os.path.splitext(input_file)[0] + f".{output_format}"
    print(f"[{config.branding}Debug] Output file: {output_file}") if config.debugging.debug_transcode else None
    ffmpeg_path = "ffmpeg"  # Default path for Linux and macOS
    if sys.platform == "win32":
        ffmpeg_path = os.path.join(this_dir, "system", "win_ffmpeg", "ffmpeg.exe")
    ffmpeg = (FFmpeg(ffmpeg_path).option("y").input(input_file).output(output_file))
    print(f"[{config.branding}Debug] Transcoding to {output_format}") if config.debugging.debug_transcode else None
    if output_format == "opus":
        print(f"[{config.branding}Debug] Configuring Opus options") if config.debugging.debug_transcode else None
        ffmpeg.output(output_file, {"codec:a": "libopus", "b:a": "128k", "vbr": "on", "compression_level": 10, "frame_duration": 60, "application": "voip"})
    elif output_format == "aac":
        print(f"[{config.branding}Debug] Configuring AAC options") if config.debugging.debug_transcode else None
        ffmpeg.output(output_file, {"codec:a": "aac", "b:a": "192k"})
    elif output_format == "flac":
        print(f"[{config.branding}Debug] Configuring FLAC options") if config.debugging.debug_transcode else None
        ffmpeg.output(output_file, {"codec:a": "flac", "compression_level": 8})
    elif output_format == "wav":
        print(f"[{config.branding}Debug] Configuring WAV options") if config.debugging.debug_transcode else None
        ffmpeg.output(output_file, {"codec:a": "pcm_s16le"})
    elif output_format == "mp3":
        print(f"[{config.branding}Debug] Configuring MP3 options") if config.debugging.debug_transcode else None
        ffmpeg.output(output_file, {"codec:a": "libmp3lame", "b:a": "192k"})
    else:
        print(f"[{config.branding}TTS] Unsupported output format: {output_format}")
        raise ValueError(f"Unsupported output format: {output_format}")
    try:
        print(f"[{config.branding}Debug] Starting transcoding process") if config.debugging.debug_transcode else None
        await ffmpeg.execute()
        print(f"[{config.branding}Debug] Transcoding completed successfully") if config.debugging.debug_transcode else None
    except Exception as e:
        print(f"[{config.branding}TTS] Error occurred during transcoding: {str(e)}")
        raise
    print(f"[{config.branding}Debug] Deleting original input file") if config.debugging.debug_transcode else None
    os.remove(input_file)
    print(f"[{config.branding}Debug] Transcoding process completed") if config.debugging.debug_transcode else None
    print(f"[{config.branding}Debug] Transcoded file: {output_file}") if config.debugging.debug_transcode else None
    return output_file

##############################
# Central Transcode function #
##############################
async def transcode_audio_if_necessary(output_file, model_audio_format, output_audio_format):
    if config.debugging.debug_transcode:
        print(f"[{config.branding}Debug] model_engine.audio_format is:", model_audio_format)
        print(f"[{config.branding}Debug] audio format is:", output_audio_format)
        print(f"[{config.branding}Debug] Entering the transcode condition")
    try:
        if config.debugging.debug_transcode:
            print(f"[{config.branding}Debug] Calling transcode_audio function")
        output_file = await transcode_audio(output_file, output_audio_format)
        if config.debugging.debug_transcode:
            print(f"[{config.branding}Debug] Transcode completed successfully")
    except Exception as e:
        print(f"[{config.branding}Debug] Error occurred during transcoding:", str(e))
        raise
    if config.debugging.debug_transcode:
        print(f"[{config.branding}Debug] Transcode condition completed")
    # Update the output file path and URLs based on the transcoded file
    if config.debugging.debug_tts:
        print(f"[{config.branding}Debug] Updating output file paths and URLs")
    if config.api_def.api_use_legacy_api:
        output_file_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audio/{os.path.basename(output_file)}'
        output_cache_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audiocache/{os.path.basename(output_file)}'
    else:
        output_file_url = f'/audio/{os.path.basename(output_file)}'
        output_cache_url = f'/audiocache/{os.path.basename(output_file)}'
    if config.debugging.debug_tts:
        print(f"[{config.branding}Debug] Output file paths and URLs updated")
    return output_file, output_file_url, output_cache_url

##############################
#### Streaming Generation ####
##############################
@app.get("/api/tts-generate-streaming", response_class=StreamingResponse)
async def apifunction_generate_streaming(text: str, voice: str, language: str, output_file: str):
    if model_engine.streaming_capable == False:
        print(f"[{config.branding}GEN] The selected TTS Engine does not support streaming. To use streaming, please select a TTS")
        print(f"[{config.branding}GEN] Engine that has streaming capability. You can find the streaming support information for")
        print(f"[{config.branding}GEN] each TTS Engine in the 'Engine Information' section of the Gradio interface.")
    try:
        output_file_path = f'{this_dir / config.get_output_directory() / output_file}.{model_engine.audio_format}'
        stream = await generate_audio(text, voice, language, model_engine.temperature_set, model_engine.repetitionpenalty_set, 1.0, 1.0, output_file_path, streaming=True)
        return StreamingResponse(stream, media_type="audio/wav")
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

@app.post("/api/tts-generate-streaming", response_class=JSONResponse)
async def tts_generate_streaming(request: Request, text: str = Form(...), voice: str = Form(...), language: str = Form(...), output_file: str = Form(...)):
    if model_engine.streaming_capable == False:
        print(f"[{config.branding}GEN] The selected TTS Engine does not support streaming. To use streaming, please select a TTS")
        print(f"[{config.branding}GEN] Engine that has streaming capability. You can find the streaming support information for")
        print(f"[{config.branding}GEN] each TTS Engine in the 'Engine Information' section of the Gradio interface.")
    try:
        output_file_path = f'{this_dir / config.get_output_directory() / output_file}.{model_engine.audio_format}'
        await generate_audio(text, voice, language, model_engine.temperature_set, model_engine.repetitionpenalty_set, "1.0", "1.0", output_file_path, streaming=False)
        return JSONResponse(content={"output_file_path": str(output_file)}, status_code=200)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

###################################
# Central generate_audio function #
###################################
async def generate_audio(text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming=False):
    if model_engine.streaming_capable == False and streaming==True:
        print(f"[{config.branding}GEN] The selected TTS Engine does not support streaming. To use streaming, please select a TTS")
        print(f"[{config.branding}GEN] Engine that has streaming capability. You can find the streaming support information for")
        print(f"[{config.branding}GEN] each TTS Engine in the 'Engine Information' section of the Gradio interface.")
    # Get the async generator from the internal function
    response = model_engine.generate_tts(text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming)
    # If streaming, then return the generator as-is, otherwise just exhaust it and return
    if streaming:
        # Return an async generator that yields from the response
        async def stream_response():
            async for chunk in response:
                yield chunk
        return stream_response()
    else:
        # Exhaust the generator and handle any errors
        try:
            async for _ in response:
                pass
        except Exception as e:
            print(f"{config.branding}[GEN] Error during audio generation: {str(e)}")
            raise

###########################
#### PREVIEW VOICE API ####
###########################
@app.post("/api/previewvoice/", response_class=JSONResponse)
async def apifunction_preview_voice(
    request: Request,
    voice: str = Form(...),
    rvccharacter_voice_gen: Optional[str] = Form(None),
    rvccharacter_pitch: Optional[float] = Form(None)
):
    try:
        # Hardcoded settings
        language = "en"
        output_file_name = "api_preview_voice"       
        # Clean the voice filename for inclusion in the text
        clean_voice_filename = re.sub(r'\.wav$', '', voice.replace(' ', '_'))
        clean_voice_filename = re.sub(r'[^a-zA-Z0-9]', ' ', clean_voice_filename)
        # Generate the audio text
        text = f"Hello, this is a preview of voice {clean_voice_filename}."
        # Set default values for new parameters if not provided
        rvccharacter_voice_gen = rvccharacter_voice_gen or "Disabled"
        rvccharacter_pitch = rvccharacter_pitch if rvccharacter_pitch is not None else 0
        # Generate the audio
        output_file_path = this_dir / config.get_output_directory() / f'{output_file_name}.{model_engine.audio_format}'
        await generate_audio(
            text, 
            voice, 
            language, 
            model_engine.temperature_set, 
            model_engine.repetitionpenalty_set, 
            1.0, 
            0, 
            output_file_path, 
            streaming=False
            )
        if config.rvc_settings.rvc_enabled:
            if rvccharacter_voice_gen.lower() in ["disabled", "disable"]:
                print(f"[{config.branding}Debug] PREVIEW VOICE - Pass rvccharacter_voice_gen") if config.debugging.debug_tts else None
                pass  # Skip RVC processing for character part
            else:
                print(f"[{config.branding}Debug] PREVIEW VOICE - send to rvc") if config.debugging.debug_tts else None
                rvccharacter_voice_gen = this_dir / "models" / "rvc_voices" / rvccharacter_voice_gen
                pth_path = rvccharacter_voice_gen if rvccharacter_voice_gen else config.rvc_settings.rvc_char_model_file
                run_rvc(output_file_path, pth_path, rvccharacter_pitch, infer_pipeline)        
        # Generate the URL
        output_file_url = f'/audio/{output_file_name}.{model_engine.audio_format}'
        # Return the response with both local file path and URL
        return JSONResponse(
            content={
                "status": "generate-success",
                "output_file_path": str(output_file_path),
                "output_file_url": str(output_file_url),
            },
            status_code=200,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

##################################################################
# API Endpoint - OpenAI Speech API compatable endpoint Validator #
##################################################################
class OpenAIInput(BaseModel):
    model: str = Field(..., description="The TTS model to use. Currently ignored.")
    input: str = Field(..., max_length=4096, description="The text to generate audio for.")
    voice: str = Field(..., description="The voice to use when generating the audio.")
    response_format: str = Field(default="wav", description="The format of the audio. Currently only 'wav' is supported.")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="The speed of the generated audio.")

    @field_validator('voice')
    def validate_voice(cls, value):
        supported_voices = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]
        if value not in supported_voices:
            raise ValueError(f'Voice must be one of {supported_voices}')
        return value

class OpenAIGenerator:
    @staticmethod
    def validate_input(json_data: dict) -> Union[None, str]:
        try:
            OpenAIInput(**json_data)
            return None
        except ValidationError as e:
            errors = []
            for err in e.errors():
                field = err['loc'][0]
                message = err['msg']
                description = OpenAIInput.model_fields[field].field_info.description
                errors.append(f"Error in field '{field}': {message}. Description: {description}")
            return ', '.join(errors)

#########################################################################
# API Endpoint - OpenAI Speech API compatable endpoint /v1/audio/speech #
#########################################################################
@app.post("/v1/audio/speech", response_class=JSONResponse)
async def openai_tts_generate(request: Request):
    try:
        json_data = await request.json()
        print(f"[{config.branding}Debug] Received JSON data: {json_data}")  if config.debugging.debug_openai else None
        validation_error = OpenAIGenerator.validate_input(json_data)
        if validation_error:
            print(f"Validation error: {validation_error}")  if config.debugging.debug_openai else None
            return JSONResponse(content={"error": validation_error}, status_code=400)
        model = json_data["model"]  # Currently ignored
        input_text = json_data["input"]
        voice = json_data["voice"]
        response_format = json_data.get("response_format", "wav").lower()
        speed = json_data.get("speed", 1.0)
        print(f"[{config.branding}Debug] Input text: {input_text}")  if config.debugging.debug_openai else None
        print(f"[{config.branding}Debug] Voice: {voice}")  if config.debugging.debug_openai else None
        print(f"[{config.branding}Debug] Speed: {speed}")  if config.debugging.debug_openai else None
        cleaned_string = html.unescape(standard_filtering(input_text))
        # Map the OpenAI voice to the corresponding internal voice
        voice_mapping = {
            "alloy": model_engine.openai_alloy,
            "echo": model_engine.openai_echo,
            "fable": model_engine.openai_fable,
            "nova": model_engine.openai_nova,
            "onyx": model_engine.openai_onyx,
            "shimmer": model_engine.openai_shimmer
        }
        mapped_voice = voice_mapping.get(voice)
        if not mapped_voice:
            print(f"Unsupported voice: {voice}")
            return JSONResponse(content={"error": "Unsupported voice"}, status_code=400)
        print(f"[{config.branding}Debug]Mapped voice: {mapped_voice}")  if config.debugging.debug_openai else None
        # Generate the audio
        # Generate a unique filename
        unique_id = uuid.uuid4()
        timestamp = int(time.time())
        output_file_path = f'{this_dir / config.get_output_directory() / f"openai_output_{unique_id}_{timestamp}.{model_engine.audio_format}"}'
        await generate_audio(cleaned_string, mapped_voice, "en", model_engine.temperature_set,
                             model_engine.repetitionpenalty_set, speed, model_engine.pitch_set,
                             output_file_path, streaming=False)
        print(f"[{config.branding}Debug] Audio generated at: {output_file_path}")  if config.debugging.debug_openai else None
        if config.rvc_settings.rvc_enabled:
            if config.rvc_settings.rvc_char_model_file.lower() in ["disabled", "disable"]:
                print(f"[{config.branding}Debug] Pass rvccharacter_voice_gen") if config.debugging.debug_openai else None
                pass  # Skip RVC processing for character part
            else:
                print(f"[{config.branding}Debug] send to rvc") if config.debugging.debug_openai else None
                pth_path = this_dir / "models" / "rvc_voices" / config.rvc_settings.rvc_char_model_file
                pitch = config.rvc_settings.pitch
                run_rvc(output_file_path, pth_path, pitch, infer_pipeline)
        # Transcode the audio to the requested format
        transcoded_file_path = await transcode_for_openai(output_file_path, response_format)
        print(f"[{config.branding}Debug] Audio transcoded to: {transcoded_file_path}")  if config.debugging.debug_openai else None
        # Return the audio file
        return FileResponse(transcoded_file_path, media_type=f"audio/{response_format}", filename=f"output.{response_format}")
    except Exception as e:
        print(f"[{config.branding}Debug] An error occurred: {str(e)}")  if config.debugging.debug_openai else None
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

###########################################################################
# API Endpoint - OpenAI Speech API compatable endpoint Transcode Function #
###########################################################################
async def transcode_for_openai(input_file, output_format):
    print(f"[{config.branding}Debug] ************************************") if config.debugging.debug_openai else None
    print(f"[{config.branding}Debug] transcode_for_openai function called") if config.debugging.debug_openai else None
    print(f"[{config.branding}Debug] ************************************") if config.debugging.debug_openai else None
    print(f"[{config.branding}Debug] Input file    : {input_file}") if config.debugging.debug_openai else None
    print(f"[{config.branding}Debug] Output format : {output_format}") if config.debugging.debug_openai else None
    if not ffmpeg_installed:
        print(f"[{config.branding}TTS] FFmpeg is not installed. Format conversion is not possible.")
        raise Exception("FFmpeg is not installed. Format conversion is not possible.")
    # Get the file extension of the input file
    input_extension = os.path.splitext(input_file)[1][1:].lower()
    print(f"[{config.branding}Debug] Input file extension: {input_extension}") if config.debugging.debug_openai else None
    # Check if the input extension matches the requested output format
    if input_extension == output_format.lower():
        print(f"[{config.branding}Debug] Input file is already in the requested format: {output_format}") if config.debugging.debug_openai else None
        return input_file
    output_file = os.path.splitext(input_file)[0] + f".{output_format}"
    print(f"[{config.branding}Debug] Output file: {output_file}") if config.debugging.debug_openai else None
    ffmpeg_path = "ffmpeg"  # Default path for Linux and macOS
    if sys.platform == "win32":
        ffmpeg_path = os.path.join(this_dir, "system", "win_ffmpeg", "ffmpeg.exe")
    ffmpeg = (FFmpeg(ffmpeg_path).option("y").input(input_file).output(output_file))
    print(f"[{config.branding}Debug] Transcoding to {output_format}") if config.debugging.debug_openai else None
    if output_format == "opus":
        print(f"[{config.branding}Debug] Configuring Opus options") if config.debugging.debug_openai else None
        ffmpeg.output(output_file, {"codec:a": "libopus", "b:a": "128k", "vbr": "on", "compression_level": 10, "frame_duration": 60, "application": "voip"})
    elif output_format == "aac":
        print(f"[{config.branding}Debug] Configuring AAC options") if config.debugging.debug_openai else None
        ffmpeg.output(output_file, {"codec:a": "aac", "b:a": "192k"})
    elif output_format == "flac":
        print(f"[{config.branding}Debug] Configuring FLAC options") if config.debugging.debug_openai else None
        ffmpeg.output(output_file, {"codec:a": "flac", "compression_level": 8})
    elif output_format == "wav":
        print(f"[{config.branding}Debug] Configuring WAV options") if config.debugging.debug_openai else None
        ffmpeg.output(output_file, {"codec:a": "pcm_s16le"})
    elif output_format == "mp3":
        print(f"[{config.branding}Debug] Configuring MP3 options") if config.debugging.debug_openai else None
        ffmpeg.output(output_file, {"codec:a": "libmp3lame", "b:a": "192k"})
    elif output_format in ["ogg", "m4a"]:
        if output_format == "ogg":
            print(f"[{config.branding}Debug] Configuring OGG options") if config.debugging.debug_openai else None
            ffmpeg.output(output_file, {"codec:a": "libvorbis"})
        elif output_format == "m4a":
            print(f"[{config.branding}Debug] Configuring M4A options") if config.debugging.debug_openai else None
            ffmpeg.output(output_file, {"codec:a": "aac", "b:a": "192k"})
    else:
        print(f"[{config.branding}TTS] Unsupported output format: {output_format}")
        raise ValueError(f"Unsupported output format: {output_format}")
    try:
        print(f"[{config.branding}Debug] Starting transcoding process") if config.debugging.debug_openai else None
        await ffmpeg.execute()
        print(f"[{config.branding}Debug] Transcoding completed successfully") if config.debugging.debug_openai else None
    except Exception as e:
        print(f"[{config.branding}TTS] Error occurred during transcoding: {str(e)}")
        raise
    print(f"[{config.branding}Debug] Transcoding process completed") if config.debugging.debug_openai else None
    print(f"[{config.branding}Debug] Transcoded file: {output_file}") if config.debugging.debug_openai else None
    return output_file

######################################################################################
# API Endpoint - OpenAI Speech API compatable endpoint change engine voices Function #
######################################################################################
class VoiceMappings(BaseModel):
    alloy: str
    echo: str
    fable: str
    nova: str
    onyx: str
    shimmer: str

@app.put("/api/openai-voicemap")
async def update_openai_voice_mappings(mappings: VoiceMappings):
    # Update in-memory versions (wont display in Gradio without a restart as gradio caches)
    model_engine.openai_alloy = mappings.alloy
    model_engine.openai_echo = mappings.echo
    model_engine.openai_fable = mappings.fable
    model_engine.openai_nova = mappings.nova
    model_engine.openai_onyx = mappings.onyx
    model_engine.openai_shimmer = mappings.shimmer
    # Update model_settings.json file for the currently loaded TTS engine
    try:
        settings_file = this_dir / "system" / "tts_engines" / model_engine.engine_loaded / "model_settings.json"
        with open(settings_file, "r+") as f:
            settings = json.load(f)
            settings["openai_voices"] = mappings.dict()
            f.seek(0)
            json.dump(settings, f, indent=4)
            f.truncate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model settings file: {str(e)}")
    return {"message": "OpenAI voice mappings updated successfully"}

#######################
# Play at the console #
#######################
def play_audio(file_path, volume):
    print(f"[{config.branding}GEN] \033[94m File path is: {file_path}") if config.debugging.debug_tts else None
    print(f"[{config.branding}GEN] \033[94m Volume is: {volume}") if config.debugging.debug_tts else None
    normalized_file_path = os.path.normpath(file_path)
    print(f"[{config.branding}GEN] \033[94m Normalized file path is: {normalized_file_path}") if config.debugging.debug_tts else None
    directory = os.path.dirname(normalized_file_path)
    if os.path.isdir(directory):
        print(f"[{config.branding}GEN] \033[94m Directory contents: {os.listdir(directory)}") if config.debugging.debug_tts else None
    else:
        print(f"[{config.branding}GEN] \033[94mError: Directory does not exist: {directory}") if config.debugging.debug_tts else None
    if not os.path.isfile(normalized_file_path):
        print(f"[{config.branding}GEN] \033[94mError: File does not exist: {normalized_file_path}\033[0m") if config.debugging.debug_tts else None
        return
    # Check for AAC format
    if normalized_file_path.lower().endswith('.aac'):
        print(f"[{config.branding}GEN] \033[94mPlay Audio  : \033[0mAAC format files cannot be played at the console. Please choose another format")
        return
    try:
        print(f"[{config.branding}GEN] \033[94mPlay Audio  : \033[0mPlaying audio at console")
        data, fs = sf.read(normalized_file_path)
        sd.play(volume * data, fs)
        sd.wait()
    except Exception as e:
        print(f"[{config.branding}GEN] \033[94mError playing audio file: {e}\033[0m")

class Request(BaseModel):
    # Define the structure of the 'Request' class if needed
    pass


class TTSGenerator:
    @staticmethod
    def validate_json_input(json_data: Union[Dict, str]) -> Union[None, str]:
        try:
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            JSONInput(**json_data)
            return None  # JSON is valid
        except ValidationError as e:
            return str(e)

#################################################################
# /api/tts-generate Generation API Endpoint Narration Filtering #
#################################################################
def process_text(text):
    # Normalize HTML encoded quotes
    text = html.unescape(text)
    # Replace ellipsis with a single dot
    text = re.sub(r'\.{3,}', '.', text)
    # Pattern to identify combined narrator and character speech
    combined_pattern = r'(\*[^*"]+\*|"[^"*]+")'
    # List to hold parts of speech along with their type
    ordered_parts = []
    # Track the start of the next segment
    start = 0
    # Find all matches
    for match in re.finditer(combined_pattern, text):
        # Add the text before the match, if any, as ambiguous
        if start < match.start():
            ambiguous_text = text[start:match.start()].strip()
            if ambiguous_text:
                ordered_parts.append(('ambiguous', ambiguous_text))
        # Add the matched part as either narrator or character
        matched_text = match.group(0)
        if matched_text.startswith('*') and matched_text.endswith('*'):
            ordered_parts.append(('narrator', matched_text.strip('*').strip()))
        elif matched_text.startswith('"') and matched_text.endswith('"'):
            ordered_parts.append(('character', matched_text.strip('"').strip()))
        else:
            # In case of mixed or improperly formatted parts
            if '*' in matched_text:
                ordered_parts.append(('narrator', matched_text.strip('*').strip('"')))
            else:
                ordered_parts.append(('character', matched_text.strip('"').strip('*')))
        # Update the start of the next segment
        start = match.end()
    # Add any remaining text after the last match as ambiguous
    if start < len(text):
        ambiguous_text = text[start:].strip()
        if ambiguous_text:
            ordered_parts.append(('ambiguous', ambiguous_text))
    return ordered_parts

def standard_filtering(text_input):
    text_output = (text_input
                        .replace("***", "")
                        .replace("**", "")
                        .replace("*", "")
                        .replace("\n\n", "\n")
                        .replace("&#x27;", "'")
                        )
    return text_output

#################################################################
# /api/tts-generate Generation API Endpoint Narration Combining #
#################################################################
def combine(output_file_timestamp, output_file_name, audio_files, target_sample_rate=44100):
    audio = np.array([]) 
    try:
        for audio_file in audio_files:
            normalized_audio_file = os.path.normpath(audio_file)
            print(f"[{config.branding}Debug] Processing file: {normalized_audio_file}") if config.debugging.debug_concat else None
            # Check if the file exists
            if not os.path.isfile(normalized_audio_file):
                print(f"[{config.branding}Debug] Error: File does not exist: {normalized_audio_file}") if config.debugging.debug_concat else None
                return None, None
            # Read the audio file
            audio_data, current_sample_rate = sf.read(normalized_audio_file)
            print(f"[{config.branding}Debug] Read file: {normalized_audio_file}, Sample rate: {current_sample_rate}, Data shape: {audio_data.shape}") if config.debugging.debug_concat else None
            # Resample if necessary
            if current_sample_rate != target_sample_rate:
                print(f"[{config.branding}Debug] Resampling file from {current_sample_rate} to {target_sample_rate} Hz") if config.debugging.debug_concat else None
                audio_data = librosa.resample(audio_data, orig_sr=current_sample_rate, target_sr=target_sample_rate)
            # Concatenate audio data
            if audio.size == 0:
                audio = audio_data
            else:
                audio = np.concatenate((audio, audio_data))
        if output_file_timestamp:
            timestamp = int(time.time())
            output_file_path = os.path.join(this_dir / "outputs" / f'{output_file_name}_{timestamp}_combined.wav')
            sf.write(output_file_path, audio, target_sample_rate)
            # Legacy API or New API return
            if config.api_def.api_use_legacy_api:
                output_file_url = f"http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audio/{output_file_name}_{timestamp}_combined.wav"
                output_cache_url = f"http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audiocache/{output_file_name}_{timestamp}_combined.wav"
            else:
                output_file_url = f'/audio/{output_file_name}_{timestamp}_combined.wav'
                output_cache_url = f'/audiocache/{output_file_name}_{timestamp}_combined.wav'
        else:
            output_file_path = os.path.join(this_dir / "outputs" / f'{output_file_name}_combined.wav')
            sf.write(output_file_path, audio, target_sample_rate)
            # Legacy API or New API return
            if config.api_def.api_use_legacy_api:
                output_file_url = f"http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audio/{output_file_name}_combined.wav"
                output_cache_url = f"http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audiocache/{output_file_name}_combined.wav"
            else:
                output_file_url = f'/audio/{output_file_name}_combined.wav'
                output_cache_url = f'/audiocache/{output_file_name}_combined.wav'
        print(f"[{config.branding}Debug] Output file changed to:", output_file_path) if config.debugging.debug_concat else None
        print(f"[{config.branding}Debug] Output file changed to:", output_file_url) if config.debugging.debug_concat else None
        print(f"[{config.branding}Debug] Output file changed to:", output_cache_url) if config.debugging.debug_concat else None
    except Exception as e:
        print(f"Error occurred: {e}")
    return output_file_path, output_file_url, output_cache_url

##################################
# Central RVC Generation request #
##################################
def run_rvc(input_tts_path, pth_path, pitch, infer_pipeline):
    generate_start_time = time.time()
    f0up_key = pitch
    filter_radius = config.rvc_settings.filter_radius
    index_rate = config.rvc_settings.index_rate
    rms_mix_rate = config.rvc_settings.rms_mix_rate
    protect = config.rvc_settings.protect
    hop_length = config.rvc_settings.hop_length
    f0method = config.rvc_settings.f0method
    split_audio = config.rvc_settings.split_audio
    f0autotune = config.rvc_settings.autotune
    embedder_model = config.rvc_settings.embedder_model
    training_data_size = config.rvc_settings.training_data_size
    # Convert path variables to strings
    input_tts_path = str(input_tts_path)
    pth_path = str(pth_path)
    # Check if the model file exists
    if not os.path.isfile(pth_path):
        print(f"Model file {pth_path} does not exist. Exiting.")
        return 
    # Get the directory of the model file
    model_dir = os.path.dirname(pth_path)
    # Get the filename of pth_path
    pth_filename = os.path.basename(pth_path)
    # Find all .index files in the model directory
    index_files = [file for file in os.listdir(model_dir) if file.endswith(".index")]
    if len(index_files) == 1:
        index_path = str(os.path.join(model_dir, index_files[0]))
        # Get the filename of index_path
        index_filename = os.path.basename(index_path)
        index_filename_print = index_filename
        index_size_print = training_data_size
    elif len(index_files) > 1:
        print(f"[{config.branding}GEN] \033[94mRVC Convert :\033[0m Multiple RVC index files found in the models folder where \033[93m{pth_filename}\033[0m is")
        print(f"[{config.branding}GEN] \033[94mRVC Convert :\033[0m located. Unable to determine which index to use. Continuing without an index file.")
        index_path = ""
        index_filename = None
        index_filename_print = "None used"
        index_size_print = "N/A"
    else:
        index_path = ""
        index_filename = None
        index_filename_print = "None used"
        index_size_print = "N/A"
    # Set the output path to be the same as the input path
    output_rvc_path = input_tts_path
    if config.debugging.debug_rvc:
        debug_lines = [
            f"[{config.branding}Debug] ************************************",
            f"[{config.branding}Debug] run_rvc function called (debug_rvc)",
            f"[{config.branding}Debug] ***********************************",
            f"[{config.branding}Debug] f0up_key        : {f0up_key}",
            f"[{config.branding}Debug] filter_radius   : {filter_radius}",
            f"[{config.branding}Debug] index_rate      : {index_rate}",
            f"[{config.branding}Debug] rms_mix_rate    : {rms_mix_rate}",
            f"[{config.branding}Debug] protect         : {protect}",
            f"[{config.branding}Debug] hop_length      : {hop_length}",
            f"[{config.branding}Debug] f0method        : {f0method}",
            f"[{config.branding}Debug] input_tts_path  : {input_tts_path}",
            f"[{config.branding}Debug] output_rvc_path : {output_rvc_path}",
            f"[{config.branding}Debug] pth_path        : {pth_path}",
            f"[{config.branding}Debug] index_path      : {index_path}",
            f"[{config.branding}Debug] split_audio     : {split_audio}",
            f"[{config.branding}Debug] f0autotune      : {f0autotune}",
            f"[{config.branding}Debug] embedder_model  : {embedder_model}",
            f"[{config.branding}Debug] training_data_size: {training_data_size}"
        ]
        debug_info = "\n".join(debug_lines)
        print(debug_info)   
    # Call the infer_pipeline function
    infer_pipeline(f0up_key, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method,
                   input_tts_path, output_rvc_path, pth_path, index_path, split_audio, f0autotune, embedder_model, training_data_size, config.debugging.debug_rvc)
    generate_end_time = time.time()
    generate_elapsed_time = generate_end_time - generate_start_time
    print(f"[{config.branding}GEN] \033[94mRVC Convert : \033[94mModel: \033[93m{pth_filename} \033[94mIndex: \033[93m{index_filename_print}\033[0m")
    print(f"[{config.branding}GEN] \033[94mRVC Convert : \033[93m{generate_elapsed_time:.2f} seconds. \033[94mMethod: \033[93m{f0method} \033[94mIndex size used \033[93m{index_size_print}\033[0m")
    return

####################################################################
# /api/tts-generate Generation API Endpoint Narration RVC handling #
####################################################################
def process_rvc_narrator(part_type, voice_gen, pitch, default_model_file, output_file, infer_pipeline):
    if voice_gen.lower() in ["disabled", "disable"]:
        return
    else:
        voice_gen_path = this_dir / "models" / "rvc_voices" / voice_gen
        pth_path = voice_gen_path if voice_gen else default_model_file
        run_rvc(output_file, pth_path, pitch, infer_pipeline)

#############################################################
# /api/tts-generate Generation API Endpoint JSON validation #
#############################################################
class JSONInput(BaseModel):
    text_input: str = Field(..., max_length=int(config.api_def.api_max_characters), description=f"text_input needs to be {config.api_def.api_max_characters} characters or less.")
    text_filtering: str = Field(..., pattern="^(none|standard|html)$", description="text_filtering needs to be 'none', 'standard' or 'html'.")
    #character_voice_gen: str = Field(..., pattern=r'^[\(\)a-zA-Z0-9\_\-./\s]+$', description="character_voice_gen needs to be the name of a valid voice for the loaded TTS engine.")
    rvccharacter_voice_gen: str = Field(..., description="rvccharacter_voice_gen needs to be the name of a valid pth file in the 'folder\\file.pth' format or the word 'Disabled'.")
    rvccharacter_pitch: float = Field(..., description="RVC Character pitch needs to be a number between -24 and 24")
    narrator_enabled: str = Field(..., pattern="^(true|false|silent)$", description="narrator_enabled needs to be 'true', 'false' or 'silent'.")
    narrator_voice_gen: str = Field(..., pattern=r'^[\(\)a-zA-Z0-9\_\-./\s]+$', description="character_voice_gen needs to be the name of a valid voice for the loaded TTS engine.")
    rvcnarrator_voice_gen: str = Field(..., description="rvcnarrator_voice_gen needs to be the name of a valid pth file in the 'folder\\file.pth' format or the word 'Disabled'.")
    rvcnarrator_pitch: float = Field(..., description="RVC Narrator pitch needs to be a number between -24 and 24")   
    text_not_inside: str = Field(..., pattern="^(character|narrator|silent)$", description="text_not_inside needs to be 'character', 'narrator' or 'silent'.")
    language: str = Field(..., pattern="^(ar|zh-cn|zh|cs|nl|en|fr|de|hu|hi|it|ja|ko|pl|pt|ru|es|tr)$", description="language needs to be one of the following: ar, zh-cn, zh, cs, nl, en, fr, de, hu, hi, it, ja, ko, pl, pt, ru, es, tr.")
    output_file_name: str = Field(..., pattern="^[a-zA-Z0-9_]+$", description="output_file_name needs to be the name without any special characters or file extension, e.g., 'filename'.")
    output_file_timestamp: bool = Field(..., description="output_file_timestamp needs to be true or false.")
    autoplay: bool = Field(..., description="autoplay needs to be a true or false value.")
    autoplay_volume: float = Field(..., ge=0.1, le=1.0, description="autoplay_volume needs to be from 0.1 to 1.0.")
    speed: float = Field(..., ge=0.25, le=2.0, description="speed needs to be between 0.25 and 2.0.")
    pitch: float = Field(..., ge=-10, le=10, description="pitch needs to be between -10 and 10.")
    temperature: float = Field(..., ge=0.1, le=1.0, description="temperature needs to be between 0.1 and 1.0.")
    repetition_penalty: float = Field(..., ge=1.0, le=20.0, description="repetition_penalty needs to be between 1.0 and 20.0.")

    @classmethod
    def validate_autoplay_volume(cls, value):
        if not (0.1 <= value <= 1.0):
            raise ValueError("Autoplay volume must be between 0.1 and 1.0")
        return value

    def validate_rvccharacter_voice_gen(cls, v):
        if v.lower() == "disabled":
            return v
        pattern = re.compile(r'^.*\.(pth)$')
        if not pattern.match(v):
            raise ValueError("rvccharacter_voice_gen needs to be the name of a valid pth file in the 'folder\\file.pth' format or 'Disabled'.")
        return v

    def validate_rvnarrator_voice_gen(cls, v):
        if v.lower() == "disabled":
            return v
        pattern = re.compile(r'^.*\.(pth)$')
        if not pattern.match(v):
            raise ValueError("rvcnarrator_voice_gen needs to be the name of a valid pth file in the 'folder\\file.pth' format or 'Disabled'.")
        return v
    
    def validate_pitches(cls, value):
        try:
            num_value = float(value)
        except ValueError:
            raise ValueError("Pitch must be a number or a string representing a number.")
        if not -24 <= num_value <= 24:
            raise ValueError("Pitch needs to be a number between -24 and 24.")
        return value

def validate_json_input(json_input_data):
    try:
        JSONInput(**json_input_data)
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            field = error["loc"][0]
            description = JSONInput.__fields__[field].description
            error_messages.append(f"{field}: {description}")
        error_message = "\n".join(error_messages)
        print(f"[{config.branding}API] \033[91mError with API request:\033[0m", error_message)
        return error_message
    return None

###############################################################
# /api/tts-generate Generation API Endpoint pull fresh values #
###############################################################
def get_api_text_filtering():
    return config.api_def.api_text_filtering
def get_api_narrator_enabled():
    return config.api_def.api_narrator_enabled
def get_api_text_not_inside():
    return config.api_def.api_text_not_inside
def get_api_language():
    return config.api_def.api_language
def get_api_output_file_name():
    return config.api_def.api_output_file_name
def get_api_output_file_timestamp():
    return config.api_def.api_output_file_timestamp
def get_api_autoplay():
    return config.api_def.api_autoplay
def get_api_autoplay_volume():
    return config.api_def.api_autoplay_volume
def get_params_speed():
    return model_engine.generationspeed_set
def get_params_temperature():
    return model_engine.temperature_set
def get_params_repetition():
    value = model_engine.repetitionpenalty_set
    return float(str(value).replace(',', '.'))
def get_params_pitch():
    return model_engine.pitch_set
def get_character_voice_gen():
    return model_engine.def_character_voice
def get_rvccharacter_voice_gen():
    return config.rvc_settings.rvc_char_model_file
def get_rvccharacter_pitch():
    return config.rvc_settings.pitch
def get_narrator_voice_gen():
    return model_engine.def_narrator_voice
def get_rvcnarrator_voice_gen():
    return config.rvc_settings.rvc_narr_model_file
def get_rvcnarrator_pitch():
    return config.rvc_settings.pitch

#############################################
# /api/tts-generate Generation API Endpoint #
#############################################
@app.post("/api/tts-generate", response_class=JSONResponse)
async def apifunction_generate_tts_standard(
    text_input: str = Form(...),
    text_filtering: str = Form(None),
    character_voice_gen: str = Form(None),
    rvccharacter_voice_gen: str = Form(None),
    rvccharacter_pitch: float = Form(None),
    narrator_enabled: str = Form(None),
    narrator_voice_gen: str = Form(None),
    rvcnarrator_voice_gen: str = Form(None),
    rvcnarrator_pitch: float = Form(None),
    text_not_inside: str = Form(None),
    language: str = Form(None),
    output_file_name: str = Form(None),
    output_file_timestamp: bool = Form(None),
    autoplay: bool = Form(None),
    autoplay_volume: float = Form(None),
    streaming: bool = Form(False),
    speed: float = Form(None),
    temperature: float = Form(None),
    repetition_penalty: float = Form(None),
    pitch: float = Form(None),
    _text_filtering: str = Depends(get_api_text_filtering),
    _character_voice_gen: str = Depends(get_character_voice_gen),
    _rvccharacter_voice_gen: str = Depends(get_rvccharacter_voice_gen),
    _rvccharacter_pitch: str = Depends(get_rvccharacter_pitch),
    _narrator_enabled: str = Depends(get_api_narrator_enabled),
    _narrator_voice_gen: str = Depends(get_narrator_voice_gen),
    _rvcnarrator_voice_gen: str = Depends(get_rvcnarrator_voice_gen),
    _rvcnarrator_pitch: str = Depends(get_rvcnarrator_pitch),
    _text_not_inside: str = Depends(get_api_text_not_inside),
    _language: str = Depends(get_api_language),
    _output_file_name: str = Depends(get_api_output_file_name),
    _output_file_timestamp: bool = Depends(get_api_output_file_timestamp),
    _autoplay: bool = Depends(get_api_autoplay),
    _autoplay_volume: float = Depends(get_api_autoplay_volume),
    _speed: float = Depends(get_params_speed),
    _temperature: float = Depends(get_params_temperature),
    _repetition_penalty: float = Depends(get_params_repetition),
    _pitch: float = Depends(get_params_pitch),
):
    if config.debugging.debug_tts or config.debugging.debug_tts_variables:
        debug_lines = [
            f"[{config.branding}Debug] *******************************************************",
            f"[{config.branding}Debug] /api/tts_generate function called (debug_tts_variables)",
            f"[{config.branding}Debug]         **** PRE validation checks ****",            
            f"[{config.branding}Debug] *******************************************************",
            f"[{config.branding}Debug] Defaults will be pulled from Settings if not specified in API Request!",
            f"[{config.branding}Debug] max_characters         : {config.api_def.api_max_characters}",
            f"[{config.branding}Debug] length_stripping       : {config.api_def.api_length_stripping}",
            f"[{config.branding}Debug] legacy_api             : {config.api_def.api_use_legacy_api}",
            f"[{config.branding}Debug] legacy_api IPadd       : {config.api_def.api_legacy_ip_address}",
            f"[{config.branding}Debug] allowed_filter         : {config.api_def.api_allowed_filter}",
            f"[{config.branding}Debug] text_filtering         : {text_filtering}",
            f"[{config.branding}Debug] character_voice_gen    : {character_voice_gen}",
            f"[{config.branding}Debug] rvccharacter_voice_gen : {rvccharacter_voice_gen}",
            f"[{config.branding}Debug] rvccharacter_pitch     : {rvccharacter_pitch}",
            f"[{config.branding}Debug] narrator_enabled       : {narrator_enabled}",
            f"[{config.branding}Debug] narrator_voice_gen     : {narrator_voice_gen}",
            f"[{config.branding}Debug] rvcnarrator_voice_gen  : {rvcnarrator_voice_gen}",
            f"[{config.branding}Debug] rvcnarrator_pitch      : {rvcnarrator_pitch}",
            f"[{config.branding}Debug] text_not_inside        : {text_not_inside}",
            f"[{config.branding}Debug] language               : {language}",
            f"[{config.branding}Debug] output_file_name       : {output_file_name}",
            f"[{config.branding}Debug] output_file_timestamp  : {output_file_timestamp}",
            f"[{config.branding}Debug] autoplay               : {autoplay}",
            f"[{config.branding}Debug] autoplay_volume        : {autoplay_volume}",
            f"[{config.branding}Debug] streaming              : {streaming}",
            f"[{config.branding}Debug] speed                  : {speed}",
            f"[{config.branding}Debug] temperature            : {temperature}",
            f"[{config.branding}Debug] repetition_penalty     : {repetition_penalty}",
            f"[{config.branding}Debug] pitch : {pitch}"
        ]
        debug_info = "\n".join(debug_lines)
        print(debug_info)  
    text_filtering = text_filtering or _text_filtering
    character_voice_gen = character_voice_gen or _character_voice_gen
    rvccharacter_voice_gen = rvccharacter_voice_gen or _rvccharacter_voice_gen
    rvccharacter_pitch = rvccharacter_pitch if rvccharacter_pitch is not None else _rvccharacter_pitch
    narrator_enabled = narrator_enabled or _narrator_enabled # NEED TO COME BACK AND LOOK AT THIS!!!
    narrator_voice_gen = narrator_voice_gen or _narrator_voice_gen
    rvcnarrator_voice_gen = rvcnarrator_voice_gen or _rvcnarrator_voice_gen
    rvcnarrator_pitch = rvcnarrator_pitch if rvcnarrator_pitch is not None else _rvcnarrator_pitch
    text_not_inside = text_not_inside or _text_not_inside
    language = language or _language
    output_file_name = output_file_name or _output_file_name
    output_file_timestamp = output_file_timestamp if output_file_timestamp is not None else _output_file_timestamp
    autoplay = autoplay if autoplay is not None else _autoplay
    autoplay_volume = autoplay_volume or _autoplay_volume
    speed = speed or _speed
    temperature = temperature or _temperature
    repetition_penalty = repetition_penalty or _repetition_penalty
    pitch = pitch or _pitch
    try:
        json_input_data = {
            "text_input": text_input,
            "text_filtering": text_filtering,
            "character_voice_gen": character_voice_gen,
            "rvccharacter_voice_gen": rvccharacter_voice_gen,
            "rvccharacter_pitch": rvccharacter_pitch,
            "narrator_enabled": narrator_enabled,
            "narrator_voice_gen": narrator_voice_gen,
            "rvcnarrator_voice_gen": rvcnarrator_voice_gen,
            "rvcnarrator_pitch": rvcnarrator_pitch,
            "text_not_inside": text_not_inside,
            "language": language,
            "output_file_name": output_file_name,
            "output_file_timestamp": output_file_timestamp,
            "autoplay": autoplay,
            "autoplay_volume": autoplay_volume,
            "streaming": streaming,
            "speed": speed,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "pitch": pitch,
        }
        JSONresult = validate_json_input(json_input_data)
        if JSONresult is None:
            pass
        else:
            return JSONResponse(content={"error": JSONresult}, status_code=400)
        
        if config.debugging.debug_tts or config.debugging.debug_tts_variables:
            debug_lines = [
                f"[{config.branding}Debug] *******************************************************",
                f"[{config.branding}Debug] /api/tts_generate function called (debug_tts_variables)",
                f"[{config.branding}Debug]        **** POST validation checks ****",
                f"[{config.branding}Debug] *******************************************************",
                f"[{config.branding}Debug] Defaults will be pulled from Settings if not specified in API Request!",
                f"[{config.branding}Debug] max_characters         : {config.api_def.api_max_characters}",
                f"[{config.branding}Debug] length_stripping       : {config.api_def.api_length_stripping}",
                f"[{config.branding}Debug] legacy_api             : {config.api_def.api_use_legacy_api}",
                f"[{config.branding}Debug] legacy_api IPadd       : {config.api_def.api_legacy_ip_address}",
                f"[{config.branding}Debug] allowed_filter         : {config.api_def.api_allowed_filter}",
                f"[{config.branding}Debug] text_filtering         : {text_filtering}",
                f"[{config.branding}Debug] character_voice_gen    : {character_voice_gen}",
                f"[{config.branding}Debug] rvccharacter_voice_gen : {rvccharacter_voice_gen}",
                f"[{config.branding}Debug] rvccharacter_pitch     : {rvccharacter_pitch}",
                f"[{config.branding}Debug] narrator_enabled       : {narrator_enabled}",
                f"[{config.branding}Debug] narrator_voice_gen     : {narrator_voice_gen}",
                f"[{config.branding}Debug] rvcnarrator_voice_gen  : {rvcnarrator_voice_gen}",
                f"[{config.branding}Debug] rvcnarrator_pitch      : {rvcnarrator_pitch}",
                f"[{config.branding}Debug] text_not_inside        : {text_not_inside}",
                f"[{config.branding}Debug] language               : {language}",
                f"[{config.branding}Debug] output_file_name       : {output_file_name}",
                f"[{config.branding}Debug] output_file_timestamp  : {output_file_timestamp}",
                f"[{config.branding}Debug] autoplay               : {autoplay}",
                f"[{config.branding}Debug] autoplay_volume        : {autoplay_volume}",
                f"[{config.branding}Debug] streaming              : {streaming}",
                f"[{config.branding}Debug] speed                  : {speed}",
                f"[{config.branding}Debug] temperature            : {temperature}",
                f"[{config.branding}Debug] repetition_penalty     : {repetition_penalty}",
                f"[{config.branding}Debug] pitch : {pitch}"
            ]
            debug_info = "\n".join(debug_lines)
            print(debug_info)        
        if narrator_enabled.lower() == "silent" and text_not_inside.lower() == "silent":
            print(f"[{config.branding}GEN] \033[92mBoth Narrator & Text-not-inside are set to \033[91msilent\033[92m. If you get no TTS, this is why.\033[0m")
        if narrator_enabled.lower() in ["true", "silent"]:
            if model_engine.lowvram_enabled and model_engine.device == "cpu":
                await model_engine.handle_lowvram_change()
            model_engine.tts_narrator_generatingtts = True
            print(f"[{config.branding}Debug] Moved into Narrator") if config.debugging.debug_tts else None
            print(f"[{config.branding}Debug] original text         : {text_input}") if config.debugging.debug_tts else None
            processed_parts = process_text(text_input)
            audio_files_all_paragraphs = []
            for part_type, part in processed_parts:
                # Skip parts that are too short
                if len(part.strip()) <= int(config.api_def.api_length_stripping):
                    continue
                # Determine the voice to use based on the part type
                if part_type == 'narrator':
                    if narrator_enabled.lower() == "silent":
                        print(f"[{config.branding}GEN] \033[95mNarrator Silent:\033[0m", part)  # Green
                        continue  # Skip generating audio for narrator parts if set to "silent"
                    voice_to_use = narrator_voice_gen
                    print(f"[{config.branding}GEN] \033[92mNarrator:\033[0m", part)  # Purple
                elif part_type == 'character':
                    voice_to_use = character_voice_gen
                    print(f"[{config.branding}GEN] \033[36mCharacter:\033[0m", part)  # Yellow
                else:
                    # Handle ambiguous parts based on user preference
                    if text_not_inside == "silent":
                        print(f"[{config.branding}GEN] \033[95mText-not-inside Silent:\033[0m", part)  # Purple
                        continue  # Skip generating audio for ambiguous parts if set to "silent"
                    voice_to_use = character_voice_gen if text_not_inside == "character" else narrator_voice_gen
                    voice_description = "\033[36mCharacter (Text-not-inside)\033[0m" if text_not_inside == "character" else "\033[92mNarrator (Text-not-inside)\033[0m"
                    print(f"[{config.branding}GEN] {voice_description}:", part)
                # Replace multiple exclamation marks, question marks, or other punctuation with a single instance
                cleaned_part = re.sub(r'([!?.])\1+', r'\1', part)
                # Further clean to remove any other unwanted characters
                cleaned_part = re.sub(rf'{config.api_def.api_allowed_filter}', '', cleaned_part)
                print(f"[{config.branding}Debug] text after api_allowed: {text_input}") if config.debugging.debug_tts else None
                # Remove all newline characters (single or multiple)
                cleaned_part = re.sub(r'\n+', ' ', cleaned_part)
                output_file = this_dir / config.get_output_directory() / f'{output_file_name}_{uuid.uuid4()}_{int(time.time())}.{model_engine.audio_format}'
                output_file_str = output_file.as_posix()
                response = await generate_audio(cleaned_part, voice_to_use, language,temperature, repetition_penalty, speed, pitch, output_file_str, streaming)
                audio_path = output_file_str
                if config.rvc_settings.rvc_enabled:
                    if part_type == 'character':
                        process_rvc_narrator(part_type, rvccharacter_voice_gen, rvccharacter_pitch, config.rvc_settings.rvc_char_model_file, output_file, infer_pipeline)
                    elif part_type == 'narrator':
                        process_rvc_narrator(part_type, rvcnarrator_voice_gen, rvcnarrator_pitch, config.rvc_settings.rvc_narr_model_file, output_file, infer_pipeline)
                    else:
                        if text_not_inside == 'character':
                            process_rvc_narrator('character', rvccharacter_voice_gen, rvccharacter_pitch, config.rvc_settings.rvc_char_model_file, output_file, infer_pipeline)
                        elif text_not_inside == 'narrator':
                            process_rvc_narrator('narrator', rvcnarrator_voice_gen, rvcnarrator_pitch, config.rvc_settings.rvc_narr_model_file, output_file, infer_pipeline)
                print(f"[{config.branding}Debug] Appending audio path to list") if config.debugging.debug_tts else None
                audio_files_all_paragraphs.append(audio_path) 
            # Combine audio files across paragraphs
            model_engine.tts_narrator_generatingtts = False
            print(f"[{config.branding}Debug] Narrator sending to combine") if config.debugging.debug_tts else None
            output_file_path, output_file_url, output_cache_url = combine(output_file_timestamp, output_file_name, audio_files_all_paragraphs)
            # Transcode audio if necessary
            model_audio_format = str(model_engine.audio_format).lower()
            output_audio_format = str(config.transcode_audio_format).lower()
            if output_audio_format == "disabled":
                pass
            else:
                if model_audio_format != output_audio_format and not model_engine.tts_narrator_generatingtts:
                    print(f"[{config.branding}Debug] Pre Transcode call - Output file: {output_file_path}") if config.debugging.debug_tts else None
                    output_file_path, output_file_url, output_cache_url = await transcode_audio_if_necessary(output_file_path, model_audio_format, output_audio_format)
                    print(f"[{config.branding}Debug] Post Transcode call - Output file: {output_file_path}") if config.debugging.debug_tts else None
            if sounddevice_installed == False:
                autoplay = False
            if autoplay:
                play_audio(output_file_path, autoplay_volume)
            # Move model back to cpu system ram if needed.
            if model_engine.lowvram_enabled and model_engine.device == "cuda" and model_engine.lowvram_capable:
                await model_engine.handle_lowvram_change()
            return JSONResponse(content={"status": "generate-success", "output_file_path": str(output_file_path), "output_file_url": str(output_file_url), "output_cache_url": str(output_cache_url)}, status_code=200)
        else:
            print(f"[{config.branding}Debug] Moved into Standard generation") if config.debugging.debug_tts else None
            print(f"[{config.branding}Debug] original text         : {text_input}") if config.debugging.debug_tts else None
            if output_file_timestamp:
                timestamp = int(time.time())
                # Generate a standard UUID
                original_uuid = uuid.uuid4()
                # Hash the UUID using SHA-256
                hash_object = hashlib.sha256(str(original_uuid).encode())
                hashed_uuid = hash_object.hexdigest()
                # Truncate to the desired length, for example, 16 characters
                short_uuid = hashed_uuid[:5]
                output_file_path = this_dir / config.get_output_directory() / f'{output_file_name}_{timestamp}{short_uuid}.{model_engine.audio_format}'
                #Legacy API or New API return
                if config.api_def.api_use_legacy_api:
                    output_file_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audio/{output_file_name}_{timestamp}{short_uuid}.{model_engine.audio_format}'
                    output_cache_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audiocache/{output_file_name}_{timestamp}{short_uuid}.{model_engine.audio_format}'
                else:
                    output_file_url = f'/audio/{output_file_name}_{timestamp}{short_uuid}.{model_engine.audio_format}'
                    output_cache_url = f'/audiocache/{output_file_name}_{timestamp}{short_uuid}.{model_engine.audio_format}'
            else:
                output_file_path = this_dir / config.get_output_directory() / f"{output_file_name}.{model_engine.audio_format}"
                # Legacy API or New API return
                if config.api_def.api_use_legacy_api:
                    output_file_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audio/{output_file_name}.{model_engine.audio_format}'
                    output_cache_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audiocache/{output_file_name}.{model_engine.audio_format}'
                else:
                    output_file_url = f'/audio/{output_file_name}.{model_engine.audio_format}'
                    output_cache_url = f'/audiocache/{output_file_name}.{model_engine.audio_format}'
            if text_filtering == "html":
                cleaned_string = html.unescape(standard_filtering(text_input))
                cleaned_string = re.sub(r'([!?.])\1+', r'\1', cleaned_string)
                # Further clean to remove any other unwanted characters
                cleaned_string = re.sub(rf'{config.api_def.api_allowed_filter}', '', cleaned_string)
                print(f"[{config.branding}Debug] HTML Filtering - text after api_allowed: {cleaned_string}") if config.debugging.debug_tts else None
                # Remove all newline characters (single or multiple)
                cleaned_string = re.sub(r'\n+', ' ', cleaned_string)
            elif text_filtering == "standard":
                cleaned_string = re.sub(r'([!?.])\1+', r'\1', text_input)
                # Further clean to remove any other unwanted characters
                cleaned_string = re.sub(rf'{config.api_def.api_allowed_filter}', '', cleaned_string)
                print(f"[{config.branding}Debug] Standard Filtering - text after api_allowed: {cleaned_string}") if config.debugging.debug_tts else None
                # Remove all newline characters (single or multiple)
                cleaned_string = re.sub(r'\n+', ' ', cleaned_string)
            else:
                print(f"[{config.branding}Debug] No filtering text     : {text_input}") if config.debugging.debug_tts else None
                cleaned_string = text_input
            print(f"[{config.branding}GEN]", cleaned_string)
            print(f"[{config.branding}Debug] Sending request to generate_audio in tts_server.py") if config.debugging.debug_tts else None
            try:
                if streaming:
                    response = await generate_audio(cleaned_string, character_voice_gen, language, temperature, repetition_penalty, speed, pitch, output_file_path, streaming)
                    return StreamingResponse(response, media_type="audio/wav")
                else:
                    response = await generate_audio(cleaned_string, character_voice_gen, language, temperature, repetition_penalty, speed, pitch, output_file_path, streaming)
                    if config.rvc_settings.rvc_enabled:
                        if rvccharacter_voice_gen.lower() in ["disabled", "disable"]:
                            print(f"[{config.branding}Debug] Pass rvccharacter_voice_gen") if config.debugging.debug_tts else None
                            pass  # Skip RVC processing for character part
                        else:
                            print(f"[{config.branding}Debug] send to rvc") if config.debugging.debug_tts else None
                            rvccharacter_voice_gen = this_dir / "models" / "rvc_voices" / rvccharacter_voice_gen
                            pth_path = rvccharacter_voice_gen if rvccharacter_voice_gen else config.rvc_settings.rvc_char_model_file
                            run_rvc(output_file_path, pth_path, rvccharacter_pitch, infer_pipeline)
                    # Transcode audio if necessary
                    model_audio_format = str(model_engine.audio_format).lower()
                    output_audio_format = str(config.transcode_audio_format).lower()
                    print(f"[{config.branding}Debug] At Model audio format") if config.debugging.debug_tts else None
                    if output_audio_format == "disabled":
                        pass
                    else:
                        if model_audio_format != output_audio_format and not model_engine.tts_narrator_generatingtts:
                            print(f"[{config.branding}Debug] Pre Transcode call - Output file: {output_file_path}") if config.debugging.debug_tts else None
                            output_file_path, output_file_url, output_cache_url = await transcode_audio_if_necessary(output_file_path, model_audio_format, output_audio_format)
                            print(f"[{config.branding}Debug] Post Transcode call - Output file: {output_file_path}") if config.debugging.debug_tts else None
                    if sounddevice_installed == False:
                        autoplay = False
                    if autoplay:
                        play_audio(output_file_path, autoplay_volume)
                    return JSONResponse(content={"status": "generate-success", "output_file_path": str(output_file_path), "output_file_url": str(output_file_url), "output_cache_url": str(output_cache_url)}, status_code=200)
            except Exception as e:
                return JSONResponse(content={"status": "generate-failure", "error": "An error occurred"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"status": "generate-failure", "error": "An error occurred"}, status_code=500)

#############################
#### Word Add-in Sharing ####
#############################
# Mount the static files from the 'word_addin' directory
app.mount("/api/word_addin", StaticFiles(directory=os.path.join(this_dir / 'system' / 'word_addin')), name="word_addin")

#############################################
#### TTS Generator Comparision Endpoints ####
#############################################
import aiofiles
class TTSItem(BaseModel):
    id: int
    fileUrl: str
    text: str
    characterVoice: str
    language: str

class TTSData(BaseModel):
    ttsList: List[TTSItem]

@app.post("/api/save-tts-data")
async def apifunction_save_tts_data(tts_data: List[TTSItem]):
    # Convert the list of Pydantic models to a list of dictionaries
    tts_data_list = [item.dict() for item in tts_data]
    # Serialize the list of dictionaries to a JSON string
    tts_data_json = json.dumps(tts_data_list, indent=4)
    async with aiofiles.open(this_dir / config.get_output_directory() / "ttsList.json", 'w') as f:
        await f.write(tts_data_json)
    return {"message": "Data saved successfully"}

########################################
# Trigger TTS Gen Text/Speech Analysis #
########################################
@app.get("/api/trigger-analysis")
async def apifunction_trigger_analysis(threshold: int = Query(default=98)):
    venv_path = sys.prefix
    env = os.environ.copy()
    env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
    ttslist_path = this_dir / config.get_output_directory() / "ttsList.json"
    wavfile_path = this_dir / config.get_output_directory()
    subprocess.run([sys.executable, "tts_diff.py", f"--threshold={threshold}", f"--ttslistpath={ttslist_path}", f"--wavfilespath={wavfile_path}"], cwd=this_dir / "system" / "tts_diff", env=env)
    # Read the analysis summary
    try:
        with open(this_dir / config.get_output_directory() / "analysis_summary.json", "r") as summary_file:
            summary_data = json.load(summary_file)
    except FileNotFoundError:
        summary_data = {"error": "Analysis summary file not found."}
    return {"message": "Analysis Completed", "summary": summary_data}

###########################################
# TTS Generator SRT Subtitiles generation #
###########################################
@app.get("/api/srt-generation")
async def apifunction_srt_generation():
    venv_path = sys.prefix
    env = os.environ.copy()
    env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
    ttslist_path = this_dir / config.get_output_directory() / "ttsList.json"
    wavfile_path = this_dir / config.get_output_directory()
    subprocess.run([sys.executable, "tts_srt.py", f"--ttslistpath={ttslist_path}", f"--wavfilespath={wavfile_path}"], cwd=this_dir / "system" / "tts_srt", env=env)
    srt_file_path = this_dir / config.get_output_directory() / "subtitles.srt"
    if not srt_file_path.exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found.")
    return FileResponse(path=srt_file_path, filename="subtitles.srt", media_type='application/octet-stream')

#################################
# Static Mount for file serving #
#################################
app.mount("/static", StaticFiles(directory=str(this_dir / "system")), name="static")
 
########################################
# Legacy JSON update settings function #
########################################
# Setup logging
logging.basicConfig(level=logging.DEBUG)

@app.get("/settings")
async def get_settings(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request, "data": config.to_dict()})

@app.get("/settings-json")
async def get_settings_json():
    return config.to_dict()

@app.post("/update-settings")
async def update_settings(
    request: Request,
    delete_output_wavs: str = Form(...),
    gradio_interface: str = Form(...),
    gradio_port_number: int = Form(...),
    api_port_number: int = Form(...),
):
    # Update the settings based on the form values
    config.delete_output_wavs = delete_output_wavs
    config.gradio_interface = gradio_interface.lower() == 'true'
    config.gradio_port_number = gradio_port_number
    config.api_def.api_port_number = api_port_number

    # Save the updated settings back to the JSON file
    config.save()

    # Redirect to the settings page to display the updated settings
    return templates.TemplateResponse("admin.html", {"request": request, "data": config.to_dict()})
 
# Create an instance of Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory=this_dir / "system")
# Get the admin interface template
template = templates.get_template("admin.html")
# Render the template with the dynamic values
rendered_html = template.render(params=config.to_dict())
 
###################################################
#### Webserver Startup & Initial model Loading ####
###################################################
@app.get("/")
async def read_root():
    return HTMLResponse(content=rendered_html, status_code=200)

# Start Uvicorn Webserver
# port_parameter = int(config.api_def.api_port_number)

if __name__ == "__main__":
    import uvicorn
    # Command line argument parser
    parser = argparse.ArgumentParser(description="AllTalk TTS Server")
    parser.add_argument("--port", type=int, help="Port number for the server")
    args = parser.parse_args()
    # Determine the port to use
    config_port = int(config.api_def.api_port_number)
    port_to_use = args.port if args.port is not None else config_port
    # Start Uvicorn Webserver
    uvicorn_server = uvicorn.run(app, host="0.0.0.0", port=port_to_use, log_level="debug")

