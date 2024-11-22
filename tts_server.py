import os
import sys
import json
import time
import inspect
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
from typing import Union, Dict, List, Optional, Tuple
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

# pylint: disable=all

##########################
# Central print function #
##########################
def print_message(message, message_type="standard", component="TTS"):
    """Centralized print function for AllTalk messages
    Args:
        message (str): The message to print
        message_type (str): Type of message (standard/warning/error/debug_*/debug)
        component (str): Component identifier (TTS/ENG/GEN/API/etc.)
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
            print(f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} Function entry:{GREEN}{message_parts[1]}{RESET} tts_server.py")
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

def debug_func_entry():
    """Print debug message for function entry if debug_func is enabled"""
    if config.debugging.debug_func:
        current_func = inspect.currentframe().f_back.f_code.co_name
        print_message(f"Function entry: {current_func}", "debug_func")

####################
# Check for FFMPEG #
####################
def check_ffmpeg(this_dir):
    """Verify FFmpeg availability in the system."""
    debug_func_entry()
    message = ""
    
    if sys.platform == "win32":
        ffmpeg_path = os.path.join(this_dir, "system", "win_ffmpeg", "ffmpeg.exe")
        if os.path.exists(ffmpeg_path):
            return True, "FFmpeg found in Windows directory"
        message = "FFmpeg not found in Windows directory"
    else:
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True, "FFmpeg found in system PATH"
        except (subprocess.CalledProcessError, FileNotFoundError):
            message = "FFmpeg not found in system PATH"
            
    print_message(message, "warning")
    return False, message

# Check if FFmpeg is installed
ffmpeg_installed = check_ffmpeg(this_dir)

if not ffmpeg_installed:
    print_message("\033[92mTranscoding       :\033[91m ffmpeg not found\033[0m", component="ENG")
    print_message("FFmpeg is not installed. Transcoding will be disabled.", "warning", "ENG")
    print_message("Please install FFmpeg on your system.", component="ENG")
    
    if sys.platform == "win32":
        print_message("\033[92mTranscoding       :\033[91m ffmpeg not found\033[0m", component="ENG")
        print_message("Installation instructions for Windows:", component="ENG")
        print_message(f"Copy the 'ffmpeg.exe' file to '{os.path.join(this_dir, 'system', 'win_ffmpeg')}'", component="ENG")
    else:
        print_message("\033[92mTranscoding       :\033[91m ffmpeg not found\033[0m", component="ENG")
        print_message("Installation instructions:", component="ENG")
        print_message("Linux (Debian-based systems): Run 'sudo apt-get install ffmpeg' in the terminal.", component="ENG")
        print_message("macOS: Run 'brew install ffmpeg' in the terminal (requires Homebrew).", component="ENG")

if ffmpeg_installed:
    from ffmpeg.asyncio import FFmpeg
    print_message("\033[92mTranscoding       :\033[93m ffmpeg found\033[0m", component="ENG")
    
################################
# Check for portaudio on Linux #
################################
try:
   import sounddevice as sd
   sounddevice_installed=True
except OSError:
    print_message("The PortAudio library is not installed. To enable audio playback for TTS in the terminal or console,", "warning")
    print_message("please install PortAudio. This will not impact other features of Alltalk.", "warning")
    print_message("You can still play audio through web browsers without PortAudio.", "warning")
    print_message("Installing PortAudio is optional and not strictly required.")
    sounddevice_installed=False
    if sys.platform.startswith('linux'):
        print_message("On Linux, you can use the following command to install PortAudio:", "warning")
        print_message("sudo apt-get install portaudio19-dev", "warning")

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
    """Initialize model engine and handle graceful shutdown. This is a context manager."""
    debug_func_entry()
    try:
        await model_engine.setup()
    except FileNotFoundError as e:
        print_message(f"Error during setup: {e}. Continuing without the TTS model.", "error")
    yield

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
    """Handle API request to change TTS model. Manages model change state and validates requested model."""
    debug_func_entry()
    
    global model_currently_changing
    if model_currently_changing:
        print_message("Model change already in progress", "debug_api", "API")
        return Response(content=json.dumps({"status": "model-is currently changing"}), media_type="application/json")
    
    model_currently_changing = True  # Fixed assignment operator
    requested_model = request.query_params.get("tts_method")
    
    if requested_model not in model_engine.available_models:
        model_currently_changing = False  # Fixed assignment operator
        print_message(f"Invalid TTS method requested: {requested_model}", "error", "API")
        return {"status": "error", "message": "Invalid TTS method specified"}
    
    print_message(f"Attempting to change model to: {requested_model}", "debug_api", "API")
    success = await model_engine.handle_tts_method_change(requested_model)
    
    model_currently_changing = False  # Fixed assignment operator
    if success:
        model_engine.current_model_loaded = requested_model
        print_message(f"Model successfully changed to: {requested_model}", "debug_api", "API")
        return Response(content=json.dumps({"status": "model-success"}), media_type="application/json")
    else:
        print_message(f"Failed to change model to: {requested_model}", "error", "API")
        return Response(content=json.dumps({"status": "model-failure"}), media_type="application/json")
    
####################################
# API Endpoint - /api/enginereload #
####################################
import asyncio
uvicorn_server = None
def restart_self():
    """Restart the current Python process."""
    debug_func_entry()
    print_message("Restarting subprocess...", component="ENG")
    os.execv(sys.executable, ['python'] + sys.argv)

async def handle_restart():
    """Handle graceful shutdown of uvicorn server before restart."""
    debug_func_entry()
    global uvicorn_server
    
    if uvicorn_server:
        print_message("Stopping uvicorn server...", component="ENG")
        uvicorn_server.should_exit = True
        uvicorn_server.force_exit = True
        
        while not uvicorn_server.is_stopped:
            await asyncio.sleep(0.1)
            
    restart_self()

@app.post("/api/enginereload")
async def apifunction_reload(request: Request):
    """Handle API request to change TTS engine. Validates engine, updates config, and triggers restart."""
    debug_func_entry()
    
    requested_engine = request.query_params.get("engine")
    if not tts_engines_config.is_valid_engine(requested_engine):
        print_message(f"Invalid engine requested: {requested_engine}", "error", "API")
        return {"status": "error", "message": "Invalid TTS engine specified"}
    
    print_message("", component="ENG")
    print_message("\033[94mChanging model loaded. Please wait.\033[00m", component="ENG")
    print_message("", component="ENG")
    
    try:
        tts_engines_config.change_engine(requested_engine).save()
        print_message(f"Engine configuration updated to: {requested_engine}", "debug_api", "API")
    finally:
        tts_engines_config.reload()
        print_message("Configuration reloaded", "debug_api", "API")
    
    asyncio.create_task(handle_restart())
    return Response(content=json.dumps({"status": "engine-success"}), media_type="application/json")


#######################################
# API Endpoint - /api/stop-generation #
#######################################
# When this endpoint it called it will set tts_stop_generation in the model_engine to True, which can be used to interrupt the current TTS generation.
@app.put("/api/stop-generation")
async def apifunction_stop_generation():
    """Handle request to stop ongoing TTS generation."""
    debug_func_entry()
    if model_engine.tts_generating_lock and not model_engine.tts_stop_generation:
        model_engine.tts_stop_generation = True
        print_message("Stopping TTS generation", "debug_api", "API")
    return {"message": "Cancelling current TTS generation"}

#############################
# API Endpoint - /api/audio #
#############################
@app.get("/audio/{filename}")
async def apifunction_get_audio(filename: str):
    """Serve audio file from output directory."""
    debug_func_entry()
    audio_path = this_dir / config.get_output_directory() / filename
    if not audio_path.is_file():
        print_message(f"Audio file not found: {filename}", "error", "API")
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(audio_path)

##################################
# API Endpoint - /api/audiocache #
##################################
@app.get("/audiocache/{filename}")
async def apifunction_get_audio(filename: str):
    """Serve cached audio file with caching headers."""
    debug_func_entry()
    audio_path = Path(config.get_output_directory()) / filename
    if not audio_path.is_file():
        print_message(f"Cached audio file not found: {filename}", "error", "API")
        raise HTTPException(status_code=404, detail="File not found")
        
    response = FileResponse(path=audio_path, media_type='audio/wav', filename=filename)
    response.headers["Cache-Control"] = "public, max-age=604800"
    response.headers["ETag"] = str(audio_path.stat().st_mtime)
    return response

##############################
# API Endpoint - /api/voices #
##############################
@app.get("/api/voices")
async def apifunction_get_voices():
    """Get available voices for current TTS engine."""
    debug_func_entry()
    
    try:
        if not model_engine.multivoice_capable:
            print_message(f"Engine '{model_engine.engine_loaded}' does not support multiple voices", "warning", "API")
            return {"status": "error", "message": f"The currently loaded TTS engine '{model_engine.engine_loaded}' does not support multiple voices."}
        
        available_voices = model_engine.voices_file_list()
        print_message(f"Successfully retrieved {len(available_voices)} available voices", "debug_api", "API")
        return {"status": "success", "voices": available_voices}
    except Exception as e:
        print_message(f"Error retrieving voices: {str(e)}", "error", "API")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

#################################
# API Endpoint - /api/rvcvoices #
#################################
@app.get("/api/rvcvoices")
async def apifunction_get_rvcvoices():
    """Get available RVC voice models."""
    debug_func_entry()
    
    try:
        load_config()
        if not config.rvc_settings.rvc_enabled:
            return {"status": "success", "rvcvoices": ["Disabled"]}
            
        directory = os.path.join(this_dir, "models", "rvc_voices")
        if not os.path.exists(directory):
            print_message("RVC voices directory not found", "warning", "API")
            return {"status": "success", "rvcvoices": ["Disabled"]}
            
        if not os.access(directory, os.R_OK):
            print_message("No read permission for RVC voices directory", "error", "API")
            return JSONResponse(content={"status": "error", "message": "Cannot access RVC voices directory"}, status_code=500)
            
        pth_files = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".pth"):
                        pth_files.append(os.path.relpath(os.path.join(root, file), start=directory))
        except OSError as e:
            print_message(f"Error scanning RVC directory: {str(e)}", "error", "API")
            return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
            
        pth_files = ["Disabled"] + sorted(pth_files) if pth_files else ["Disabled"]
        return {"status": "success", "rvcvoices": pth_files}
        
    except Exception as e:
        print_message(f"Error processing RVC voices: {str(e)}", "error", "API")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

#####################################
# API Endpoint - /api/reload_config #
#####################################
@app.get("/api/reload_config")
async def apifunction_reload_config():
    """Reload configuration settings."""
    debug_func_entry()
    
    try:
        load_config(True)
        print_message("Configuration reloaded successfully", "debug_api", "API")
        return Response("Config file reloaded successfully")
    except Exception as e:
        print_message(f"Error reloading config: {str(e)}", "error", "API")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

#############################
# API Endpoint - /api/ready #
#############################
@app.get("/api/ready")
async def apifunction_ready():
    """Check if model engine setup has completed."""
    debug_func_entry()
    status = "Ready" if model_engine.setup_has_run else "Unloaded"
    print_message(f"Engine status: {status}", "debug_api", "API")
    return Response(status)

#######################################
# API Endpoint - /api/currentsettings #
#######################################
@app.get('/api/currentsettings')
def apifunction_get_current_settings():
    """Get comprehensive dictionary of current engine settings and capabilities."""
    debug_func_entry()
    load_config()
    try:
        settings = {
            "engines_available": tts_engines_config.get_engine_names_available(),
            "current_engine_loaded": model_engine.engine_loaded,
            "models_available": [{"name": name} for name in model_engine.available_models.keys()],
            "current_model_loaded": model_engine.current_model_loaded,
            "manufacturer_name": model_engine.manufacturer_name,
            "audio_format": model_engine.audio_format,
            "deepspeed_capable": model_engine.deepspeed_capable,
            "deepspeed_available": model_engine.deepspeed_available,
            "deepspeed_enabled": model_engine.deepspeed_enabled,
            "generationspeed_capable": model_engine.generationspeed_capable,
            "generationspeed_set": model_engine.generationspeed_set,
            "lowvram_capable": model_engine.lowvram_capable,
            "lowvram_enabled": model_engine.lowvram_enabled,
            "pitch_capable": model_engine.pitch_capable,
            "pitch_set": model_engine.pitch_set,
            "repetitionpenalty_capable": model_engine.repetitionpenalty_capable,
            "repetitionpenalty_set": model_engine.repetitionpenalty_set,
            "streaming_capable": model_engine.streaming_capable,
            "temperature_capable": model_engine.temperature_capable,
            "temperature_set": model_engine.temperature_set,
            "ttsengines_installed": model_engine.engine_installed,
            "languages_capable": model_engine.languages_capable,
            "multivoice_capable": model_engine.multivoice_capable,
            "multimodel_capable": model_engine.multimodel_capable,
        }
        print_message("Current settings retrieved successfully", "debug_api", "API")
        return settings
    except Exception as e:
        print_message(f"Error retrieving settings: {str(e)}", "error", "API")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

######################################
# API Endpoint - /api/lowvramsetting #
######################################    
@app.post("/api/lowvramsetting")
async def apifunction_low_vram(request: Request, new_low_vram_value: bool):
    """Handle LowVRAM mode toggle. Updates model device allocation between VRAM and system RAM."""
    debug_func_entry()
    
    if not model_engine.lowvram_capable:
        print_message(f"Engine '{model_engine.engine_loaded}' does not support lowvram", "error", "API")
        return {"status": "error", "message": f"The currently loaded TTS engine '{model_engine.engine_loaded}' does not support lowvram."}
        
    try:
        if new_low_vram_value is None:
            raise ValueError("Missing 'low_vram' parameter")
            
        if model_engine.lowvram_enabled == new_low_vram_value:
            msg = f"LowVRAM is already {'enabled' if new_low_vram_value else 'disabled'}"
            print_message(msg, "debug_api", "API")
            return Response(content=json.dumps({"status": "success", "message": msg}))
            
        model_engine.lowvram_enabled = new_low_vram_value
        await model_engine.unload_model()
        
        if model_engine.cuda_is_available:
            if model_engine.lowvram_enabled:
                model_engine.device = "cpu"
                print_message("\033[94mLowVRAM Enabled.\033[0m Model will move between \033[93mVRAM(cuda) <> System RAM(cpu)\033[0m", component="ENG")
            else:
                model_engine.device = "cuda"
                print_message("\033[94mLowVRAM Disabled.\033[0m Model will stay in \033[93mVRAM(cuda)\033[0m", component="ENG")
            await model_engine.setup()
        else:
            print_message("Nvidia CUDA is not available on this system. Unable to use LowVRAM mode.", "error", "ENG")
            model_engine.lowvram_enabled = False
        
        return Response(content=json.dumps({"status": "lowvram-success"}))
        
    except Exception as e:
        print_message(f"Error setting LowVRAM mode: {str(e)}", "error", "API")
        return Response(content=json.dumps({"status": "error", "message": str(e)}))

#################################
# API Endpoint - /api/deepspeed #
#################################
@app.post("/api/deepspeed")
async def deepspeed(request: Request, new_deepspeed_value: bool):
    """Handle DeepSpeed mode toggle. Verifies capability and handles model changes."""
    debug_func_entry()
    
    if not model_engine.deepspeed_capable or not model_engine.deepspeed_available:
        print_message("DeepSpeed not supported or available", "error", "API")
        return {"status": "error", "message": f"The currently loaded TTS engine '{model_engine.engine_loaded}' does not support DeepSpeed or DeepSpeed is not available on this system."}
        
    try:
        if new_deepspeed_value is None:
            raise ValueError("Missing 'deepspeed' parameter")
            
        if model_engine.deepspeed_enabled == new_deepspeed_value:
            msg = f"DeepSpeed is already {'enabled' if new_deepspeed_value else 'disabled'}"
            print_message(msg, "debug_api", "API")
            return Response(content=json.dumps({"status": "success", "message": msg}))
            
        model_engine.deepspeed_enabled = new_deepspeed_value
        await model_engine.handle_deepspeed_change(new_deepspeed_value)
        print_message(f"DeepSpeed {'enabled' if new_deepspeed_value else 'disabled'}", "debug_api", "API")
        return Response(content=json.dumps({"status": "deepspeed-success"}))
        
    except Exception as e:
        print_message(f"Error setting DeepSpeed mode: {str(e)}", "error", "API")
        return Response(content=json.dumps({"status": "error", "message": str(e)}))

#################################
# API Endpoint - /api/voice2rvc #
#################################
@app.post("/api/voice2rvc")
async def voice2rvc(input_tts_path: str = Form(...), output_rvc_path: str = Form(...), 
                   pth_name: str = Form(...), pitch: str = Form(...), method: str = Form(...)):
    """Handle voice conversion using RVC. Processes input audio through specified RVC model."""
    debug_func_entry()
    
    try:
        if pth_name.lower() in ["disabled", "disable"]:
            print_message("\033[94mVoice2RVC Convert: No voice was specified or the name was Disabled\033[0m")
            return {"status": "error", "message": "No voice was specified or the name was Disabled"}
            
        input_tts_path = Path(input_tts_path)
        output_rvc_path = Path(output_rvc_path)
        
        if not input_tts_path.is_file():
            print_message(f"Input file not found: {input_tts_path}", "error")
            raise HTTPException(status_code=400, detail=f"Input file {input_tts_path} does not exist.")
            
        pth_path = this_dir / "models" / "rvc_voices" / pth_name
        if not pth_path.is_file():
            print_message(f"RVC model file not found: {pth_path}", "error")
            raise HTTPException(status_code=400, detail=f"Model file {pth_path} does not exist.")
            
        print_message(f"Starting RVC conversion with model: {pth_name}", "debug_rvc")
        result_path = run_voice2rvc(input_tts_path, output_rvc_path, pth_path, pitch, method)
        
        if result_path:
            print_message("RVC conversion completed successfully", "debug_rvc")
            return {"status": "success", "output_path": str(result_path)}
        else:
            print_message("RVC conversion failed", "error")
            raise HTTPException(status_code=500, detail="RVC conversion failed.")
            
    except Exception as e:
        print_message(f"Error during Voice2RVC conversion: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))

def run_voice2rvc(input_tts_path, output_rvc_path, pth_path, pitch, method):
    """Run RVC conversion on input audio using specified model and parameters."""
    debug_func_entry()
    
    print_message("\033[94mVoice2RVC Convert: Started\033[0m", component="GEN")
    generate_start_time = time.time()
    
    # Get settings from config
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
    input_tts_path = str(input_tts_path)
    pth_path = str(pth_path)
    
    if not os.path.isfile(pth_path):
        print_message(f"Model file {pth_path} does not exist. Exiting.", "error", "GEN")
        return
        
    model_dir = os.path.dirname(pth_path)
    index_files = [file for file in os.listdir(model_dir) if file.endswith(".index")]
    index_path = str(os.path.join(model_dir, index_files[0])) if len(index_files) == 1 else ""
    
    print_message(f"Using RVC model: {os.path.basename(pth_path)}", "debug_rvc", "GEN")
    print_message(f"Using index file: {os.path.basename(index_path) if index_path else 'None'}", "debug_rvc", "GEN")
    
    infer_pipeline(f0up_key, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method,
                input_tts_path, output_rvc_path, pth_path, index_path, split_audio, f0autotune, 
                embedder_model, training_data_size, config.debugging.debug_rvc)
                
    generate_elapsed_time = time.time() - generate_start_time
    print_message(f"\033[94mVoice2RVC Convert: \033[91m{generate_elapsed_time:.2f} seconds.\033[0m", component="GEN")
    
    return output_rvc_path

##################################
# Transcode between file formats #
##################################
async def transcode_audio(input_file, output_format):
    """Transcode audio files between different formats using FFmpeg. Supports wav, mp3, flac, aac, and opus formats."""
    debug_func_entry()
    
    print_message("*************************************************", "debug_transcode")
    print_message("transcode_audio function called (debug_transcode)", "debug_transcode")
    print_message("*************************************************", "debug_transcode")
    print_message(f"Input file    : {input_file}", "debug_transcode")
    print_message(f"Output format : {output_format}", "debug_transcode")
    
    if output_format == "Disabled":
        print_message("Transcode format is set to Disabled so skipping transcode.", "debug_transcode")
        return input_file
        
    if not ffmpeg_installed:
        print_message("FFmpeg is not installed. Format conversion is not possible.", "error")
        raise Exception("FFmpeg is not installed. Format conversion is not possible.")
        
    input_extension = os.path.splitext(input_file)[1][1:].lower()
    print_message(f"Input file extension: {input_extension}", "debug_transcode")
    
    if input_extension == output_format.lower():
        print_message(f"Input file is already in the requested format: {output_format}", "debug_transcode")
        return input_file
        
    output_file = os.path.splitext(input_file)[0] + f".{output_format}"
    print_message(f"Output file: {output_file}", "debug_transcode")
    
    ffmpeg_path = os.path.join(this_dir, "system", "win_ffmpeg", "ffmpeg.exe") if sys.platform == "win32" else "ffmpeg"
    ffmpeg = FFmpeg(ffmpeg_path).option("y").input(input_file).output(output_file)
    
    print_message(f"Transcoding to {output_format}", "debug_transcode")
    
    # Configure format-specific options
    if output_format == "opus":
        print_message("Configuring Opus options", "debug_transcode")
        ffmpeg.output(output_file, {
            "codec:a": "libopus", 
            "b:a": "128k", 
            "vbr": "on", 
            "compression_level": 10, 
            "frame_duration": 60, 
            "application": "voip"
        })
    elif output_format == "aac":
        print_message("Configuring AAC options", "debug_transcode")
        ffmpeg.output(output_file, {"codec:a": "aac", "b:a": "192k"})
    elif output_format == "flac":
        print_message("Configuring FLAC options", "debug_transcode")
        ffmpeg.output(output_file, {"codec:a": "flac", "compression_level": 8})
    elif output_format == "wav":
        print_message("Configuring WAV options", "debug_transcode")
        ffmpeg.output(output_file, {"codec:a": "pcm_s16le"})
    elif output_format == "mp3":
        print_message("Configuring MP3 options", "debug_transcode")
        ffmpeg.output(output_file, {"codec:a": "libmp3lame", "b:a": "192k"})
    else:
        print_message(f"Unsupported output format: {output_format}", "error")
        raise ValueError(f"Unsupported output format: {output_format}")
        
    try:
        print_message("Starting transcoding process", "debug_transcode")
        await ffmpeg.execute()
        print_message("Transcoding completed successfully", "debug_transcode")
    except Exception as e:
        print_message(f"Error occurred during transcoding: {str(e)}", "error")
        raise
        
    print_message("Deleting original input file", "debug_transcode")
    os.remove(input_file)
    
    print_message("Transcoding process completed", "debug_transcode")
    print_message(f"Transcoded file: {output_file}", "debug_transcode")
    return output_file

##############################
# Central Transcode function #
##############################
async def transcode_audio_if_necessary(output_file, model_audio_format, output_audio_format):
    """Transcode audio to requested format if needed and generate appropriate URLs."""
    debug_func_entry()
    
    print_message(f"model_engine.audio_format is: {model_audio_format}", "debug_transcode")
    print_message(f"audio format is: {output_audio_format}", "debug_transcode")
    print_message("Entering the transcode condition", "debug_transcode")
    
    try:
        print_message("Calling transcode_audio function", "debug_transcode")
        output_file = await transcode_audio(output_file, output_audio_format)
        print_message("Transcode completed successfully", "debug_transcode")
    except Exception as e:
        print_message(f"Error occurred during transcoding: {str(e)}", "error")
        raise
        
    print_message("Transcode condition completed", "debug_transcode")
    print_message("Updating output file paths and URLs", "debug_tts")
    
    # Generate appropriate URLs based on API configuration
    if config.api_def.api_use_legacy_api:
        output_file_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audio/{os.path.basename(output_file)}'
        output_cache_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}/audiocache/{os.path.basename(output_file)}'
    else:
        output_file_url = f'/audio/{os.path.basename(output_file)}'
        output_cache_url = f'/audiocache/{os.path.basename(output_file)}'
        
    print_message("Output file paths and URLs updated", "debug_tts")
    return output_file, output_file_url, output_cache_url

##############################
#### Streaming Generation ####
##############################
@app.get("/api/tts-generate-streaming", response_class=StreamingResponse)
async def apifunction_generate_streaming(text: str, voice: str, language: str, output_file: str):
    """Handle streaming TTS generation via GET request."""
    debug_func_entry()
    
    if not model_engine.streaming_capable:
        print_message("The selected TTS Engine does not support streaming. To use streaming, please select a TTS", "warning", "GEN")
        print_message("Engine that has streaming capability. You can find the streaming support information for", "warning", "GEN")
        print_message("each TTS Engine in its 'Engine Information' section of the Gradio interface.", "warning", "GEN")
        
    try:
        output_file_path = f'{this_dir / config.get_output_directory() / output_file}.{model_engine.audio_format}'
        print_message(f"Starting streaming TTS generation for: {text[:50]}{'...' if len(text) > 50 else ''}")
        stream = await generate_audio(text, voice, language, model_engine.temperature_set, 
                                   model_engine.repetitionpenalty_set, 1.0, 1.0, 
                                   output_file_path, streaming=True)
        return StreamingResponse(stream, media_type="audio/wav")
    except Exception as e:
        print_message(f"An error occurred: {e}", "error", "GEN")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

@app.post("/api/tts-generate-streaming", response_class=JSONResponse)
async def tts_generate_streaming(request: Request, text: str = Form(...), voice: str = Form(...), 
                               language: str = Form(...), output_file: str = Form(...)):
    """Handle streaming TTS generation via POST request."""
    debug_func_entry()
    
    if not model_engine.streaming_capable:
        print_message("The selected TTS Engine does not support streaming. To use streaming, please select a TTS", "warning", "GEN")
        print_message("Engine that has streaming capability. You can find the streaming support information for", "warning", "GEN")
        print_message("each TTS Engine in its 'Engine Information' section of the Gradio interface.", "warning", "GEN")
        
    try:
        output_file_path = f'{this_dir / config.get_output_directory() / output_file}.{model_engine.audio_format}'
        print_message(f"Starting TTS generation for file: {os.path.basename(output_file)}")
        await generate_audio(text, voice, language, model_engine.temperature_set, 
                           model_engine.repetitionpenalty_set, "1.0", "1.0", 
                           output_file_path, streaming=False)
        return JSONResponse(content={"output_file_path": str(output_file)}, status_code=200)
    except Exception as e:
        print_message(f"An error occurred: {e}", "error", "GEN")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

###################################
# Central generate_audio function #
###################################
async def generate_audio(text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming=False):
    """Generate TTS audio with specified parameters. Supports streaming and non-streaming modes."""
    debug_func_entry()
    
    if not model_engine.streaming_capable and streaming:
        print_message("The selected TTS Engine does not support streaming. To use streaming, please select a TTS", "warning", "GEN")
        print_message("Engine that has streaming capability. You can find the streaming support information for", "warning", "GEN")
        print_message("each TTS Engine in the 'Engine Information' section of the Gradio interface.", "warning", "GEN")
        raise ValueError("Streaming not supported by current TTS engine")
            
    response = model_engine.generate_tts(text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming)
    
    if streaming:
        async def stream_response():
            try:
                async for chunk in response:
                    yield chunk
            except Exception as e:
                print_message(f"Error during streaming audio generation: {str(e)}", "error", "GEN")
                raise
        return stream_response()
    else:
        try:
            async for _ in response:
                pass
        except Exception as e:
            print_message(f"Error during audio generation: {str(e)}", "error", "GEN")
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
    """Generate preview audio for selected voice with optional RVC processing."""
    debug_func_entry()
    
    try:
        language = "en"
        output_file_name = "api_preview_voice"
        
        clean_voice_filename = re.sub(r'\.wav$', '', voice.replace(' ', '_'))
        clean_voice_filename = re.sub(r'[^a-zA-Z0-9]', ' ', clean_voice_filename)
        text = f"Hello, this is a preview of voice {clean_voice_filename}."
        
        rvccharacter_voice_gen = rvccharacter_voice_gen or "Disabled"
        rvccharacter_pitch = rvccharacter_pitch if rvccharacter_pitch is not None else 0
        
        print_message(f"Generating preview for voice: {clean_voice_filename}", "debug_tts", "GEN")
        output_file_path = this_dir / config.get_output_directory() / f'{output_file_name}.{model_engine.audio_format}'
        
        await generate_audio(
            text, voice, language, 
            model_engine.temperature_set, 
            model_engine.repetitionpenalty_set, 
            1.0, 0, output_file_path, 
            streaming=False
        )
        
        if config.rvc_settings.rvc_enabled:
            if rvccharacter_voice_gen.lower() in ["disabled", "disable"]:
                print_message("def apifunction_preview_voice RVC processing skipped", "debug_tts")
            else:
                print_message(f"def apifunction_preview_voice processing with RVC: {rvccharacter_voice_gen}", "debug_tts")
                rvccharacter_voice_gen = this_dir / "models" / "rvc_voices" / rvccharacter_voice_gen
                pth_path = rvccharacter_voice_gen if rvccharacter_voice_gen else config.rvc_settings.rvc_char_model_file
                
                if not os.path.exists(pth_path):
                    raise FileNotFoundError(f"RVC model not found: {pth_path}")
                    
                run_rvc(output_file_path, pth_path, rvccharacter_pitch, infer_pipeline)
                
        output_file_url = f'/audio/{output_file_name}.{model_engine.audio_format}'
        return JSONResponse(
            content={
                "status": "generate-success",
                "output_file_path": str(output_file_path),
                "output_file_url": str(output_file_url),
            },
            status_code=200,
        )
        
    except FileNotFoundError as e:
        print_message(str(e), "error", "GEN")
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        print_message(f"Error generating preview: {str(e)}", "error", "GEN")
        return JSONResponse(content={"error": str(e)}, status_code=500)

##################################################################
# API Endpoint - OpenAI Speech API compatable endpoint Validator #
##################################################################
class OpenAIInput(BaseModel):
    """Validates input parameters for OpenAI TTS generation."""
    model: str = Field(..., description="The TTS model to use. Currently ignored.")
    input: str = Field(..., max_length=4096, description="The text to generate audio for.")
    voice: str = Field(..., description="The voice to use when generating the audio.")
    response_format: str = Field(default="wav", description="The format of the audio. Currently only 'wav' is supported.")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="The speed of the generated audio.")

    @field_validator('voice')
    def validate_voice(cls, value):
        """Validate that the requested voice is supported by OpenAI TTS."""
        supported_voices = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]
        if value not in supported_voices:
            print_message(f"Invalid OpenAI voice requested: {value}", "debug_openai", "TTS")
            raise ValueError(f'Voice must be one of {supported_voices}')
        print_message(f"OpenAI voice validated: {value}", "debug_openai", "TTS")
        return value

class OpenAIGenerator:
    @staticmethod
    def validate_input(json_data: dict) -> Union[None, str]:
        """Validate OpenAI TTS request parameters."""
        debug_func_entry()
        try:
            OpenAIInput(**json_data)
            print_message("OpenAI input validation successful", "debug_openai", "TTS")
            return None
        except ValidationError as e:
            errors = []
            for err in e.errors():
                field = err['loc'][0]
                message = err['msg']
                description = OpenAIInput.model_fields[field].field_info.description
                errors.append(f"Error in field '{field}': {message}. Description: {description}")
            error_msg = ', '.join(errors)
            print_message(f"OpenAI input validation failed: {error_msg}", "error", "TTS")
            return error_msg

#########################################################################
# API Endpoint - OpenAI Speech API compatable endpoint /v1/audio/speech #
#########################################################################
@app.post("/v1/audio/speech", response_class=JSONResponse)
async def openai_tts_generate(request: Request):
    """Handle OpenAI-compatible TTS generation requests."""
    debug_func_entry()
    
    try:
        json_data = await request.json()
        print_message(f"Received JSON data: {json_data}", "debug_openai", "TTS")
        
        validation_error = OpenAIGenerator.validate_input(json_data)
        if validation_error:
            print_message(f"Validation error: {validation_error}", "debug_openai", "TTS")
            return JSONResponse(content={"error": validation_error}, status_code=400)
        
        input_text = json_data["input"]
        voice = json_data["voice"]
        response_format = json_data.get("response_format", "wav").lower()
        speed = json_data.get("speed", 1.0)
        
        print_message(f"Input text: {input_text}", "debug_openai", "TTS")
        print_message(f"Voice: {voice}", "debug_openai", "TTS")
        print_message(f"Speed: {speed}", "debug_openai", "TTS")
        
        cleaned_string = html.unescape(standard_filtering(input_text))        
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
            print_message(f"Unsupported voice: {voice}", "error", "TTS")
            return JSONResponse(content={"error": "Unsupported voice"}, status_code=400)
            
        print_message(f"Mapped voice: {mapped_voice}", "debug_openai", "TTS")
        
        unique_id = uuid.uuid4()
        timestamp = int(time.time())
        output_file_path = f'{this_dir / config.get_output_directory() / f"openai_output_{unique_id}_{timestamp}.{model_engine.audio_format}"}'

        if config.debugging.debug_fullttstext:
            print_message(cleaned_string, component="TTS")
        else:
            print_message(f"{cleaned_string[:90]}{'...' if len(cleaned_string) > 90 else ''}", component="TTS")
        
        await generate_audio(cleaned_string, mapped_voice, "en", model_engine.temperature_set,
                           model_engine.repetitionpenalty_set, speed, model_engine.pitch_set,
                           output_file_path, streaming=False)
                           
        print_message(f"Audio generated at: {output_file_path}", "debug_openai", "TTS")
        
        if config.rvc_settings.rvc_enabled:
            if config.rvc_settings.rvc_char_model_file.lower() in ["disabled", "disable"]:
                print_message("Pass rvccharacter_voice_gen", "debug_openai", "TTS")
            else:
                print_message("send to rvc", "debug_openai", "TTS")
                pth_path = this_dir / "models" / "rvc_voices" / config.rvc_settings.rvc_char_model_file
                pitch = config.rvc_settings.pitch
                run_rvc(output_file_path, pth_path, pitch, infer_pipeline)
        
        transcoded_file_path = await transcode_for_openai(output_file_path, response_format)
        print_message(f"Audio transcoded to: {transcoded_file_path}", "debug_openai", "TTS")
        
        return FileResponse(transcoded_file_path, media_type=f"audio/{response_format}", 
                          filename=f"output.{response_format}")
                          
    except Exception as e:
        print_message(f"An error occurred: {str(e)}", "error", "TTS")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

###########################################################################
# API Endpoint - OpenAI Speech API compatable endpoint Transcode Function #
###########################################################################
async def transcode_for_openai(input_file, output_format):
    """Transcode audio files for OpenAI API compatibility. Handles additional formats like ogg and m4a."""
    debug_func_entry()
    
    print_message("************************************", "debug_openai", "TTS")
    print_message("transcode_for_openai function called", "debug_openai", "TTS")
    print_message("************************************", "debug_openai", "TTS")
    print_message(f"Input file    : {input_file}", "debug_openai", "TTS")
    print_message(f"Output format : {output_format}", "debug_openai", "TTS")
    
    if not ffmpeg_installed:
        print_message("FFmpeg is not installed. Format conversion is not possible.", "error")
        raise Exception("FFmpeg is not installed. Format conversion is not possible.")
        
    input_extension = os.path.splitext(input_file)[1][1:].lower()
    print_message(f"Input file extension: {input_extension}", "debug_openai", "TTS")
    
    if input_extension == output_format.lower():
        print_message(f"Input file is already in the requested format: {output_format}", "debug_openai", "TTS")
        return input_file
        
    output_file = os.path.splitext(input_file)[0] + f".{output_format}"
    print_message(f"Output file: {output_file}", "debug_openai", "TTS")
    
    ffmpeg_path = os.path.join(this_dir, "system", "win_ffmpeg", "ffmpeg.exe") if sys.platform == "win32" else "ffmpeg"
    ffmpeg = FFmpeg(ffmpeg_path).option("y").input(input_file).output(output_file)
    
    print_message(f"Transcoding to {output_format}", "debug_openai", "TTS")
    
    codec_settings = {
        "opus": {"codec:a": "libopus", "b:a": "128k", "vbr": "on", "compression_level": 10, 
                "frame_duration": 60, "application": "voip"},
        "aac": {"codec:a": "aac", "b:a": "192k"},
        "flac": {"codec:a": "flac", "compression_level": 8},
        "wav": {"codec:a": "pcm_s16le"},
        "mp3": {"codec:a": "libmp3lame", "b:a": "192k"},
        "ogg": {"codec:a": "libvorbis"},
        "m4a": {"codec:a": "aac", "b:a": "192k"}
    }
    
    if output_format in codec_settings:
        print_message(f"Configuring {output_format.upper()} options", "debug_openai", "TTS")
        ffmpeg.output(output_file, codec_settings[output_format])
    else:
        print_message(f"Unsupported output format: {output_format}", "error", "TTS")
        raise ValueError(f"Unsupported output format: {output_format}")
        
    try:
        print_message("Starting transcoding process", "debug_openai", "TTS")
        await ffmpeg.execute()
        print_message("Transcoding completed successfully", "debug_openai", "TTS")
    except Exception as e:
        print_message(f"Error occurred during transcoding: {str(e)}", "error", "TTS")
        raise
        
    print_message("Transcoding process completed", "debug_openai", "TTS")
    print_message(f"Transcoded file: {output_file}", "debug_openai", "TTS")
    return output_file


######################################################################################
# API Endpoint - OpenAI Speech API compatable endpoint change engine voices Function #
######################################################################################
class VoiceMappings(BaseModel):
    """OpenAI to engine voice mapping configuration."""
    alloy: str
    echo: str
    fable: str
    nova: str
    onyx: str
    shimmer: str

@app.put("/api/openai-voicemap")
async def update_openai_voice_mappings(mappings: VoiceMappings):
    """Update OpenAI voice mappings in memory and settings file."""
    debug_func_entry()
    
    try:
        # Update in-memory mappings
        print_message("Updating in-memory voice mappings", "debug_openai", "TTS")
        model_engine.openai_alloy = mappings.alloy
        model_engine.openai_echo = mappings.echo
        model_engine.openai_fable = mappings.fable
        model_engine.openai_nova = mappings.nova
        model_engine.openai_onyx = mappings.onyx
        model_engine.openai_shimmer = mappings.shimmer
        
        # Update settings file
        settings_file = this_dir / "system" / "tts_engines" / model_engine.engine_loaded / "model_settings.json"
        with open(settings_file, "r+") as f:
            settings = json.load(f)
            settings["openai_voices"] = mappings.dict()
            f.seek(0)
            json.dump(settings, f, indent=4)
            f.truncate()
            
        print_message("Voice mappings updated successfully", "debug_openai", "TTS")
        return {"message": "OpenAI voice mappings updated successfully"}
        
    except Exception as e:
        print_message(f"Failed to update voice mappings: {str(e)}", "error", "TTS")
        raise HTTPException(status_code=500, detail=f"Failed to update model settings file: {str(e)}")

#######################
# Play at the console #
#######################
def play_audio(file_path, volume):
    """Play audio file with specified volume through system speakers at the console/terminal window"""
    debug_func_entry()
    
    try:
        print_message(f"\033[94mFile path is: {file_path}", "debug_tts", "GEN")
        print_message(f"\033[94mVolume is: {volume}", "debug_tts", "GEN")
        
        normalized_file_path = os.path.normpath(file_path)
        print_message(f"\033[94mNormalized file path is: {normalized_file_path}", "debug_tts", "GEN")
        
        directory = os.path.dirname(normalized_file_path)
        if not os.path.isdir(directory):
            print_message(f"\033[94mError: Directory does not exist: {directory}", "error", "GEN")
            return
            
        print_message(f"\033[94mDirectory contents: {os.listdir(directory)}", "debug_tts", "GEN")
        
        if not os.path.isfile(normalized_file_path):
            print_message(f"\033[94mError: File does not exist: {normalized_file_path}\033[0m", "error", "GEN")
            return
            
        if normalized_file_path.lower().endswith('.aac'):
            print_message("\033[94mPlay Audio  : \033[0mAAC format files cannot be played at the console. Please choose another format", "error", "GEN")
            return
            
        print_message("\033[94mPlay Audio  : \033[0mPlaying audio at console", "debug_tts", "GEN")
        data, fs = sf.read(normalized_file_path)
        sd.play(volume * data, fs)
        sd.wait()
        
    except Exception as e:
        print_message(f"\033[94mError playing audio file: {e}\033[0m", "error", "GEN")
        

class Request(BaseModel):
    """Define the structure of the 'Request' class if needed"""
    pass

class TTSGenerator:
    @staticmethod
    def validate_json_input(json_data: Union[Dict, str]) -> Union[None, str]:
        """Validate TTS generation input parameters."""
        debug_func_entry()
        
        try:
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            JSONInput(**json_data)
            print_message("JSON input validation successful", "debug_tts", "GEN")
            return None
        except ValidationError as e:
            print_message(f"JSON validation failed: {str(e)}", "error", "GEN")
            return str(e)

#################################################################
# /api/tts-generate Generation API Endpoint Narration Filtering #
#################################################################
def process_text(text):
    """Parse text into narrator, character, and ambiguous speech segments. Handles mixed formatting and nested quotes."""
    debug_func_entry()
    
    # Normalize and clean text
    text = html.unescape(text)
    text = re.sub(r'\.{3,}', '.', text)
    
    # Pattern for identifying speech segments
    combined_pattern = r'(\*[^*"]+\*|"[^"*]+")'
    ordered_parts = []
    start = 0
    
    # Process text segments
    for match in re.finditer(combined_pattern, text):
        # Handle pre-match ambiguous text
        if start < match.start():
            ambiguous_text = text[start:match.start()].strip()
            if ambiguous_text:
                ordered_parts.append(('ambiguous', ambiguous_text))
                print_message(f"Added ambiguous segment: {ambiguous_text}", "debug_tts", "GEN")
        
        # Process matched segment
        matched_text = match.group(0)
        if matched_text.startswith('*') and matched_text.endswith('*'):
            text_part = matched_text.strip('*').strip()
            ordered_parts.append(('narrator', text_part))
            print_message(f"Added narrator segment: {text_part}", "debug_tts", "GEN")
        elif matched_text.startswith('"') and matched_text.endswith('"'):
            text_part = matched_text.strip('"').strip()
            ordered_parts.append(('character', text_part))
            print_message(f"Added character segment: {text_part}", "debug_tts", "GEN")
        else:
            text_part = matched_text.strip('*').strip('"')
            if '*' in matched_text:
                ordered_parts.append(('narrator', text_part))
                print_message(f"Added mixed narrator segment: {text_part}", "debug_tts", "GEN")
            else:
                ordered_parts.append(('character', text_part))
                print_message(f"Added mixed character segment: {text_part}", "debug_tts", "GEN")
        
        start = match.end()
    
    # Handle remaining text
    if start < len(text):
        ambiguous_text = text[start:].strip()
        if ambiguous_text:
            ordered_parts.append(('ambiguous', ambiguous_text))
            print_message(f"Added final ambiguous segment: {ambiguous_text}", "debug_tts", "GEN")
    
    return ordered_parts

def standard_filtering(text_input):
    """Remove special characters and normalize text formatting."""
    debug_func_entry()
    
    text_output = (text_input
                  .replace("***", "")
                  .replace("**", "")
                  .replace("*", "")
                  .replace("\n\n", "\n")
                  .replace("&#x27;", "'")
                  )
    
    print_message(f"Filtered text: {text_output[:50]}{'...' if len(text_output) > 50 else ''}", 
                 "debug_tts", "GEN")
    return text_output

#################################################################
# /api/tts-generate Generation API Endpoint Narration Combining #
#################################################################
def combine(output_file_timestamp, output_file_name, audio_files, target_sample_rate=44100):
    """Combine multiple audio files into one, with optional resampling and timestamping."""
    debug_func_entry()
    
    audio = np.array([])
    try:
        for audio_file in audio_files:
            normalized_audio_file = os.path.normpath(audio_file)
            print_message(f"Processing file: {normalized_audio_file}", "debug_concat", "TTS")
            
            if not os.path.isfile(normalized_audio_file):
                print_message(f"Error: File does not exist: {normalized_audio_file}", "error", "TTS")
                return None, None, None
                
            audio_data, current_sample_rate = sf.read(normalized_audio_file)
            print_message(f"Read file: {normalized_audio_file}, Sample rate: {current_sample_rate}, Data shape: {audio_data.shape}", 
                         "debug_concat", "TTS")
            
            if current_sample_rate != target_sample_rate:
                print_message(f"Resampling file from {current_sample_rate} to {target_sample_rate} Hz", "debug_concat", "TTS")
                audio_data = librosa.resample(audio_data, orig_sr=current_sample_rate, target_sr=target_sample_rate)
                
            audio = audio_data if audio.size == 0 else np.concatenate((audio, audio_data))
            
        # Prepare output paths
        if output_file_timestamp:
            timestamp = int(time.time())
            filename = f'{output_file_name}_{timestamp}_combined.wav'
        else:
            filename = f'{output_file_name}_combined.wav'
            
        output_file_path = os.path.join(this_dir / "outputs" / filename)
        sf.write(output_file_path, audio, target_sample_rate)
        
        # Generate URLs based on API configuration
        base_url = f"http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}" if config.api_def.api_use_legacy_api else ""
        output_file_url = f"{base_url}/audio/{filename}"
        output_cache_url = f"{base_url}/audiocache/{filename}"
        
        print_message(f"Output file path: {output_file_path}", "debug_concat", "TTS")
        print_message(f"Output file URL: {output_file_url}", "debug_concat", "TTS")
        print_message(f"Output cache URL: {output_cache_url}", "debug_concat", "TTS")
        
        return output_file_path, output_file_url, output_cache_url
        
    except Exception as e:
        print_message(f"Error combining audio files: {str(e)}", "error", "TTS")
        return None, None, None

##################################
# Central RVC Generation request #
##################################
def run_rvc(input_tts_path, pth_path, pitch, infer_pipeline):
    """Run RVC (Retrieval-based Voice Conversion) processing on input audio."""
    debug_func_entry()
    
    generate_start_time = time.time()
    
    # Config settings
    settings = {
        "f0up_key": pitch,
        "filter_radius": config.rvc_settings.filter_radius,
        "index_rate": config.rvc_settings.index_rate,
        "rms_mix_rate": config.rvc_settings.rms_mix_rate,
        "protect": config.rvc_settings.protect,
        "hop_length": config.rvc_settings.hop_length,
        "f0method": config.rvc_settings.f0method,
        "split_audio": config.rvc_settings.split_audio,
        "f0autotune": config.rvc_settings.autotune,
        "embedder_model": config.rvc_settings.embedder_model,
        "training_data_size": config.rvc_settings.training_data_size,
    }
    
    input_tts_path = str(input_tts_path)
    pth_path = str(pth_path)
    output_rvc_path = input_tts_path
    
    if not os.path.isfile(pth_path):
        print_message(f"Model file {pth_path} does not exist. Exiting.", "error", "GEN")
        return
        
    model_dir = os.path.dirname(pth_path)
    pth_filename = os.path.basename(pth_path)
    index_files = [file for file in os.listdir(model_dir) if file.endswith(".index")]
    
    if len(index_files) == 1:
        index_path = str(os.path.join(model_dir, index_files[0]))
        index_filename_print = os.path.basename(index_path)
        index_size_print = settings["training_data_size"]
    else:
        if len(index_files) > 1:
            print_message(f"RVC Convert: Multiple RVC index files found in the models folder where {pth_filename} is", "warning", "GEN")
            print_message("RVC Convert: located. Unable to determine which index to use. Continuing without an index file.", "warning", "GEN")
        index_path = ""
        index_filename_print = "None used"
        index_size_print = "N/A"
    
    if config.debugging.debug_rvc:
        print_message("RVC Pipeline Configuration:", "debug_rvc", "GEN")
        for key, value in settings.items():
            print_message(f"{key}: {value}", "debug_rvc", "GEN")
        print_message(f"input_tts_path: {input_tts_path}", "debug_rvc", "GEN")
        print_message(f"output_rvc_path: {output_rvc_path}", "debug_rvc", "GEN")
        print_message(f"pth_path: {pth_path}", "debug_rvc", "GEN")
        print_message(f"index_path: {index_path}", "debug_rvc", "GEN")
    
    infer_pipeline(settings["f0up_key"], settings["filter_radius"], settings["index_rate"], 
                  settings["rms_mix_rate"], settings["protect"], settings["hop_length"], 
                  settings["f0method"], input_tts_path, output_rvc_path, pth_path, index_path, 
                  settings["split_audio"], settings["f0autotune"], settings["embedder_model"], 
                  settings["training_data_size"], config.debugging.debug_rvc)
    
    generate_elapsed_time = time.time() - generate_start_time
    print_message(f"RVC Convert: Model: {pth_filename} Index: {index_filename_print}", "debug_rvc", "GEN")
    print_message(f"RVC Convert: {generate_elapsed_time:.2f} seconds. Method: {settings['f0method']} Index size used {index_size_print}", 
                 "debug_rvc", "GEN")
    return

####################################################################
# /api/tts-generate Generation API Endpoint Narration RVC handling #
####################################################################
def process_rvc_narrator(part_type, voice_gen, pitch, default_model_file, output_file, infer_pipeline):
    """Process RVC voice conversion for narrator segments."""
    debug_func_entry()
    
    if voice_gen.lower() in ["disabled", "disable"]:
        print_message(f"RVC processing skipped for {part_type} segment - voice disabled", "debug_rvc", "GEN")
        return
        
    try:
        voice_gen_path = this_dir / "models" / "rvc_voices" / voice_gen
        pth_path = voice_gen_path if voice_gen else default_model_file
        
        if not os.path.exists(pth_path):
            print_message(f"RVC model not found: {pth_path}", "error", "GEN")
            return
            
        print_message(f"Processing {part_type} segment with RVC model: {os.path.basename(pth_path)}", "debug_rvc", "GEN")
        run_rvc(output_file, pth_path, pitch, infer_pipeline)
        
    except Exception as e:
        print_message(f"Error processing RVC for {part_type} segment: {str(e)}", "error", "GEN")
        
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
        """Validate autoplay volume is within acceptable range (0.1-1.0)."""
        debug_func_entry()
        try:
            if not (0.1 <= value <= 1.0):
                print_message(f"Invalid autoplay volume: {value}", "error", "GEN")
                raise ValueError("Autoplay volume must be between 0.1 and 1.0")
            print_message(f"Autoplay volume validated: {value}", "debug_tts", "GEN")
            return value
        except Exception as e:
            print_message(f"Error validating autoplay volume: {str(e)}", "error", "GEN")
            raise

    @classmethod
    def validate_rvccharacter_voice_gen(cls, v):
        """Validate RVC character voice file format (folder/file.pth or 'Disabled')."""
        debug_func_entry()
        try:
            if v.lower() == "disabled":
                print_message("RVC character voice disabled", "debug_rvc", "GEN")
                return v
            pattern = re.compile(r'^.*\.(pth)$')
            if not pattern.match(v):
                print_message(f"Invalid RVC character voice format: {v}", "error", "GEN")
                raise ValueError("rvccharacter_voice_gen needs to be the name of a valid pth file in the 'folder\\file.pth' format or 'Disabled'.")
            print_message(f"RVC character voice validated: {v}", "debug_rvc", "GEN")
            return v
        except Exception as e:
            print_message(f"Error validating RVC character voice: {str(e)}", "error", "GEN")
            raise

    @classmethod
    def validate_rvnarrator_voice_gen(cls, v):
        """Validate RVC narrator voice file format (folder/file.pth or 'Disabled')."""
        debug_func_entry()
        try:
            if v.lower() == "disabled":
                print_message("RVC narrator voice disabled", "debug_rvc", "GEN")
                return v
            pattern = re.compile(r'^.*\.(pth)$')
            if not pattern.match(v):
                print_message(f"Invalid RVC narrator voice format: {v}", "error", "GEN")
                raise ValueError("rvcnarrator_voice_gen needs to be the name of a valid pth file in the 'folder\\file.pth' format or 'Disabled'.")
            print_message(f"RVC narrator voice validated: {v}", "debug_rvc", "GEN")
            return v
        except Exception as e:
            print_message(f"Error validating RVC narrator voice: {str(e)}", "error", "GEN")
            raise

    @classmethod
    def validate_pitches(cls, value):
        """Validate pitch value is within acceptable range (-24 to 24)."""
        debug_func_entry()
        try:
            num_value = float(value)
            if not -24 <= num_value <= 24:
                print_message(f"Invalid pitch value: {value}", "error", "GEN")
                raise ValueError("Pitch needs to be a number between -24 and 24.")
            print_message(f"Pitch validated: {value}", "debug_rvc", "GEN")
            return value
        except ValueError:
            print_message(f"Invalid pitch format: {value}", "error", "GEN")
            raise ValueError("Pitch must be a number or a string representing a number.")
        except Exception as e:
            print_message(f"Error validating pitch: {str(e)}", "error", "GEN")
            raise

def validate_json_input(json_input_data):
    """Validate JSON input against JSONInput model schema."""
    debug_func_entry()
    try:
        JSONInput(**json_input_data)
        return None
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            field = error["loc"][0]
            description = JSONInput.__fields__[field].description
            error_messages.append(f"{field}: {description}")
        error_message = "\n".join(error_messages)
        print_message(f"Error with API request: {error_message}", "error", "API")
        return error_message

# API Configuration Getters
def get_api_text_filtering(): """Get text filtering setting."""; return config.api_def.api_text_filtering
def get_api_narrator_enabled(): """Get narrator mode status."""; return config.api_def.api_narrator_enabled
def get_api_text_not_inside(): """Get text parsing mode."""; return config.api_def.api_text_not_inside
def get_api_language(): """Get API language setting."""; return config.api_def.api_language
def get_api_output_file_name(): """Get output filename template."""; return config.api_def.api_output_file_name
def get_api_output_file_timestamp(): """Get file timestamp setting."""; return config.api_def.api_output_file_timestamp
def get_api_autoplay(): """Get autoplay status."""; return config.api_def.api_autoplay
def get_api_autoplay_volume(): """Get autoplay volume level."""; return config.api_def.api_autoplay_volume

# Engine Parameter Getters
def get_params_speed(): """Get speech generation speed."""; return model_engine.generationspeed_set
def get_params_temperature(): """Get model temperature."""; return model_engine.temperature_set
def get_params_repetition(): """Get repetition penalty."""; return float(str(model_engine.repetitionpenalty_set).replace(',', '.'))
def get_params_pitch(): """Get voice pitch setting."""; return model_engine.pitch_set
def get_character_voice_gen(): """Get default character voice."""; return model_engine.def_character_voice
def get_narrator_voice_gen(): """Get default narrator voice."""; return model_engine.def_narrator_voice

# RVC Settings Getters
def get_rvccharacter_voice_gen(): """Get RVC character model."""; return config.rvc_settings.rvc_char_model_file
def get_rvccharacter_pitch(): """Get RVC character pitch."""; return config.rvc_settings.pitch
def get_rvcnarrator_voice_gen(): """Get RVC narrator model."""; return config.rvc_settings.rvc_narr_model_file
def get_rvcnarrator_pitch(): """Get RVC narrator pitch."""; return config.rvc_settings.pitch

#############################################
# /api/tts-generate Generation API Endpoint #
#############################################
async def tts_validate_and_prepare_input(
    text_input: str,
    request_params: dict,
    default_params: dict
) -> Tuple[dict, Union[None, str]]:
    """Validate TTS input parameters and prepare settings."""
    debug_func_entry()
    
    if config.debugging.debug_tts or config.debugging.debug_tts_variables:
        print_message("\033[94mPre-validation parameter check > tts_validate_and_prepare_input > tts_server.py\033[0m", "debug_tts_variables", "GEN")
        print_message("API Configuration:", "debug_tts_variables", "GEN")
        for key, value in {
            "max_characters": config.api_def.api_max_characters,
            "length_stripping": config.api_def.api_length_stripping,
            "legacy_api": config.api_def.api_use_legacy_api,
            "legacy_api_ip": config.api_def.api_legacy_ip_address,
            "allowed_filter": config.api_def.api_allowed_filter
        }.items():
            print_message(f"{key}: {value}", "debug_tts_variables", "GEN")
            
        print_message("\033[94mIncoming TTS Variables > tts_validate_and_prepare_input > tts_server.py\033[0m", "debug_tts_variables", "GEN")
        for key, value in request_params.items():
            print_message(f"{key}: {value}", "debug_tts_variables", "GEN")
    
    # Merge request parameters with defaults
    params = {}
    for key, default_value in default_params.items():
        params[key] = request_params.get(key) if request_params.get(key) is not None else default_value
    
    # Validate merged parameters
    json_input_data = {
        "text_input": text_input,
        **params
    }
    
    validation_result = validate_json_input(json_input_data)
    
    if config.debugging.debug_tts or config.debugging.debug_tts_variables:
        print_message("\033[94mPost Validation Variables > tts_validate_and_prepare_input > tts_server.py\033[0m", "debug_tts_variables", "GEN")
        for key, value in params.items():
            print_message(f"{key}: {value}", "debug_tts_variables", "GEN")
                
    return params, validation_result

async def tts_handle_output_paths(output_file_name: str, timestamp: bool = True) -> Tuple[Path, str, str]:
    """Generate output paths and URLs for TTS files."""
    debug_func_entry()

    if timestamp:
        hash_object = hashlib.sha256(str(uuid.uuid4()).encode())
        short_uuid = hash_object.hexdigest()[:5]
        timestamp_str = str(int(time.time()))
        filename = f'{output_file_name}_{timestamp_str}{short_uuid}.{model_engine.audio_format}'
    else:
        filename = f"{output_file_name}.{model_engine.audio_format}"

    output_path = this_dir / config.get_output_directory() / filename
    
    if config.api_def.api_use_legacy_api:
        base_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}'
        file_url = f'{base_url}/audio/{filename}'
        cache_url = f'{base_url}/audiocache/{filename}'
    else:
        file_url = f'/audio/{filename}'
        cache_url = f'/audiocache/{filename}'

    print_message(f"Generated output path: {output_path}", "debug_tts", "GEN")
    print_message(f"Generated URLs: {file_url}, {cache_url}", "debug_tts", "GEN")

    return output_path, file_url, cache_url

def tts_clean_text(text: str, filtering_type: str) -> str:
    """Clean and filter input text based on specified filtering type."""
    debug_func_entry()
    
    print_message(f"Cleaning text with filter type: {filtering_type}", "debug_tts", "GEN")
    
    if filtering_type == "html":
        cleaned = html.unescape(standard_filtering(text))
    elif filtering_type == "standard":
        cleaned = text
    else:
        print_message("No filtering applied", "debug_tts", "GEN")
        return text
        
    # Common cleaning steps
    cleaned = re.sub(r'([!?.])\1+', r'\1', cleaned)
    cleaned = re.sub(rf'{config.api_def.api_allowed_filter}', '', cleaned)
    cleaned = re.sub(r'\n+', ' ', cleaned)
    
    print_message(f"Cleaned text: {cleaned}", "debug_tts", "GEN")

    return cleaned

async def tts_process_narrator_mode(params: dict, text_input: str) -> Tuple[Path, str, str]:
    """Process text with narrator mode, handling different voice types."""
    debug_func_entry()
    
    if config.debugging.debug_fullttstext:
        print_message(text_input, component="TTS")
    else:
        print_message(f"{text_input[:90]}{'...' if len(text_input) > 90 else ''}", component="TTS")    
    
    if params['narrator_enabled'].lower() == "silent" and params['text_not_inside'].lower() == "silent":
        print_message("Both Narrator & Text-not-inside are set to silent. If you get no TTS, this is why.", "warning", "GEN")
    
    if model_engine.lowvram_enabled and model_engine.device == "cpu":
        await model_engine.handle_lowvram_change()
        
    model_engine.tts_narrator_generatingtts = True
    print_message("Processing with narrator mode", "debug_tts", "GEN")
    
    processed_parts = process_text(text_input)
    audio_files = []
    
    for part_type, part in processed_parts:
        if len(part.strip()) <= int(config.api_def.api_length_stripping):
            continue
            
        voice_to_use = tts_get_voice_for_part(part_type, params, part)
        if not voice_to_use:
            continue
            
        output_file = await tts_generate_part(part, voice_to_use, params)
        if output_file:
            await tts_apply_rvc(output_file, part_type, params)
            audio_files.append(output_file)
    
    model_engine.tts_narrator_generatingtts = False
    
    if audio_files:
        print_message("\033[92mNarrated TTS generation complete\033[0m", component="GEN")
        return await tts_finalize_output(audio_files, params)
    
    print_message("\033[92mNarrated TTS generation complete\033[0m", component="GEN")
    return await tts_finalize_output(audio_files, params)

def tts_get_voice_for_part(part_type: str, params: dict, part: str) -> Optional[str]:
    """Determine appropriate voice for text part."""
    preview = part[:50] + ('...' if len(part) > 50 else '') if not config.debugging.debug_fullttstext else part
    
    if part_type == 'narrator':
        if params['narrator_enabled'].lower() == "silent":
            print_message(f"\033[95mNarrator Silent:\033[0m {preview}", component="GEN")
            return None
        voice = params['narrator_voice_gen']
        print_message(f"\033[92mNarrator:\033[0m {preview}", component="GEN")
        
    elif part_type == 'character':
        voice = params['character_voice_gen']
        print_message(f"\033[96mCharacter:\033[0m {preview}", component="GEN")
        
    else:
        if params['text_not_inside'] == "silent":
            print_message(f"\033[95mText-not-inside Silent:\033[0m {preview}", component="GEN")
            return None
            
        voice = params['character_voice_gen'] if params['text_not_inside'] == "character" else params['narrator_voice_gen']
        voice_type = "\033[96mCharacter (Text-not-inside)\033[0m" if params['text_not_inside'] == "character" else "\033[92mNarrator (Text-not-inside)\033[0m"
        print_message(f"{voice_type}: {preview}", component="GEN")
    
    return voice

async def tts_generate_part(part: str, voice: str, params: dict) -> Optional[Path]:
    """Generate audio for text part."""
    cleaned_part = tts_clean_text(part, params['text_filtering'])
    output_file = await tts_handle_output_paths(params['output_file_name'])
    
    try:
        await generate_audio(
            cleaned_part, voice, params['language'],
            params['temperature'], params['repetition_penalty'],
            params['speed'], params['pitch'],
            output_file[0], False
        )
        return output_file[0]
    except Exception as e:
        print_message(f"Error generating audio: {str(e)}", "error", "GEN")
        return None

async def tts_apply_rvc(output_file: Path, part_type: str, params: dict):
    """Apply RVC processing if enabled."""
    if not config.rvc_settings.rvc_enabled:
        return
        
    if part_type == 'character':
        process_rvc_narrator(part_type, params['rvccharacter_voice_gen'], 
                           params['rvccharacter_pitch'], 
                           config.rvc_settings.rvc_char_model_file, 
                           output_file, infer_pipeline)
    elif part_type == 'narrator':
        process_rvc_narrator(part_type, params['rvcnarrator_voice_gen'], 
                           params['rvcnarrator_pitch'], 
                           config.rvc_settings.rvc_narr_model_file, 
                           output_file, infer_pipeline)
    else:
        if params['text_not_inside'] == 'character':
            process_rvc_narrator('character', params['rvccharacter_voice_gen'], 
                               params['rvccharacter_pitch'], 
                               config.rvc_settings.rvc_char_model_file, 
                               output_file, infer_pipeline)
        elif params['text_not_inside'] == 'narrator':
            process_rvc_narrator('narrator', params['rvcnarrator_voice_gen'], 
                               params['rvcnarrator_pitch'], 
                               config.rvc_settings.rvc_narr_model_file, 
                               output_file, infer_pipeline)
            
async def tts_process_standard_mode(params: dict, text_input: str) -> Union[StreamingResponse, Tuple[Path, str, str]]:
    """Process text with standard mode, without narrator functionality."""
    debug_func_entry()
    
    print_message("Standard generation mode", "debug_tts", "GEN")
    
    output_file_path, output_file_url, output_cache_url = await tts_handle_output_paths(
        params['output_file_name'], 
        params['output_file_timestamp']
    )
    
    cleaned_text = tts_clean_text(text_input, params['text_filtering'])
    
    if config.debugging.debug_fullttstext:
        print_message(cleaned_text, component="TTS")
    else:
        print_message(f"{cleaned_text[:90]}{'...' if len(cleaned_text) > 90 else ''}", component="TTS")
    
    try:
        if params.get('streaming', False):
            stream = await generate_audio(
                cleaned_text, params['character_voice_gen'], params['language'],
                params['temperature'], params['repetition_penalty'],
                params['speed'], params['pitch'],
                output_file_path, True
            )
            return StreamingResponse(stream, media_type="audio/wav")
            
        await generate_audio(
            cleaned_text, params['character_voice_gen'], params['language'],
            params['temperature'], params['repetition_penalty'],
            params['speed'], params['pitch'],
            output_file_path, False
        )
        
        if config.rvc_settings.rvc_enabled:
            await tts_handle_rvc_processing(output_file_path, params)
            
        return await tts_handle_audio_output(output_file_path, params['autoplay'], params['autoplay_volume'])
        
    except Exception as e:
        print_message(f"Error in standard processing: {str(e)}", "error", "GEN")
        raise

async def tts_handle_rvc_processing(output_file_path: Path, params: dict):
    """Handle RVC voice conversion if enabled."""
    if params['rvccharacter_voice_gen'].lower() in ["disabled", "disable"]:
        print_message("RVC processing skipped", "debug_tts", "GEN")
        return
        
    print_message("Processing with RVC", "debug_tts", "GEN")
    rvc_model_path = this_dir / "models" / "rvc_voices" / params['rvccharacter_voice_gen']
    pth_path = rvc_model_path if rvc_model_path else config.rvc_settings.rvc_char_model_file
    run_rvc(output_file_path, pth_path, params['rvccharacter_pitch'], infer_pipeline)

async def tts_handle_audio_output(output_file_path: Path, autoplay: bool, volume: float) -> Tuple[Path, str, str]:
    """Handle audio transcoding and playback."""
    model_format = str(model_engine.audio_format).lower()
    output_format = str(config.transcode_audio_format).lower()
    
    output_file_url = f'/audio/{output_file_path.name}'
    output_cache_url = f'/audiocache/{output_file_path.name}'
    
    if config.api_def.api_use_legacy_api:
        base_url = f'http://{config.api_def.api_legacy_ip_address}:{config.api_def.api_port_number}'
        output_file_url = f'{base_url}{output_file_url}'
        output_cache_url = f'{base_url}{output_cache_url}'
    
    if output_format != "disabled" and model_format != output_format:
        output_file_path, output_file_url, output_cache_url = await transcode_audio_if_necessary(
            output_file_path, model_format, output_format
        )
    
    if sounddevice_installed and autoplay:
        play_audio(output_file_path, volume)
        
    return output_file_path, output_file_url, output_cache_url

async def tts_finalize_output(audio_files: List[Path], params: dict) -> Tuple[Path, str, str]:
    """Combine audio files and handle final processing."""
    debug_func_entry()
    
    output_file_path, output_file_url, output_cache_url = combine(
        params['output_file_timestamp'],
        params['output_file_name'],
        audio_files
    )
    
    model_format = str(model_engine.audio_format).lower()
    output_format = str(config.transcode_audio_format).lower()
    
    if output_format != "disabled" and model_format != output_format:
        if not model_engine.tts_narrator_generatingtts:
            output_file_path, output_file_url, output_cache_url = await transcode_audio_if_necessary(
                output_file_path, model_format, output_format
            )
    
    if sounddevice_installed and params['autoplay']:
        play_audio(output_file_path, params['autoplay_volume'])
        
    return output_file_path, output_file_url, output_cache_url

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
    """Generate TTS audio with optional narrator mode and RVC processing."""
    try:
        # Prepare request parameters
        request_params = {
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
            "pitch": pitch
        }

        default_params = {
            "text_filtering": _text_filtering,
            "character_voice_gen": _character_voice_gen,
            "rvccharacter_voice_gen": _rvccharacter_voice_gen,
            "rvccharacter_pitch": _rvccharacter_pitch,
            "narrator_enabled": _narrator_enabled,
            "narrator_voice_gen": _narrator_voice_gen,
            "rvcnarrator_voice_gen": _rvcnarrator_voice_gen,
            "rvcnarrator_pitch": _rvcnarrator_pitch,
            "text_not_inside": _text_not_inside,
            "language": _language,
            "output_file_name": _output_file_name,
            "output_file_timestamp": _output_file_timestamp,
            "autoplay": _autoplay,
            "autoplay_volume": _autoplay_volume,
            "speed": _speed,
            "temperature": _temperature,
            "repetition_penalty": _repetition_penalty,
            "pitch": _pitch
        }

        # Validate and prepare input
        params, validation_error = await tts_validate_and_prepare_input(
            text_input, request_params, default_params
        )
        
        if validation_error:
            return JSONResponse(content={"error": validation_error}, status_code=400)

        # Process based on mode
        if params["narrator_enabled"].lower() in ["true", "silent"]:
            try:
                output_file_path, output_file_url, output_cache_url = await tts_process_narrator_mode(
                    params, text_input
                )
            finally:
                # Ensure model returns to CPU if using low VRAM
                if model_engine.lowvram_enabled and model_engine.device == "cuda" and model_engine.lowvram_capable:
                    await model_engine.handle_lowvram_change()
        else:
            output_file_path, output_file_url, output_cache_url = await tts_process_standard_mode(
                params, text_input
            )

        if isinstance(output_file_path, StreamingResponse):
            return output_file_path

        return JSONResponse(
            content={
                "status": "generate-success",
                "output_file_path": str(output_file_path),
                "output_file_url": str(output_file_url),
                "output_cache_url": str(output_cache_url)
            },
            status_code=200
        )

    except Exception as e:
        print_message(f"Error in TTS generation: {str(e)}", "error", "GEN")
        return JSONResponse(
            content={"status": "generate-failure", "error": "An error occurred"},
            status_code=500
        )

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
    """Individual TTS item structure for saving."""
    id: int
    fileUrl: str
    text: str
    characterVoice: str
    language: str

class TTSData(BaseModel):
    """Collection of TTS items."""
    ttsList: List[TTSItem]

@app.post("/api/save-tts-data")
async def apifunction_save_tts_data(tts_data: List[TTSItem]):
    """Save TTS data to JSON file in output directory."""
    debug_func_entry()
    
    try:
        print_message(f"Saving {len(tts_data)} TTS items", "debug_tts", "GEN")
        
        # Convert Pydantic models to dictionaries
        tts_data_list = [item.dict() for item in tts_data]
        tts_data_json = json.dumps(tts_data_list, indent=4)
        
        output_path = this_dir / config.get_output_directory() / "ttsList.json"
        
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(tts_data_json)
            
        print_message(f"TTS data saved to {output_path}", "debug_tts", "GEN")
        return {"message": "Data saved successfully"}
        
    except Exception as e:
        print_message(f"Error saving TTS data: {str(e)}", "error", "GEN")
        raise HTTPException(status_code=500, detail="Failed to save TTS data")

########################################
# Trigger TTS Gen Text/Speech Analysis #
########################################
@app.get("/api/trigger-analysis")
async def apifunction_trigger_analysis(threshold: int = Query(default=98)):
    """Trigger Whisper analysis to compare generated TTS against original text.
    
    Args:
        threshold: Minimum acceptable match percentage (default: 98)
    """
    debug_func_entry()
    
    try:
        # Setup environment
        venv_path = sys.prefix
        env = os.environ.copy()
        env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
        
        # Define paths
        ttslist_path = this_dir / config.get_output_directory() / "ttsList.json"
        wavfile_path = this_dir / config.get_output_directory()
        
        if not ttslist_path.exists():
            print_message("TTS list file not found", "error", "GEN")
            raise HTTPException(status_code=404, detail="TTS list file not found")
            
        print_message(f"Starting TTS analysis with threshold: {threshold}%", "debug_tts", "GEN")
        
        # Run analysis script
        result = subprocess.run(
            [
                sys.executable,
                "tts_diff.py",
                f"--threshold={threshold}",
                f"--ttslistpath={ttslist_path}",
                f"--wavfilespath={wavfile_path}"
            ],
            cwd=this_dir / "system" / "tts_diff",
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print_message(f"Analysis script error: {result.stderr}", "error", "GEN")
            raise HTTPException(status_code=500, detail="Analysis script failed")
            
        # Read analysis results
        try:
            summary_path = this_dir / config.get_output_directory() / "analysis_summary.json"
            with open(summary_path, "r") as summary_file:
                summary_data = json.load(summary_file)
                print_message("Analysis summary loaded successfully", "debug_tts", "GEN")
        except FileNotFoundError:
            print_message("Analysis summary file not found", "error", "GEN")
            summary_data = {"error": "Analysis summary file not found."}
        except json.JSONDecodeError as e:
            print_message(f"Error parsing analysis summary: {str(e)}", "error", "GEN")
            summary_data = {"error": "Invalid analysis summary format."}
            
        return {"message": "Analysis Completed", "summary": summary_data}
        
    except Exception as e:
        print_message(f"Error during analysis: {str(e)}", "error", "GEN")
        raise HTTPException(status_code=500, detail=str(e))

###########################################
# TTS Generator SRT Subtitiles generation #
###########################################
@app.get("/api/srt-generation")
async def apifunction_srt_generation():
    """Generate SRT subtitles for TTS audio files using speech-to-text analysis."""
    debug_func_entry()
    
    try:
        # Setup environment
        venv_path = sys.prefix
        env = os.environ.copy()
        env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
        
        # Define paths
        ttslist_path = this_dir / config.get_output_directory() / "ttsList.json"
        wavfile_path = this_dir / config.get_output_directory()
        srt_file_path = this_dir / config.get_output_directory() / "subtitles.srt"
        
        # Validate input file existence
        if not ttslist_path.exists():
            print_message("TTS list file not found", "error", "GEN")
            raise HTTPException(status_code=404, detail="TTS list file not found")
            
        print_message("Starting SRT generation process", "debug_tts", "GEN")
        
        # Run SRT generation script
        result = subprocess.run(
            [
                sys.executable,
                "tts_srt.py",
                f"--ttslistpath={ttslist_path}",
                f"--wavfilespath={wavfile_path}"
            ],
            cwd=this_dir / "system" / "tts_srt",
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print_message(f"SRT generation script error: {result.stderr}", "error", "GEN")
            raise HTTPException(status_code=500, detail="SRT generation failed")
        
        # Verify output file
        if not srt_file_path.exists():
            print_message("Generated SRT file not found", "error", "GEN")
            raise HTTPException(status_code=404, detail="Subtitle file not found")
            
        print_message("SRT generation completed successfully", "debug_tts", "GEN")
        return FileResponse(
            path=srt_file_path,
            filename="subtitles.srt",
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        print_message(f"Error during SRT generation: {str(e)}", "error", "GEN")
        raise HTTPException(status_code=500, detail=str(e))

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
    """Return HTML template with current configuration settings."""
    debug_func_entry()
    
    try:
        settings_data = config.to_dict()
        print_message("Rendering admin settings page", "debug_api", "API")
        return templates.TemplateResponse("admin.html", {
            "request": request, 
            "data": settings_data
        })
    except Exception as e:
        print_message(f"Error rendering settings page: {str(e)}", "error", "API")
        raise HTTPException(status_code=500, detail="Failed to load settings page")

@app.get("/settings-json")
async def get_settings_json():
    """Return current configuration settings as JSON."""
    debug_func_entry()
    
    try:
        settings_data = config.to_dict()
        print_message("Returning JSON settings data", "debug_api", "API")
        return settings_data
    except Exception as e:
        print_message(f"Error retrieving settings data: {str(e)}", "error", "API")
        raise HTTPException(status_code=500, detail="Failed to retrieve settings")

@app.post("/update-settings")
async def update_settings(
    request: Request,
    delete_output_wavs: str = Form(...),
    gradio_interface: str = Form(...),
    gradio_port_number: int = Form(...),
    api_port_number: int = Form(...),
):
    """Update configuration settings from admin form submission."""
    debug_func_entry()
    
    try:
        print_message("Updating configuration settings", "debug_api", "API")
        
        # Log current values before update
        if config.debugging.debug_api:
            print_message("Current settings:", "debug_api", "API")
            print_message(f"delete_output_wavs: {config.delete_output_wavs}", "debug_api", "API")
            print_message(f"gradio_interface: {config.gradio_interface}", "debug_api", "API")
            print_message(f"gradio_port: {config.gradio_port_number}", "debug_api", "API")
            print_message(f"api_port: {config.api_def.api_port_number}", "debug_api", "API")
        
        # Validate port numbers
        if not (1024 <= gradio_port_number <= 65535 and 1024 <= api_port_number <= 65535):
            print_message("Invalid port numbers provided", "error", "API")
            raise HTTPException(status_code=400, detail="Port numbers must be between 1024 and 65535")
            
        # Update configuration
        config.delete_output_wavs = delete_output_wavs
        config.gradio_interface = gradio_interface.lower() == 'true'
        config.gradio_port_number = gradio_port_number
        config.api_def.api_port_number = api_port_number
        
        # Log new values after update
        if config.debugging.debug_api:
            print_message("Updated settings:", "debug_api", "API")
            print_message(f"delete_output_wavs: {config.delete_output_wavs}", "debug_api", "API")
            print_message(f"gradio_interface: {config.gradio_interface}", "debug_api", "API")
            print_message(f"gradio_port: {config.gradio_port_number}", "debug_api", "API")
            print_message(f"api_port: {config.api_def.api_port_number}", "debug_api", "API")
        
        # Save configuration
        config.save()
        print_message("Settings saved successfully", "debug_api", "API")
        
        # Return updated settings page
        return templates.TemplateResponse("admin.html", {
            "request": request, 
            "data": config.to_dict()
        })
        
    except ValueError as e:
        print_message(f"Validation error in settings update: {str(e)}", "error", "API")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print_message(f"Error updating settings: {str(e)}", "error", "API")
        raise HTTPException(status_code=500, detail="Failed to update settings")
 
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
    """Serve the main application HTML page."""
    debug_func_entry()
    
    try:
        print_message("Serving main application page", "debug_api", "API")
        return HTMLResponse(content=rendered_html, status_code=200)
    except Exception as e:
        print_message(f"Error serving main page: {str(e)}", "error", "API")
        raise HTTPException(status_code=500, detail="Failed to load application page")

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

