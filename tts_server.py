import json
import time
import os
from pathlib import Path
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

##########################
#### Webserver Imports####
##########################
from fastapi import (
    FastAPI,
    Form,
    Request,
    Response,
    Depends,
)
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from jinja2 import Template
from contextlib import asynccontextmanager

###########################
#### STARTUP VARIABLES ####
###########################
# STARTUP VARIABLE - Create "this_dir" variable as the current script directory
this_dir = Path(__file__).parent.resolve()
# STARTUP VARIABLE - Set "device" to cuda if exists, otherwise cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
# STARTUP VARIABLE - Import languges file for Gradio to be able to display them in the interface
with open(this_dir / "languages.json", encoding="utf8") as f:
    languages = json.load(f)


#################################################################
#### LOAD PARAMS FROM confignew.json - REQUIRED FOR BRANDING ####
#################################################################
# Load config file and get settings
def load_config(file_path):
    with open(file_path, "r") as configfile_path:
        configfile_data = json.load(configfile_path)
    return configfile_data


# Define the path to the confignew.json file
configfile_path = this_dir / "confignew.json"

# Load confignew.json and assign it to a different variable (config_data)
params = load_config(configfile_path)
# check someone hasnt enabled lowvram on a system thats not cuda enabled
params["low_vram"] = "false" if not torch.cuda.is_available() else params["low_vram"]

# Define the path to the JSON file
config_file_path = this_dir / "modeldownload.json"

#############################################
#### LOAD PARAMS FROM MODELDOWNLOAD.JSON ####
############################################
# This is used only in the instance that someone has changed their model path
# Define the path to the JSON file
modeldownload_config_file_path = this_dir / "modeldownload.json"

# Check if the JSON file exists
if modeldownload_config_file_path.exists():
    with open(modeldownload_config_file_path, "r") as modeldownload_config_file:
        modeldownload_settings = json.load(modeldownload_config_file)

    # Extract settings from the loaded JSON
    modeldownload_base_path = Path(modeldownload_settings.get("base_path", ""))
    modeldownload_model_path = Path(modeldownload_settings.get("model_path", ""))
else:
    # Default settings if the JSON file doesn't exist or is empty
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m modeldownload.config is missing so please re-download it and save it in the alltalk_tts main folder."
    )


########################
#### STARTUP CHECKS ####
########################
try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    logger.error(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Could not find the TTS module. Make sure to install the requirements for the alltalk_tts extension."
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Linux / Mac:\npip install -r extensions/alltalk_tts/requirements.txt\n"
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Windows:\npip install -r extensions\\alltalk_tts\\requirements.txt\n"
        f"[{params['branding']}Startup] \033[91mWarning\033[0m If you used the one-click installer, paste the command above in the terminal window launched after running the cmd_ script. On Windows, that's cmd_windows.bat."
    )
    raise

# DEEPSPEED Import - Check for DeepSpeed and import it if it exists
try:
    import deepspeed

    deepspeed_installed = True
    print(f"[{params['branding']}Startup] DeepSpeed \033[93mDetected\033[0m")
    print(
        f"[{params['branding']}Startup] Activate DeepSpeed in {params['branding']} settings"
    )
except ImportError:
    deepspeed_installed = False
    print(
        f"[{params['branding']}Startup] DeepSpeed \033[93mNot Detected\033[0m. See https://github.com/microsoft/DeepSpeed"
    )


@asynccontextmanager
async def startup_shutdown(no_actual_value_it_demanded_something_be_here):
    await setup()
    yield
    # Shutdown logic


# Create FastAPI app with lifespan
app = FastAPI(lifespan=startup_shutdown)


#####################################
#### MODEL LOADING AND UNLOADING ####
#####################################
# MODEL LOADERS Picker For API TTS, API Local, XTTSv2 Local
async def setup():
    global device
    # Set a timer to calculate load times
    generate_start_time = time.time()  # Record the start time of loading the model
    # Start loading the correct model as set by "tts_method_api_tts", "tts_method_api_local" or "tts_method_xtts_local" being True/False
    if params["tts_method_api_tts"]:
        print(
            f"[{params['branding']}Model] \033[94mAPI TTS Loading\033[0m {params['tts_model_name']} \033[94minto\033[93m",
            device,
            "\033[0m",
        )
        model = await api_load_model()
    elif params["tts_method_api_local"]:
        print(
            f"[{params['branding']}Model] \033[94mAPI Local Loading\033[0m {modeldownload_model_path} \033[94minto\033[93m",
            device,
            "\033[0m",
        )
        model = await api_manual_load_model()
    elif params["tts_method_xtts_local"]:
        print(
            f"[{params['branding']}Model] \033[94mXTTSv2 Local Loading\033[0m {modeldownload_model_path} \033[94minto\033[93m",
            device,
            "\033[0m",
        )
        model = await xtts_manual_load_model()
    # Create an end timer for calculating load times
    generate_end_time = time.time()
    # Calculate start time minus end time
    generate_elapsed_time = generate_end_time - generate_start_time
    # Print out the result of the load time
    print(
        f"[{params['branding']}Model] \033[94mModel Loaded in \033[93m{generate_elapsed_time:.2f} seconds.\033[0m"
    )
    # Set "tts_model_loaded" to true
    params["tts_model_loaded"] = True
    # Set the output path for wav files
    output_directory = this_dir / params["output_folder_wav_standalone"]
    output_directory.mkdir(parents=True, exist_ok=True)
    #Path(f'this_folder/outputs/').mkdir(parents=True, exist_ok=True)


# MODEL LOADER For "API TTS"
async def api_load_model():
    global model
    model = TTS(params["tts_model_name"]).to(device)
    return model


# MODEL LOADER For "API Local"
async def api_manual_load_model():
    global model
    # check to see if a custom path has been set in modeldownload.json and use that path to load the model if so
    if str(modeldownload_base_path) == "models":
        model = TTS(
            model_path=this_dir / "models" / modeldownload_model_path,
            config_path=this_dir / "models" / modeldownload_model_path / "config.json",
        ).to(device)
    else:
        print(
            f"[{params['branding']}Model] \033[94mInfo\033[0m Loading your custom model set in \033[93mmodeldownload.json\033[0m:",
            modeldownload_base_path / modeldownload_model_path,
        )
        model = TTS(
            model_path=modeldownload_base_path / modeldownload_model_path,
            config_path=modeldownload_base_path / modeldownload_model_path / "config.json",
        ).to(device)
    return model


# MODEL LOADER For "XTTSv2 Local"
async def xtts_manual_load_model():
    global model
    config = XttsConfig()
    # check to see if a custom path has been set in modeldownload.json and use that path to load the model if so
    if str(modeldownload_base_path) == "models":
        config_path = this_dir / "models" / modeldownload_model_path / "config.json"
        vocab_path_dir = this_dir / "models" / modeldownload_model_path / "vocab.json"
        checkpoint_dir = this_dir / "models" / modeldownload_model_path
    else:
        print(
            f"[{params['branding']}Model] \033[94mInfo\033[0m Loading your custom model set in \033[93mmodeldownload.json\033[0m:",
            modeldownload_base_path / modeldownload_model_path,
        )
        config_path = modeldownload_base_path / modeldownload_model_path / "config.json"
        vocab_path_dir = modeldownload_base_path / modeldownload_model_path / "vocab.json"
        checkpoint_dir = modeldownload_base_path / modeldownload_model_path
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(checkpoint_dir),
        vocab_path=str(vocab_path_dir),
        use_deepspeed=params["deepspeed_activate"],
    )
    model.to(device)
    return model


# MODEL UNLOADER
async def unload_model(model):
    print(f"[{params['branding']}Model] \033[94mUnloading model \033[0m")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    params["tts_model_loaded"] = False
    return None


# MODEL - Swap model based on Gradio selection API TTS, API Local, XTTSv2 Local
async def handle_tts_method_change(tts_method):
    global model
    # Update the params dictionary based on the selected radio button
    print(
        f"[{params['branding']}Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m"
    )
    # Set other parameters to False
    if tts_method == "API TTS":
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = False
        params["tts_method_api_tts"] = True
        params["deepspeed_activate"] = False
    elif tts_method == "API Local":
        params["tts_method_api_tts"] = False
        params["tts_method_xtts_local"] = False
        params["tts_method_api_local"] = True
        params["deepspeed_activate"] = False
    elif tts_method == "XTTSv2 Local":
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = True

    # Unload the current model
    model = await unload_model(model)

    # Load the correct model based on the updated params
    await setup()


# MODEL WEBSERVER- API Swap Between Models
@app.route("/api/reload", methods=["POST"])
async def reload(request: Request):
    tts_method = request.query_params.get("tts_method")
    if tts_method not in ["API TTS", "API Local", "XTTSv2 Local"]:
        return {"status": "error", "message": "Invalid TTS method specified"}
    await handle_tts_method_change(tts_method)
    return Response(
        content=json.dumps({"status": "model-success"}), media_type="application/json"
    )


##################
#### LOW VRAM ####
##################
# LOW VRAM - MODEL MOVER VRAM(cuda)<>System RAM(cpu) for Low VRAM setting
async def switch_device():
    global model, device
    # Check if CUDA is available before performing GPU-related operations
    if torch.cuda.is_available():
        if device == "cuda":
            device = "cpu"
            model.to(device)
            torch.cuda.empty_cache()
        else:
            device == "cpu"
            device = "cuda"
            model.to(device)


@app.post("/api/lowvramsetting")
async def set_low_vram(request: Request, new_low_vram_value: bool):
    global device
    try:
        if new_low_vram_value is None:
            raise ValueError("Missing 'low_vram' parameter")

        if params["low_vram"] == new_low_vram_value:
            return Response(
                content=json.dumps(
                    {
                        "status": "success",
                        "message": f"[{params['branding']}Model] LowVRAM is already {'enabled' if new_low_vram_value else 'disabled'}.",
                    }
                )
            )
        params["low_vram"] = new_low_vram_value
        if params["low_vram"]:
            await unload_model(model)
            if torch.cuda.is_available():
                device = "cpu"
                print(
                    f"[{params['branding']}Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m"
                )
                print(
                    f"[{params['branding']}Model] \033[94mLowVRAM Enabled.\033[0m Model will move between \033[93mVRAM(cuda) <> System RAM(cpu)\033[0m"
                )
                await setup()
            else:
                # Handle the case where CUDA is not available
                print(
                    f"[{params['branding']}Model] \033[91mError:\033[0m Nvidia CUDA is not available on this system. Unable to use LowVRAM mode."
                )
                params["low_vram"] = False
        else:
            await unload_model(model)
            if torch.cuda.is_available():
                device = "cuda"
                print(
                    f"[{params['branding']}Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m"
                )
                print(
                    f"[{params['branding']}Model] \033[94mLowVRAM Disabled.\033[0m Model will stay in \033[93mVRAM(cuda)\033[0m"
                )
                await setup()
            else:
                # Handle the case where CUDA is not available
                print(
                    f"[{params['branding']}Model] \033[91mError:\033[0m Nvidia CUDA is not available on this system. Unable to use LowVRAM mode."
                )
                params["low_vram"] = False
        return Response(content=json.dumps({"status": "lowvram-success"}))
    except Exception as e:
        return Response(content=json.dumps({"status": "error", "message": str(e)}))


###################
#### DeepSpeed ####
###################
# DEEPSPEED - Reload the model when DeepSpeed checkbox is enabled/disabled
async def handle_deepspeed_change(value):
    global model
    if value:
        # DeepSpeed enabled
        print(f"[{params['branding']}Model] \033[93mDeepSpeed Activating\033[0m")

        print(
            f"[{params['branding']}Model] \033[94mChanging model \033[92m(DeepSpeed can take 30 seconds to activate)\033[0m"
        )
        print(
            f"[{params['branding']}Model] \033[91mInformation\033[0m If you have not set CUDA_HOME path, DeepSpeed may fail to load/activate"
        )
        print(
            f"[{params['branding']}Model] \033[91mInformation\033[0m DeepSpeed needs to find nvcc from the CUDA Toolkit. Please check your CUDA_HOME path is"
        )
        print(
            f"[{params['branding']}Model] \033[91mInformation\033[0m pointing to the correct location and use 'set CUDA_HOME=putyoutpathhere' (Windows) or"
        )
        print(
            f"[{params['branding']}Model] \033[91mInformation\033[0m 'export CUDA_HOME=putyoutpathhere' (Linux) within your Python Environment"
        )
        model = await unload_model(model)
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = True
        params["deepspeed_activate"] = True
        await setup()
    else:
        # DeepSpeed disabled
        print(f"[{params['branding']}Model] \033[93mDeepSpeed De-Activating\033[0m")
        print(
            f"[{params['branding']}Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m"
        )
        params["deepspeed_activate"] = False
        model = await unload_model(model)
        await setup()

    return value  # Return new checkbox value


# DEEPSPEED WEBSERVER- API Enable/Disable DeepSpeed
@app.post("/api/deepspeed")
async def deepspeed(request: Request, new_deepspeed_value: bool):
    try:
        if new_deepspeed_value is None:
            raise ValueError("Missing 'deepspeed' parameter")
        if params["deepspeed_activate"] == new_deepspeed_value:
            return Response(
                content=json.dumps(
                    {
                        "status": "success",
                        "message": f"DeepSpeed is already {'enabled' if new_deepspeed_value else 'disabled'}.",
                    }
                )
            )
        params["deepspeed_activate"] = new_deepspeed_value
        await handle_deepspeed_change(params["deepspeed_activate"])
        return Response(content=json.dumps({"status": "deepspeed-success"}))
    except Exception as e:
        return Response(content=json.dumps({"status": "error", "message": str(e)}))


########################
#### TTS GENERATION ####
########################
# TTS VOICE GENERATION METHODS (called from voice_preview and output_modifer)
async def generate_audio(text, voice, language, output_file):
    global model
    if params["low_vram"] and device == "cpu":
        await switch_device()
    generate_start_time = time.time()  # Record the start time of generating TTS
    # XTTSv2 LOCAL Method
    if params["tts_method_xtts_local"]:
        print(f"[{params['branding']}TTSGen] {text}")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[f"{this_dir}/voices/{voice}"],
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs,
        )
        out = model.inference(
            text,
            language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=float(params["local_temperature"]),
            length_penalty=float(model.config.length_penalty),
            repetition_penalty=float(params["local_repetition_penalty"]),
            top_k=int(model.config.top_k),  # Convert to int if necessary
            top_p=float(model.config.top_p),
            enable_text_splitting=True,
        )
        torchaudio.save(output_file, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    # API LOCAL Methods
    elif params["tts_method_api_local"]:
        # Set the correct output path (different from the if statement)
        print(f"[{params['branding']}TTSGen] Using API Local")
        model.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=[f"{this_dir}/voices/{voice}"],
            language=language,
            temperature=float(params["local_temperature"]),
            length_penalty=model.config.length_penalty,
            repetition_penalty=float(params["local_repetition_penalty"]),
            top_k=model.config.top_k,
            top_p=model.config.top_p,
        )
    # API TTS
    elif params["tts_method_api_tts"]:
        print(f"[{params['branding']}TTSGen] Using API TTS")
        model.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=[f"{this_dir}/voices/{voice}"],
            language=language,
        )
    # Print Generation time and settings
    generate_end_time = time.time()  # Record the end time to generate TTS
    generate_elapsed_time = generate_end_time - generate_start_time
    print(
        f"[{params['branding']}TTSGen] \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{params['low_vram']} \033[94mDeepSpeed: \033[33m{params['deepspeed_activate']}\033[0m"
    )
    # Move model back to cpu system ram if needed.
    if params["low_vram"] and device == "cuda":
        await switch_device()
    return


# TTS VOICE GENERATION METHODS - generate TTS API
@app.route("/api/generate", methods=["POST"])
async def generate(request: Request):
    try:
        # Get parameters from JSON body
        data = await request.json()
        text = data["text"]
        voice = data["voice"]
        language = data["language"]
        output_file = data["output_file"]
        # Generation logic
        await generate_audio(text, voice, language, output_file)
        return JSONResponse(
            content={"status": "generate-success", "data": {"audio_path": output_file}}
        )
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})


###################################################
#### POPULATE FILES LIST FROM VOICES DIRECTORY ####
###################################################
# List files in the "voices" directory
def list_files(directory):
    files = [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(".wav")
    ]
    return files

#############################
#### JSON CONFIG UPDATER ####
#############################

# Create an instance of Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory=this_dir / "templates")

# Create a dependency to get the current JSON data
def get_json_data():
    with open(this_dir / "confignew.json", "r") as json_file:
        data = json.load(json_file)
    return data


# Define an endpoint function
@app.get("/settings")
async def get_settings(request: Request):
    wav_files = list_files(this_dir / "voices")
    # Render the template with the current JSON data and list of WAV files
    return templates.TemplateResponse(
        "generate_form.html",
        {
            "request": request,
            "data": get_json_data(),
            "modeldownload_model_path": modeldownload_model_path,
            "wav_files": wav_files,
        },
    )

# Define an endpoint to serve static files
app.mount("/static", StaticFiles(directory=str(this_dir / "templates")), name="static")

@app.post("/update-settings")
async def update_settings(
    request: Request,
    activate: bool = Form(...),
    autoplay: bool = Form(...),
    deepspeed_activate: bool = Form(...),
    delete_output_wavs: str = Form(...),
    ip_address: str = Form(...),
    language: str = Form(...),
    local_temperature: str = Form(...),
    local_repetition_penalty: str = Form(...),
    low_vram: bool = Form(...),
    tts_model_loaded: bool = Form(...),
    tts_model_name: str = Form(...),
    narrator_enabled: bool = Form(...),
    narrator_voice: str = Form(...),
    output_folder_wav: str = Form(...),
    port_number: str = Form(...),
    remove_trailing_dots: bool = Form(...),
    show_text: bool = Form(...),
    tts_method: str = Form(...),
    voice: str = Form(...),
    data: dict = Depends(get_json_data),
):
    # Update the settings based on the form values
    data["activate"] = activate
    data["autoplay"] = autoplay
    data["deepspeed_activate"] = deepspeed_activate
    data["delete_output_wavs"] = delete_output_wavs
    data["ip_address"] = ip_address
    data["language"] = language
    data["local_temperature"] = local_temperature
    data["local_repetition_penalty"] = local_repetition_penalty
    data["low_vram"] = low_vram
    data["tts_model_loaded"] = tts_model_loaded
    data["tts_model_name"] = tts_model_name
    data["narrator_enabled"] = narrator_enabled
    data["narrator_voice"] = narrator_voice
    data["output_folder_wav"] = output_folder_wav
    data["port_number"] = port_number
    data["remove_trailing_dots"] = remove_trailing_dots
    data["show_text"] = show_text
    data["tts_method_api_local"] = tts_method == "api_local"
    data["tts_method_api_tts"] = tts_method == "api_tts"
    data["tts_method_xtts_local"] = tts_method == "xtts_local"
    data["voice"] = voice

    # Save the updated settings back to the JSON file
    with open(this_dir / "confignew.json", "w") as json_file:
        json.dump(data, json_file)

    # Redirect to the settings page to display the updated settings
    return RedirectResponse(url="/settings", status_code=303)


##################################
#### SETTINGS PAGE DEMO VOICE ####
##################################
# Submit demo post here
@app.post("/tts-demo-request", response_class=JSONResponse)
async def tts_demo_request(request: Request, text: str = Form(...), voice: str = Form(...), language: str = Form(...), output_file: str = Form(...)):
    try:
        output_file_path = this_dir / "outputs" / output_file
        await generate_audio(text, voice, language, output_file_path)
        return JSONResponse(content={"output_file_path": str(output_file)}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

# Gives web access to the output files
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = this_dir / "outputs" / filename
    return FileResponse(audio_path)


#########################
#### VOICES LIST API ####
#########################
# Define the new endpoint
@app.get("/api/voices")
async def get_voices():
    wav_files = list_files(this_dir / "voices")
    return {"voices": wav_files}

###########################
#### PREVIEW VOICE API ####
###########################
@app.post("/api/previewvoice/", response_class=JSONResponse)
async def preview_voice(request: Request, voice: str = Form(...)):
    try:
        # Hardcoded settings
        language = "en"
        output_file_name = "api_preview_voice"

        # Clean the voice filename for inclusion in the text
        clean_voice_filename = re.sub(r'\.wav$', '', voice.replace(' ', '_'))
        clean_voice_filename = re.sub(r'[^a-zA-Z0-9]', ' ', clean_voice_filename)
        
        # Generate the audio
        text = f"Hello, this is a preview of voice {clean_voice_filename}."

        # Generate the audio
        output_file_path = this_dir / "outputs" / f"{output_file_name}.wav"
        await generate_audio(text, voice, language, output_file_path)

        # Generate the URL
        output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}.wav'

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

########################
#### GENERATION API ####
########################
import html
import re
import uuid
import numpy as np
import soundfile as sf
import sys

# Check for PortAudio library on Linux
try:
    import sounddevice as sd
    sounddevice_installed=True
except OSError:
    print(f"[{params['branding']}Startup] \033[91mInfo\033[0m PortAudio library not found. If you wish to play TTS in standalone mode through the API suite")
    print(f"[{params['branding']}Startup] \033[91mInfo\033[0m please install PortAudio. This will not affect any other features or use of Alltalk.")
    print(f"[{params['branding']}Startup] \033[91mInfo\033[0m If you don't know what the API suite is, then this message is nothing to worry about.")
    sounddevice_installed=False
    if sys.platform.startswith('linux'):
        print(f"[{params['branding']}Startup] \033[91mInfo\033[0m On Linux, you can use the following command to install PortAudio:")
        print(f"[{params['branding']}Startup] \033[91mInfo\033[0m sudo apt-get install portaudio19-dev")

from typing import Union, Dict
from pydantic import BaseModel, ValidationError, Field

def play_audio(file_path, volume):
    data, fs = sf.read(file_path)
    sd.play(volume * data, fs)
    sd.wait()

class Request(BaseModel):
    # Define the structure of the 'Request' class if needed
    pass

class JSONInput(BaseModel):
    text_input: str = Field(..., max_length=1000, description="text_input needs to be 1000 characters or less.")
    text_filtering: str = Field(..., pattern="^(none|standard|html)$", description="text_filtering needs to be 'none', 'standard' or 'html'.")
    character_voice_gen: str = Field(..., pattern="^.*\.wav$", description="character_voice_gen needs to be the name of a wav file e.g. mysample.wav.")
    narrator_enabled: bool = Field(..., description="narrator_enabled needs to be true or false.")
    narrator_voice_gen: str = Field(..., pattern="^.*\.wav$", description="narrator_voice_gen needs to be the name of a wav file e.g. mysample.wav.")
    text_not_inside: str = Field(..., pattern="^(character|narrator)$", description="text_not_inside needs to be 'character' or 'narrator'.")
    language: str = Field(..., pattern="^(ar|zh-cn|cs|nl|en|fr|de|hu|it|ja|ko|pl|pt|ru|es|tr)$", description="language needs to be one of the following ar|zh-cn|cs|nl|en|fr|de|hu|it|ja|ko|pl|pt|ru|es|tr.")
    output_file_name: str = Field(..., pattern="^[a-zA-Z0-9_]+$", description="output_file_name needs to be the name without any special characters or file extension e.g. 'filename'")
    output_file_timestamp: bool = Field(..., description="output_file_timestamp needs to be true or false.")
    autoplay: bool = Field(..., description="autoplay needs to be a true or false value.")
    autoplay_volume: float = Field(..., ge=0.1, le=1.0, description="autoplay_volume needs to be from 0.1 to 1.0")

    @classmethod
    def validate_autoplay_volume(cls, value):
        if not (0.1 <= value <= 1.0):
            raise ValueError("Autoplay volume must be between 0.1 and 1.0")
        return value


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

def standard_filtering(text_input):
    text_output = (text_input
                        .replace("***", "")
                        .replace("**", "")
                        .replace("*", "")
                        .replace("\n\n", "\n")
                        .replace("&#x27;", "'")
                        )
    return text_output

def narrator_filtering(narrator_text_input):
    processed_string = re.sub(r'\.\*\n\*', '. ', narrator_text_input)
    processed_string = (
        processed_string
        .replace("\n", " ")
        .replace('&quot;', '&quot;<')
        .replace('"', '&quot;<')
    )
    processed_string = processed_string.replace('&quot;<. *', '&quot;< *"')
    processed_string = processed_string.replace('< *"', '< *')
    processed_string = processed_string.replace('. *', '< *')
    text_output = html.unescape(processed_string)
    return text_output

def combine(output_file_timestamp, output_file_name, audio_files):
    audio = np.array([])
    sample_rate = None
    try:
        for audio_file in audio_files:
            audio_data, current_sample_rate = sf.read(audio_file)
            if audio.size == 0:
                audio = audio_data
                sample_rate = current_sample_rate
            elif sample_rate == current_sample_rate:
                audio = np.concatenate((audio, audio_data))
            else:
                raise ValueError("Sample rates of input files are not consistent.")
    except Exception as e:
        # Handle exceptions (e.g., file not found, invalid audio format)
        return None, None
    if output_file_timestamp:
        timestamp = int(time.time())
        output_file_path = os.path.join(this_dir / "outputs" / f'{output_file_name}_{timestamp}_combined.wav')
        output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}_{timestamp}_combined.wav'
    else:
        output_file_path = os.path.join(this_dir / "outputs" / f'{output_file_name}_combined.wav')
        output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}.wav'
    try:
        sf.write(output_file_path, audio, samplerate=sample_rate)
        # Clean up unnecessary files
        for audio_file in audio_files:
            os.remove(audio_file)
    except Exception as e:
        # Handle exceptions (e.g., failed to write output file)
        return None, None
    return output_file_path, output_file_url

# Generation API (separate from text-generation-webui)
@app.post("/api/tts-generate", response_class=JSONResponse)
async def tts_generate(
    text_input: str = Form(...),
    text_filtering: str = Form(...),
    character_voice_gen: str = Form(...),
    narrator_enabled: bool = Form(...),
    narrator_voice_gen: str = Form(...),
    text_not_inside: str = Form(...),
    language: str = Form(...),
    output_file_name: str = Form(...),
    output_file_timestamp: bool = Form(...),
    autoplay: bool = Form(...),
    autoplay_volume: float = Form(...),
):
    try:
        #print(f"text_filtering: {text_filtering}")
        #print(f"narrator_enabled: {narrator_enabled}")
        #print(f"text_not_inside: {text_not_inside}")
        #print(f"output_file_timestamp: {output_file_timestamp}")
        json_input_data = {
            "text_input": text_input,
            "text_filtering": text_filtering,
            "character_voice_gen": character_voice_gen,
            "narrator_enabled": narrator_enabled,
            "narrator_voice_gen": narrator_voice_gen,
            "text_not_inside": text_not_inside,
            "language": language,
            "output_file_name": output_file_name,
            "output_file_timestamp": output_file_timestamp,
            "autoplay": autoplay,
            "autoplay_volume": autoplay_volume,
        }
        JSONresult = TTSGenerator.validate_json_input(json_input_data)
        if JSONresult is None:
            pass
        else:
            return JSONResponse(content={"error": JSONresult}, status_code=400)
        if narrator_enabled:
            print("ORIGINAL UNTOUCHED STRING IS:", text_input,"\n")
            if text_filtering in ["standard", "none"]:
                    cleaned_string = (
                        text_input
                        .replace('"', '"<')
                        .replace('*', '<*')
                    )
                    print("STANDARD FILTERING IS NOW:", cleaned_string,"\n")
                    #parts = re.split(r'\.(?<!<[*"])', cleaned_string)
                    parts = re.split(r'(?<=<\*\s)|(?<=\.\s)|(?<=\."\<)|(?<=\."<)', cleaned_string)
                    parts = list(filter(lambda x: x.strip(), parts))
            elif text_filtering == "html":
                cleaned_string = standard_filtering(text_input)
                cleaned_string = narrator_filtering(cleaned_string)
                print("HTML FILTERING IS NOW:", cleaned_string,"\n")
                parts = re.split(r'&quot;|\.\*', cleaned_string)
            audio_files_all_paragraphs = []
            audio_files_paragraph = []
            for i, part in enumerate(parts):
                if len(part.strip()) <= 1:
                    continue
                print("THIS IS A PART", part)
                # Figure out which type of line it is, then replace characters as necessary to avoid TTS trying to pronunce them, htmlunescape after. 
                # Character will always be a < with a letter immediately after it
                if '<' in part and '<*' not in part and '< *' not in part and '<  *' not in part and '< ' not in part and '<  ' not in part:
                    print("IF - Character\n")
                    cleaned_part = html.unescape(part.replace('<', ''))
                    voice_to_use = character_voice_gen
                #Narrator will always be an * or < with an * a position or two after it.
                elif '<*' in part or '< *' in part or '<  *' in part or '*' in part:
                    print("IF - Narrator\n")
                    cleaned_part = html.unescape(part.replace('<*', '').replace('< *', '').replace('<  *', '').replace('*', '').replace('<. ', '')) 
                    voice_to_use = narrator_voice_gen
                #If the other two dont capture it, aka, the AI gave no * or &quot; on the line, use non_quoted_text_is aka user interface, user can choose Char or Narrator
                elif text_not_inside == "character":
                    print("ELSE - CHARACTER\n")
                    cleaned_part = html.unescape(part.replace('< ', '').replace('<  ', '').replace('<  ', ''))
                    voice_to_use = character_voice_gen
                elif text_not_inside == "narrator":
                    print("ELSE - NARRATOR\n")
                    cleaned_part = html.unescape(part.replace('< ', '').replace('<  ', '').replace('<  ', ''))
                    voice_to_use = narrator_voice_gen
                output_file = this_dir / "outputs" / f"{output_file_name}_{uuid.uuid4()}_{int(time.time())}_{i}.wav"
                output_file_str = output_file.as_posix()
                await generate_audio(cleaned_part, voice_to_use, language, output_file_str)
                audio_path = output_file_str
                audio_files_paragraph.append(audio_path)
            # Accumulate audio files within the paragraph
            audio_files_all_paragraphs.extend(audio_files_paragraph)
            # Combine audio files across paragraphs
            output_file_path, output_file_url = combine(output_file_timestamp, output_file_name, audio_files_all_paragraphs)
        else:
            if output_file_timestamp:
                timestamp = int(time.time())
                output_file_path = this_dir / "outputs" / f"{output_file_name}_{timestamp}.wav"
                output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}_{timestamp}.wav'
            else:
                output_file_path = this_dir / "outputs" / f"{output_file_name}.wav"
                output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}.wav'
            if text_filtering == "html":
                cleaned_string = html.unescape(standard_filtering(text_input))
            elif text_filtering == "standard":
                cleaned_string = standard_filtering(text_input)
            else:
                cleaned_string = text_input
            await generate_audio(cleaned_string, character_voice_gen, language, output_file_path)
        if sounddevice_installed == False:
            autoplay = False
        if autoplay:
            play_audio(output_file_path, autoplay_volume)
        return JSONResponse(content={"status": "generate-success", "output_file_path": str(output_file_path), "output_file_url": str(output_file_url)}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "generate-failure", "error": "An error occurred"}, status_code=500)

#############################################################
#### DOCUMENTATION - README ETC - PRESENTED AS A WEBPAGE ####
#############################################################

simple_webpage = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AllTalk TTS for Text generation webUI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px; /* Adjusted max-width for better readability */
            margin: 40px auto;
            padding: 20px;
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        h1, h2 {
            color: black;
            text-decoration: underline;
        }

    p, span {
        color: #555;
        font-size: 16px; /* Increased font size for better readability */
        margin-top: 0; /* Remove top margin for paragraphs */
    }

        code {
            background-color: #f8f8f8;
            border: 2px solid #ddd;
            border-radius: 4px;
            padding: 0px 0px;
            font-size: 14px;
            margin-bottom: 20px;
            color: #3366ff;
        }

        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
            font-size: 14px; /* Adjusted font size for better visibility */
        }

        ul {
            color: #555;
            list-style-type: square; /* Set the bullet style */
            margin-left: 2px; /* Adjust the left margin to create an indent */
        } 

        li {
            font-size: 16px; /* Set the font size for list items */
            margin-bottom: 8px; /* Add some space between list items */
        }

        .key {
        color: black; /* Color for keys */
        font-size: 14px; /* Increased font size for better readability */
        }

        .value {
        font-size: 14px; /* Increased font size for better readability */
        color: blue; /* Color for values */
        }


        a {
            color: #0077cc;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        strong {
            font-weight: bold;
        }
        
        .option-a {
            color: #33cc33;
            font-weight: bold;
        }

        .option-b {
            color: red;
            font-weight: bold;
        }

                /* New styles for TTS Request Page */
        #container {
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        form {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        label {
            font-weight: bold;
            font-size: 14px;
            padding: 2px;
        }

        textarea, input, select, button {
            padding: 4px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px;
        }

        button {
            background-color: #4caf50;
            color: #fff;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #45a049;
        }

        p {
            margin-top: 20px;
        }

        #outputFilePath {
            font-weight: bold;
        }

        #audioSource {
        display: block;
        margin: auto;
        }

        #outputFilePath {
        display: none;
        }

       
        table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        /* Style for the first table */
        #configuration-details table {
            font-size: 16px;
        }

        /* Style for the nested table */
        #modeldownload-table table {
            font-size: 16px;
        }

        #modeldownload-table table td {
            padding: 4px; /* Adjust padding as needed for the nested table */
        }




    </style>
</head>

<body>
<h1 id="toc">AllTalk TTS for Text generation webUI</h1>
<iframe src="http://{{ params["ip_address"] }}:{{ params["port_number"] }}/settings" width="100%" height="500" frameborder="0" style="margin: 0; padding: 0;"></iframe>
<h2>Table of Contents</h2>
<ul>
<li><a href="#getting-started">Getting Started</a></li>
<li><a href="#server-information">Server Information</a></li>
<li><a href="#demotesttts">Demo/Test TTS output</a></li>
<li><a href="#using-voice-samples">Using Voice Samples</a></li>
<li><a href="#text-not-inside">Text Not inside</a></li>
<li><a href="#low-vram">Low VRAM</a></li>
<li><a href="#finetune">Finetuning (Training the model)</a></li>
<li><a href="#deepspeed">DeepSpeed</a>
<ul>
<li><a href="#deepspeed-linux">DeepSpeed Setup Linux</a></li>
<li><a href="#deepspeed-windows">DeepSpeed Setup Windows</a></li>
</ul>
</li>
<li><a href="#TTSmodels">TTS Models/Methods</a></li>
<li><a href="#temperature-and-repetition-settings">Model temperature and repetition settings</a></li>
<li><a href="#start--up-checks">Start-up checks</a></li>
<li><a href="#customTTSmodels">Custom TTS Models and Model path</a></li>
<li><a href="#configuration-details">Configuration file settings</a></li>
<li><a href="#curl-commands">JSON calls and CURL Commands</a></li>
<li><a href="#debugging-and-tts-generation-information">Debugging and TTS Generation Information</a></li>
<li><a href="#references">Thanks &amp; References</a></li>
</ul>

<h2 id="getting-started">Getting Started</h2>
<p style="padding-left: 30px; text-align: justify;">AllTalk is a web interface, based around the Coqui TTS voice cloning/speech generation system. To generate TTS, you can use the provided gradio interface or interact with the server using JSON/CURL commands.&nbsp;</p>
<p style="padding-left: 30px; text-align: justify;"><span style="color: #ff0000;">Note:</span> When loading up a new character in Text generation webUI it may look like nothing is happening for 20-30 seconds. Its actually processing the introduction section (greeting message) of the text and once that is completed, it will appear. You can see the activity occuring in the console window. Refreshing the page multiple times will try force the TTS engine to keep re-generating the text, so please just wait and check the console if needed.</p>
<p style="padding-left: 30px; text-align: justify;"><span style="color: #ff0000;">Note: </span>Ensure that your RP character card has asterisks around anything for the narration and double quotes around anything spoken. There is a complication ONLY with the greeting card so, ensuring it has the correct use of quotes and asterisks will help make sure that the greeting card sounds correct. I will aim to address this issue in a future update. In Text-generation-webUI <span style="color: #3366ff;">parameters menu</span> &gt; <span style="color: #3366ff;">character tab</span> &gt; <span style="color: #3366ff;">greeting</span>&nbsp;make sure that anything in there that is the narrator is in asterisks and anything spoken is in double quotes, then hit the <span style="color: #3366ff;">save</span> (💾) button.</p>
<p style="padding-left: 30px;"><span style="text-decoration: underline;"><strong>AllTalk <span style="color: #ff0000; text-decoration: underline;">Minor Bug Fixes Changelog &amp; Known issues</span></strong></span></p>
<p style="padding-left: 30px;">If I squash any minor bugs or find any issues, I will try to apply an update asap. If you think something isnt working correctly or you have a problem, check these two links below first.</p>
<ul>
<li>Minor bug fixes changelog&nbsp;<a href="https://github.com/erew123/alltalk_tts/issues/25" target="_blank" rel="noopener">link here</a>.</li>
<li>Help and known issues <a href="https://github.com/erew123/alltalk_tts?#-help-with-problems" target="_blank" rel="noopener">link here</a>.</li>
</ul>
<p style="padding-left: 30px;"><span style="text-decoration: underline;"><strong>AllTalk</strong></span> <br />Github <a href="https://github.com/erew123/alltalk_tts" target="_blank" rel="noopener">link here</a><br />Update instructions <a href="https://github.com/erew123/alltalk_tts?#-updating" target="_blank" rel="noopener">link here</a><br />Help and issues <a href="https://github.com/erew123/alltalk_tts?#-help-with-problems" target="_blank" rel="noopener">link here<br /><br /></a><span style="text-decoration: underline;"><strong>Text generation webUI</strong></span> <br />Web interface&nbsp;<a href="http://{{ params["ip_address"] }}:7860" target="_blank" rel="noopener">link here</a> <br />Documentation <a href="https://github.com/oobabooga/text-generation-webui/wiki" target="_blank" rel="noopener">link here</a></p>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>
<h2 id="server-information">Server Information</h2>
<ul>

        <li>Base URL: <code>http://{{ params["ip_address"] }}:{{ params["port_number"] }}</code></li>
        <li>Server Status: <code><a href="http://{{ params["ip_address"] }}:{{ params["port_number"] }}/ready">http://{{ params["ip_address"] }}:{{ params["port_number"] }}/ready</a></code></li>

</ul>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>

<h2 id="demotesttts">Demo/Test TTS</h2>
    <div id="container">
        <form method="post" action="/tts-demo-request" id="ttsForm">
            <label for="text">Text:</label>
            <textarea id="text" name="text" rows="4" required></textarea>

            <label for="voice">Voice:</label>
            <input type="text" id="voice" name="voice" value="female_01.wav" required>

            <label for="language">Language:</label>
            <select id="language" name="language" value="English" required>
                <option value="en" selected>English</option>
                <option value="ar">Arabic</option>
                <option value="zh-cn">Chinese</option>
                <option value="cs">Czech</option>
                <option value="nl">Dutch</option>
                <option value="fr">French</option>
                <option value="de">German</option>
                <option value="hu">Hungarian</option>
                <option value="it">Italian</option>
                <option value="ja">Japanese</option>
                <option value="ko">Korean</option>
                <option value="pl">Polish</option>
                <option value="pt">Portuguese</option>
                <option value="ru">Russian</option>
                <option value="es">Spanish</option>
                <option value="tr">Turkish</option>
            </select>

            <label for="outputFile">Output File:</label>
            <input type="text" id="outputFile" name="output_file" value="demo_output.wav" required>

            <!-- Audio player with autoplay -->
            <audio controls autoplay id="audioSource">
                <source type="audio/wav" />
                Your browser does not support the audio element.
            </audio>
            <span id="outputFilePath" style="height: 0px;">{{ output_file_path }}</span>
            <button type="submit">Generate TTS</button>
        </form>
    </div>

    <script>
        const form = document.getElementById('ttsForm');
        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            const formData = new FormData(form);
            const response = await fetch('/tts-demo-request', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            // Update the generated audio file path
            const outputFilePath = document.getElementById('outputFilePath');
            outputFilePath.textContent = result.output_file_path;

            // Update the audio player source
            const audioPlayer = document.getElementById('audioSource');
            audioPlayer.src = `/audio/${result.output_file_path}`;
            audioPlayer.load(); // Reload the audio player
            audioPlayer.play(); // Play the audio;
        });
    </script>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>

<h2 id="using-voice-samples"><strong>Using Voice Samples</strong></h2>
<h4 id="where-are-the-sample-voices-stored">Where are the sample voices stored?</h4>
<p style="padding-left: 30px;">Voice samples are stored in <span style="color: #3366ff;">/alltalk_tts/voices/</span> and should be named using the following format <span style="color: #3366ff;">name.wav</span></p>
<h4 id="where-are-the-outputs-stored">Where are the outputs stored &amp; Automatic output wav file deletion.</h4>
<p style="padding-left: 30px; text-align: justify;">Voice outputs are stored in&nbsp;<span style="color: #3366ff;">/alltalk_tts/outputs/</span></p>
<p style="padding-left: 30px; text-align: justify;">You can configure automatic maintenence deletion of old wav files by setting <span style="color: #3366ff;">Del WAV's older than</span> in the settings above.</p>
<p style="padding-left: 30px; text-align: justify;">When <span style="color: #3366ff;">Disabled</span> your output wav files will be left untouched. When set to a setting <span style="color: #3366ff;">1 Day</span> or greater, your output wav files older than that time period will be automatically deleted on start-up of AllTalk.</p>
<h4>Where are the models stored?</h4>
<p style="padding-left: 30px; text-align: justify;">This extension will download the 2.0.2 model to <span style="color: #3366ff;">/alltalk_tts/models/</span></p>
<p style="padding-left: 30px; text-align: justify;">This TTS engine will also download the latest available model and store it wherever your OS normally stores it (Windows/Linux/Mac).</p>
<h4>How do I create a new voice sample?</h4>
<p style="padding-left: 30px; text-align: justify;">To create a new voice sample you need to make a wav file that is <span style="color: #3366ff;">22050Hz</span>, <span style="color: #3366ff;">Mono</span>, <span style="color: #3366ff;">16 bit</span> and between 6 to 30 seconds long, though 8 to 10 seconds is usually good enough. The model can handle up to a 30 second samples, however Ive not noticed any improvement in voice output from a much longer clips.</p>
<p style="padding-left: 30px; text-align: justify;">You want to find a nice clear selection of audio, so lets say you wanted to clone your favourite celebrity. You may go looking for an interview where they are talking. Pay close attention to the audio you are listening to and trying to sample. Are there noises in the backgroud, hiss on the soundtrack, a low humm, some quiet music playing or something? The better quality the audio the better the final TTS result. Dont forget, the AI that processes the sounds can hear everything in your sample and it will use them in the voice its trying to recreate.</p>
<p style="padding-left: 30px; text-align: justify;">Try make your clip one of nice flowing speech, like the included example files. No big pauses, gaps or other sounds. Preferably a sample that the person you are trying to copy will show a little vocal range and emotion in their voice. Also, try to avoid a clip starting or ending with breathy sounds (breathing in/out etc).</p>
<h4>Editing your sample!</h4>
<p style="padding-left: 30px; text-align: justify;">So youve downloaded your favoutie celebrity interview off Youtube, from here you need to chop it down to 6 to 30 seconds in length and resample it.</p>
<p style="text-align: justify; padding-left: 30px;">If you need to clean it up, do audio processing, volume level changes etc, do this before downsampling.<br /><br />Using the latest version of Audacity <span style="color: #3366ff;">select/highlight</span> your 6 to 30 second clip and:<br /><br /><span style="color: #3366ff;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Tracks</span> &gt;<span style="color: #3366ff;"> Resample to 22050Hz</span>&nbsp;then<br /><span style="color: #3366ff;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Tracks</span> &gt; <span style="color: #3366ff;">Mix</span> &gt; <span style="color: #3366ff;">Stereo to Mono&nbsp;</span>then<br /><span style="color: #3366ff;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; File</span> &gt; <span style="color: #3366ff;">Export Audio</span>&nbsp;saving it as a <span style="color: #3366ff;">WAV</span> of <span style="color: #3366ff;">22050Hz</span>.</p>
<p style="padding-left: 30px; text-align: justify;">Save your generated wav file in the&nbsp;<span style="color: #3366ff;">/alltalk_tts/voices/ <span style="color: #808080;">folder.</span></span></p>
<p style="padding-left: 30px; text-align: justify;">Its worth mentioning that using AI generated audio clips may introduce unwanted sounds as its already a copy/simulation of a voice.</p>
<h4>Why doesnt it sound like XXX Person?</h4>
<p style="padding-left: 30px; text-align: justify;">Maybe you might be interested in trying <a href="#finetune">Finetuning of the model</a>. Otherwise, the reasons can be that you:</p>
<ul style="text-align: justify;">
<li>Didn't downsample it as above.</li>
<li>Have a bad quality voice sample.</li>
<li>Try using the 3x different generation methods: <span style="color: #3366ff;">API TTS</span>, <span style="color: #3366ff;">API Local</span>, and <span style="color: #3366ff;">XTTSv2 Local</span> within the web interface, as they generate output in different ways and sound different.</li>
</ul>
<p style="padding-left: 30px; text-align: justify;">Some samples just never seem to work correctly, so maybe try a different sample. Always remember though, this is an AI model attempting to re-create a voice, so you will never get a 100% match.</p>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>


<h2 id="text-not-inside"><strong>Text Not inside</strong></h2>
<p style="padding-left: 30px; text-align: justify;">This only affects the Narrator function. Most AI models should be using asterisks or double quotes to differentiate between the Narrator or the Character, however, many models sometimes switch between using asterisks and double quotes or sometimes nothing at all for the text it outputs. <br /><br />This leaves a bit of a mess because sometimes un-marked text is narration and sometimes its the character talking, leaving no clear way to know where to split sentences and what voice to use. Whilst there is no 100% solution at the moment many models lean more one way or the other as to what that unmarked text will be (character or narrator).</p>
<p style="padding-left: 30px; text-align: justify;">As such, the "Text not inside" function at least gives you the choice to set how you want the TTS engine to handle situations of un-marked text.</p>
<div style="text-align: center;"><img src="/static/textnotinside.jpg" alt="When the AI doesnt use an asterisk or a quote" /></div>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>


<h2 id="low-vram"><strong>Low VRAM</strong></h2>
<p style="padding-left: 30px; text-align: justify;">The Low VRAM option is a crucial feature designed to enhance performance under constrained (VRAM) conditions, as the TTS models require 2GB-3GB of VRAM to run effectively. This feature strategically manages the relocation of the Text-to-Speech (TTS) model between your system's Random Access Memory (RAM) and VRAM, moving it between the two on the fly. Obviously, this is very useful for people who have smaller graphics cards and will use all their VRAM to load in their LLM.</p>
<p style="padding-left: 30px; text-align: justify;">When you dont have enough VRAM free after loading your LLM model into your VRAM (Normal Mode example below), you can see that with so little working space, your GPU will have to swap in and out bits of the TTS model, which causes horrible slowdown.</p>
<p style="padding-left: 30px; text-align: justify;"><span style="color: #ff0000;">Note:</span> An Nvidia Graphics card is required for the LowVRAM option to work, as you will just be using system RAM otherwise.&nbsp;</p>
<h4>How It Works:</h4>
<p style="padding-left: 30px; text-align: justify;">The Low VRAM mode intelligently orchestrates the relocation of the entire TTS model and stores the TTS model in your system RAM. When the TTS engine requires VRAM for processing, the entire model seamlessly moves into VRAM, causing your LLM to unload/displace some layers, ensuring optimal performance of the TTS engine.</p>
<p style="padding-left: 30px; text-align: justify;">Post-TTS processing, the model moves back to system RAM, freeing up VRAM space for your Language Model (LLM) to load back in the missing layers. This adds about 1-2 seconds to both text generation by the LLM and the TTS engine.</p>
<p style="padding-left: 30px; text-align: justify;">By transferring the entire model between RAM and VRAM, the Low VRAM option avoids fragmentation, ensuring the TTS model remains cohesive and has all the working space it needs in your GPU, without having to just work on small bits of the TTS model at a time (which causes terrible slow down).</p>
<p style="padding-left: 30px; text-align: justify;">This creates a TTS generation performance Boost for Low VRAM Users and is particularly beneficial for users with less than 2GB of free VRAM after loading their LLM, delivering a substantial 5-10x improvement in TTS generation speed.</p>
<div style="text-align: center;"><img src="/static/lowvrammode.png" alt="How Low VRAM Works" /></div>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>


<h2 id="finetune"><strong>Finetuning (Training the model)</strong></h2>

<p style="margin-left:40px; text-align:justify">If you have a voice that the model doesnt quite reproduce correctly, or indeed you just want to improve the reproduced voice, then finetuning is a way to train your &quot;XTTSv2 local&quot; model <span style="color:#3366ff">(stored in /alltalk_tts/models/xxxxx/)</span> on a specific voice. For this you will need:</p>

<ul style="margin-left:40px">
	<li>An Nvidia graphics card To install a few portions of the Nvidia CUDA <strong>11.8</strong> Toolkit (this will not impact text-generation-webui&#39;s cuda setup.</li>
	<li>18GB of disk space free (most of this is used temporarily)</li>
	<li>At least 2 minutes of good quality speech from your chosen speaker in mp3, wav or flacc format, in one or more files (have tested as far as 20 minutes worth of audio).</li>
</ul>

<h4>How will this work/How complicated is it?</h4>

<p style="margin-left:40px; text-align:justify">Everything has been done to make this as simple as possible. At its simplest, you can literally just download a large chunk of audio from an interview, and tell the finetuning to strip through it, find spoken parts and build your dataset. You can literally click 4 buttons, then copy a few files and you are done. At it&#39;s more complicated end you will clean up the audio a little beforehand, but its still only 4x buttons and copying a few files.</p>

<h4>The audio you will use:</h4>

<p style="margin-left:40px; text-align:justify">I would suggest that if its in an interview format, you cut out the interviewer speaking in audacity or your chosen audio editing package. You dont have to worry about being perfect with your cuts, the finetuning Step 1 will go and find spoken audio and cut it out for you. Is there is music over the spoken parts, for best quality you would cut out those parts, though its not 100% necessary. As always, try to avoid bad quality audio with noises in it (humming sounds, hiss etc). You can try something like Audioenhancer to try clean up noisier audio. There is no need to down-sample any of the audio, all of that is handled for you. Just give the finetuning some good quality audio to work with.</p>

<h4>Important requirements CUDA 11.8:</h4>

<p style="margin-left:40px; text-align:justify">As mentioned you must have a small portion of the <span style="color:#3366ff">Nvidia CUDA Toolkit&nbsp;<strong>11.8</strong></span>&nbsp;installed. Not higher or lower versions. Specifically&nbsp;<strong>11.8</strong>. You do not have to uninstall any other versions, change any graphics drivers, reinstall torch or anything like that. There are instructions within the finetuning interface on doing this or you can also find them on this link&nbsp;<a href="https://github.com/erew123/alltalk_tts#-important-requirements-cuda-118">here</a></p>

<h4>Starting Finetuning:</h4>

<p style="margin-left:40px"><strong>Ensure</strong> you have followed the instructions on setting up the Nvidia CUDA Toolkit 11.8 <a href="https://github.com/erew123/alltalk_tts#-important-requirements-cuda-118">here</a>&nbsp;or the below procedure will fail.</p>

<p style="margin-left:40px">The below instructions are also available online&nbsp;<a href="https://github.com/erew123/alltalk_tts#-finetuning-a-model">here</a></p>

<ol>
	<li>
	<p>Close all other applications that are using your GPU/VRAM and copy your audio samples into:<br />
	<br />
	<span style="color:#3366ff">/alltalk_tts/finetune/put-voice-samples-in-here/</span></p>
	</li>
	<li>
	<p>In a command prompt/terminal window you need to move into your Text generation webUI folder:<br />
	<br />
	<span style="color:#3366ff">cd text-generation-webui</span></p>
	</li>
	<li>
	<p>Start the Text generation webUI Python environment for your OS:<br />
	<br />
	<span style="color:#3366ff">cmd_windows.bat,&nbsp;./cmd_linux.sh,&nbsp;cmd_macos.sh&nbsp;or&nbsp;cmd_wsl.bat</span></p>
	</li>
	<li>
	<p>You can double check your search path environment still works correctly with&nbsp;<span style="color:#3366ff">nvcc --version</span>. It should report back <span style="color:#3366ff">11.8</span>:<br />
	<br />
	<span style="color:#3366ff">Cuda compilation tools, release 11.8.</span></p>
	</li>
	<li>
	<p>Move into your extensions folder:<br />
	<br />
	<span style="color:#3366ff">cd extensions</span></p>
	</li>
	<li>
	<p>Move into the&nbsp;<span style="color:#3366ff">alltalk_tts</span>&nbsp;folder:<br />
	<br />
	<span style="color:#3366ff">cd alltalk_tts</span></p>
	</li>
	<li>
	<p>Install the finetune requirements file:&nbsp;<span style="color:#3366ff">pip install -r requirements_finetune.txt</span></p>
	</li>
	<li>
	<p>Type<span style="color:#3366ff">&nbsp;python finetune.py</span>&nbsp;and it should start up.</p>
	</li>
	<li>
	<p>Follow the on-screen instructions when the web interface starts up.</p>
	</li>
	<li>
	<p>When you have finished finetuning, the final tab will tell you what to do with your files and how to move your newly trained model to the correct location on disk.</p>
	</li>
</ol>

<p><a href="#toc">Back to top of page</a></p>


<h2 id="deepspeed"><strong>DeepSpeed</strong></h2>
<p style="padding-left: 30px; text-align: justify;">DeepSpeed provides a 2x-3x speed boost for Text-to-Speech and AI tasks. It's all about making AI and TTS happen faster and more efficiently.</p>
<ul>
<li><strong>Model Parallelism:</strong> Spreads work across multiple GPUs, making AI/TTS models handle tasks more efficiently.</li>
<li><strong>Memory Magic:</strong> Optimizes how memory is used, reducing the memory needed for large AI/TTS models.</li>
<li><strong>Handles More Load:</strong> Scales up to handle larger workloads with improved performance.</li>
<li><strong>Smart Resource Use:</strong> Uses your computer's resources smartly, getting the most out of your hardware.</li>
</ul>
<div style="text-align: center;"><img src="/static/deepspeedexample.jpg" alt="DeepSpeed on vs off" /></div>
<p style="padding-left: 30px; text-align: justify;">DeepSpeed only works with the <span style="color: #3366ff;">XTTSv2 Local</span> model and will deactivate when other models are selected, even if the checkbox still shows as being selected.</p>
<p style="padding-left: 30px; text-align: justify;"><span style="color: #ff0000;">Note:</span> DeepSpeed/AllTalk may warn if the Nvidia Cuda Toolkit and CUDA_HOME environment variable isnt set correctly. On Linux you need CUDA_HOME configured correctly; on Windows, if you use the pre-built wheel its ok without.</p>
<p style="padding-left: 30px; text-align: justify;"><span style="color: #ff0000;">Note:</span><strong>&nbsp;</strong>You do <strong>not</strong> need to set Text-generation-webUi's --deepspeed setting for AllTalk to be able to use DeepSpeed.</p>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>


<h3 id="deepspeed-linux">DeepSpeed Setup - Linux</h3>
<p style="padding-left: 30px;">➡️DeepSpeed requires an Nvidia Graphics card!⬅️</p>
<ol>
<li>Preferably use your built in package manager to install the CUDA toolkit. Alternatively download and install the Nvidia Cuda Toolkit for Linux <a href="https://developer.nvidia.com/cuda-toolkit-archive">Nvidia Cuda Toolkit 11.8 or 12.1</a></li>
<li>Open a terminal console.</li>
<li>Install <span style="color: #3366ff;">libaio-dev</span> (however your Linux version installs things) e.g. <span style="color: #3366ff;">sudo apt install libaio-dev</span></li>
<li>Move into your Text generation webUI folder e.g. <span style="color: #0000ff;"><span style="color: #3366ff;">cd text-generation-webui</span><br /></span></li>
<li>Start the Text generation webUI Python environment <span style="color: #0000ff;"><span style="color: #3366ff;">./cmd_linux.sh</span><br /></span></li>
<li style="text-align: left;">Text generation webUI <strong>overwrites</strong> the <strong>CUDA_HOME</strong> environment variable each time you <span style="color: #3366ff;">./cmd_linux.sh</span> or <span style="color: #3366ff;">./start_linux.sh&nbsp;</span>so you will need to either permanently change within the python environment OR set CUDA_HOME it each time you <span style="color: #3366ff;">./cmd_linux.sh</span>. Details to change it each time are on the next step. Below is a link to Conda's manual and changing environment variables permanently though its possible changing it permanently could affect other extensions, you would have to test.<a href="https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#set-env-vars"> Conda manual - Environment variables<br /></a></li>
<li>You can temporarily set the <span style="color: #3366ff;">CUDA_HOME</span> environment with (Standard Ubuntu path below, but it could vary on other Linux flavours):<br /><br /><span style="color: #0000ff;"><span style="color: #3366ff;">export CUDA_HOME=/etc/alternatives/cuda</span>&nbsp;</span><strong>every</strong> time you run <span style="color: #0000ff;"><span style="color: #808080;">.<span style="color: #3366ff;">/cmd_linux.sh</span></span><br /><br /></span>If you try to start DeepSpeed with the CUDA_HOME path set incorrectly, expect an error similar to:<br /><br /><span style="color: #ffcc00;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;[Errno 2] No such file or directory: /home/yourname/text-generation-webui/installer_files/env/bin/nvcc<br /><br /></span></li>
<li>Now install deepspeed with <span style="color: #0000ff;"><span style="color: #3366ff;">pip install deepspeed</span><br /></span></li>
<li>You can now start Text generation webUI <span style="color: #3366ff;">python server.py</span> ensuring to activate your extensions.<br /><br />Just to reiterate, starting Text-generation-webUI with <span style="color: #3366ff;">./start_linux.sh</span> will overwrite the CUDA_HOME variable unless you have permanently changed it, hence always starting it with <span style="color: #3366ff;">./cmd_linux.sh</span>&nbsp;<strong>then</strong> setting the environment variable manually <span style="color: #3366ff;">export CUDA_HOME=/etc/alternatives/cuda</span> and <strong>then</strong> <span style="color: #3366ff;">python server.py</span>&nbsp;which is how you would need to run it each time, unless you permanently set the environment variable for CUDA_HOME within Text-generation-webUI's standard Python environment.<br /><br /><span style="color: #ff0000;">Removal</span><strong> -</strong> If it became necessary to uninstall DeepSpeed, you can do so with <span style="color: #3366ff;">./cmd_linux.sh</span> and then <span style="color: #3366ff;">pip uninstall deepspeed</span></li>
</ol>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>


<h3 id="deepspeed-windows">DeepSpeed Setup - Windows</h3>
<p style="padding-left: 30px;">➡️DeepSpeed requires an Nvidia Graphics card!⬅️</p>
<p style="padding-left: 30px;">DeepSpeed v11.2 will work on the current default text-generation-webui Python 3.11 environment! You have 2x options for how to setup DeepSpeed on Windows. A quick way <span style="color: #339966;"><strong>Option 1</strong></span>&nbsp;and a long way <span style="color: #ff9900;"><strong>Option 2</strong></span>.</p>
<p style="padding-left: 30px;">Thanks to <a href="https://github.com/S95Sedan">@S95Sedan</a> They managed to get DeepSpeed 11.2 working on Windows via making some edits to the original Microsoft DeepSpeed v11.2 installation. The original post is <a href="https://github.com/oobabooga/text-generation-webui/issues/4734#issuecomment-1843984142">here</a>.<br /><br /></p>
<p style="padding-left: 30px;"><strong><span style="text-decoration: underline;"><span style="color: #339966; text-decoration: underline;">OPTION 1 - <span style="color: #000000; text-decoration: underline;">Pre-Compiled Wheel Deepspeed v11.2 (Python 3.11 and 3.10)</span></span></span></strong></p>
<ol>
<li>Download the correct wheel version for your Python/Cuda from <a href="https://github.com/erew123/alltalk_tts/releases/tag/deepspeed" target="_blank">here</a> and save the file it inside your <span style="color: #3366ff;">text-generation-webui folder</span>.</li>
<li>Open a command prompt window, move into your <span style="color: #3366ff;">text-generation-webui folder</span> you can now start the Python environment for text-generation-webui <span style="color: #3366ff;">cmd_windows.bat<br /></span></li>
<li>With the file that you saved in the <span style="color: #3366ff;">text-generation-webui folder</span> you now type the following, replacing <span style="color: #99cc00;">your-version</span> with the name of the file you have:<br /><br /><span style="color: #3366ff;">pip install "deepspeed-0.11.2+<span style="color: #99cc00;">your-version</span>-win_amd64.whl"<br /><br /></span></li>
<li>This should install through cleanly and you should now have DeepSpeed v11.2 installed within the Python 3.11/3.10 environment of text-generation-webui.</li>
<li>When you start up text-generation-webui, and AllTalk starts, you should see:<br /><br /><span style="background-color: #999999;"><span style="background-color: #ffffff;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [AllTalk Startup] DeepSpeed</span></span><strong><span style="background-color: #999999;"><span style="color: #ffff00; background-color: #ffffff;"> Detected<br /></span></span></strong><strong><span style="background-color: #999999;"><span style="color: #ffff00; background-color: #ffffff;"><br /></span></span></strong></li>
<li>Within AllTalk, you will now have a checkbox for <span style="color: #3366ff;">Activate DeepSpeed</span> though remember you can only change 1x setting every 15 or so seconds, so dont try to activate DeepSpeed and LowVRAM simultaneously. When you are happy it works, you can set the default start-up settings in the settings page.<br /><br /><span style="color: #ff0000;">Removal</span> - If it became necessary to uninstall DeepSpeed, you can do so with <span style="color: #3366ff;">cmd_windows.bat</span> and then <span style="color: #3366ff;">pip uninstall deepspeed<br /></span><span style="color: #3366ff;"><br /><br /></span><span style="color: #3366ff;"><span style="text-decoration: underline;"><span style="color: #ff9900; text-decoration: underline;"><strong>OPTION 2 - <span style="color: #000000; text-decoration: underline;">Manual build of DeepSpeed&nbsp;v11.2 (Python 3.11 and 3.10)</span></strong></span></span><br /><br /></span>Due to the complexity of this, and the complicated formatting, instructions can be found on this <a href="https://github.com/erew123/alltalk_tts?tab=readme-ov-file#-option-2---a-bit-more-complicated" target="_blank">link</a></li>
</ol>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>



<h2 id="TTSmodels">TTS Models/Methods</h2>
<p style="padding-left: 30px; text-align: justify;">It's worth noting that all models and methods can and do sound different from one another. Many people complained about the quality of audio produced by the 2.0.3 model, so this extension will download the 2.0.2 model to your models folder and give you the choice to use 2.0.2 (<span style="color: #3366ff;">API Local</span> and <span style="color: #3366ff;">XTTSv2 Local</span>) or use the most current model 2.0.3 (<span style="color: #3366ff;">API TTS</span>). As/When a new model is released by Coqui it will be downloaded by the TTS service on startup and stored wherever the TTS service keeps new models for your operating system.</p>
<ul>
<li><span style="color: #3366ff;">API TTS:</span> Uses the current TTS model available that's downloaded by the TTS API process ( version 2.0.3 at the time of writing). This model is not stored in your "models" folder, but elsewhere on your system and managed by the TTS software.<br /></li>
<li><span style="color: #3366ff;">API Local:</span> Utilizes the 2.0.2 local model stored at <span style="color: #3366ff;">/alltalk_tts/models/xttsv2_2.0.2<br /></span></li>
<li><span style="color: #3366ff;">XTTSv2 Local: </span>Utilizes the 2.0.2 local model <span style="color: #3366ff;">/alltalk_tts/models/xttsv2_2.0.2</span> and utilizes a distinct TTS generation method. <span style="color: #99cc00;">Supports DeepSpeed acceleration</span>.</li>
</ul>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>


<h2 id="temperature-and-repetition-settings"><strong>Model Temperature and Repetition Settings</strong></h3>
<p style="padding-left: 30px; text-align: justify;">It is recommended not to modify these settings unless you fully comprehend their effects. A general overview is provided below for reference.</p>
<p style="padding-left: 30px; text-align: justify;">Changes to these settings won't take effect until you restart AllTalk/Text generation webUI.</p>
<p style="padding-left: 30px; text-align: justify;">These settings only affect API Local and XTTSv2 Local methods.</p>
<h4 style="text-align: justify;">Repetition Penalty:</h4>
<p style="padding-left: 30px; text-align: justify;">In the context of text-to-speech (TTS), the Repetition Penalty<strong>&nbsp;</strong>influences how the model handles the repetition of sounds, phonemes, or intonation patterns. Here's how it works:</p>
<ul style="text-align: justify;">
<li><strong>Higher Repetition Penalty (e.g. 16.0):</strong> The model is less likely to repeat sounds or patterns. It promotes diversity in the generated speech. This can result in a more varied and expressive output, though introduce elements of unpredictability in the TTS output.<br /></li>
<li><strong>Lower Repetition Penalty (e.g. 2.0):</strong> The model is more tolerant of repeating sounds or patterns. This might lead to more repetition in the generated speech, potentially making it sound more structured or rhythmically consistent. Lower values can still introduce expressive variations, but to a lesser extent. This tendency means that the generated speech may remain closer to the original sample.</li>
</ul>
<p style="padding-left: 30px; text-align: justify;">The factory setting for repetition penalty is 10.0</p>
<h4>Temperature:</h4>
<p style="padding-left: 30px; text-align: justify;">Temperature&nbsp;influences the randomness of the generated speech. Here's how it affects the output:</p>
<ul style="text-align: justify;">
<li><strong>Higher Temperature (e.g. 0.95):</strong> Increases randomness in how the model selects and pronounces phonemes or intonation patterns. This can result in more creative, but potentially less controlled or "stable," speech that may deviate from the input sample. It adds an element of unpredictability and variety, contributing to expressiveness in the voice output created.<br /></li>
<li><strong>Lower Temperature (e.g. 0.20):</strong> Reduces randomness, making the model more likely to closely mimic the input sample's voice, intonation, and overall style. This tends to produce more "coherent" speech that aligns closely with the characteristics of the training data or input voice sample. It adds a level of predictability and consistency, potentially reducing expressive variations. So it could end up sounging too monotone.</li>
</ul>
<p style="padding-left: 30px; text-align: justify;">The factory setting for temperature is 0.70</p>


<h4><strong>Temperature and Repetition Settings Examples:</strong></h4>
<ul style="text-align: justify;">
<li><strong>Temp High (0.90) and Repetition High (16.0):</strong><br /> Result: Speech may sound highly creative and diverse, with reduced repetition. It could be more expressive and unpredictable.<br /></li>
<li><strong>Temp Low (0.20) and Repetition High (16.0):</strong><br /> Result: Output tends to be focused and deterministic, but with reduced repetition. It may sound structured and less expressive.<br /></li>
<li><strong>Temp High (0.90) and Repetition Low (2.0):</strong><br /> Result: Speech may be more creative and diverse, with tolerance for repeating sounds. It could have expressive variations but with some structured patterns.<br /></li>
<li><strong>Temp Low (0.20) and Repetition Low (2.0):</strong><br /> Result: Output is focused and deterministic, with tolerance for repeating sounds. It may sound more structured and less expressive.</li>
</ul>
<p style="text-align: justify;">Factory settings should be fine for most people, however if you choose to experiment, setting extremely high or low values, especially without a good understanding of their effects, may lead to flat-sounding output or very strange-sounding output. It's advisable to experiment with adjustments incrementally and observe the impact on the generated speech to find a balance that suits your desired outcome.</p>
<p style="text-align: justify;"><a href="#toc">Back to top of page</a><br /></p>


<h2 id="start-up-checks">Start-up Checks</h2>
<p style="padding-left: 30px; text-align: justify;">AllTalk performs a variety of checks on startup and will warn out messages at the console should you need to do something such as update your TTS version.&nbsp;</p>
<p style="padding-left: 30px; text-align: justify;">A basic environment check to ensure everything should work e.g. is the model already downloaded, are the configuration files set correctly etc.</p>
<p style="padding-left: 30px; text-align: justify;">AllTalk will download the Xtts model (version 2.0.2) into your models folder. Many people didnt like the quality of the 2.0.3 model, however the latest model will be accessable on the API TTS setting (2.0.3 at the time of writing) so you have the best of both worlds.</p>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>

<h2 id="customTTSmodels">Custom TTS Models and Model path</h2>
<p style="padding-left: 30px; text-align: justify;">Its possible to set a custom model for the <span style="color: #3366ff;">API Local</span> and <span style="color: #3366ff;">XTTSv2 Local</span> methods, or indeed point it at the same model that <span style="color: #3366ff;">API TTS</span> uses (wherever it is stored on your OS of choice).</p>
<p style="padding-left: 30px; text-align: justify;">Many people did not like the sound quality of the Coqui <span style="color: #000000;">2.0.3</span> model, and as such the AllTalk downloads the <span style="color: #000000;">2.0.2</span> model seperately from the <span style="color: #000000;">2.0.3</span> model that TTS service downloads and manages.</p>
<p style="padding-left: 30px; text-align: justify;">Typically the <span style="color: #000000;">2.0.2</span> model is stored in your <span style="color: #3366ff;">/alltalk_tts/models</span> folder and it is downloaded on first start-up of the&nbsp;AllTalk_tts extension. However, you may either want to use a custom model version of your choosing, or re-point AllTalk to a different path on your system, or even point it so that&nbsp;<span style="color: #3366ff;">API Local</span> and <span style="color: #3366ff;">XTTSv2 Local</span> both use the same model that <span style="color: #3366ff;">API TTS</span> is using.</p>
<p style="padding-left: 30px; text-align: justify;">If you do choose to change the location there are a couple of things to note.&nbsp;</p>
<ul style="text-align: justify;">
<li>The folder you place the model in, <span style="color: #3366ff;">cannot </span>be called "<span style="color: #3366ff;">models</span>". This name is reserved solely for the system to identify you are or are not using a custom model.</li>
<li>On each startup, the&nbsp;AllTalk tts extension will check the custom location and if it does not exist, it will create it and download the files it needs. It will also re download any missing files in that location that are needed for the model to function.</li>
<li>There will be extra output at the console to inform you that you are using a custom model and each time you load up AllTalk extension or switch between models.</li>
</ul>
<p style="padding-left: 30px; text-align: justify;">To change the model path, there are at minimum 2x settings you need to alter in the <span style="color: #3366ff;">modeldownload.json</span> file, <span style="color: #3366ff;">base_path </span>and <span style="color: #3366ff;">model_path</span>.</p>
<p style="padding-left: 30px; text-align: justify;">You would edit the settings in the <span style="color: #3366ff;">modeldownload.json</span> file as follows (make a backup of your current file in case):<br /><br />&nbsp; &nbsp; &nbsp; &nbsp; Windows path example:&nbsp;<span style="color: #2980b9;">c:&bsol;&bsol;mystuff&bsol;&bsol;mydownloads&bsol;&bsol;myTTSmodel&bsol;&bsol;</span><span style="color: #00ccff;"><em>{files in here}<br /></em></span><span style="color: #3366ff;">&nbsp; &nbsp; &nbsp; &nbsp; base_path</span> would be "<span style="color: #2980b9;">c:&bsol;&bsol;mystuff&bsol;&bsol;mydownloads"<br /></span><span style="color: #3366ff;">&nbsp; &nbsp; &nbsp; &nbsp; model_path</span> would be:<span style="color: #2980b9;">&nbsp;"myTTSmodel"</span></p>
<p style="padding-left: 30px;"><span style="color: #ff0000;">Note: </span>On Windows systems, you have to specify a <span style="color: #e74c3c;"></span><span style="color: #e74c3c;">double backslash &bsol;&bsol;</span> for each folder level in the path (as above)</p>
<p style="padding-left: 30px;">&nbsp; &nbsp; &nbsp; &nbsp; Linux path example:&nbsp;<span style="color: #2980b9;">/home/myaccount/myTTSmodel/</span><span style="color: #00ccff;"><em>{files in here}<br /></em></span><span style="color: #3366ff;">&nbsp; &nbsp; &nbsp; &nbsp; base_path</span> would be "<span style="color: #2980b9;">/home/myaccount"<br /></span><span style="color: #3366ff;">&nbsp; &nbsp; &nbsp; &nbsp; model_path</span> would be:<span style="color: #2980b9;">&nbsp;"TTSmodel"<br /><br /></span>Once you restart Alltalk, it will check this path for the files and output any details at the console.</p>
<p style="padding-left: 30px;">When you are happy it's' working correctly, you are welcome to go delete the models folder stored at&nbsp;<span style="color: #3366ff;">/alltalk_tts/models</span><strong>.</strong></p>
<p style="padding-left: 30px;">If you wish to change the files that the modeldownloader is pulling at startup, you can futher edit the <span style="color: #3366ff;">modeldownload.json</span> and change the https addresses within this files&nbsp;<span style="color: #3366ff;">files_to_download</span> section &nbsp;e.g.</p>
<p style="padding-left: 30px;">"files_to_download": {<br /> &nbsp; &nbsp; &nbsp; &nbsp; "LICENSE.txt": "<span style="color: #00ccff;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/LICENSE.txt?download=true</span>",<br /> &nbsp; &nbsp; &nbsp; &nbsp; "README.md": "<span style="color: #00ccff;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/README.md?download=true</span>",<br /> &nbsp; &nbsp; &nbsp; &nbsp; "config.json": "<span style="color: #00ccff;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/config.json?download=true</span>",<br /> &nbsp; &nbsp; &nbsp; &nbsp; "model.pth": "<span style="color: #00ccff;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth?download=true</span>",<br /> &nbsp; &nbsp; &nbsp; &nbsp; "vocab.json": "<span style="color: #00ccff;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/vocab.json?download=true</span>"<br />&nbsp;}</p>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page<br /></a></p>

<h2 id="configuration-details"><strong>Configuration file settings</strong></h2>
<p><span style="color: #3366ff;">confignew.json</span> file:</p>
<table>
    <tr>
        <th>Key</th>
        <th>Default Value</th>
        <th>Explanation</th>
    </tr>
    <tr>
        <td>"activate"</td>
        <td>true</td>
        <td>Sets activation state within Text-generation-webUI.</td>
    </tr>
    <tr>
        <td>"autoplay"</td>
        <td>true</td>
        <td>Sets autoplay within Text generation webUI.</td>
    </tr>
    <tr>
        <td>"branding"</td>
        <td>"AllTalk"</td>
        <td>Used to change the default name. Requires a space e.g. "Mybrand ".</td>
    </tr>
    <tr>
        <td>"deepspeed_activate"</td>
        <td>false</td>
        <td>Sets DeepSpeed activation on startup.</td>
    </tr>
    <tr>
        <td>"delete_output_wavs"</td>
        <td>"Disabled"</td>
        <td>Sets duration of outputs to delete.</td>
    </tr>
    <tr>
        <td>"ip_address"</td>
        <td>"127.0.0.1"</td>
        <td>Sets default IP address.</td>
    </tr>
    <tr>
        <td>"language"</td>
        <td>"English"</td>
        <td>Sets default language for Text-generation-webUI TTS.</td>
    </tr>
    <tr>
        <td>"low_vram"</td>
        <td>false</td>
        <td>Sets default setting for LowVRAM mode.</td>
    </tr>
    <tr>
        <td>"local_temperature"</td>
        <td>"0.70"</td>
        <td>Sets default model temp for API Local and XTTSv2 Local.</td>
    </tr>
    <tr>
        <td>"local_repetition_penalty"</td>
        <td>"10.0"</td>
        <td>Sets default model repetition for API Local and XTTSv2 Local.</td>
    </tr>
    <tr>
        <td>"tts_model_loaded"</td>
        <td>true</td>
        <td>AllTalk internal use only. Do not change.</td>
    </tr>
    <tr>
        <td>"tts_model_name"</td>
        <td>"tts_models/multilingual/multi-dataset/xtts_v2"</td>
        <td>Sets default model that API TTS is looking for through the TTS service (separate to API Local and XTTSv2 Local).</td>
    </tr>
    <tr>
        <td>"narrator_enabled"</td>
        <td>true</td>
        <td>Sets default narrator on/off in Text-generation-webUI TTS.</td>
    </tr>
    <tr>
        <td>"narrator_voice"</td>
        <td>"female_02.wav"</td>
        <td>Sets default wav to use for narrator in Text-generation-webUI TTS.</td>
    </tr>
    <tr>
        <td>"port_number"</td>
        <td>"7851"</td>
        <td>Sets default port number for AllTalk.</td>
    </tr>
    <tr>
        <td>"output_folder_wav"</td>
        <td>"extensions/alltalk_tts/outputs/"</td>
        <td>Sets default output path Text-generation-webUI should use for finding outputs.</td>
    </tr>
    <tr>
        <td>"output_folder_wav_standalone"</td>
        <td>"outputs/"</td>
        <td>Sets default output path in standalone mode.</td>
    </tr>
    <tr>
        <td>"remove_trailing_dots"</td>
        <td>false</td>
        <td>Sets trailing dot removal pre-generating TTS.</td>
    </tr>
    <tr>
        <td>"show_text"</td>
        <td>true</td>
        <td>Sets if text should be displayed below audio in Text-generation-webUI.</td>
    </tr>
    <tr>
        <td>"tts_method_api_local"</td>
        <td>false</td>
        <td>Sets API Local as the default model/method for TTS.</td>
    </tr>
    <tr>
        <td>"tts_method_api_tts"</td>
        <td>false</td>
        <td>Sets API TTS as the default model/method for TTS.</td>
    </tr>
    <tr>
        <td>"tts_method_xtts_local"</td>
        <td>true</td>
        <td>Sets XTTSv2 Local as the default model/method for TTS.</td>
    </tr>
    <tr>
        <td>"voice"</td>
        <td>"female_01.wav"</td>
        <td>Sets default voice for TTS.</td>
    </tr>
</table>

<p><span style="color: #3366ff;">modeldownload.json</span> file:</p>
<table id="modeldownload-table">
    <tr>
        <th>Key</th>
        <th>Value</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><span class="key">"base_path"</span></td>
        <td>"models"</td>
        <td>Sets local model base path for API Local and XTTSv2 Local.</td>
    </tr>
    <tr>
        <td><span class="key">"model_path"</span></td>
        <td>"xttsv2_2.0.2"</td>
        <td>Sets local model folder for API Local and XTTSv2 Local below the base path.</td>
    </tr>
    <tr>
        <td><span class="key">"files_to_download"</span></td>
        <td>
            <table>
                <tr>
                    <th>File</th>
                    <th>Download URL</th>
                </tr>
                <tr>
                    <td>"LICENSE.txt"</td>
                    <td><span style="color: #333399;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/LICENSE.txt?download=true</span></td>
                </tr>
                <tr>
                    <td>"README.md"</td>
                    <td><span style="color: #333399;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/README.md?download=true</span></td>
                </tr>
                <tr>
                    <td>"config.json"</td>
                    <td><span style="color: #333399;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/config.json?download=true</span></td>
                </tr>
                <tr>
                    <td>"model.pth"</td>
                    <td><span style="color: #333399;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth?download=true</span></td>
                </tr>
                <tr>
                    <td>"vocab.json"</td>
                    <td><span style="color: #333399;">https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/vocab.json?download=true</span></td>
                </tr>
            </table>
        </td>
        <td>Sets the model files required to be downloaded into &bsol;base_path&bsol;model_path&bsol; and where to download them from.</td>
    </tr>
</table>

<h2 id="curl-commands"><strong>JSON calls &amp; CURL Commands</strong></h2>

<h3>Overview</h3>
<p style="margin-left: 40px; text-align: justify;">The Text-to-Speech (TTS) Generation API allows you to generate speech from text input using various configuration options. This API supports both character and narrator voices, providing flexibility for creating dynamic and engaging audio content.</p>
<h3>TTS Generation Endpoint</h3>
<ul>
<li><strong>URL</strong>: <span style="color: #3366ff;">http://127.0.0.1:7851/api/tts-generate</span></li>
<li><strong>Method</strong>: <span style="color: #3366ff;">POST</span></li>
<li><strong>Content-Type</strong>: <span style="color: #3366ff;">application/x-www-form-urlencoded</span></li>
</ul>
<h3>Example command line</h3>
<p style="padding-left: 30px;">Standard TTS speech Example (standard text) generating a time-stamped file</p>
<p style="padding-left: 30px;"><span style="color: #3366ff;">curl -X POST "http://127.0.0.1:7851/api/tts-generate" -d "text_input=</span><span style="color: #ff9900;">All of this is text spoken by the character. This is text not inside quotes, though that doesnt matter in the slightest</span><span style="color: #3366ff;">" -d "text_filtering=</span><span style="color: #339966;">standard</span><span style="color: #3366ff;">" -d "character_voice_gen=</span><span style="color: #ff9900;">female_01.wav</span><span style="color: #3366ff;">" -d "narrator_enabled=</span><span style="color: #339966;">false</span><span style="color: #3366ff;">" -d "narrator_voice_gen=male_01.wav" -d "text_not_inside=character" -d "language=en" -d "output_file_name=</span><span style="color: #ff9900;">myoutputfile</span><span style="color: #3366ff;">" -d "output_file_timestamp=</span><span style="color: #ff9900;">true</span><span style="color: #3366ff;">" -d "autoplay=true" -d "autoplay_volume=0.8"</span></p>
<p style="padding-left: 30px;">Narrator Example (standard text)&nbsp;generating a time-stamped file</p>
<p style="padding-left: 30px;"><span style="color: #3366ff;">curl -X POST "http://127.0.0.1:7851/api/tts-generate" -d "text_input=</span><span style="color: #ff9900;">*This is text spoken by the narrator* &bsol;"This is text spoken by the character&bsol;". This is text not inside quotes.</span><span style="color: #3366ff;">" -d "text_filtering=</span><span style="color: #339966;">standard</span><span style="color: #3366ff;">" -d "character_voice_gen=</span><span style="color: #ff9900;">female_01.wav</span><span style="color: #3366ff;">" -d "narrator_enabled=</span><span style="color: #339966;">true</span><span style="color: #3366ff;">" -d "narrator_voice_gen=</span><span style="color: #339966;">male_01.wav</span><span style="color: #3366ff;">" -d "text_not_inside=</span><span style="color: #339966;">character</span><span style="color: #3366ff;">" -d "language=en" -d "output_file_name=</span><span style="color: #ff9900;">myoutputfile</span><span style="color: #3366ff;">" -d "output_file_timestamp=</span><span style="color: #ff9900;">true</span><span style="color: #3366ff;">" -d "autoplay=true" -d "autoplay_volume=0.8"</span></p>
<p style="padding-left: 30px; text-align: justify;"><strong><span style="color: #ff0000;">Note</span></strong> that if your text that needs to be generated contains double quotes you will need to escape them with&nbsp;&bsol;" (Please see the narrator example).</p>
<h3>&nbsp;Request Parameters</h3>
<ul>
<li>
<p style="text-align: justify;"><strong>text_input</strong>: The text you want the TTS engine to produce. Use escaped double quotes for character speech and asterisks for narrator speech if using the narrator function. Example:</p>
<p>-<span style="color: #3366ff;">d "text_input=*This is text spoken by the narrator* &bsol;"This is text spoken by the character&bsol;". This is text not inside quotes." </span></p>
</li>
<li>
<p><strong>text_filtering</strong>: Filter for text. Options:</p>
<ul>
<li style="text-align: justify;"><span style="color: #3366ff;">none</span>&nbsp;No filtering. Whatever is sent will go over to the TTS engine as raw text, which may result in some odd sounds with some special characters.</li>
<li><span style="color: #3366ff;">standard</span>&nbsp;Human-readable text and a basic level of filtering, just to clean up some special characters.</li>
<li><span style="color: #3366ff;">html</span>&nbsp;HTML content. Where you are using HTML entity's like &amp;quot;</li>
</ul>
<p><span style="color: #3366ff;">-d "text_filtering=none" </span><br /><span style="color: #3366ff;">-d "text_filtering=standard" </span><br /><span style="color: #3366ff;">-d "text_filtering=html"</span></p>
<p><strong>Example:</strong></p>
<ul>
<li>Standard Example: <span style="color: #3366ff;">*This is text spoken by the narrator* "This is text spoken by the character" This is text not inside quotes.</span></li>
<li>HTML Example:<span style="color: #3366ff;"> *This is text spoken by the narrator*&nbsp;&amp;quot;This is text spoken by the character&amp;quot;&nbsp;This is text not inside quotes.</span></li>
<li>None will just pass whatever characters/text you send at it.</li>
</ul>
</li>
<li>
<p><strong>character_voice_gen</strong>: The WAV file name for the character's voice.</p>
<p><span style="color: #3366ff;">-d "character_voice_gen=female_01.wav" </span></p>
</li>
<li>
<p><strong>narrator_enabled</strong>: Enable or disable the narrator function. If true, minimum text filtering is set to standard. Anything between double quotes is considered the character's speech, and anything between asterisks is considered the narrator's speech.</p>
<p><span style="color: #3366ff;">-d "narrator_enabled=true"<br />-d "narrator_enabled=false" </span><span style="font-family: Verdana, Arial, Helvetica, sans-serif;">&nbsp;</span></p>
</li>
<li>
<p><strong>narrator_voice_gen</strong>: The WAV file name for the narrator's voice.</p>
<p><span style="color: #3366ff;">-d "narrator_voice_gen=male_01.wav"</span></p>
</li>
<li>
<p><strong>text_not_inside</strong>: Specify the handling of lines not inside double quotes or asterisks, for the narrator feature. Options:</p>
<ul>
<li><span style="color: #3366ff;">character</span>: Treat as character speech.</li>
<li><span style="color: #3366ff;">narrator</span>: Treat as narrator speech.</li>
</ul>
<p><span style="color: #3366ff;">-d "text_not_inside=character" </span><br /><span style="color: #3366ff;">-d "text_not_inside=narrator"</span></p>
</li>
<li>
<p><strong>language</strong>: Choose the language for TTS. Options:</p>
<ul>
<li><span style="color: #3366ff;">ar</span>&nbsp;Arabic</li>
<li><span style="color: #3366ff;">zh-cn</span>&nbsp;Chinese (Simplified)</li>
<li><span style="color: #3366ff;">cs</span>&nbsp;Czech</li>
<li><span style="color: #3366ff;">nl</span> Dutch</li>
<li><span style="color: #3366ff;">en</span> English</li>
<li><span style="color: #3366ff;">fr</span> French</li>
<li><span style="color: #3366ff;">de</span> German</li>
<li><span style="color: #3366ff;">hu</span> Hungarian</li>
<li><span style="color: #3366ff;">it</span> Italian</li>
<li><span style="color: #3366ff;">ja</span> Japanese</li>
<li><span style="color: #3366ff;">ko</span> Korean</li>
<li><span style="color: #3366ff;">pl</span> Polish</li>
<li><span style="color: #3366ff;">pt</span> Portuguese</li>
<li><span style="color: #3366ff;">ru </span>Russian</li>
<li><span style="color: #3366ff;">es</span> Spanish</li>
<li><span style="color: #3366ff;">tr</span> Turkish</li>
</ul>
<p><span style="color: #3366ff;">-d "language=en"</span></p>
</li>
<li>
<p><strong>output_file_name</strong>: The name of the output file (excluding the .wav extension).</p>
<p><span style="color: #3366ff;">-d "output_file_name=myoutputfile" </span></p>
</li>
<li>
<p><strong>output_file_timestamp</strong>: Add a timestamp to the output file name. If true, each file will have a unique timestamp; otherwise, the same file name will be overwritten each time you generate TTS.</p>
<p><span style="color: #3366ff;">-d "output_file_timestamp=true" <br />-d "output_file_timestamp=false" </span></p>
</li>
<li>
<p><strong>autoplay</strong>: <span style="color: #ff0000;">Feature not yet available</span>. Enable or disable autoplay. Still needs to be specified in the JSON request.</p>
<p><span style="color: #3366ff;">-d "autoplay=true" <br />-d "autoplay=false"</span></p>
</li>
<li>
<p><strong>autoplay_volume</strong>: <span style="color: #ff0000;">Feature not yet available</span>. Set the autoplay volume. Should be between 0.1 and 1.0.&nbsp;Still needs to be specified in the JSON request.</p>
<p><span style="color: #3366ff;">-d "autoplay_volume=0.8" </span></p>
</li>
</ul>
<h3>TTS Generation Response</h3>
<p style="padding-left: 30px;">The API returns a JSON object with the following properties:</p>
<ul>
<li><span style="color: #3366ff;"><strong>status</strong></span>&nbsp;Indicates whether the generation was successful (<span style="color: #3366ff;">generate-success</span>) or failed (<span style="color: #3366ff;">generate-failure</span>).</li>
<li><span style="color: #3366ff;"><strong>output_file_path</strong></span>&nbsp;The on-disk location of the generated WAV file.</li>
<li><span style="color: #3366ff;"><strong>output_file_url</strong></span>&nbsp;The HTTP location for accessing the generated WAV file.</li>
</ul>
<p><strong>Example JSON TTS Generation Response:</strong></p>
<p style="padding-left: 30px;"><span style="color: #339966;">{"status": "generate-success", "output_file_path": "C:\\text-generation-webui\\extensions\\alltalk_tts\\outputs\\myoutputfile_1703149973.wav", "output_file_url": "http://127.0.0.1:7851/audio/myoutputfile_1703149973.wav"}</span></p>


<h4>Switching Model</h4>
<p style="padding-left: 30px;"><span style="color: #3366ff;">curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=API%20Local"</span><br /><span style="color: #3366ff;"> curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=API%20TTS"</span><br /><span style="color: #3366ff;"> curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=XTTSv2%20Local"</span></p>
<p style="padding-left: 30px;">Switch between the 3 models respectively.</p>
<p style="padding-left: 30px;">JSON return <span style="color: #339966;">{"status": "model-success"}</span></p>
<h4>Switch DeepSpeed</h4>
<p style="padding-left: 30px;"><span style="color: #3366ff;">curl -X POST "http://127.0.0.1:7851/api/deepspeed?new_deepspeed_value=True"</span></p>
<p style="padding-left: 30px;">Replace True with False to disable DeepSpeed mode.</p>
<p style="padding-left: 30px;">JSON return <span style="color: #339966;">{"status": "deepspeed-success"}</span></p>
<h4>Switching Low VRAM</h4>
<p style="padding-left: 30px;"><span style="color: #3366ff;">curl -X POST "http://127.0.0.1:7851/api/lowvramsetting?new_low_vram_value=True"</span></p>
<p style="padding-left: 30px;">Replace True with False to disable Low VRAM mode.</p>
<p style="padding-left: 30px;">JSON return <span style="color: #339966;">{"status": "lowvram-success"}</span></p>
<h4>Ready Endpoint</strong></h4>
<p>Check if the Text-to-Speech (TTS) service is ready to accept requests.</p>
<ul>
  <li>URL: <span style="color: #3366ff;">http://127.0.0.1:7851/api/ready</span></li>
  <li>Method: <span style="color: #3366ff;">GET</span></li>
  <li>Response: <span style="color: #339966;">Ready</span></li>
</ul>
<p style="padding-left: 30px;"><span style="color: #3366ff;">curl -X GET "http://127.0.0.1:7851/api/ready"</span></p>
<h4>Voices List Endpoint</strong></h4>
<p>Retrieve a list of available voices for generating speech.</p>
<ul>
  <li>URL: <span style="color: #3366ff;">http://127.0.0.1:7851/api/voices</span></li>
  <li>Method: <span style="color: #3366ff;">GET</span></li>
</ul>
<p style="padding-left: 30px;"><span style="color: #3366ff;">curl -X GET "http://127.0.0.1:7851/api/voices"</span></p>
<p style="padding-left: 30px;">JSON return: <span style="color: #339966;">{"voices": ["voice1.wav", "voice2.wav", "voice3.wav"]}</span></p>
<h4><strong>Preview Voice Endpoint</strong></h4>
<p>Generate a preview of a specified voice with hardcoded settings.</p>
<ul>
  <li>URL: <span style="color: #3366ff;">http://127.0.0.1:7851/api/previewvoice/</span></li>
  <li>Method: <span style="color: #3366ff;">POST</span></li>
  <li>Content-Type: <span style="color: #3366ff;">application/x-www-form-urlencoded</span></li>
</ul>
<p style="padding-left: 30px;"><span style="color: #3366ff;">curl -X POST "http://127.0.0.1:7851/api/previewvoice/" -F "voice=female_01.wav"</span></p>
<p style="padding-left: 30px;">Replace <span style="color: #3366ff;">female_01.wav</span> with the name of the voice sample you want to hear.</p>
<p style="padding-left: 30px;">JSON return: <span style="color: #339966;">{"status": "generate-success", "output_file_path": "/path/to/outputs/api_preview_voice.wav", "output_file_url": "http://127.0.0.1:7851/audio/api_preview_voice.wav"}</span></p>

<p><a href="#toc">Back to top of page<br /></a></p>

<h2 id="debugging-and-tts-generation-information"><strong>Debugging and TTS Generation Information</strong></h2>
<p style="padding-left: 30px; text-align: justify;">Command line outputs are more verbose to assist in understanding backend processes and debugging.</p>
<p style="padding-left: 30px; text-align: justify;">Its possible during startup you can get a warning message such as: <br /><br /><span style="color: black;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AllTalk Startup]</span> <span style="color: #ff0000;">Warning</span> <span style="color: black;">TTS Subprocess has NOT started up yet, Will keep trying for 80 seconds maximum</span> <br /><br />This is normal behavior if the subprocess is taking a while to start, however, if there is an issue starting the subprocess, you may see multiples of this message and an it will time out after 80 seconds, resulting in the TTS engine not starting. Its likely that you are not in the correct python environment or one that has a TTS engine inside, if this happens, though the system will output a warning about that ahead of this message</p>
<p style="padding-left: 30px; text-align: justify;">Typically the command line console will output any warning or error messages. If you need to reset your default configuation, the settings are all listed above in the configuration details.</p>
<p style="padding-left: 30px;"><a href="#toc">Back to top of page</a></p>
<h2 id="references"><strong>Thanks &amp; References</strong></h2>
<h4>Coqui TTS Engine</h4>
<ul>
<li><a href="https://coqui.ai/cpml.txt" target="_blank" rel="noopener">Coqui License</a></li>
<li><a href="https://github.com/coqui-ai/TTS" target="_blank" rel="noopener">Coqui TTS GitHub Repository</a></li>
</ul>
<h4>Extension coded by</h4>
<ul>
<li><a href="https://github.com/erew123" target="_blank" rel="noopener">Erew123 GitHub Profile</a></li>
</ul>
<h4>Thanks to &amp; Text generation webUI</h4>
<ul>
<li><a href="https://github.com/oobabooga/text-generation-webui" target="_blank" rel="noopener">Ooobabooga GitHub Repository</a> (Portions of orginal Coquii_TTS extension)</li>
</ul>
<h4>Thanks to</h4>
<ul>
<li><a href="https://github.com/daswer123" target="_blank" rel="noopener">daswer123 GitHub Profile</a> (Assistance with cuda to cpu moving)</li>
<li><a href="https://github.com/S95Sedan" target="_blank" rel="noopener">S95Sedan GitHub Profile</a> (Editing the Microsoft DeepSpeed v11.x installation files so they work)</li>
<li><a href="https://github.com/kanttouchthis" target="_blank" rel="noopener">kanttouchthis GitHub Profile</a> (Portions of orginal Coquii_TTS extension)</li>
<li><a href="https://github.com/Wuzzooy" target="_blank" rel="noopener">Wuzzooy GitHub Profile</a> (Trying out the code while in development)</li>
</ul>
<p><a href="#toc">Back to top of page</a></p>

</body>

</html>
"""

###################################################
#### Webserver Startup & Initial model Loading ####
###################################################
# Create a Jinja2 template object
template = Template(simple_webpage)

# Render the template with the dynamic values
rendered_html = template.render(params=params)

###############################
#### Internal script ready ####
###############################
@app.get("/ready")
async def ready():
    return Response("Ready endpoint")

############################
#### External API ready ####
############################
@app.get("/api/ready")
async def ready():
    return Response("Ready")

@app.get("/")
async def read_root():
    return HTMLResponse(content=rendered_html, status_code=200)

# Start Uvicorn Webserver
host_parameter = {params["ip_address"]}
port_parameter = str(params["port_number"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host_parameter, port=port_parameter, log_level="warning")
