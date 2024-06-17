import json
import time
import os
from pathlib import Path
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import io
import wave
import logging
logging.disable(logging.WARNING)
##########################
#### Webserver Imports####
##########################
from fastapi import (
    FastAPI,
    Form,
    Request,
    Response,
    Depends,
    HTTPException,
    Query,
)
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

###########################
#### STARTUP VARIABLES ####
###########################
# STARTUP VARIABLE - Create "this_dir" variable as the current script directory
this_dir = Path(__file__).parent.resolve()
# STARTUP VARIABLE - Set "device" to cuda if exists, otherwise cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
# STARTUP VARIABLE - Import languges file for Gradio to be able to display them in the interface
with open(this_dir / "system" / "config" / "languages.json", encoding="utf8") as f:
    languages = json.load(f)
# Base setting for a possible FineTuned model existing and the loader being available
tts_method_xtts_ft = False

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

# Load values for temperature and repetition_penalty
temperature = params["local_temperature"]
repetition_penalty = params["local_repetition_penalty"]

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

##################################################
#### Check to see if a finetuned model exists ####
##################################################
# Set the path to the directory
trained_model_directory = this_dir / "models" / "trainedmodel"
# Check if the directory "trainedmodel" exists
finetuned_model = trained_model_directory.exists()
# If the directory exists, check for the existence of the required files
if finetuned_model:
    required_files = ["model.pth", "config.json", "vocab.json"]
    finetuned_model = all((trained_model_directory / file).exists() for file in required_files)

########################
#### STARTUP CHECKS ####
########################
try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Could not find the TTS module. Make sure to install the requirements for the alltalk_tts extension.",
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Linux / Mac:\npip install -r extensions/alltalk_tts/requirements.txt\n",
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Windows:\npip install -r extensions\\alltalk_tts\\requirements.txt\n",
        f"[{params['branding']}Startup] \033[91mWarning\033[0m If you used the one-click installer, paste the command above in the terminal window launched after running the cmd_ script. On Windows, that's cmd_windows.bat."
    )
    raise

# DEEPSPEED Import - Check for DeepSpeed and import it if it exists
deepspeed_available = False
try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    pass

@asynccontextmanager
async def startup_shutdown(no_actual_value_it_demanded_something_be_here):
    await setup()
    yield
    # Shutdown logic


# Create FastAPI app with lifespan
app = FastAPI(lifespan=startup_shutdown)
# Allow all origins, and set other CORS options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#####################################
#### MODEL LOADING AND UNLOADING ####
#####################################
# MODEL LOADERS Picker For API TTS, API Local, XTTSv2 Local, XTTSv2 FT
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
    elif tts_method_xtts_ft:
        print(
            f"[{params['branding']}Model] \033[94mXTTSv2 FT Loading\033[0m /models/fintuned/model.pth \033[94minto\033[93m",
            device,
            "\033[0m",
        )
        model = await xtts_ft_manual_load_model()
    # Create an end timer for calculating load times
    generate_end_time = time.time()
    # Calculate start time minus end time
    generate_elapsed_time = generate_end_time - generate_start_time
    # Print out the result of the load time
    print(f"[{params['branding']}Model] \033[94mModel Loaded in \033[93m{generate_elapsed_time:.2f} seconds.\033[0m")
    print(f"[{params['branding']}Model] Ready")
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
        print(f"[{params['branding']}Model] \033[94mCoqui Public Model License\033[0m")
        print(f"[{params['branding']}Model] \033[94mhttps://coqui.ai/cpml.txt\033[0m")
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
    print(f"[{params['branding']}Model] \033[94mCoqui Public Model License\033[0m")
    print(f"[{params['branding']}Model] \033[94mhttps://coqui.ai/cpml.txt\033[0m")
    return model

# MODEL LOADER For "XTTSv2 FT"
async def xtts_ft_manual_load_model():
    global model
    config = XttsConfig()
    config_path = this_dir / "models" / "trainedmodel" / "config.json"
    vocab_path_dir = this_dir / "models" / "trainedmodel" / "vocab.json"
    checkpoint_dir = this_dir / "models" / "trainedmodel"
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(checkpoint_dir),
        vocab_path=str(vocab_path_dir),
        use_deepspeed=params["deepspeed_activate"],
    )
    model.to(device)
    print(f"[{params['branding']}Model] \033[94mCoqui Public Model License\033[0m")
    print(f"[{params['branding']}Model] \033[94mhttps://coqui.ai/cpml.txt\033[0m")
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
    global tts_method_xtts_ft
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
        tts_method_xtts_ft = False
    elif tts_method == "API Local":
        params["tts_method_api_tts"] = False
        params["tts_method_xtts_local"] = False
        params["tts_method_api_local"] = True
        params["deepspeed_activate"] = False
        tts_method_xtts_ft = False
    elif tts_method == "XTTSv2 Local":
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = True
        tts_method_xtts_ft = False
    elif tts_method == "XTTSv2 FT":
        tts_method_xtts_ft = True
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = False

    # Unload the current model
    model = await unload_model(model)

    # Load the correct model based on the updated params
    await setup()


# MODEL WEBSERVER- API Swap Between Models
@app.route("/api/reload", methods=["POST"])
async def reload(request: Request):
    tts_method = request.query_params.get("tts_method")
    if tts_method not in ["API TTS", "API Local", "XTTSv2 Local", "XTTSv2 FT"]:
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
debug_generate_audio = False
tts_stop_generation = False # Called to stop generation of the current text at whatever stage its at. currently only set for streaming.
tts_generation_lock = False # Tracks locking of the generaetion process.
tts_narrator_generatingtts = False # Tracks if the current tts processes are narrator based, to avoid moving model on each chunk of text generated.
# Not in use tts_queue_behaviour = False # True is to queue the current generation request. False is to cancel the current generation and start a new one.

# TTS VOICE GENERATION METHODS (called from voice_preview and output_modifer)
async def generate_audio(text, voice, language, temperature, repetition_penalty, output_file, streaming=False):
    # Get the async generator from the internal function
    response = generate_audio_internal(text, voice, language, temperature, repetition_penalty, output_file, streaming)
    # If streaming, then return the generator as-is, otherwise just exhaust it and return
    if streaming:
        return response
    async for _ in response:
        pass


async def generate_audio_internal(text, voice, language, temperature, repetition_penalty, output_file, streaming):
    global model, tts_stop_generation, tts_generation_lock
    tts_generation_lock = True
    if params["low_vram"] and device == "cpu":
        await switch_device()
    generate_start_time = time.time()  # Record the start time of generating TTS
    
    # XTTSv2 LOCAL & Xttsv2 FT Method
    if params["tts_method_xtts_local"] or tts_method_xtts_ft:
        print(f"[{params['branding']}TTSGen] {text}")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[f"{this_dir}/voices/{voice}"],
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs,
        )

        # Common arguments for both functions
        common_args = {
            "text": text,
            "language": language,
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
            "temperature": float(temperature),
            "length_penalty": float(model.config.length_penalty),
            "repetition_penalty": float(repetition_penalty),
            "top_k": int(model.config.top_k),
            "top_p": float(model.config.top_p),
            "enable_text_splitting": True
        }

        # Determine the correct inference function and add streaming specific argument if needed
        inference_func = model.inference_stream if streaming else model.inference
        if streaming:
            common_args["stream_chunk_size"] = 20

        # Call the appropriate function
        output = inference_func(**common_args)

        # Process the output based on streaming or non-streaming
        if streaming:
            # Streaming-specific operations
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
                if tts_stop_generation:
                    print(f"[{params['branding']}TTSGen] Stopping audio generation.")
                    file_chunks.clear()  # Clear the file_chunks list
                    tts_stop_generation = False
                    tts_generation_lock = False
                    break
                file_chunks.append(chunk)
                if isinstance(chunk, list):
                    chunk = torch.cat(chunk, dim=0)
                chunk = chunk.clone().detach().cpu().numpy()
                chunk = chunk[None, : int(chunk.shape[0])]
                chunk = np.clip(chunk, -1, 1)
                chunk = (chunk * 32767).astype(np.int16)
                yield chunk.tobytes()
                print(f"[{params['branding']}Debug] Stream audio generation: Yielded audio chunk {i}.") if debug_generate_audio else None   
        else:
            # Non-streaming-specific operation
            torchaudio.save(output_file, torch.tensor(output["wav"]).unsqueeze(0), 24000)

    
    # API LOCAL Methods
    elif params["tts_method_api_local"]:
        # Streaming only allowed for XTTSv2 local
        if streaming:
            raise ValueError("Streaming is only supported in XTTSv2 local")

        # Set the correct output path (different from the if statement)
        print(f"[{params['branding']}TTSGen] Using API Local")
        model.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=[f"{this_dir}/voices/{voice}"],
            language=language,
            temperature=float(temperature),
            length_penalty=float(model.config.length_penalty),
            repetition_penalty=float(repetition_penalty),
            top_k=model.config.top_k,
            top_p=model.config.top_p,
        )

    # API TTS
    elif params["tts_method_api_tts"]:
        # Streaming only allowed for XTTSv2 local
        if streaming:
            raise ValueError("Streaming is only supported in XTTSv2 local")

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
    print(f"[{params['branding']}TTSGen] \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{params['low_vram']} \033[94mDeepSpeed: \033[33m{params['deepspeed_activate']}\033[0m")
    # Move model back to cpu system ram if needed.
    if params["low_vram"] and device == "cuda" and tts_narrator_generatingtts == False:
        await switch_device()
    tts_generation_lock = False
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
        temperature = data["temperature"]
        repetition_penalty = data["repetition_penalty"]
        output_file = data["output_file"]
        streaming = False
        # Generation logic
        response = await generate_audio(text, voice, language, temperature, repetition_penalty, output_file, streaming)
        if streaming:
            return StreamingResponse(response, media_type="audio/wav")
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
templates = Jinja2Templates(directory=this_dir / "system")

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
        "/at_admin/at_settings.html",
        {
            "request": request,
            "data": get_json_data(),
            "modeldownload_model_path": modeldownload_model_path,
            "wav_files": wav_files,
        },
    )

# Define an endpoint to serve static files
app.mount("/static", StaticFiles(directory=str(this_dir / "system")), name="static")

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

@app.get("/tts-demo-request", response_class=StreamingResponse)
async def tts_demo_request_streaming(text: str, voice: str, language: str, output_file: str):
    try:
        output_file_path = this_dir / "outputs" / output_file
        stream = await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=True)
        return StreamingResponse(stream, media_type="audio/wav")
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

@app.post("/tts-demo-request", response_class=JSONResponse)
async def tts_demo_request(request: Request, text: str = Form(...), voice: str = Form(...), language: str = Form(...), output_file: str = Form(...)):
    try:
        output_file_path = this_dir / "outputs" / output_file
        await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=False)
        return JSONResponse(content={"output_file_path": str(output_file)}, status_code=200)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)


#####################
#### Audio feeds ####
#####################

# Gives web access to the output files
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = this_dir / "outputs" / filename
    return FileResponse(audio_path)

@app.get("/audiocache/{filename}")
async def get_audio(filename: str):
    audio_path = Path("outputs") / filename
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    response = FileResponse(
        path=audio_path,
        media_type='audio/wav',
        filename=filename
    )
    # Set caching headers
    response.headers["Cache-Control"] = "public, max-age=604800"  # Cache for one week
    response.headers["ETag"] = str(audio_path.stat().st_mtime)  # Use the file's last modified time as a simple ETag

    return response

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
        await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=False)

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
import hashlib

##############################
#### Streaming Generation ####
##############################

@app.get("/api/tts-generate-streaming", response_class=StreamingResponse)
async def tts_generate_streaming(text: str, voice: str, language: str, output_file: str):
    try:
        output_file_path = this_dir / "outputs" / output_file
        stream = await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=True)
        return StreamingResponse(stream, media_type="audio/wav")
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

@app.post("/api/tts-generate-streaming", response_class=JSONResponse)
async def tts_generate_streaming(request: Request, text: str = Form(...), voice: str = Form(...), language: str = Form(...), output_file: str = Form(...)):
    try:
        output_file_path = this_dir / "outputs" / output_file
        await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=False)
        return JSONResponse(content={"output_file_path": str(output_file)}, status_code=200)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

@app.put("/api/stop-generation")
async def stop_generation_endpoint():
    global tts_stop_generation, tts_generation_lock
    if tts_generation_lock and not tts_stop_generation:
        tts_stop_generation = True
    return {"message": "Generation stopped"}

##############################
#### Standard Generation ####
##############################

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

from typing import Union, Dict, List
from pydantic import BaseModel, ValidationError, Field

def play_audio(file_path, volume):
    data, fs = sf.read(file_path)
    sd.play(volume * data, fs)
    sd.wait()

class Request(BaseModel):
    # Define the structure of the 'Request' class if needed
    pass

class JSONInput(BaseModel):
    text_input: str = Field(..., max_length=2000, description="text_input needs to be 2000 characters or less.")
    text_filtering: str = Field(..., pattern="^(none|standard|html)$", description="text_filtering needs to be 'none', 'standard' or 'html'.")
    character_voice_gen: str = Field(..., pattern="^.*\.wav$", description="character_voice_gen needs to be the name of a wav file e.g. mysample.wav.")
    narrator_enabled: bool = Field(..., description="narrator_enabled needs to be true or false.")
    narrator_voice_gen: str = Field(..., pattern="^.*\.wav$", description="narrator_voice_gen needs to be the name of a wav file e.g. mysample.wav.")
    text_not_inside: str = Field(..., pattern="^(character|narrator)$", description="text_not_inside needs to be 'character' or 'narrator'.")
    language: str = Field(..., pattern="^(ar|zh-cn|cs|nl|en|fr|de|hu|hi|it|ja|ko|pl|pt|ru|es|tr)$", description="language needs to be one of the following ar|zh-cn|cs|nl|en|fr|de|hu|hi|it|ja|ko|pl|pt|ru|es|tr.")
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
        output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}_{timestamp}_combined.wav'
    else:
        output_file_path = os.path.join(this_dir / "outputs" / f'{output_file_name}_combined.wav')
        output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}_combined.wav'
        output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}_combined.wav'
    try:
        sf.write(output_file_path, audio, samplerate=sample_rate)
        # Clean up unnecessary files
        for audio_file in audio_files:
            os.remove(audio_file)
    except Exception as e:
        # Handle exceptions (e.g., failed to write output file)
        return None, None
    return output_file_path, output_file_url, output_cache_url

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
    streaming: bool = Form(False),
):
    try:
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
            "streaming": streaming,
        }
        global tts_narrator_generatingtts
        JSONresult = TTSGenerator.validate_json_input(json_input_data)
        if JSONresult is None:
            pass
        else:
            print(f"[{params['branding']}API] \033[91mError with API request:\033[0m", JSONresult)
            return JSONResponse(content={"error": JSONresult}, status_code=400)
        if narrator_enabled:
            if params["low_vram"] and device == "cpu":
                await switch_device()
            tts_narrator_generatingtts = True
            processed_parts = process_text(text_input)
            audio_files_all_paragraphs = []
            for part_type, part in processed_parts:
                # Skip parts that are too short
                if len(part.strip()) <= 3:
                    continue
                # Determine the voice to use based on the part type
                if part_type == 'narrator':
                    voice_to_use = narrator_voice_gen
                    print(f"[{params['branding']}TTSGen] \033[92mNarrator\033[0m")  # Green
                elif part_type == 'character':
                    voice_to_use = character_voice_gen
                    print(f"[{params['branding']}TTSGen] \033[36mCharacter\033[0m")  # Yellow
                else:
                    # Handle ambiguous parts based on user preference
                    voice_to_use = character_voice_gen if text_not_inside == "character" else narrator_voice_gen
                    voice_description = "\033[36mCharacter (Text-not-inside)\033[0m" if text_not_inside == "character" else "\033[92mNarrator (Text-not-inside)\033[0m"
                    print(f"[{params['branding']}TTSGen] {voice_description}")
                # Replace multiple exclamation marks, question marks, or other punctuation with a single instance
                cleaned_part = re.sub(r'([!?.\u3002\uFF1F\uFF01\uFF0C])\1+', r'\1', part)
                # Replace "Chinese ellipsis" with a single dot
                cleaned_part = re.sub(r"\u2026{1,2}", ". ", cleaned_part)
                # Further clean to remove any other unwanted characters
                cleaned_part = re.sub(r'[^a-zA-Z0-9\s.,;:!?\-\'"$\u0400-\u04FF\u00C0-\u00FF\u0150\u0151\u0170\u0171\u0900-\u097F\u2018\u2019\u201C\u201D\u3001\u3002\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\uFF01\uFF0c\uFF1A\uFF1B\uFF1F]', '', cleaned_part)
                # Remove all newline characters (single or multiple)
                cleaned_part = re.sub(r'\n+', ' ', cleaned_part)
                output_file = this_dir / "outputs" / f"{output_file_name}_{uuid.uuid4()}_{int(time.time())}.wav"
                output_file_str = output_file.as_posix()
                response = await generate_audio(cleaned_part, voice_to_use, language,temperature, repetition_penalty, output_file_str, streaming)
                audio_path = output_file_str
                audio_files_all_paragraphs.append(audio_path)
            # Combine audio files across paragraphs
            output_file_path, output_file_url, output_cache_url = combine(output_file_timestamp, output_file_name, audio_files_all_paragraphs)
            tts_narrator_generatingtts = False
            # Move model back to cpu system ram if needed.
            if params["low_vram"] and device == "cuda":
                await switch_device()
        else:
            if output_file_timestamp:
                timestamp = int(time.time())
                # Generate a standard UUID
                original_uuid = uuid.uuid4()
                # Hash the UUID using SHA-256
                hash_object = hashlib.sha256(str(original_uuid).encode())
                hashed_uuid = hash_object.hexdigest()
                # Truncate to the desired length, for example, 16 characters
                short_uuid = hashed_uuid[:5]
                output_file_path = this_dir / "outputs" / f"{output_file_name}_{timestamp}{short_uuid}.wav"
                output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}_{timestamp}{short_uuid}.wav'
                output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}_{timestamp}{short_uuid}.wav'
            else:
                output_file_path = this_dir / "outputs" / f"{output_file_name}.wav"
                output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}.wav'
                output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}.wav'
            if text_filtering == "html":
                cleaned_string = html.unescape(standard_filtering(text_input))
                cleaned_string = re.sub(r'([!?.\u3002\uFF1F\uFF01\uFF0C])\1+', r'\1', text_input)
                # Replace "Chinese ellipsis" with a single dot
                cleaned_string = re.sub(r"\u2026{1,2}", ". ", cleaned_string)
                # Further clean to remove any other unwanted characters
                cleaned_string = re.sub(r'[^a-zA-Z0-9\s.,;:!?\-\'"$\u0400-\u04FF\u00C0-\u00FF\u0150\u0151\u0170\u0171\u0900-\u097F\u2018\u2019\u201C\u201D\u3001\u3002\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\uFF01\uFF0c\uFF1A\uFF1B\uFF1F]', '', cleaned_string)
                # Remove all newline characters (single or multiple)
                cleaned_string = re.sub(r'\n+', ' ', cleaned_string)
            elif text_filtering == "standard":
                cleaned_string = re.sub(r'([!?.\u3002\uFF1F\uFF01\uFF0C])\1+', r'\1', text_input)
                # Replace "Chinese ellipsis" with a single dot
                cleaned_string = re.sub(r"\u2026{1,2}", ". ", cleaned_string)
                # Further clean to remove any other unwanted characters
                cleaned_string = re.sub(r'[^a-zA-Z0-9\s.,;:!?\-\'"$\u0400-\u04FF\u00C0-\u00FF\u0150\u0151\u0170\u0171\u0900-\u097F\u2018\u2019\u201C\u201D\u3001\u3002\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\uFF01\uFF0c\uFF1A\uFF1B\uFF1F]', '', cleaned_string)
                # Remove all newline characters (single or multiple)
                cleaned_string = re.sub(r'\n+', ' ', cleaned_string)
            else:
                cleaned_string = text_input
            response = await generate_audio(cleaned_string, character_voice_gen, language, temperature, repetition_penalty, output_file_path, streaming)
        if sounddevice_installed == False or streaming == True:
            autoplay = False
        if autoplay:
            play_audio(output_file_path, autoplay_volume)       
        if streaming:
            return StreamingResponse(response, media_type="audio/wav")
        return JSONResponse(content={"status": "generate-success", "output_file_path": str(output_file_path), "output_file_url": str(output_file_url), "output_cache_url": str(output_cache_url)}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "generate-failure", "error": "An error occurred"}, status_code=500)


##########################
#### Current Settings ####
##########################
# Define the available models
models_available = [
    {"name": "Coqui", "model_name": "API TTS"},
    {"name": "Coqui", "model_name": "API Local"},
    {"name": "Coqui", "model_name": "XTTSv2 Local"}
]

@app.get('/api/currentsettings')
def get_current_settings():
    # Determine the current model loaded
    if params["tts_method_api_tts"]:
        current_model_loaded = "API TTS"
    elif params["tts_method_api_local"]:
        current_model_loaded = "API Local"
    elif params["tts_method_xtts_local"]:
        current_model_loaded = "XTTSv2 Local"
    else:
        current_model_loaded = None  # or a default value if no method is active

    settings = {
        "models_available": models_available,
        "current_model_loaded": current_model_loaded,
        "deepspeed_available": deepspeed_available,
        "deepspeed_status": params["deepspeed_activate"],
        "low_vram_status": params["low_vram"],
        "finetuned_model": finetuned_model
    }
    return settings  # Automatically converted to JSON by Fas

#############################
#### Word Add-in Sharing ####
#############################
# Mount the static files from the 'word_addin' directory
app.mount("/api/word_addin", StaticFiles(directory=os.path.join(this_dir / 'system' / 'word_addin')), name="word_addin")

#############################################
#### TTS Generator Comparision Endpoints ####
#############################################
import subprocess
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
async def save_tts_data(tts_data: List[TTSItem]):
    # Convert the list of Pydantic models to a list of dictionaries
    tts_data_list = [item.dict() for item in tts_data]
    # Serialize the list of dictionaries to a JSON string
    tts_data_json = json.dumps(tts_data_list, indent=4)
    async with aiofiles.open(this_dir / "outputs" / "ttsList.json", 'w') as f:
        await f.write(tts_data_json)
    return {"message": "Data saved successfully"}

import sys

@app.get("/api/trigger-analysis")
async def trigger_analysis(threshold: int = Query(default=98)):
    venv_path = sys.prefix
    env = os.environ.copy()
    env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
    ttslist_path = this_dir / "outputs" / "ttsList.json"
    wavfile_path = this_dir / "outputs"
    subprocess.run(["python", "tts_diff.py", f"--threshold={threshold}", f"--ttslistpath={ttslist_path}", f"--wavfilespath={wavfile_path}"], cwd=this_dir / "system" / "tts_diff", env=env)
    # Read the analysis summary
    try:
        with open(this_dir / "outputs" / "analysis_summary.json", "r") as summary_file:
            summary_data = json.load(summary_file)
    except FileNotFoundError:
        summary_data = {"error": "Analysis summary file not found."}
    return {"message": "Analysis Completed", "summary": summary_data}

#################################################
#### TTS Generator SRT Subtitiles generation ####
#################################################
@app.get("/api/srt-generation")
async def srt_generation():
    venv_path = sys.prefix
    env = os.environ.copy()
    env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
    ttslist_path = this_dir / "outputs" / "ttsList.json"
    wavfile_path = this_dir / "outputs"
    subprocess.run(["python", "tts_srt.py", f"--ttslistpath={ttslist_path}", f"--wavfilespath={wavfile_path}"], cwd=this_dir / "system" / "tts_srt", env=env)
    
    srt_file_path = this_dir / "outputs" / "subtitles.srt"
    if not srt_file_path.exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found.")
    
    return FileResponse(path=srt_file_path, filename="subtitles.srt", media_type='application/octet-stream')

###################################################
#### Webserver Startup & Initial model Loading ####
###################################################

# Get the admin interface template
template = templates.get_template("admin.html")
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
host_parameter = params["ip_address"]
port_parameter = int(params["port_number"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host_parameter, port=port_parameter, log_level="warning")
