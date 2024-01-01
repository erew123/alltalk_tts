import html
import json
import random
import subprocess
import time
import os
import requests
import threading
import signal
import sys
import atexit
from pathlib import Path
from datetime import datetime, timedelta
import re
import numpy as np
import soundfile as sf
import uuid

#################################################################
#### LOAD PARAMS FROM confignew.json - REQUIRED FOR BRANDING ####
#################################################################
# STARTUP VARIABLE - Create "this_dir" variable as the current script directory
this_dir = Path(__file__).parent.resolve()

# load config file in and get settings
def load_config(file_path):
    with open(file_path, "r") as config_file:
        config = json.load(config_file)
    return config

config_file_path = this_dir / "confignew.json"
# Load the params dictionary from the confignew.json file
params = load_config(config_file_path)

# Required for sentence splitting
try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    logger.error(
        f"[{params['branding']}]\033[91mWarning\033[0m Could not find the TTS module. Make sure to install the requirements for the {params['branding']} extension."
        f"[{params['branding']}]\033[91mWarning\033[0m Linux / Mac:\npip install -r /alltalk_tts/requirements.txt\n"
        f"[{params['branding']}]\033[91mWarning\033[0m Windows:\npip install -r \\alltalk_tts\\requirements.txt\n"
        f"[{params['branding']}]\033[91mWarning\033[0m If you used the one-click installer, paste the command above in the terminal window launched after running the cmd_ script. On Windows, that's cmd_windows.bat."
    )
    raise

# IMPORT - Attempt Importing DeepSpeed (required for displaying Deepspeed checkbox in gradio)
try:
    import deepspeed
    deepspeed_installed = True
except ImportError:
    deepspeed_installed = False

# Import gradio if being used within text generation webUI
try:
    import gradio as gr

    from modules import chat, shared, ui_chat
    from modules.logging_colors import logger
    from modules.ui import create_refresh_button
    from modules.utils import gradio
    # This is set to check if the script is being run within text generation webui or as a standalone script. False is running as part of text gen web ui or a gradio interface
    running_in_standalone = False
    output_folder_wav = params["output_folder_wav"]
except ModuleNotFoundError:
    output_folder_wav = params["output_folder_wav_standalone"]
    print(f"[{params['branding']}Startup] Running script.py in standalone mode")
    # This is set to check if the script is being run within text generation webui or as a standalone script. true means standalone
    running_in_standalone = True

###########################
#### STARTUP VARIABLES ####
###########################
# STARTUP VARIABLE - Import languges file for Gradio to be able to display them in the interface
with open(this_dir / "languages.json", encoding="utf8") as f:
    languages = json.load(f)
# Create a global lock
process_lock = threading.Lock()
# Base setting for a possible FineTuned model existing and the loader being available
tts_method_xtts_ft = False

# Gather the voice files
def get_available_voices():
    return sorted([voice.name for voice in Path(f"{this_dir}/voices").glob("*.wav")])

#########################
#### LICENSE DISPLAY ####
#########################
# STARTUP Display Licence Information
print(f"[{params['branding']}Startup] \033[94mCoqui Public Model License\033[0m")
print(f"[{params['branding']}Startup] \033[94mhttps://coqui.ai/cpml.txt\033[0m")


############################################
#### DELETE OLD OUTPUT WAV FILES IF SET ####
############################################
def delete_old_files(folder_path, days_to_keep):
    current_time = datetime.now()
    print(
        f"[{params['branding']}Startup] Deletion of old output folder WAV files is currently enabled and set at",
        delete_output_wavs_setting,
    )
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            age = current_time - file_creation_time
            if age > timedelta(days=days_to_keep):
                os.remove(file_path)


# Extract settings using params dictionary
delete_output_wavs_setting = params["delete_output_wavs"]
output_folder_wav = os.path.normpath(output_folder_wav)

# Check and perform file deletion
if delete_output_wavs_setting.lower() == "disabled":
    print(
        "["
        + params["branding"]
        + "Startup] Old output wav file deletion is set to disabled."
    )
else:
    try:
        days_to_keep = int(delete_output_wavs_setting.split()[0])
        delete_old_files(output_folder_wav, days_to_keep)
    except ValueError:
        print(
            f"[{params['branding']}Startup] Invalid setting for deleting old wav files. Please use 'Disabled' or 'X Days' format."
        )


########################
#### STARTUP CHECKS ####
########################
# STARTUP Checks routine
def check_required_files():
    this_dir = Path(__file__).parent.resolve()
    download_script_path = this_dir / "modeldownload.py"
    subprocess.run(["python", str(download_script_path)])
    print(f"[{params['branding']}Startup] All required files are present.")

# STARTUP Call Check routine
check_required_files()

##################################################
#### Check to see if a finetuned model exists ####
##################################################
# Set the path to the directory
trained_model_directory = this_dir / "models" / "trainedmodel"
# Check if the directory "trainedmodel" exists
finetuned_model = trained_model_directory.exists()
# If the directory exists, check for the existence of the required files
# If true, this will add a extra option in the Gradio interface for loading Xttsv2 FT
if finetuned_model:
    required_files = ["model.pth", "config.json", "vocab.json"]
    finetuned_model = all((trained_model_directory / file).exists() for file in required_files)
if finetuned_model:
    print(f"[{params['branding']}Startup] Finetuned model \033[93mDetected\033[0m")

####################################################
#### SET GRADIO BUTTONS BASED ON confignew.json ####
####################################################

if params["tts_method_api_tts"] == True:
    gr_modelchoice = "API TTS"
elif params["tts_method_api_local"] == True:
    gr_modelchoice = "API Local"
elif params["tts_method_xtts_local"] == True:
    gr_modelchoice = "XTTSv2 Local"

# Set the default for Narrated text without asterisk or quotes to be Narrator
non_quoted_text_is = True

######################
#### GRADIO STUFF ####
######################
def remove_tts_from_history(history):
    for i, entry in enumerate(history["internal"]):
        history["visible"][i] = [history["visible"][i][0], entry[1]]
    return history


def toggle_text_in_history(history):
    for i, entry in enumerate(history["visible"]):
        visible_reply = entry[1]
        if visible_reply.startswith("<audio"):
            if params["show_text"]:
                reply = history["internal"][i][1]
                history["visible"][i] = [
                    history["visible"][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}",
                ]
            else:
                history["visible"][i] = [
                    history["visible"][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>",
                ]
    return history


def history_modifier(history):
    # Remove autoplay from the last reply
    if len(history["internal"]) > 0:
        history["visible"][-1] = [
            history["visible"][-1][0],
            history["visible"][-1][1].replace("controls autoplay>", "controls>"),
        ]

    return history


######################################
#### SUBPROCESS/WEBSERVER STARTUP ####
######################################
base_url = f"http://{params['ip_address']}:{params['port_number']}"
script_path = this_dir / "tts_server.py"


def signal_handler(sig, frame):
    print(
        f"[{params['branding']}Shutdown] \033[94mReceived Ctrl+C, terminating subprocess\033[92m"
    )
    if process.poll() is None:
        process.terminate()
        process.wait()  # Wait for the subprocess to finish
    sys.exit(0)


# Attach the signal handler to the SIGINT signal (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Start the subprocess
process = subprocess.Popen(["python", script_path])

# Check if the subprocess has started successfully
if process.poll() is None:
    print(f"[{params['branding']}Startup] TTS Subprocess starting")
    print(
        f"[{params['branding']}Startup] Readme available here:",
        f"http://{params['ip_address']}:{params['port_number']}",
    )
else:
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m TTS Subprocess Webserver failing to start process"
    )
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m It could be that you have something on port:",
        params["port_number"],
    )
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Or you have not started in a Python environement with all the necesssary bits installed"
    )
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Check you are starting Text-generation-webui with either the start_xxxxx file or the Python environment with cmd_xxxxx file."
    )
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m xxxxx is the type of OS you are on e.g. windows, linux or mac."
    )
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Alternatively, you could check no other Python processes are running that shouldnt be e.g. Restart your computer is the simple way."
    )
    # Cleanly kill off this script, but allow text-generation-webui to keep running, albeit without this alltalk_tts
    sys.exit(1)

timeout = 120  # Adjust the timeout as needed

# Introduce a delay before starting the check loop
time.sleep(26)  # Wait 26 secs before checking if the tts_server.py has started up.
start_time = time.time()
while time.time() - start_time < timeout:
    try:
        response = requests.get(f"{base_url}/ready")
        if response.status_code == 200:
            break
    except requests.RequestException as e:
        # Print the exception for debugging purposes
        print(
            f"[{params['branding']}Startup] \033[91mWarning\033[0m TTS Subprocess has NOT started up yet, Will keep trying for 120 seconds maximum. Please wait."
        )
    time.sleep(4)
else:
    print(
        f"[{params['branding']}Startup] Startup timed out. Check the server logs for more information."
    )
    # Cleanly kill off this script, but allow text-generation-webui to keep running, albeit without this alltalk_tts
    sys.exit(1)


#####################################
#### MODEL LOADING AND UNLOADING ####
#####################################
# MODEL - Swap model based on Gradio selection API TTS, API Local, XTTSv2 Local
def send_reload_request(tts_method):
    global tts_method_xtts_ft
    try:
        params["tts_model_loaded"] = False
        url = f"{base_url}/api/reload"
        payload = {"tts_method": tts_method}
        response = requests.post(url, params=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        json_response = response.json()
        # Check if the reload operation was successful
        if json_response.get("status") == "model-success":
            # Update tts_tts_model_loaded to True if the reload was successful
            params["tts_model_loaded"] = True
            # Update local script parameters based on the tts_method
            if tts_method == "API TTS":
                params["tts_method_api_local"] = False
                params["tts_method_xtts_local"] = False
                params["tts_method_api_tts"] = True
                params["deepspeed_activate"] = False
                audio_path = this_dir / "templates" / "apitts.wav"
                tts_method_xtts_ft = False
            elif tts_method == "API Local":
                params["tts_method_api_tts"] = False
                params["tts_method_xtts_local"] = False
                params["tts_method_api_local"] = True
                params["deepspeed_activate"] = False
                audio_path = this_dir / "templates" / "apilocal.wav"
                tts_method_xtts_ft = False
            elif tts_method == "XTTSv2 Local":
                params["tts_method_api_tts"] = False
                params["tts_method_api_local"] = False
                params["tts_method_xtts_local"] = True
                audio_path = this_dir / "templates" / "xttslocal.wav"
                tts_method_xtts_ft = False
            elif tts_method == "XTTSv2 FT":
                params["tts_method_api_tts"] = False
                params["tts_method_api_local"] = False
                params["tts_method_xtts_local"] = False
                audio_path = this_dir / "templates" / "xttsfinetuned.wav"
                tts_method_xtts_ft = True
        return f'<audio src="file/{audio_path}" controls autoplay></audio>'
    except requests.exceptions.RequestException as e:
        # Handle the HTTP request error
        print(
            f"[{params['branding']}Server] \033[91mWarning\033[0m Error during request to webserver process: {e}"
        )
        return {"status": "error", "message": str(e)}

##################
#### LOW VRAM ####
##################
# LOW VRAM - Gradio Checkbox handling
def send_lowvram_request(low_vram):
    try:
        params["tts_model_loaded"] = False
        if low_vram:
            audio_path = this_dir / "templates" / "lowvramenabled.wav"
        else:
            audio_path = this_dir / "templates" / "lowvramdisabled.wav"
        url = f"{base_url}/api/lowvramsetting?new_low_vram_value={low_vram}"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        json_response = response.json()
        # Check if the low VRAM request was successful
        if json_response.get("status") == "lowvram-success":
            # Update any relevant variables or perform other actions on success
            params["tts_model_loaded"] = True
        return f'<audio src="file/{audio_path}" controls autoplay></audio>'
    except requests.exceptions.RequestException as e:
        # Handle the HTTP request error
        print(
            f"[{params['branding']}Server] \033[91mWarning\033[0m Error during request to webserver process: {e}"
        )
        return {"status": "error", "message": str(e)}


###################
#### DeepSpeed ####
###################
# DEEPSPEED - Reload the model when DeepSpeed checkbox is enabled/disabled
def send_deepspeed_request(deepspeed_param):
    try:
        params["tts_model_loaded"] = False
        if deepspeed_param:
            audio_path = this_dir / "templates" / "deepspeedenabled.wav"
        else:
            audio_path = this_dir / "templates" / "deepspeeddisabled.wav"
        
        url = f"{base_url}/api/deepspeed?new_deepspeed_value={deepspeed_param}"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        json_response = response.json()
        # Check if the deepspeed request was successful
        if json_response.get("status") == "deepspeed-success":
            # Update any relevant variables or perform other actions on success
            params["tts_model_loaded"] = True
        return f'<audio src="file/{audio_path}" controls autoplay></audio>'
    except requests.exceptions.RequestException as e:
        # Handle the HTTP request error
        print(
            f"[{params['branding']}Server] \033[91mWarning\033[0m Error during request to webserver process: {e}"
        )
        return {"status": "error", "message": str(e)}


# DEEPSPEED - Display DeepSpeed Checkbox Yes or No
deepspeed_condition = params["tts_method_xtts_local"] == "True" and deepspeed_installed


#############################################################
#### TTS STRING CLEANING & PROCESSING PRE SENDING TO TTS ####
#############################################################
def new_split_into_sentences(self, text):
    sentences = self.seg.segment(text)
    if params["remove_trailing_dots"]:
        sentences_without_dots = []
        for sentence in sentences:
            if sentence.endswith(".") and not sentence.endswith("..."):
                sentence = sentence[:-1]

            sentences_without_dots.append(sentence)

        return sentences_without_dots
    else:
        return sentences


Synthesizer.split_into_sentences = new_split_into_sentences


# Check model is loaded and string isnt empty, before sending a TTS request.
def before_audio_generation(string, params):
    # Check Model is loaded into cuda or cpu and error if not
    if not params["tts_model_loaded"]:
        print(
            f"[{params['branding']}Model] \033[91mWarning\033[0m Model is still loading, please wait before trying to generate TTS"
        )
        return
    string = html.unescape(string) or random_sentence()
    if string == "":
        return "*Empty string*"
    return string


##################
#### Narrator ####
##################
def combine(audio_files, output_folder, state):
    audio = np.array([])

    for audio_file in audio_files:
        audio_data, sample_rate = sf.read(audio_file)
        # Ensure all audio files have the same sample rate
        if audio.size == 0:
            audio = audio_data
        else:
            audio = np.concatenate((audio, audio_data))

    # Save the combined audio to a file with a specified sample rate
    if "character_menu" in state:
        output_file_path = os.path.join(output_folder, f'{state["character_menu"]}_{int(time.time())}_combined.wav')
    else:
        output_file_path = os.path.join(output_folder, f'TTSOUT_{int(time.time())}_combined.wav')
    sf.write(output_file_path, audio, samplerate=sample_rate)
    # Clean up unnecessary files
    for audio_file in audio_files:
        os.remove(audio_file)

    return output_file_path


################################
#### TTS PREVIEW GENERATION ####
################################
# PREVIEW VOICE - Generate Random Sentence if Voice Preview box is empty
def random_sentence():
    with open(this_dir / "harvard_sentences.txt") as f:
        return random.choice(list(f))


# PREVIEW VOICE- Generate TTS Function
def voice_preview(string):
    if not params["activate"]:
        return string
    # Clean the string, capture model not loaded, and move model to cuda if needed
    cleaned_string = before_audio_generation(string, params)
    if cleaned_string is None:
        return
    string = cleaned_string
    # Setup the output file
    output_file = Path(params["output_folder_wav"]) / "voice_preview.wav"
    # Generate the audio
    language_code = languages.get(params["language"])
    # Convert the WindowsPath object to a string before using it in JSON payload
    output_file_str = output_file.as_posix()
    # Lock before making the generate request
    with process_lock:
        generate_response = send_generate_request(
            string, params["voice"], language_code, output_file_str
        )
    # Check if lock is already acquired
    if process_lock.locked():
        print(
            f"[{params['branding']}Model] \033[91mWarning\033[0m Audio generation is already in progress. Please wait."
        )
        return
    if generate_response.get("status") == "generate-success":
        # Handle Gradio and playback
        autoplay = "autoplay" if params["autoplay"] else ""
        return f'<audio src="file/{output_file_str}?{int(time.time())}" controls {autoplay}></audio>'
    else:
        # Handle the case where audio generation was not successful
        return f"[{params['branding']}Server] Audio generation failed. Status: {generate_response.get('status')}"


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

#################################
#### TTS STANDARD GENERATION ####
#################################
# STANDARD VOICE - Generate TTS Function
def output_modifier(string, state):
    if not params["activate"]:
        return string
    original_string = string
    #print("ORIGINAL STRING IS:", original_string)
    cleaned_string = before_audio_generation(string, params)
    if cleaned_string is None:
        return
    string = cleaned_string
    language_code = languages.get(params["language"])
    # Create a list to store generated audio paths
    audio_files = []
    if process_lock.acquire(blocking=False):
        try:
            if params["narrator_enabled"]:
                processed_parts = process_text(original_string)

                audio_files_all_paragraphs = []
                for part_type, part in processed_parts:
                    # Skip parts that are too short
                    if len(part.strip()) <= 3:
                        continue

                    # Determine the voice to use based on the part type
                    if part_type == 'narrator':
                        voice_to_use = params["narrator_voice"]
                        print(f"[{params['branding']}TTSGen] \033[92mNarrator\033[0m")  # Green
                    elif part_type == 'character':
                        voice_to_use = params["voice"]
                        print(f"[{params['branding']}TTSGen] \033[36mCharacter\033[0m")  # Yellow
                    else:
                        # Handle ambiguous parts based on user preference
                        voice_to_use = params["voice"] if non_quoted_text_is else params["narrator_voice"]
                        voice_description = "\033[36mCharacter (Text-not-inside)\033[0m" if non_quoted_text_is else "\033[92mNarrator (Text-not-inside)\033[0m"
                        print(f"[{params['branding']}TTSGen] {voice_description}")

                    # Replace multiple exclamation marks, question marks, or other punctuation with a single instance
                    cleaned_part = re.sub(r'([!?.])\1+', r'\1', part)
                    # Further clean to remove any other unwanted characters
                    cleaned_part = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\-\'"]', '', cleaned_part)
                    # Remove all newline characters (single or multiple)
                    cleaned_part = re.sub(r'\n+', ' ', cleaned_part)

                    # Generate TTS and output to a file
                    output_filename = get_output_filename(state)
                    generate_response = send_generate_request(
                        cleaned_part, voice_to_use, language_code, output_filename
                    )
                    audio_path = generate_response.get("data", {}).get("audio_path")
                    audio_files_all_paragraphs.append(audio_path)

                # Combine audio files across paragraphs
                final_output_file = combine(
                    audio_files_all_paragraphs, params["output_folder_wav"], state
                )
            else:
                # Decode HTML entities first
                cleaned_part = html.unescape(original_string)
                # Replace multiple instances of certain punctuation marks with a single instance
                cleaned_part = re.sub(r'([!?.])\1+', r'\1', cleaned_part)
                # Further clean to remove any other unwanted characters
                cleaned_part = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\-\'"]', '', cleaned_part)
                # Remove all newline characters (single or multiple)
                cleaned_part = re.sub(r'\n+', ' ', cleaned_part)
                # Process the part and give it a non-character name if being used vai API or standalone.
                if "character_menu" in state:
                    output_file = Path(f'{params["output_folder_wav"]}/{state["character_menu"]}_{int(time.time())}.wav')
                else:
                    output_file = Path(f'{params["output_folder_wav"]}/TTSOUT_{int(time.time())}.wav')
                output_file_str = output_file.as_posix()
                output_file = get_output_filename(state)
                generate_response = send_generate_request(
                    cleaned_part, params["voice"], language_code, output_file_str
                )
                audio_path = generate_response.get("data", {}).get("audio_path")
                final_output_file = audio_path
        finally:
            # Always release the lock, whether an exception occurs or not
            process_lock.release()
    else:
        # The lock is already acquired
        print(
            f"[{params['branding']}Model] \033[91mWarning\033[0m Audio generation is already in progress. Please wait."
        )
        return

    if generate_response.get("status") == "generate-success":
        audio_path = generate_response.get("data", {}).get("audio_path")
        if audio_path:
            # Handle Gradio and playback
            autoplay = "autoplay" if params["autoplay"] else ""
            string = (
                f'<audio src="file/{final_output_file}" controls {autoplay}></audio>'
            )

            if params["show_text"]:
                string += f"\n\n{original_string}"
                shared.processing_message = "*Is typing...*"

            return string
        else:
            print(
                f"[{params['branding']}Server] \033[91mWarning\033[0m No audio path in the response."
            )
    else:
        print(
            f"[{params['branding']}Server] \033[91mWarning\033[0m Audio generation failed. Status:",
            generate_response.get("message"),
        )


def get_output_filename(state):
    if "character_menu" in state:
        return Path(f'{params["output_folder_wav"]}/{state["character_menu"]}_{str(uuid.uuid4())[:8]}.wav').as_posix()
    else:
        return Path(f'{params["output_folder_wav"]}/TTSOUT_{str(uuid.uuid4())[:8]}.wav').as_posix()


###############################################
#### SEND GENERATION REQUEST TO TTS ENGINE ####
###############################################
def send_generate_request(text, voice, language, output_file):
    url = f"{base_url}/api/generate"
    payload = {
        "text": text,
        "voice": voice,
        "language": language,
        "output_file": output_file,
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()


################################
#### SUBPORCESS TERMINATION ####
################################
# Register the termination code to be executed at exit
atexit.register(lambda: process.terminate() if process.poll() is None else None)


######################
#### GRADIO STUFF ####
######################
def state_modifier(state):
    if not params["activate"]:
        return state

    state["stream"] = False
    return state

def update_narrator_enabled(value):
    if value == "Enabled":
        params["narrator_enabled"] = True
    elif value == "Disabled":
        params["narrator_enabled"] = False

def update_non_quoted_text_is(value):
    global non_quoted_text_is
    if value == "Narrator":
        non_quoted_text_is = False
    elif value == "Char":
        non_quoted_text_is = True

def input_modifier(string, state):
    if not params["activate"]:
        return string

    shared.processing_message = "*Is recording a voice message...*"
    return string

def custom_css():
    path_to_css = Path(f"{this_dir}/style.css")
    return open(path_to_css, "r").read()


def ui():
    with gr.Accordion(params["branding"] + " TTS (XTTSv2)"):
        with gr.Row():
            activate = gr.Checkbox(value=params["activate"], label="Activate TTS")
            autoplay = gr.Checkbox(
                value=params["autoplay"], label="Play TTS automatically"
            )

        with gr.Row():
            show_text = gr.Checkbox(
                value=params["show_text"], label="Show message text under audio player"
            )
            remove_trailing_dots = gr.Checkbox(
                value=params["remove_trailing_dots"],
                label='Remove trailing "." from text segments before generation',
            )

        with gr.Row():
            with gr.Row():
                available_voices = get_available_voices()
                default_voice = params["voice"]
                # Check if the default voice is in the list of available voices
                if default_voice not in available_voices:
                    # Handle the case where the default voice is not in the list (choose a default value or update it)
                    default_voice = available_voices[
                        0
                    ]  # Choose the first available voice as the default
                # Add allow_custom_value=True to the Dropdown
                voice = gr.Dropdown(
                    available_voices,
                    label="Default Voice",
                    value=default_voice,
                    allow_custom_value=True,
                )
                create_refresh_button(
                    voice,
                    lambda: None,
                    lambda: {
                        "choices": get_available_voices(),
                        "value": params["voice"],
                    },
                    "refresh-button",
                )

            language = gr.Dropdown(
                languages.keys(),
                label="Language",
                allow_custom_value=True,
                value=params["language"],
            )

        with gr.Row():
            with gr.Row():
                narrator_voice_gr = gr.Dropdown(
                    get_available_voices(),
                    label="Narrator Voice",
                    allow_custom_value=True,
                    value=params["narrator_voice"],
                )
                create_refresh_button(
                    narrator_voice_gr,
                    lambda: None,
                    lambda: {
                        "choices": get_available_voices(),
                        "value": params["narrator_voice"],
                    },
                    "refresh-button",
                )
                narrator_enabled_gr = gr.Radio(
                    choices={"Enabled": "true", "Disabled": "false"},
                    label="Narrator Activation",
                    value="Enabled" if params.get("narrator_enabled") else "Disabled",
                )
                non_quoted_text_is_gr = gr.Radio(
                    choices={"Char": "true", "Narrator": "false"},
                    label='Text NOT inside of * or " is',
                    value="Char" if non_quoted_text_is else "Narrator",
                )

        with gr.Row():
            low_vram = gr.Checkbox(
                value=params["low_vram"], label="Low VRAM mode (Read NOTE)"
            )
            low_vram_play = gr.HTML(visible=False)
            deepspeed_checkbox = gr.Checkbox(
                value=params["deepspeed_activate"],
                label="Activate DeepSpeed (Read NOTE)",
                visible=deepspeed_installed,
            )
            deepspeed_checkbox_play = gr.HTML(visible=False)

        with gr.Row():
            model_loader_choices=["API TTS", "API Local", "XTTSv2 Local"]
            if finetuned_model:
                model_loader_choices.append("XTTSv2 FT")
            tts_radio_buttons = gr.Radio(
                choices=model_loader_choices,
                label="Select TTS Generation Method (Read NOTE)",
                value=gr_modelchoice,  # Set the default value
            )
            tts_radio_buttons_play = gr.HTML(visible=False)
            explanation_text = gr.HTML(
                f"<p>⚙️ <a href='http://{params['ip_address']}:{params['port_number']}'>Click here for the Settings page, help & instructions</a><a href='http://{params['ip_address']}:{params['port_number']}'></a>⚙️<br>NOTE: Switching Model Type, Low VRAM & DeepSpeed, each takes 15 seconds. Each TTS generation method has a slightly different sound. DeepSpeed checkbox is only visible if DeepSpeed is installed on your system."
            )

        with gr.Row():
            preview_text = gr.Text(
                show_label=False,
                placeholder="Preview text",
                elem_id="silero_preview_text",
            )
            preview_play = gr.Button("Preview")
            preview_audio = gr.HTML(visible=False)

        with gr.Row():
            convert = gr.Button("Permanently replace audios with the message texts")
            convert_cancel = gr.Button("Cancel", visible=False)
            convert_confirm = gr.Button(
                "Confirm (cannot be undone)", variant="stop", visible=False
            )

    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(
        lambda: [
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        ],
        None,
        convert_arr,
    )
    convert_confirm.click(
        lambda: [
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        ],
        None,
        convert_arr,
    ).then(remove_tts_from_history, gradio("history"), gradio("history")).then(
        chat.save_history,
        gradio("history", "unique_id", "character_menu", "mode"),
        None,
    ).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio("display")
    )

    convert_cancel.click(
        lambda: [
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        ],
        None,
        convert_arr,
    )

    # Toggle message text in history
    show_text.change(lambda x: params.update({"show_text": x}), show_text, None).then(
        toggle_text_in_history, gradio("history"), gradio("history")
    ).then(
        chat.save_history,
        gradio("history", "unique_id", "character_menu", "mode"),
        None,
    ).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio("display")
    )

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    low_vram.change(lambda x: params.update({"low_vram": x}), low_vram, None)
    low_vram.change(lambda x: send_lowvram_request(x), low_vram, low_vram_play, None)
    tts_radio_buttons.change(send_reload_request, tts_radio_buttons, tts_radio_buttons_play, None)
    deepspeed_checkbox.change(send_deepspeed_request, deepspeed_checkbox, deepspeed_checkbox_play, None)
    remove_trailing_dots.change(
        lambda x: params.update({"remove_trailing_dots": x}), remove_trailing_dots, None
    )
    voice.change(lambda x: params.update({"voice": x}), voice, None)
    language.change(lambda x: params.update({"language": x}), language, None)

    # Narrator selection actions
    narrator_enabled_gr.change(update_narrator_enabled, narrator_enabled_gr, None)
    non_quoted_text_is_gr.change(update_non_quoted_text_is, non_quoted_text_is_gr, None)
    narrator_voice_gr.change(lambda x: params.update({"narrator_voice": x}), narrator_voice_gr, None)

    # Play preview
    preview_text.submit(voice_preview, preview_text, preview_audio)
    preview_play.click(voice_preview, preview_text, preview_audio)


##########################################
#### STANDALONE MODE LOOP TERMINATION ####
##########################################
## Loop to keep the script from exiting out if its being run as a standalone script and not part of text-generation-webui
if running_in_standalone:
    while True:
        try:
            time.sleep(1)  # Add a small delay to avoid high CPU usage
        except KeyboardInterrupt:
            break  # Allow graceful exit on Ctrl+C
