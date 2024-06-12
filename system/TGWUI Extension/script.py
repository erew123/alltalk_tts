import re
import json
import time
import random
import logging
import requests
import threading
import gradio as gr
from pathlib import Path
from modules import chat, shared, ui_chat
from modules.utils import gradio
from requests.exceptions import RequestException, ConnectionError
# Store the current disable level
current_disable_level = logging.getLogger().manager.disable

####################################################################
# Load in the confignew.json and put the main settings into params #
####################################################################
this_dir = Path(__file__).parent.resolve()
def load_config(file_path):
    with open(file_path, "r") as config_file:
        config = json.load(config_file)
    return config
# Set confiig file path       
config_file_path = this_dir / "tgwui_remote_config.json"
# Load the params dictionary from the tgwui_remote_config.json file
params = load_config(config_file_path)
# Set branding
branding = params['branding']

################################
# Print Spashscreen to console #
################################
print(f"[{branding}Startup]\033[94m     _    _ _ \033[1;35m_____     _ _     \033[0m  _____ _____ ____  ")
print(f"[{branding}Startup]\033[94m    / \  | | |\033[1;35m_   _|_ _| | | __ \033[0m |_   _|_   _/ ___| ")
print(f"[{branding}Startup]\033[94m   / _ \ | | |\033[1;35m | |/ _` | | |/ / \033[0m   | |   | | \___ \ ")
print(f"[{branding}Startup]\033[94m  / ___ \| | |\033[1;35m | | (_| | |   <  \033[0m   | |   | |  ___) |")
print(f"[{branding}Startup]\033[94m /_/   \_\_|_|\033[1;35m |_|\__,_|_|_|\_\ \033[0m   |_|   |_| |____/ ")
print(f"[{branding}Startup]")
print(f"[{branding}Startup] \033[92m{branding}startup Mode   : \033[93mText-Gen-webui Remote\033[0m")
print(f"[{branding}Startup]")
           
##########################
# Setup global variables #
##########################
# STARTUP VARIABLE - Import languges file for Gradio to be able to display them in the interface
with open(this_dir / "languages.json", encoding="utf8") as f:
    languages = json.load(f)
# Create a global lock for tracking TTS generation occuring
process_lock = threading.Lock()
# Pull the values for IP address, port and protocol for communication with the AllTalk renote server
alltalk_protocol = params["api_def"]["api_alltalk_protocol"]
alltalk_ip_port = params["api_def"]["api_alltalk_ip_port"]
# Pull the connection timeout value for communication requests with the AllTalk remote server
connection_timeout = params["api_def"]["api_connection_timeout"]
# Create a few base global variables that are required
models_available = None         # Gets populated with the list of all models/engines available on the AllTalk Server
alltalk_settings = None         # Gets populated with the list of all settings currently set on the AllTalk Server
current_model_loaded = None     # Gets populated with the name of the current loaded TTS engine/model on the AllTalk Server
# Used to detect if a model is loaded in to AllTalk server to block TTS genereation if needed.
tts_model_loaded = None

#################################################
# Pull all the settings from the AllTalk Server #
#################################################
def get_alltalk_settings():
    global current_model_loaded, models_available
    voices_url = f"{alltalk_protocol}{alltalk_ip_port}/api/voices"
    settings_url = f"{alltalk_protocol}{alltalk_ip_port}/api/currentsettings"
    rvcvoices_url = f"{alltalk_protocol}{alltalk_ip_port}/api/rvcvoices"

    try:
        voices_response = requests.get(voices_url, timeout=connection_timeout)
        rvcvoices_response = requests.get(rvcvoices_url, timeout=connection_timeout)
        settings_response = requests.get(settings_url, timeout=connection_timeout)

        if voices_response.status_code == 200 and settings_response.status_code == 200:
            voices_data = voices_response.json()
            rvcvoices_data = rvcvoices_response.json() 
            settings_data = settings_response.json()

            models_available = [model["name"] for model in settings_data["models_available"]]
            current_model_loaded = settings_data["current_model_loaded"]

            return {
                "voices": sorted(voices_data["voices"]),
                "rvcvoices": (rvcvoices_data["rvcvoices"]),
                "models_available": models_available,
                "current_model_loaded": current_model_loaded,
                "manufacturer_name": settings_data.get("manufacturer_name", ""),
                "deepspeed_capable": settings_data.get("deepspeed_capable", False),
                "deepspeed_available": settings_data.get("deepspeed_available", False),
                "deepspeed_enabled": settings_data.get("deepspeed_enabled", False),
                "generationspeed_capable": settings_data.get("generationspeed_capable", False),
                "generationspeed_set": settings_data.get("generationspeed_set", 1.0),
                "lowvram_capable": settings_data.get("lowvram_capable", False),
                "lowvram_enabled": settings_data.get("lowvram_enabled", False),
                "pitch_capable": settings_data.get("pitch_capable", False),
                "pitch_set": settings_data.get("pitch_enabled", False),
                "repetitionpenalty_capable": settings_data.get("repetitionpenalty_capable", False),
                "repetitionpenalty_set": settings_data.get("repetitionpenalty_set", 10.0),
                "streaming_capable": settings_data.get("streaming_capable", False),
                "temperature_capable": settings_data.get("temperature_capable", False),
                "temperature_set": settings_data.get("temperature_set", 0.75),
                "ttsengines_installed": settings_data.get("ttsengines_installed", False),
                "languages_capable": settings_data.get("languages_capable", False),
                "multivoice_capable": settings_data.get("multivoice_capable", False),
                "multimodel_capable": settings_data.get("multimodel_capable", False),
            }
        else:
            print(f"[{branding}Server] \033[91mWarning\033[0m Failed to retrieve {branding}settings from API.")
            if voices_response.status_code != 200:
                print(f"[{branding}Server] \033[91mWarning\033[0m Failed to retrieve voices from API. Status code:\n{voices_response.status_code}")
            if settings_response.status_code != 200:
                print(f"[{branding}Server] \033[91mWarning\033[0m Failed to retrieve current settings from API. Status code:\n{settings_response.status_code}")
            return {
                "voices": ["Please Refresh Settings"],
                "rvcvoices": ["Please Refresh Settings"],
                "models_available": ["Please Refresh Settings"],
                "current_model_loaded": "Please Refresh Settings",
                "manufacturer_name": "",
                "deepspeed_capable": False,
                "deepspeed_available": False,
                "deepspeed_enabled": False,
                "generationspeed_capable": False,
                "generationspeed_set": 1.0,
                "lowvram_capable": False,
                "lowvram_enabled": False,
                "pitch_capable": False,
                "pitch_set": 0,
                "repetitionpenalty_capable": False,
                "repetitionpenalty_set": 10.0,
                "streaming_capable": False,
                "temperature_capable": False,
                "temperature_set": 0.75,
                "ttsengines_installed": False,
                "languages_capable": False,
                "multivoice_capable": False,
                "multimodel_capable": False,
            }
    except (RequestException, ConnectionError) as e:
        print(f"[{branding}Server] \033[91mWarning\033[0m Unable to connect to the {branding}server. Status code:\n{str(e)}")
        return {
            "voices": ["Please Refresh Settings"],
            "rvcvoices": ["Please Refresh Settings"],            
            "models_available": ["Please Refresh Settings"],
            "current_model_loaded": "Please Refresh Settings",
            "manufacturer_name": "",
            "deepspeed_capable": False,
            "deepspeed_available": False,
            "deepspeed_enabled": False,
            "generationspeed_capable": False,
            "generationspeed_set": 1.0,
            "lowvram_capable": False,
            "lowvram_enabled": False,
            "pitch_capable": False,
            "pitch_set": 0,
            "repetitionpenalty_capable": False,
            "repetitionpenalty_set": 10.0,
            "streaming_capable": False,
            "temperature_capable": False,
            "temperature_set": 0.75,
            "ttsengines_installed": False,
            "languages_capable": False,
            "multivoice_capable": False,
            "multimodel_capable": False,
        }

#############################
#### TTS STOP GENERATION ####
#############################
def stop_generate_tts():
    api_url = f"{alltalk_protocol}{alltalk_ip_port}/api/stop-generation"
    try:
        response = requests.put(api_url, timeout=connection_timeout)  
        if response.status_code == 200:
            return response.json()["message"]
        else:
            print(f"[{branding}Server] \033[91mWarning\033[0m Failed to stop generation. Status code:\n{response.status_code}")
            return {"message": "Failed to stop generation"}
    except (RequestException, ConnectionError) as e:
        print(f"[{branding}Server] \033[91mWarning\033[0m Unable to connect to the {branding}server. Status code:\n{str(e)}")
        return {"message": "Failed to stop generation"}

#####################################
#### MODEL LOADING AND UNLOADING ####
#####################################
# MODEL - Swap model based on Gradio selection
def send_reload_request(value_sent):
    try:
        url = f"{alltalk_protocol}{alltalk_ip_port}/api/reload"
        payload = {"tts_method": value_sent}
        response = requests.post(url, params=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        # Handle the HTTP request error
        print(f"[{branding}Server] \033[91mWarning\033[0m Error during request to webserver process: Status code:\n{e}")
        return {"status": "error", "message": str(e)}

####################################################################################################
# TGWUI # Saves all settings back to the config file for the REMOTE version of the TGWUI extension #
####################################################################################################
def tgwui_save_settings():
    settings = {
        "branding": params['branding'],
        "tgwui": {
            "tgwui_activate_tts": params["tgwui"]['tgwui_activate_tts'],
            "tgwui_autoplay_tts": params["tgwui"]['tgwui_autoplay_tts'],
            "tgwui_narrator_enabled": params["tgwui"]['tgwui_narrator_enabled'],
            "tgwui_non_quoted_text_is": params["tgwui"]["tgwui_non_quoted_text_is"],
            "tgwui_deepspeed_enabled": params["tgwui"]['tgwui_deepspeed_enabled'],
            "tgwui_language": params["tgwui"]['tgwui_language'],
            "tgwui_lowvram_enabled": params["tgwui"]['tgwui_lowvram_enabled'],
            "tgwui_pitch_set": params["tgwui"]['tgwui_pitch_set'],
            "tgwui_temperature_set": params["tgwui"]['tgwui_temperature_set'],
            "tgwui_repetitionpenalty_set": params["tgwui"]['tgwui_repetitionpenalty_set'],
            "tgwui_generationspeed_set": params["tgwui"]['tgwui_generationspeed_set'],
            "tgwui_narrator_voice": params["tgwui"]['tgwui_narrator_voice'],
            "tgwui_show_text": params["tgwui"]['tgwui_show_text'],
            "tgwui_character_voice": params["tgwui"]['tgwui_character_voice'],
            "tgwui_rvc_char_voice": params["tgwui"]['tgwui_rvc_char_voice'],
            "tgwui_rvc_narr_voice": params["tgwui"]['tgwui_rvc_narr_voice']
        },
        "api_def": {
            "api_use_legacy_api": params['api_def']['api_use_legacy_api'],
            "api_alltalk_protocol": alltalk_protocol,
            "api_alltalk_ip_port": alltalk_ip_port,
            "api_connection_timeout": 5,
        }
    }
    
    with open(config_file_path, "w") as file:
        json.dump(settings, file, indent=4)
    
#############################################
# Low VRAM change request to enable/disable #
#############################################
# LOW VRAM - Gradio Checkbox handling
def send_lowvram_request(value_sent):
    try:
        params["tgwui"]["tts_model_loaded"] = False
        if value_sent:
            audio_path = this_dir / "lowvramenabled.wav"
        else:
            audio_path = this_dir / "lowvramdisabled.wav"
        url = f"{alltalk_protocol}{alltalk_ip_port}/api/lowvramsetting?new_low_vram_value={value_sent}"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        json_response = response.json()
        # Check if the low VRAM request was successful
        if json_response.get("status") == "lowvram-success":
            # Update any relevant variables or perform other actions on success
            params["tgwui"]["tts_model_loaded"] = True
        return f'<audio src="file/{audio_path}" controls autoplay></audio>'
    except requests.exceptions.RequestException as e:
        # Handle the HTTP request error
        print(f"[{branding}Server] \033[91mWarning\033[0m Error during request to webserver process: Status code:\n{e}")
        return {"status": "error", "message": str(e)}

###################
#### DeepSpeed ####
###################
def send_deepspeed_request(value_sent):
    try:
        params["tgwui"]["tts_model_loaded"] = False
        if value_sent:
            audio_path = this_dir / "deepspeedenabled.wav"
        else:
            audio_path = this_dir / "deepspeeddisabled.wav"
        url = f"{alltalk_protocol}{alltalk_ip_port}/api/deepspeed?new_deepspeed_value={value_sent}"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        json_response = response.json()
        # Check if the deepspeed request was successful
        if json_response.get("status") == "deepspeed-success":
            # Update any relevant variables or perform other actions on success
            params["tgwui"]["tts_model_loaded"] = True
        return f'<audio src="file/{audio_path}" controls autoplay></audio>'
    except requests.exceptions.RequestException as e:
        # Handle the HTTP request error
        print(f"[{branding}Server] \033[91mWarning\033[0m Error during request to webserver process: Status code:\n{e}")
        return {"status": "error", "message": str(e)}

#################################
#### TTS STANDARD GENERATION ####
#################################
def send_and_generate(gen_text, gen_character_voice, gen_rvccharacter_voice, gen_narrator_voice, gen_rvcnarrator_voice, gen_narrator_activated, gen_textnotinisde, gen_repetition, gen_language, gen_filter, gen_speed, gen_pitch, gen_autoplay, gen_autoplay_vol, gen_file_name, gen_temperature, gen_filetimestamp, gen_stream, gen_stopcurrentgen):
    api_url = f"{alltalk_protocol}{alltalk_ip_port}/api/tts-generate"
    if gen_stopcurrentgen:
        stop_generate_tts() # Call for the current generation to stop
    tgwui_save_settings()   # Save the current TGWUI settings to the config file
    if gen_stream == "true":
        api_url = f"{alltalk_protocol}{alltalk_ip_port}/api/tts-generate-streaming"
        encoded_text = requests.utils.quote(gen_text)
        streaming_url = f"{api_url}?text={encoded_text}&voice={gen_character_voice}&language={gen_language}&output_file={gen_file_name}"
        return streaming_url, str("TTS Streaming Audio Generated")
    else:
        data = {
            "text_input": gen_text,
            "text_filtering": gen_filter,
            "character_voice_gen": gen_character_voice,
            "rvccharacter_voice_gen": gen_rvccharacter_voice,
            "narrator_enabled": str(gen_narrator_activated).lower(),
            "narrator_voice_gen": gen_narrator_voice,
            "rvcnarrator_voice_gen": gen_rvcnarrator_voice,
            "text_not_inside": gen_textnotinisde,
            "language": gen_language,
            "output_file_name": str(gen_file_name),
            "output_file_timestamp": str(gen_filetimestamp).lower(),
            "autoplay": str(gen_autoplay).lower(),
            "autoplay_volume": str(gen_autoplay_vol),
            "speed": str(gen_speed),
            "pitch": str(gen_pitch),
            "temperature": str(gen_temperature),
            "repetition_penalty": str(gen_repetition),
        }
        #print(f"Debug: Generate request param:", data) if params["debug_tts"] else None
        try:
            response = requests.post(api_url, data=data)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            result = response.json()
            if gen_autoplay == "true":
                return None, str("TTS Audio Generated (Played remotely)")
            else:
                if params["api_def"]["api_use_legacy_api"]:
                    return result['output_file_url'], str("TTS Audio Generated")
                else:
                    # Prepend the URL and PORT to the output_file_url
                    output_file_url = f"{alltalk_protocol}{alltalk_ip_port}{result['output_file_url']}"
                    return output_file_url, str("TTS Audio Generated")
        except (RequestException, ConnectionError) as e:
            print(f"[{branding}Server] \033[91mWarning\033[0m Error occurred during the API request: Status code:\n{str(e)}")
            return None, str("Error occurred during the API request")       

####################################################################
# TGWUI # Text-generation-webui sends text for TTS generation here #
####################################################################
def output_modifier(string, state):
    if not params["tgwui"]["tgwui_activate_tts"]:
        return string
    # Strip out Images
    img_info = ""
    cleaned_text, img_info = tgwui_extract_and_remove_images(string)
    if cleaned_text is None:
        return
    # Get current settings
    language_code = languages.get(params["tgwui"]["tgwui_language"])
    character_voice = params["tgwui"]["tgwui_character_voice"]
    rvc_character_voice = params["tgwui"]["tgwui_rvc_char_voice"]
    narrator_voice = params["tgwui"]["tgwui_narrator_voice"]
    rvc_narrator_voice = params["tgwui"]["tgwui_rvc_narr_voice"]
    narrator_enabled = params["tgwui"]["tgwui_narrator_enabled"]
    text_not_inside = params["tgwui"]["tgwui_non_quoted_text_is"]
    repetition_policy = params["tgwui"]["tgwui_repetitionpenalty_set"]
    text_filtering = "html"
    speed = params["tgwui"]["tgwui_generationspeed_set"]
    pitch = params["tgwui"]["tgwui_pitch_set"]
    autoplay = False        # This is never used in TGWUI so a base False setting is sent
    autoplay_volume = 0.8   # This is never used in TGWUI so a base False setting is sent
    temperature = params["tgwui"]["tgwui_temperature_set"]
    # Lock and process TTS request
    if process_lock.acquire(blocking=False):
        try:
            if "character_menu" in state:
                output_file = state["character_menu"]
            else:
                output_file = str("TTSOUT_")
            generate_response, status_message = send_and_generate(cleaned_text, character_voice, rvc_character_voice, narrator_voice, rvc_narrator_voice, narrator_enabled, text_not_inside, repetition_policy, language_code, text_filtering, speed, pitch, autoplay, autoplay_volume, output_file, temperature, True, False, False)
            if status_message == "TTS Audio Generated":
                # Handle Gradio and playback
                autoplay = "autoplay" if params["tgwui"]["tgwui_autoplay_tts"] else ""
                string = f'<audio src="{generate_response}" controls {autoplay}></audio>'
                if params["tgwui"]["tgwui_show_text"]:
                    string += tgwui_reinsert_images(cleaned_text, img_info)
                shared.processing_message = "*Is typing...*"
                return string
            else:
                print(f"[{branding}Server] \033[91mWarning\033[0m Audio generation failed.  Status code:\n{status_message}")
        finally:
            # Always release the lock, whether an exception occurs or not
            process_lock.release()
    else:
        # The lock is already acquired
        print(f"[{branding}Model] \033[91mWarning\033[0m Audio generation is already in progress. Please wait.")
        return

###########################################################################################################
# TGWUI # Strips out images from the TGWUI string if needed and re-inserts the image after TTS generation #
###########################################################################################################
img_pattern = r'<img[^>]*src\s*=\s*["\'][^"\'>]+["\'][^>]*>'

def tgwui_extract_and_remove_images(text):
    """
    Extracts all image data from the text and removes it for clean TTS processing.
    Returns the cleaned text and the extracted image data.
    """
    img_matches = re.findall(img_pattern, text)
    img_info = "\n".join(img_matches)  # Store extracted image data
    cleaned_text = re.sub(img_pattern, '', text)  # Remove images from text
    return cleaned_text, img_info

def tgwui_reinsert_images(text, img_info):
    """
    Reinserts the previously extracted image data back into the text.
    """
    if img_info:  # Check if there are images to reinsert
        text += f"\n\n{img_info}"
    return text

################################################################
# TGWUI # Used to generate a preview voice sample within TGWUI #
################################################################
def random_sentence():
    with open(this_dir / "harvard_sentences.txt") as f:
        return random.choice(list(f)).rstrip()

def voice_preview(string):
    if not params["tgwui"]["tgwui_activate_tts"]:
        return string
    language_code = languages.get(params["tgwui"]["tgwui_language"])
    if not string:
        string = random_sentence()
    generate_response, status_message = send_and_generate(string, params["tgwui"]["tgwui_character_voice"], params["tgwui"]["tgwui_rvc_char_voice"], params["tgwui"]["tgwui_narrator_voice"], params["tgwui"]["tgwui_rvc_narr_voice"], False, "character", params["tgwui"]["tgwui_repetitionpenalty_set"], language_code, "standard", params["tgwui"]["tgwui_generationspeed_set"], params["tgwui"]["tgwui_pitch_set"], False, 0.8, "previewvoice", params["tgwui"]["tgwui_temperature_set"], False, False, False)
    if status_message == "TTS Audio Generated":
        # Handle Gradio and playback
        autoplay = "autoplay" if params["tgwui"]["tgwui_autoplay_tts"] else ""
        return f'<audio src="{generate_response}?{int(time.time())}" controls {autoplay}></audio>'
    else:
        # Handle the case where audio generation was not successful
        return f"[{branding}Server] Audio generation failed. Status code:\n{status_message}"

###################################################################
# TGWUI # Used to inform TGWUI that TTS is disabled/not activated #
###################################################################
def state_modifier(state):
    if not params["tgwui"]["tgwui_activate_tts"]:
        return state
    state["stream"] = False
    return state

###################################################################
# TGWUI #  Sends message to TGWUI interface during TTS generation #
###################################################################
def input_modifier(string, state):
    if not params["tgwui"]["tgwui_activate_tts"]:
        return string
    shared.processing_message = "*Is recording a voice message...*"
    return string

########################################################################
# TGWUI # Used to delete historic TTS audios from TGWUI chat interface #
########################################################################
def remove_tts_from_history(history):
    for i, entry in enumerate(history["internal"]):
        history["visible"][i] = [history["visible"][i][0], entry[1]]
    return history

def toggle_text_in_history(history):
    for i, entry in enumerate(history["visible"]):
        visible_reply = entry[1]
        if visible_reply.startswith("<audio"):
            if params["tgwui"]["tgwui_show_text"]:
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

###########################################################
# TGWUI # Update IP address and Port from TGWUI interface #
###########################################################
def tgwui_update_alltalk_ip_port(value_sent):
    global alltalk_ip_port
    alltalk_ip_port = value_sent

###########################################################
# TGWUI # Update http/https protocol from TGWUI interface #
###########################################################
# Update Protocol from Gradio
def tgwui_update_alltalk_protocol(value_sent):
    global alltalk_protocol
    alltalk_protocol = value_sent

###########################################################################################################################
# TGWUI # Debounces model updates when tgwui_update_dropdowns is called, preventing an update from causing a model reload #
###########################################################################################################################
def tgwui_handle_ttsmodel_dropdown_change(model_name):
    if not getattr(tgwui_handle_ttsmodel_dropdown_change, "skip_reload", False):
        send_reload_request(model_name)
    tgwui_handle_ttsmodel_dropdown_change.skip_reload = False

##################################################################################################
# TGWUI # Pulls the current AllTalk Server settings & updates gradio when Refresh button pressed #
##################################################################################################
def tgwui_update_dropdowns():
    global at_settings
    at_settings = get_alltalk_settings()  # Pull all the current settings from the AllTalk server, if its online.
    current_voices = at_settings["voices"]
    rvccurrent_voices = at_settings["rvcvoices"]
    rvccurrent_character_voice = params["tgwui"]["tgwui_rvc_char_voice"]
    rvccurrent_narrator_voice = params["tgwui"]["tgwui_rvc_char_voice"]    
    current_models_available = sorted(at_settings["models_available"])
    current_model_loaded = at_settings["current_model_loaded"]
    current_character_voice = params["tgwui"]["tgwui_character_voice"]
    current_narrator_voice = params["tgwui"]["tgwui_narrator_voice"]
    current_lowvram_capable = at_settings["lowvram_capable"]
    current_lowvram_enabled = at_settings["lowvram_enabled"]
    current_temperature_capable = at_settings["temperature_capable"]
    current_repetitionpenalty_capable = at_settings["repetitionpenalty_capable"]
    current_generationspeed_capable = at_settings["generationspeed_capable"]
    current_pitch_capable = at_settings["pitch_capable"]
    current_deepspeed_capable = at_settings["deepspeed_capable"]
    current_deepspeed_enabled = at_settings["deepspeed_enabled"]
    current_non_quoted_text_is = params.get("tgwui")["tgwui_non_quoted_text_is"]  # Use the correct parameter path
    current_languages_capable = at_settings["languages_capable"]
    if at_settings["languages_capable"]:
        language_label = "Languages"
    else:
        language_label = "Model not multi language"
    if current_character_voice not in current_voices:
        current_character_voice = current_voices[0] if current_voices else ""
        params["tgwui"]["tgwui_character_voice"] = current_character_voice
    if current_narrator_voice not in current_voices:
        current_narrator_voice = current_voices[0] if current_voices else ""
        params["tgwui"]["tgwui_narrator_voice"] = current_narrator_voice
    if rvccurrent_character_voice not in rvccurrent_voices:
        rvccurrent_character_voice = rvccurrent_voices[0] if rvccurrent_voices else ""
        params["tgwui"]["tgwui_rvc_char_voice"] = rvccurrent_character_voice
    if rvccurrent_narrator_voice not in rvccurrent_voices:
        rvccurrent_narrator_voice = rvccurrent_voices[0] if rvccurrent_voices else ""
        params["tgwui"]["tgwui_rvc_char_voice"] = rvccurrent_narrator_voice
    tgwui_handle_ttsmodel_dropdown_change.skip_reload = True  # Debounce tgwui_tts_dropdown_gr and stop it sending a model reload when it is updated.

    return (
        gr.Checkbox(interactive=current_lowvram_capable, value=current_lowvram_enabled),
        gr.Checkbox(interactive=current_deepspeed_capable, value=current_deepspeed_enabled),
        gr.Dropdown(choices=current_voices, value=current_character_voice),
        gr.Dropdown(choices=rvccurrent_voices, value=rvccurrent_character_voice, interactive=True),        
        gr.Dropdown(choices=current_voices, value=current_narrator_voice),
        gr.Dropdown(choices=rvccurrent_voices, value=rvccurrent_narrator_voice, interactive=True),        
        gr.Dropdown(choices=current_models_available, value=current_model_loaded),
        gr.Dropdown(interactive=current_temperature_capable),
        gr.Dropdown(interactive=current_repetitionpenalty_capable),
        gr.Dropdown(interactive=current_languages_capable, label=language_label),
        gr.Dropdown(interactive=current_generationspeed_capable),
        gr.Dropdown(interactive=current_pitch_capable),
        gr.Dropdown(value=current_non_quoted_text_is),
    )

###############################################################
# TGWUI # Gradio interface & layout for Text-generation-webui #
###############################################################
def ui():
    global alltalk_settings
    alltalk_settings = get_alltalk_settings()   # Pull all the current settings from the AllTalk server, if its online.
    with gr.Accordion(params["branding"] + " TTS (Text-gen-webui Remote)"):
        tgwui_available_voices_gr = alltalk_settings["voices"]
        tgwui_rvc_available_voices_gr = alltalk_settings["rvcvoices"]        
        # Activate alltalk_tts, Enable autoplay, Show text
        with gr.Row():
            tgwui_activate_tts_gr = gr.Checkbox(value=params["tgwui"]["tgwui_activate_tts"], label="Enable TGWUI TTS")
            tgwui_autoplay_gr = gr.Checkbox(value=params["tgwui"]["tgwui_autoplay_tts"], label="Autoplay TTS Generated")
            tgwui_show_text_gr = gr.Checkbox(value=params["tgwui"]["tgwui_show_text"], label="Show Text in chat")

        # Low vram enable, Deepspeed enable, Link
        with gr.Row():
            tgwui_lowvram_enabled_gr = gr.Checkbox(value=alltalk_settings["lowvram_enabled"] if alltalk_settings["lowvram_capable"] else False, label="Enable Low VRAM Mode", interactive=alltalk_settings["lowvram_capable"])
            tgwui_lowvram_enabled_play_gr = gr.HTML(visible=False)
            tgwui_deepspeed_enabled_gr = gr.Checkbox(value=params["tgwui"]["tgwui_deepspeed_enabled"], label="Enable DeepSpeed", interactive=alltalk_settings["deepspeed_capable"],)
            tgwui_deepspeed_enabled_play_gr = gr.HTML(visible=False)
            tgwui_empty_space_gr = gr.HTML(f"<p><a href='{alltalk_protocol}{alltalk_ip_port}'>AllTalk Server & Documentation Link</a><a href='{alltalk_protocol}{alltalk_ip_port}'></a>")

        # Model, Language, Character voice
        with gr.Row():
            tgwui_tts_dropdown_gr = gr.Dropdown(choices=models_available, label="TTS Engine/Model", value=current_model_loaded,)
            tgwui_language_gr = gr.Dropdown(languages.keys(), label="Languages" if alltalk_settings["languages_capable"] else "Model not multi language", interactive=alltalk_settings["languages_capable"], value=params["tgwui"]["tgwui_language"])
            tgwui_narrator_enabled_gr = gr.Dropdown(choices=[("Enabled", "true"), ("Disabled", "false"), ("Silent", "silent")], label="Narrator Enable", value="true" if params.get("tgwui_narrator_enabled") == "true" else ("silent" if params.get("tgwui_narrator_enabled") == "silent" else "false"))            

        # Narrator
        with gr.Row():
            tgwui_available_voices_gr = alltalk_settings["voices"]
            tgwui_default_voice_gr = params["tgwui"]["tgwui_character_voice"]
            if tgwui_default_voice_gr not in tgwui_available_voices_gr:
                tgwui_default_voice_gr = tgwui_available_voices_gr[0] if tgwui_available_voices_gr else ""
            tgwui_character_voice_gr = gr.Dropdown(choices=tgwui_available_voices_gr, label="Character Voice", value=tgwui_default_voice_gr, allow_custom_value=True,)
            tgwui_narr_voice_gr = params["tgwui"]["tgwui_narrator_voice"]
            if tgwui_narr_voice_gr not in tgwui_available_voices_gr:
                tgwui_narr_voice_gr = tgwui_available_voices_gr[0] if tgwui_available_voices_gr else ""
            tgwui_narrator_voice_gr = gr.Dropdown(choices=tgwui_available_voices_gr, label="Narrator Voice", value=tgwui_narr_voice_gr, allow_custom_value=True,)                      
            tgwui_non_quoted_text_is_gr = gr.Dropdown(choices=[("Character", "character"), ("Narrator", "narrator"), ("Silent", "silent")], label='Narrator unmarked text is', value=params.get("tgwui_non_quoted_text_is", "character"))

        # RVC voices
        with gr.Row():
            tgwui_rvc_default_voice_gr = params["tgwui"]["tgwui_rvc_char_voice"]
            tgwui_rvc_narrator_voice_gr = params["tgwui"]["tgwui_rvc_narr_voice"]            
            if tgwui_rvc_default_voice_gr not in tgwui_rvc_available_voices_gr:
                tgwui_rvc_default_voice_gr = tgwui_rvc_available_voices_gr[0] if tgwui_rvc_available_voices_gr else ""
            tgwui_rvc_char_voice_gr = gr.Dropdown(choices=tgwui_rvc_available_voices_gr, label="RVC Character Voice", value=tgwui_rvc_default_voice_gr, allow_custom_value=True,)
            if tgwui_rvc_narrator_voice_gr not in tgwui_rvc_available_voices_gr:
                tgwui_rvc_narrator_voice_gr = tgwui_rvc_available_voices_gr[0] if tgwui_rvc_available_voices_gr else ""
            tgwui_rvc_narr_voice_gr = gr.Dropdown(choices=tgwui_rvc_available_voices_gr, label="RVC Narrator Voice", value=tgwui_rvc_narrator_voice_gr, allow_custom_value=True,)                   

        # Temperature, Repetition Penalty, Speed, pitch
        with gr.Row():
            tgwui_temperature_set_gr = gr.Slider(minimum=0.05, maximum=1, step=0.05, label="Temperature", value=params["tgwui"]["tgwui_temperature_set"], interactive=alltalk_settings["temperature_capable"])
            tgwui_repetitionpenalty_set_gr = gr.Slider(minimum=0.5, maximum=20, step=0.5, label="Repetition Penalty", value=params["tgwui"]["tgwui_repetitionpenalty_set"], interactive=alltalk_settings["repetitionpenalty_capable"])
        with gr.Row():            
            tgwui_generationspeed_set_gr = gr.Slider(minimum=0.30, maximum=2.00, step=0.10, label="TTS Speed", value=params["tgwui"]["tgwui_generationspeed_set"], interactive=alltalk_settings["generationspeed_capable"])
            tgwui_pitch_set_gr = gr.Slider(minimum=-10, maximum=10, step=1, label="Pitch", value=params["tgwui"]["tgwui_pitch_set"], interactive=alltalk_settings["pitch_capable"])

        # Preview speech
        with gr.Row():
            tgwui_preview_text_gr = gr.Text(show_label=False, placeholder="Preview text", elem_id="silero_preview_text", scale=2,)
            tgwui_preview_play_gr = gr.Button("Generate Preview", scale=1)
            tgwui_preview_audio_gr = gr.HTML(visible=False)

        with gr.Row():
            tgwui_protocol_gr = gr.Dropdown(choices=["http://", "https://"], label="AllTalk Server Protocol", value=alltalk_protocol)
            tgwui_protocol_gr.change(tgwui_update_alltalk_protocol, tgwui_protocol_gr, None)
            tgwui_ip_address_port_gr = gr.Textbox(label="AllTalk Server IP:Port", value=alltalk_ip_port)
            tgwui_ip_address_port_gr.change(tgwui_update_alltalk_ip_port, tgwui_ip_address_port_gr, None)
            tgwui_refresh_settings_gr = gr.Button("Refresh settings & voices")
            tgwui_refresh_settings_gr.click(tgwui_update_dropdowns, None, [tgwui_lowvram_enabled_gr, tgwui_deepspeed_enabled_gr, tgwui_character_voice_gr, tgwui_rvc_char_voice_gr, tgwui_narrator_voice_gr, tgwui_rvc_narr_voice_gr, tgwui_tts_dropdown_gr, tgwui_temperature_set_gr, tgwui_repetitionpenalty_set_gr, tgwui_language_gr, tgwui_generationspeed_set_gr, tgwui_pitch_set_gr, tgwui_non_quoted_text_is_gr])

        with gr.Row():
            tgwui_convert_gr = gr.Button("Remove old TTS audio and leave only message texts")
            tgwui_convert_cancel_gr = gr.Button("Cancel", visible=False)
            tgwui_convert_confirm_gr = gr.Button("Confirm (cannot be undone)", variant="stop", visible=False)
            tgwui_stop_generation_gr = gr.Button("Stop current TTS generation")
            tgwui_stop_generation_gr.click(stop_generate_tts, None, None,)

    # Convert history with confirmation
    convert_arr = [tgwui_convert_confirm_gr, tgwui_convert_gr, tgwui_convert_cancel_gr]
    tgwui_convert_gr.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True),], None, convert_arr,)
    tgwui_convert_confirm_gr.click(lambda: [ gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),], None, convert_arr,
                          ).then(remove_tts_from_history, gradio("history"), gradio("history")
                                 ).then(chat.save_history, gradio("history", "unique_id", "character_menu", "mode"), None,
                                        ).then(chat.redraw_html, gradio(ui_chat.reload_arr), gradio("display"))
    tgwui_convert_cancel_gr.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),], None, convert_arr,)

    # Toggle message text in history
    tgwui_show_text_gr.change(lambda x: params["tgwui"].update({"tgwui_show_text": x}), tgwui_show_text_gr, None
                     ).then(toggle_text_in_history, gradio("history"), gradio("history")
                            ).then(chat.save_history, gradio("history", "unique_id", "character_menu", "mode"), None,
                                   ).then(chat.redraw_html, gradio(ui_chat.reload_arr), gradio("display"))

    # Event functions to update the parameters in the backend
    tgwui_activate_tts_gr.change(lambda x: params["tgwui"].update({"tgwui_activate_tts": x}), tgwui_activate_tts_gr, None)
    tgwui_autoplay_gr.change(lambda x: params["tgwui"].update({"tgwui_autoplay_tts": x}), tgwui_autoplay_gr, None)
    tgwui_lowvram_enabled_gr.change(lambda x: params["tgwui"].update({"tgwui_lowvram_enabled": x}), tgwui_lowvram_enabled_gr, None)
    tgwui_lowvram_enabled_gr.change(lambda x: send_lowvram_request(x), tgwui_lowvram_enabled_gr, tgwui_lowvram_enabled_play_gr, None)

    # Trigger the send_reload_request function when the dropdown value changes
    tgwui_tts_dropdown_gr.change(tgwui_handle_ttsmodel_dropdown_change, tgwui_tts_dropdown_gr, None)

    tgwui_deepspeed_enabled_gr.change(lambda x: params["tgwui"].update({"tgwui_deepspeed_enabled": x}), tgwui_deepspeed_enabled_gr, None)
    tgwui_deepspeed_enabled_gr.change(send_deepspeed_request, tgwui_deepspeed_enabled_gr, tgwui_deepspeed_enabled_play_gr, None)
    tgwui_character_voice_gr.change(lambda x: params["tgwui"].update({"tgwui_character_voice": x}), tgwui_character_voice_gr, None)
    tgwui_language_gr.change(lambda x: params["tgwui"].update({"tgwui_language": x}), tgwui_language_gr, None)

    # TSS Settings
    tgwui_temperature_set_gr.change(lambda x: params["tgwui"].update({"tgwui_temperature_set": x}), tgwui_temperature_set_gr, None)
    tgwui_repetitionpenalty_set_gr.change(lambda x: params["tgwui"].update({"tgwui_repetitionpenalty_set": x}), tgwui_repetitionpenalty_set_gr, None)
    tgwui_generationspeed_set_gr.change(lambda x: params["tgwui"].update({"tgwui_generationspeed_set": x}), tgwui_generationspeed_set_gr, None)
    tgwui_pitch_set_gr.change(lambda x: params["tgwui"].update({"tgwui_pitch_set": x}), tgwui_pitch_set_gr, None)

    # Narrator selection actions
    tgwui_narrator_enabled_gr.change(lambda x: params["tgwui"].update({"tgwui_narrator_enabled": x}), tgwui_narrator_enabled_gr, None)
    tgwui_non_quoted_text_is_gr.change(lambda x: params["tgwui"].update({"tgwui_non_quoted_text_is": x}), tgwui_non_quoted_text_is_gr, None)
    tgwui_narrator_voice_gr.change(lambda x: params["tgwui"].update({"tgwui_narrator_voice": x}), tgwui_narrator_voice_gr, None)
    
    # RVC selection actions
    tgwui_rvc_char_voice_gr.change(lambda x: params["tgwui"].update({"tgwui_rvc_char_voice": x}), tgwui_rvc_char_voice_gr, None)
    tgwui_rvc_narr_voice_gr.change(lambda x: params["tgwui"].update({"tgwui_rvc_narr_voice": x}), tgwui_rvc_narr_voice_gr, None)

    # Play preview
    tgwui_preview_text_gr.submit(voice_preview, tgwui_preview_text_gr, tgwui_preview_audio_gr)
    tgwui_preview_play_gr.click(voice_preview, tgwui_preview_text_gr, tgwui_preview_audio_gr)

