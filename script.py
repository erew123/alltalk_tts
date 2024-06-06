import os
import re
import sys
import json
import time
import shutil
import random
import atexit
import signal
import logging
import requests
import platform
import subprocess
import threading
import soundfile as sf
from pathlib import Path
from datetime import datetime, timedelta
from requests.exceptions import RequestException, ConnectionError
###########################################################################
# START-UP # Silence Character Normaliser when it checks the Ready Status #
###########################################################################
import warnings
warnings.filterwarnings('ignore', message='Trying to detect encoding from a tiny portion')
###########################################
# START-UP # AllTalk allowed startup time #
###########################################
startup_wait_time = 240

# You can change the above setting to a larger number to allow AllTAlk more time to start up. 
# The default setting is 240 seconds (4 minutes). If its taking longer though, you may have a
# Very old system or system issue.

##############################################
# START-UP # Load confignew.json into params #
##############################################
this_dir = Path(__file__).parent.resolve()

def load_config(file_path):
    with open(file_path, "r") as config_file:
        config = json.load(config_file)
    return config

config_file_path = this_dir / "confignew.json"
params = load_config(config_file_path)
branding = params['branding']
github_site = "erew123"
github_repository = "alltalk_tts"
current_folder = os.path.basename(os.getcwd())
output_folder = this_dir / params["output_folder"]
delete_output_wavs_setting = params["delete_output_wavs"]
gradio_enabled = params["gradio_interface"]
script_path = this_dir / "tts_server.py"
tunnel_url = None
running_on_google_colab = False

############################################
# START-UP # Display initial splash screen #
############################################
print(f"[{branding}TTS]\033[94m     _    _ _ \033[1;35m_____     _ _     \033[0m  _____ _____ ____  ")
print(f"[{branding}TTS]\033[94m    / \  | | |\033[1;35m_   _|_ _| | | __ \033[0m |_   _|_   _/ ___| ")
print(f"[{branding}TTS]\033[94m   / _ \ | | |\033[1;35m | |/ _` | | |/ / \033[0m   | |   | | \___ \ ")
print(f"[{branding}TTS]\033[94m  / ___ \| | |\033[1;35m | | (_| | |   <  \033[0m   | |   | |  ___) |")
print(f"[{branding}TTS]\033[94m /_/   \_\_|_|\033[1;35m |_|\__,_|_|_|\_\ \033[0m   |_|   |_| |____/ ")
print(f"[{branding}TTS]")

######################################################
# START-UP # Check if this is a first time start-up  #
######################################################
def run_firsttime_script():
    try:
        subprocess.run(['python', 'system/config/firstrun.py'], check=True)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Call the function to run the startup script
run_firsttime_script()

###########################################################
# START-UP # Check for updates needed to the config file  #
###########################################################
update_config_path = this_dir / "system" / "config" / "at_configupdate.json"
downgrade_config_path = this_dir / "system" / "config" / "at_configdowngrade.json"
def changes_needed(main_config, update_config, downgrade_config):
    """Check if there are any changes to be made to the main configuration."""
    for key in downgrade_config.keys():
        if key in main_config:
            return True
    for key, value in update_config.items():
        if key not in main_config:
            return True
    return False

def update_config(config_file_path, update_config_path, downgrade_config_path):
    try:
        with open(config_file_path, 'r') as file:
            main_config = json.load(file)
        with open(update_config_path, 'r') as file:
            update_config = json.load(file)
        with open(downgrade_config_path, 'r') as file:
            downgrade_config = json.load(file)

        # Determine if changes are needed
        if changes_needed(main_config, update_config, downgrade_config):
            # Backup with timestamp to avoid overwriting
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = config_file_path.with_suffix(f".{timestamp}.bak")
            logging.info(f"Creating backup of the main config to {backup_path}")
            shutil.copy(config_file_path, backup_path)

            # Proceed with updates and downgrades
            for key, value in update_config.items():
                if key not in main_config:
                    main_config[key] = value
            for key in downgrade_config.keys():
                if key in main_config:
                    del main_config[key]

            # Save the updated configuration
            with open(config_file_path, 'w') as file:
                json.dump(main_config, file, indent=4)

            print(f"[{branding}TTS] \033[92mConfig file update: \033[91mUpdates applied\033[0m")
        else:
            print(f"[{branding}TTS] \033[92mConfig file update: \033[93mNo Updates required\033[0m")

    except Exception as e:
        print(f"[{branding}TTS] \033[92mConfig file update: \033[91mError updating\033[0m")

# Update the configuration
update_config(config_file_path, update_config_path, downgrade_config_path)

###########################################################################################
# START-UP # Check for updates needed to the TTS engines list when a new engine is added  #
###########################################################################################
tts_engines_path = this_dir / "system" / "tts_engines" / "tts_engines.json"
new_engines_path = this_dir / "system" / "tts_engines" / "new_engines.json"

# Load the JSON files
with open(tts_engines_path, 'r') as f:
    tts_engines_data = json.load(f)

with open(new_engines_path, 'r') as f:
    new_engines_data = json.load(f)

# Extract the list of current and new engines
current_engines = {engine['name'] for engine in tts_engines_data['engines_available']}
new_engines = new_engines_data['engines_available']

# Iterate over the new engines to see if they need to be added
for engine in new_engines:
    engine_name = engine['name']
    if engine_name not in current_engines:
        # Check if the directory for the new engine exists
        engine_dir = this_dir / "system" / "tts_engines" / engine_name
        if engine_dir.is_dir():
            # Add the new engine to the list
            tts_engines_data['engines_available'].append(engine)
            print(f"[{branding}TTS] \033[92mNew TTS Engines   : \033[91mAdded {engine_name}\033[0m")

# Save the updated tts_engines.json
with open(tts_engines_path, 'w') as f:
    json.dump(tts_engines_data, f, indent=4)

####################################################
# START-UP # Re-load from confignew.json to params #
####################################################
params = load_config(config_file_path)

#############################################################################
# START-UP # Check current folder name has dashes '-' in it and error if so #
#############################################################################
if "-" in current_folder:
    print(f"[{branding}TTS]")
    print(f"[{branding}TTS] \033[91mWarning\033[0m The current folder name contains a dash ('\033[93m-\033[0m') and this causes errors/issues. Please ensure")
    print(f"[{branding}TTS] \033[91mWarning\033[0m the folder name does not have a dash e.g. rename ('\033[93malltalk_tts-main\033[0m') to ('\033[93malltalk_tts\033[0m').")
    print(f"[{branding}TTS]")
    print(f"[{branding}TTS] \033[92mCurrent folder:\033[0m {current_folder}")
    sys.exit(1)

###############################################################################
# START-UP # Test if we are running within Text-gen-webui or as a Stanadalone #
###############################################################################
try:
    import gradio as gr
    from modules import chat, shared, ui_chat
    from modules.logging_colors import logger
    from modules.ui import create_refresh_button
    from modules.utils import gradio
    print(f"[{branding}TTS] \033[92mStart-up Mode     : \033[93mText-gen-webui mode\033[0m")
    running_in_standalone = False
except ModuleNotFoundError:
    running_in_standalone = True
    print(f"[{branding}TTS] \033[92mStart-up Mode     : \033[93mStandalone mode\033[0m")

###########################################################
# START-UP # Delete files in outputs folder if configured #
###########################################################
def delete_old_files(folder_path, days_to_keep):
    current_time = datetime.now()
    print(f"[{branding}TTS] \033[92mWAV file deletion    :\033[93m", delete_output_wavs_setting,"\033[0m")
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            age = current_time - file_creation_time
            if age > timedelta(days=days_to_keep):
                os.remove(file_path)

# Check and perform file deletion
if delete_output_wavs_setting.lower() == "disabled":
    print(f"[{branding}TTS] \033[92mWAV file deletion :\033[93m Disabled\033[0m")
else:
    try:
        days_to_keep = int(delete_output_wavs_setting.split()[0])
        delete_old_files(output_folder, days_to_keep)
    except ValueError:
        print(f"[{branding}TTS] \033[92mWAV file deletion :\033[93m Invalid setting for deleting old wav files. Please use 'Disabled' or 'X Days' format\033[0m")

#####################################################################
# START-UP # Check Githubs last update and output into splashscreen #
#####################################################################
def ordinal(n):
    return "%d%s" % (n, "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))

def format_datetime(iso_str):
    # Parse the ISO 8601 string to datetime
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ")
    # Format the datetime into a more readable string
    return dt.strftime(f"{ordinal(dt.day)} %B %Y at %H:%M")

def fetch_latest_commit_sha_and_date(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            latest_commit = response.json()[0]
            latest_commit_sha = latest_commit['sha']
            latest_commit_date = latest_commit['commit']['committer']['date']
            return latest_commit_sha, latest_commit_date
        else:
            print(f"[{branding}TTS] \033[92mGithub updated    :\033[91m Failed to fetch the latest commits due to an unexpected response from GitHub")
            return None, None
    except ConnectionError:
        # This block is executed when a connection error occurs
        print(f"[{branding}TTS] \033[92mGithub updated    :\033[91m Could not reach GitHub to check for updates\033[0m")
        return None, None

def read_or_initialize_sha(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data.get("last_known_commit_sha")
    else:
        # File doesn't exist, fetch the latest SHA and create the file
        latest_commit_sha = fetch_latest_commit_sha_and_date(github_site, github_repository)
        if latest_commit_sha:
            with open(file_path, 'w') as file:
                json.dump({"last_known_commit_sha": latest_commit_sha}, file)
            return latest_commit_sha
        else:
            # Handle the case where GitHub couldn't be reached
            return None

def update_sha_file(file_path, new_sha):
    with open(file_path, 'w') as file:
        json.dump({"last_known_commit_sha": new_sha}, file)

# Define the file path based on your directory structure
sha_file_path = this_dir / "system" / "config" / "at_github_sha.json"

# Read or initialize the SHA (adjusted for handling both SHA and date)
last_known_commit_sha = read_or_initialize_sha(sha_file_path)  # Assuming adjustment for date

# Assuming you have fetched the latest commit SHA and date
latest_commit_sha, latest_commit_date = fetch_latest_commit_sha_and_date(github_site, github_repository)

formatted_date = format_datetime(latest_commit_date) if latest_commit_date else "an unknown date"

if latest_commit_sha and latest_commit_sha != last_known_commit_sha:
    #print(f"There's an update available for alltalk_tts.")
    print(f"[{branding}TTS] \033[92mGithub updated    :\033[93m {formatted_date}\033[0m")
    # Update the file with the new SHA
    update_sha_file(sha_file_path, latest_commit_sha)
elif latest_commit_sha == last_known_commit_sha:
    #print(f"Your alltalk_tts software is up to date.")
    print(f"[{branding}TTS] \033[92mGithub updated    :\033[93m {formatted_date}\033[0m")

##################################################
# START-UP # Configure the subprocess hanlder ####
##################################################
def signal_handler(sig, frame):
    print(f"[{branding}Shutdown] \033[94mReceived Ctrl+C, terminating subprocess. Kill your Python processes if this fails to exit.\033[92m")
    if process.poll() is None:
        process.terminate()
        process.wait()  # Wait for the subprocess to finish
    sys.exit(0)

#####################################################################
# START-UP # Start the Subprocess and Check for Google Colab/Docker #
#####################################################################
try:
    import google.colab
    running_on_google_colab = True
    #if deepspeed_installed:
        #params["deepspeed_activate"] = True
    with open('/content/alltalk_tts/googlecolab.json', 'r') as f:
        data = json.load(f)
        google_ip_address = data.get('google_ip_address', tunnel_url)
except FileNotFoundError:
    print("Could not find IP address")
    google_ip_address = tunnel_url
except ImportError:
    pass

# Attach the signal handler to the SIGINT signal (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Check if we're running in docker
if os.path.isfile("/.dockerenv") and 'google.colab' not in sys.modules:
    print(f"[{branding}TTS] \033[94mRunning in Docker. Please wait.\033[0m")
else:
    # Start the subprocess
    process = subprocess.Popen(["python", script_path])

    # Check if the subprocess has started successfully
    if process.poll() is None:
        if running_on_google_colab:
            print(f"[{branding}TTS]")
            print(f"[{branding}TTS] \033[94m{branding}Google Colab Address:\033[00m",f"\033[92m{google_ip_address}:443\033[00m")
            print(f"[{branding}TTS]")
    else:
        print(f"[{branding}TTS] \033[91mWarning\033[0m TTS Subprocess Webserver failing to start process")
        print(f"[{branding}TTS] \033[91mWarning\033[0m It could be that you have something on port:",params["port_number"],)
        print(f"[{branding}TTS] \033[91mWarning\033[0m Or you have not started in a Python environement with all the necesssary bits installed")
        print(f"[{branding}TTS] \033[91mWarning\033[0m Check you are starting Text-generation-webui with either the start_xxxxx file or the Python environment with cmd_xxxxx file.")
        print(f"[{branding}TTS] \033[91mWarning\033[0m xxxxx is the type of OS you are on e.g. windows, linux or mac.")
        print(f"[{branding}TTS] \033[91mWarning\033[0m Alternatively, you could check no other Python processes are running that shouldnt be e.g. Restart your computer is the simple way.")
        # Cleanly kill off this script, but allow text-generation-webui to keep running, albeit without this alltalk_tts
        sys.exit(1)

    timeout = startup_wait_time  # Gather timeout setting from startup_wait_time
    initial_delay = 5  # Initial delay before starting the check loop
    warning_delay = 60  # Delay before displaying warnings

    # Introduce a delay before starting the check loop
    time.sleep(initial_delay)

    start_time = time.time()
    warning_displayed = False

    url = f"http://localhost:{params['api_def']['api_port_number']}/api/ready"
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200 and response.text == "Ready":
                break
        except requests.RequestException as e:
            # Log the exception if needed
            pass

        if not warning_displayed and time.time() - start_time >= warning_delay:
            print(f"[{branding}TTS] \033[91mWarning\033[0m TTS Engine has NOT started up yet. Will keep trying for {timeout} seconds maximum. Please wait.")
            print(f"[{branding}TTS] \033[91mWarning\033[0m Mechanical hard drives and a slow PCI BUS are examples of things that can affect load times.")
            print(f"[{branding}TTS] \033[91mWarning\033[0m Some TTS engines index their AI TTS models on loading, which can be slow on CPU or old systems.")
            print(f"[{branding}TTS] \033[91mWarning\033[0m Using one of the other TTS engines on slower systems can help ease this issue.")
            warning_displayed = True
        
        time.sleep(1)
    else:
        print(f"[{branding}TTS]")
        print(f"[{branding}TTS] Startup timed out. Full help available here \033[92mhttps://github.com/erew123/alltalk_tts#-help-with-problems\033[0m")
        print(f"[{branding}TTS] On older systems, you may wish to open and edit \033[94mscript.py\033[0m with a text editor and change the")
        print(f"[{branding}TTS] \033[94mstartup_wait_time = 240\033[0m setting to something like \033[94mstartup_wait_time = 460\033[0m as this will allow")
        print(f"[{branding}TTS] AllTalk more time (6 mins) to try load the model into your VRAM. Otherwise, please visit the GitHub for")
        print(f"[{branding}TTS] a list of other possible troubleshooting options.")
        # Cleanly kill off this script, but allow text-generation-webui to keep running, albeit without this alltalk_tts
        sys.exit(1)

    print(f"[{branding}TTS]")
    print(f"[{branding}TTS] \033[94mAPI Address :\033[00m",f"\033[92m127.0.0.1:{params['api_def']['api_port_number']}\033[00m")
    print(f"[{branding}TTS] \033[94mGradio Light:\033[00m",f"\033[92mhttp://127.0.0.1:{params['gradio_port_number']}\033[00m")
    print(f"[{branding}TTS] \033[94mGradio Dark :\033[00m \033[92mhttp://127.0.0.1:{params['gradio_port_number']}?__theme=dark\033[00m")
    print(f"[{branding}TTS]")


#########################################
# START-UP # Espeak-ng check on Windows #
#########################################
def check_espeak_ng():
    if platform.system() == "Windows":
        try:
            result = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # If the command returns an error, print the error message
            print(f"[{branding}TTS]")
            print(f"[{branding}TTS] Espeak-ng for Windows\033[91m WAS NOT FOUND. \033[0mYou can install\033[0m")
            print(f"[{branding}TTS] \033[0mit from this location \033[93m\\alltalk_tts\\system\espeak-ng\\\033[0m")
            print(f"[{branding}TTS] Then close this command prompt window and open a new")
            print(f"[{branding}TTS] command prompt, before starting {branding}again.")
    elif platform.system() == "Darwin":  # macOS
        try:
            result = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, check=True)
            return
        except FileNotFoundError:
            print(f"[{branding}TTS]")
            print(f"[{branding}TTS] Espeak-ng for macOS\033[91m WAS NOT FOUND. \033[0mYou can install\033[0m")
            print(f"[{branding}TTS] \033[0mit using Homebrew: \033[93mbrew install espeak-ng\033[0m")
    else:  # Linux
        try:
            result = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, check=True)
            return
        except FileNotFoundError:
            print(f"[{branding}TTS]")
            print(f"[{branding}TTS] Espeak-ng for Linux\033[91m WAS NOT FOUND. \033[0mYou can install\033[0m")
            print(f"[{branding}TTS] \033[0mit using your package manager, e.g., \033[93mapt-get install espeak-ng\033[0m")
    print(f"[{branding}TTS]")
    print(f"[{branding}TTS]")

check_espeak_ng()
####################################
# START-UP # Subprecess management #
####################################

def start_subprocess():
    global process
    if process is None or process.poll() is not None:
        process = subprocess.Popen(["python", script_path])
        return "Subprocess started."
    else:
        return "Subprocess is already running."

def stop_subprocess():
    global process
    if process is not None:
        process.terminate()
        process.wait()
        process = None
        return "Subprocess stopped."
    else:
        return "Subprocess is not running."

def restart_subprocess():
    stop_subprocess()
    print(f"[{branding}ENG]")
    print(f"[{branding}ENG] \033[94mSwapping TTS Engine. Please wait.\033[00m")
    print(f"[{branding}ENG]")
    return start_subprocess()

def check_subprocess_status():
    global process
    if process is None or process.poll() is not None:
        return "Subprocess is not running."
    else:
        return "Subprocess is running."


###################################################################
# START-UP # Register the termination code to be executed at exit #
###################################################################
atexit.register(lambda: process.terminate() if process.poll() is None else None)

#############################################################################################################
#  _____         _                                           _           _    ____               _ _        #
# |_   _|____  _| |_       __ _  ___ _ __      __      _____| |__  _   _(_)  / ___|_ __ __ _  __| (_) ___   #
#   | |/ _ \ \/ / __|____ / _` |/ _ \ '_ \ ____\ \ /\ / / _ \ '_ \| | | | | | |  _| '__/ _` |/ _` | |/ _ \  #
#   | |  __/>  <| ||_____| (_| |  __/ | | |_____\ V  V /  __/ |_) | |_| | | | |_| | | | (_| | (_| | | (_) | #
#   |_|\___/_/\_\\__|     \__, |\___|_| |_|      \_/\_/ \___|_.__/ \__,_|_|  \____|_|  \__,_|\__,_|_|\___/  #
#                         |___/                                                                             #
#############################################################################################################
##########################
# Setup global variables #
##########################
# Load the language options from languages
with open(os.path.join(this_dir, "system", "config", "languages.json"), "r") as f:
    languages_list = json.load(f) 
current_model_loaded = None
#deepspeed_installed = True
tgwui_lovram = False
tgwui_deepspeed = False
# Create a global lock for tracking TTS generation occuring
process_lock = threading.Lock()
# Pull the values for IP address, port and protocol for communication with the AllTalk renote server
alltalk_protocol = "http://"
alltalk_ip_port = "127.0.0.1:7851"
# Pull the connection timeout value for communication requests with the AllTalk remote server
connection_timeout = 10
# Create a few base global variables that are required
models_available = None         # Gets populated with the list of all models/engines available on the AllTalk Server
at_settings = None    # Gets populated with the list of all settings currently set on the AllTalk Server
current_model_loaded = None     # Gets populated with the name of the current loaded TTS engine/model on the AllTalk Server
# Used to detect if a model is loaded in to AllTalk server to block TTS genereation if needed.
tts_model_loaded = None

#################################################
# Pull all the settings from the AllTalk Server #
#################################################
def get_alltalk_settings():
    global current_model_loaded, models_available, engines_available, current_engine_loaded
    voices_url = f"{alltalk_protocol}{alltalk_ip_port}/api/voices"
    rvcvoices_url = f"{alltalk_protocol}{alltalk_ip_port}/api/rvcvoices"
    settings_url = f"{alltalk_protocol}{alltalk_ip_port}/api/currentsettings"

    try:
        voices_response = requests.get(voices_url, timeout=connection_timeout)
        rvcvoices_response = requests.get(rvcvoices_url, timeout=connection_timeout)
        settings_response = requests.get(settings_url, timeout=connection_timeout)

        if voices_response.status_code == 200 and settings_response.status_code == 200:
            voices_data = voices_response.json()
            rvcvoices_data = rvcvoices_response.json() 
            settings_data = settings_response.json()

            engines_available = sorted(settings_data["engines_available"])
            current_engine_loaded = settings_data["current_engine_loaded"]
            models_available = sorted([model["name"] for model in settings_data["models_available"]])
            current_model_loaded = settings_data["current_model_loaded"]

            return {
                "voices": sorted(voices_data["voices"], key=lambda s: s.lower()),
                "rvcvoices": (rvcvoices_data["rvcvoices"]),
                "engines_available": engines_available,
                "current_engine_loaded": current_engine_loaded,
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
                "engines_available": ["Please Refresh Settings"],
                "current_engine_loaded": ["Please Refresh Settings"],
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
            "engines_available": ["Please Refresh Settings"],
            "current_engine_loaded": ["Please Refresh Settings"],
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

at_settings = get_alltalk_settings()   # Pull all the current settings from the AllTalk server, if its online.

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
            print(f"[{branding}TTS] \033[91mWarning\033[0m Failed to stop generation. Status code:\n{response.status_code}")
            return {"message": "Failed to stop generation"}
    except (RequestException, ConnectionError) as e:
        print(f"[{branding}TTS] \033[91mWarning\033[0m Unable to connect to the {branding}server. Status code:\n{str(e)}")
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
        print(f"[{branding}TTS] \033[91mWarning\033[0m Error during request to webserver process: Status code:\n{e}")
        return {"status": "error", "message": str(e)}

####################################################################################################
# TGWUI # Saves all settings back to the config file for the REMOTE version of the TGWUI extension #
####################################################################################################
def tgwui_save_settings():
    # Read the existing settings from the file
    try:
        with open(config_file_path, "r") as file:
            existing_settings = json.load(file)
    except FileNotFoundError:
        existing_settings = {}
    # Update the specific settings you want to change
    existing_settings.setdefault("tgwui", {})
    existing_settings["tgwui"]["tgwui_activate_tts"] = params["tgwui"]['tgwui_activate_tts']
    existing_settings["tgwui"]["tgwui_autoplay_tts"] = params["tgwui"]['tgwui_autoplay_tts']
    existing_settings["tgwui"]["tgwui_narrator_enabled"] = params["tgwui"]['tgwui_narrator_enabled']
    existing_settings["tgwui"]["tgwui_non_quoted_text_is"] = params["tgwui"]["tgwui_non_quoted_text_is"]
    existing_settings["tgwui"]["tgwui_language"] = params["tgwui"]['tgwui_language']
    existing_settings["tgwui"]["tgwui_temperature_set"] = params["tgwui"]['tgwui_temperature_set']
    existing_settings["tgwui"]["tgwui_repetitionpenalty_set"] = params["tgwui"]['tgwui_repetitionpenalty_set']
    existing_settings["tgwui"]["tgwui_generationspeed_set"] = params["tgwui"]['tgwui_generationspeed_set']
    existing_settings["tgwui"]["tgwui_pitch_set"] = params["tgwui"]['tgwui_pitch_set']
    existing_settings["tgwui"]["tgwui_narrator_voice"] = params["tgwui"]['tgwui_narrator_voice']
    existing_settings["tgwui"]["tgwui_show_text"] = params["tgwui"]['tgwui_show_text']
    existing_settings["tgwui"]["tgwui_character_voice"] = params["tgwui"]['tgwui_character_voice']
    # Save the updated settings back to the file
    with open(config_file_path, "w") as file:
        json.dump(existing_settings, file, indent=4)
        
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
def send_and_generate(gen_text, gen_character_voice, gen_narrartor_voice, gen_narrator_activated, gen_textnotinisde, gen_repetition, gen_language, gen_filter, gen_speed, gen_pitch, gen_autoplay, gen_autoplay_vol, gen_file_name, gen_temperature, gen_filetimestamp, gen_stream, gen_stopcurrentgen):
    api_url = f"{alltalk_protocol}{alltalk_ip_port}/api/tts-generate"
    if gen_text == "":
        print(f"[{branding}TTS] No Text was sent to generate as TTS")
        return None, str("No Text was sent to generate as TTS")
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
            "narrator_enabled": str(gen_narrator_activated).lower(),
            "narrator_voice_gen": gen_narrartor_voice,
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
        print(f"Debug: Generate request param:", data) if params["debug_tts"] else None
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
            print(f"[{branding}TTS] \033[91mWarning\033[0m Error occurred during the API request: Status code:\n{str(e)}")
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
    language_code = languages_list.get(params["tgwui"]["tgwui_language"])
    character_voice = params["tgwui"]["tgwui_character_voice"]
    narrator_voice = params["tgwui"]["tgwui_narrator_voice"]
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
            generate_response, status_message = send_and_generate(cleaned_text, character_voice, narrator_voice, narrator_enabled, text_not_inside, repetition_policy, language_code, text_filtering, speed, pitch, autoplay, autoplay_volume, output_file, temperature, True, False, False)
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
    with open(this_dir / "system" / "config" / "harvard_sentences.txt") as f:
        return random.choice(list(f)).rstrip()

def voice_preview(string):
    if not params["tgwui"]["tgwui_activate_tts"]:
        return string
    language_code = languages_list.get(params["tgwui"]["tgwui_language"])
    if not string:
        string = random_sentence()
    generate_response, status_message = send_and_generate(string, params["tgwui"]["tgwui_character_voice"], params["tgwui"]["tgwui_narrator_voice"], params["tgwui"]["tgwui_narrator_enabled"], "character", params["tgwui"]["tgwui_repetitionpenalty_set"], language_code, "standard", params["tgwui"]["tgwui_generationspeed_set"], params["tgwui"]["tgwui_pitch_set"], False, 0.8, "previewvoice", params["tgwui"]["tgwui_temperature_set"], False, False, False)
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
    if current_narrator_voice not in current_voices:
        current_narrator_voice = current_voices[0] if current_voices else ""

    tgwui_handle_ttsmodel_dropdown_change.skip_reload = True  # Debounce tgwui_tts_dropdown_gr and stop it sending a model reload when it is updated.

    return (
        gr.Checkbox(interactive=current_lowvram_capable, value=current_lowvram_enabled),
        gr.Checkbox(interactive=current_deepspeed_capable, value=current_deepspeed_enabled),
        gr.Dropdown(choices=current_voices, value=current_character_voice),
        gr.Dropdown(choices=current_voices, value=current_narrator_voice),
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
    global at_settings
    with gr.Accordion(params["branding"] + " TTS (Text-gen-webui Remote)"):
        # Activate alltalk_tts, Enable autoplay, Show text
        with gr.Row():
            tgwui_activate_tts_gr = gr.Checkbox(value=params["tgwui"]["tgwui_activate_tts"], label="Enable TGWUI TTS")
            tgwui_autoplay_gr = gr.Checkbox(value=params["tgwui"]["tgwui_autoplay_tts"], label="Autoplay TTS Generated")
            tgwui_show_text_gr = gr.Checkbox(value=params["tgwui"]["tgwui_show_text"], label="Show Text in chat")

        # Low vram enable, Deepspeed enable, Link
        with gr.Row():
            tgwui_lowvram_enabled_gr = gr.Checkbox(value=at_settings["lowvram_enabled"] if at_settings["lowvram_capable"] else False, label="Enable Low VRAM Mode", interactive=at_settings["lowvram_capable"])
            tgwui_lowvram_enabled_play_gr = gr.HTML(visible=False)
            tgwui_deepspeed_enabled_gr = gr.Checkbox(value=params["tgwui"]["tgwui_deepspeed_enabled"], label="Enable DeepSpeed", interactive=at_settings["deepspeed_capable"],)
            tgwui_deepspeed_enabled_play_gr = gr.HTML(visible=False)
            tgwui_empty_space_gr = gr.HTML(f"<p><a href='{alltalk_protocol}{alltalk_ip_port}'>AllTalk Server & Documentation Link</a><a href='{alltalk_protocol}{alltalk_ip_port}'></a>")

        # Model, Language, Character voice
        with gr.Row():
            tgwui_tts_dropdown_gr = gr.Dropdown(choices=models_available, label="TTS Engine/Model", value=current_model_loaded,)
            tgwui_language_gr = gr.Dropdown(languages_list.keys(), label="Languages" if at_settings["languages_capable"] else "Model not multi language", interactive=at_settings["languages_capable"], value=params["tgwui"]["tgwui_language"])
            tgwui_available_voices_gr = at_settings["voices"]
            tgwui_default_voice_gr = params["tgwui"]["tgwui_character_voice"]
            if tgwui_default_voice_gr not in tgwui_available_voices_gr:
                tgwui_default_voice_gr = tgwui_available_voices_gr[0] if tgwui_available_voices_gr else ""
            tgwui_character_voice_gr = gr.Dropdown(choices=tgwui_available_voices_gr, label="Character Voice", value=tgwui_default_voice_gr, allow_custom_value=True,)

        # Narrator
        with gr.Row():
            with gr.Row():
                tgwui_narrator_voice_gr = gr.Dropdown(tgwui_available_voices_gr, label="Narrator Voice", allow_custom_value=True, value=params["tgwui"]["tgwui_narrator_voice"],)
                tgwui_narrator_enabled_gr = gr.Dropdown(choices=[("Enabled", "true"), ("Disabled", "false"), ("Enabled (Silent)", "silent")], label="Narrator Enable", value="true" if params.get("tgwui_narrator_enabled") == "true" else ("silent" if params.get("tgwui_narrator_enabled") == "silent" else "false"))
                tgwui_non_quoted_text_is_gr = gr.Dropdown(choices=[("Character", "character"), ("Narrator", "narrator"), ("Silent", "silent")], label='Narrator unmarked text is', value=params.get("tgwui_non_quoted_text_is", "character"))

        # Temperature, Repetition Penalty, Speed, pitch
        with gr.Row():
            tgwui_temperature_set_gr = gr.Slider(minimum=0.05, maximum=1, step=0.05, label="Temperature", value=params["tgwui"]["tgwui_temperature_set"], interactive=at_settings["temperature_capable"])
            tgwui_repetitionpenalty_set_gr = gr.Slider(minimum=0.5, maximum=20, step=0.5, label="Repetition Penalty", value=params["tgwui"]["tgwui_repetitionpenalty_set"], interactive=at_settings["repetitionpenalty_capable"])
        with gr.Row():            
            tgwui_generationspeed_set_gr = gr.Slider(minimum=0.30, maximum=2.00, step=0.10, label="TTS Speed", value=params["tgwui"]["tgwui_generationspeed_set"], interactive=at_settings["generationspeed_capable"])
            tgwui_pitch_set_gr = gr.Slider(minimum=-10, maximum=10, step=1, label="Pitch", value=params["tgwui"]["tgwui_pitch_set"], interactive=at_settings["pitch_capable"])

        # Preview speech
        with gr.Row():
            tgwui_preview_text_gr = gr.Text(show_label=False, placeholder="Preview text", elem_id="silero_preview_text", scale=2,)
            tgwui_preview_play_gr = gr.Button("Generate Preview", scale=1)
            tgwui_preview_audio_gr = gr.HTML(visible=False)

        with gr.Row():
            tgwui_protocol_gr = gr.Dropdown(choices=["http://", "https://"], label="AllTalk Server Protocol", value=alltalk_protocol, interactive=False, visible=False)
            tgwui_protocol_gr.change(tgwui_update_alltalk_protocol, tgwui_protocol_gr, None)
            tgwui_ip_address_port_gr = gr.Textbox(label="AllTalk Server IP:Port", value=alltalk_ip_port, interactive=False, visible=False)
            tgwui_ip_address_port_gr.change(tgwui_update_alltalk_ip_port, tgwui_ip_address_port_gr, None)
            tgwui_convert_gr = gr.Button("Remove old TTS audio and leave only message texts")
            tgwui_convert_cancel_gr = gr.Button("Cancel", visible=False)
            tgwui_convert_confirm_gr = gr.Button("Confirm (cannot be undone)", variant="stop", visible=False)
            tgwui_stop_generation_gr = gr.Button("Stop current TTS generation")
            tgwui_stop_generation_gr.click(stop_generate_tts, None, None,)
            tgwui_refresh_settings_gr = gr.Button("Refresh settings & voices")
            tgwui_refresh_settings_gr.click(tgwui_update_dropdowns, None, [tgwui_lowvram_enabled_gr, tgwui_deepspeed_enabled_gr, tgwui_character_voice_gr, tgwui_narrator_voice_gr, tgwui_tts_dropdown_gr, tgwui_temperature_set_gr, tgwui_repetitionpenalty_set_gr, tgwui_language_gr, tgwui_generationspeed_set_gr, tgwui_pitch_set_gr, tgwui_non_quoted_text_is_gr])

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
    #tgwui_tts_dropdown_gr.change(send_reload_request, tgwui_tts_dropdown_gr, None)
    tgwui_tts_dropdown_gr.change(tgwui_handle_ttsmodel_dropdown_change, tgwui_tts_dropdown_gr, None)

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
    # Play preview
    tgwui_preview_text_gr.submit(voice_preview, tgwui_preview_text_gr, tgwui_preview_audio_gr)
    tgwui_preview_play_gr.click(voice_preview, tgwui_preview_text_gr, tgwui_preview_audio_gr)

##################################################################
#     _    _ _ _____     _ _       ____               _ _        #
#    / \  | | |_   _|_ _| | | __  / ___|_ __ __ _  __| (_) ___   #
#   / _ \ | | | | |/ _` | | |/ / | |  _| '__/ _` |/ _` | |/ _ \  #
#  / ___ \| | | | | (_| | |   <  | |_| | | | (_| | (_| | | (_) | #
# /_/   \_\_|_| |_|\__,_|_|_|\_\  \____|_|  \__,_|\__,_|_|\___/  #
#                                                                #
##################################################################

if gradio_enabled == True:
    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent
    my_current_url = "null"
    at_default_voice_gr = params["tgwui"]["tgwui_character_voice"]
    # Load the theme list and select a theme
    import system.gradio_pages.themes.loadThemes as loadThemes
    # Load the theme list from JSON file
    theme_list = loadThemes.get_list()

    # Load the selected theme from configuration
    selected_theme = loadThemes.load_json()

    # Check if the script is running in standalone mode
    if script_dir.name == "alltalk_tts":
        # Running in standalone mode, add the script's directory to the Python module search path
        sys.path.insert(0, str(script_dir))
    else:
        # Running as part of text-generation-webui, add the parent directory of "alltalk_tts" to the Python module search path
        sys.path.insert(0, str(script_dir.parent))

    import importlib
    import gradio as gr
    from system.gradio_pages.alltalk_documentation import alltalk_documentation
    from system.gradio_pages.alltalk_generation_help import alltalk_generation_help
    from system.gradio_pages.alltalk_about import alltalk_about
    from system.gradio_pages.alltalk_diskspace import get_disk_interface
    from system.gradio_pages.api_documentation import api_documentation
    if params["firstrun_splash"]:
        from system.gradio_pages.alltalk_welcome import alltalk_welcome
    
    def get_tts_engines_data():
        global engines_available, engine_loaded, selected_model
        tts_engines_file = os.path.join(this_dir, "system", "tts_engines", "tts_engines.json")
        with open(tts_engines_file, "r") as f:
            tts_engines_data = json.load(f)
        engines_available = [engine["name"] for engine in tts_engines_data["engines_available"]]
        engine_loaded = tts_engines_data["engine_loaded"]
        selected_model = tts_engines_data["selected_model"]
        return engines_available, engine_loaded, selected_model

    get_tts_engines_data()
    # Dynamically import modules and load JSON data for each available engine
    for engine_name in engines_available:
        module_name = f"system.tts_engines.{engine_name}.{engine_name}_settings_page"
        module = importlib.import_module(module_name)
        globals()[f"{engine_name}_at_gradio_settings_page"] = getattr(module, f"{engine_name}_at_gradio_settings_page")
        globals()[f"{engine_name}_model_update_settings"] = getattr(module, f"{engine_name}_model_update_settings")
        json_file_path = os.path.join(this_dir, "system", "tts_engines", engine_name, "model_settings.json")
        with open(json_file_path, "r") as f:
            globals()[f"{engine_name}_model_config_data"] = json.load(f)

    def confirm(message):
        return gr.Interface.fn("confirmation", f"""<script>
            var confirmation = confirm("{message}");
            gr.Interface.send(confirmation);
        </script>""")

    def save_config_data():
        try:
            # Save the updated JSON data to confignew.json
            with open(os.path.join(this_dir, "confignew.json"), "w") as f:
                json.dump(params, f, indent=4)
            reload_config()
        except Exception as e:
            error_message = "Failed to save config data."
            print(f"[{branding}TTS] \033[91mError:\033[0m {error_message}")
            print(f"[{branding}TTS] \033[91mError Details:\033[0m {str(e)}")

    def reload_config():
        # Send a GET request to the /reload_config endpoint
        reload_config_url = f"http://localhost:{params['api_def']['api_port_number']}/api/reload_config"
        try:
            response = requests.get(reload_config_url)
            if response.status_code == 200:
                print(f"[{branding}TTS] Reloaded config")
            else:
                print(f"[{branding}TTS] Failed to reload config file")
        except ConnectionError as e:
            error_message = "Connection to the server has been lost. Please reload the webpage."
            print(f"[{branding}TTS] \033[91mError:\033[0m {error_message}")
        except Exception as e:
            error_message = "An error occurred while reloading the config file."
            print(f"[{branding}TTS] \033[91mError:\033[0m {error_message}")
            print(f"[{branding}TTS] \033[91mError Details:\033[0m {str(e)}")
    
    ##########################################################################################
    # Pulls the current AllTalk Server settings & updates gradio when Refresh button pressed #
    ##########################################################################################
    def at_update_dropdowns():
        global at_settings
        at_settings = get_alltalk_settings()   # Pull all the current settings from the AllTalk server, if its online.
        engines_available = at_settings["engines_available"]
        current_engine_loaded = at_settings["current_engine_loaded"]
        current_voices = at_settings["voices"]
        current_character_voice = params["tgwui"]["tgwui_character_voice"]
        current_narrator_voice = params["tgwui"]["tgwui_narrator_voice"]
        rvccurrent_voices = at_settings["rvcvoices"]
        rvccurrent_character_voice = params["rvc_settings"]["rvc_char_model_file"]
        rvccurrent_narrator_voice = params["rvc_settings"]["rvc_narr_model_file"]
        current_temperature_capable = at_settings["temperature_capable"]
        current_repetitionpenalty_capable = at_settings["repetitionpenalty_capable"]
        current_languages_capable = at_settings["languages_capable"]
        current_generationspeed_capable = at_settings["generationspeed_capable"]
        current_pitch_capable = at_settings["pitch_capable"]
        current_languages_capable = at_settings["languages_capable"]
        if at_settings["languages_capable"]:
            language_label = "Languages"
        else:
            language_label = "Model not multi language"
        if at_settings["streaming_capable"]:
            gen_choices = [("Standard", "false"), ("Streaming (Disable Narrator)", "true")]
        else:
            gen_choices = [("Standard", "false")]
        if current_character_voice not in current_voices:
            current_character_voice = current_voices[0] if current_voices else ""
        if current_narrator_voice not in current_voices:
            current_narrator_voice = current_voices[0] if current_voices else ""
        if rvccurrent_character_voice not in rvccurrent_voices:
            rvccurrent_character_voice = rvccurrent_voices[0] if rvccurrent_voices else ""
        if rvccurrent_narrator_voice not in rvccurrent_voices:
            rvccurrent_narrator_voice = rvccurrent_voices[0] if rvccurrent_voices else ""
        return (
            gr.Dropdown(choices=gen_choices, interactive=True),
            gr.Dropdown(choices=current_voices, value=current_character_voice, interactive=True),
            gr.Dropdown(choices=rvccurrent_voices, value=rvccurrent_character_voice, interactive=True),
            gr.Dropdown(choices=current_voices, value=current_narrator_voice, interactive=True),
            gr.Dropdown(choices=rvccurrent_voices, value=rvccurrent_narrator_voice, interactive=True),
            gr.Slider(interactive=current_generationspeed_capable),
            gr.Slider(interactive=current_pitch_capable),
            gr.Slider(interactive=current_temperature_capable),
            gr.Slider(interactive=current_repetitionpenalty_capable),
            gr.Dropdown(interactive=current_languages_capable, label=language_label),
            gr.Dropdown(choices=models_available, value=current_model_loaded),
            gr.Dropdown(choices=engines_available, value=current_engine_loaded),
        ) 
   
    ######################################################################################
    # Sends request to reload the current TTS engine & set the default TTS engine loaded #
    ######################################################################################
    def set_engine_loaded(engine_name):
        global engine_loaded, selected_model, tts_engines_data, models_available, at_update_dropdowns
        tts_engines_file = os.path.join(this_dir, "system", "tts_engines", "tts_engines.json")
        with open(tts_engines_file, "r") as f:
            tts_engines_data = json.load(f)
        for engine in tts_engines_data["engines_available"]:
            if engine["name"] == engine_name:
                tts_engines_data["engine_loaded"] = engine_name
                tts_engines_data["selected_model"] = engine["selected_model"]
                break
        with open(tts_engines_file, "w") as f:
            json.dump(tts_engines_data, f)
        engine_loaded = engine_name
        selected_model = tts_engines_data["selected_model"]
        # Restart the subprocess
        restart_subprocess()
        # Wait for the engine to be ready with error handling and retries
        max_retries = 40
        retry_delay = 1  # seconds
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url)
                if response.status_code == 200 and response.text == "Ready":
                    break
            except requests.exceptions.RequestException:
                pass
            
            retries += 1
            if retries == max_retries:
                raise Exception("Failed to connect to the TTS engine after multiple retries.")
            
            time.sleep(retry_delay)
        models_available = at_settings["models_available"]
        # Update the dropdowns directly
        print(f"[{branding}ENG]")
        print(f"[{branding}ENG] {branding}Server Ready")
        gen_stream_value, gen_char_choices, rvcgen_char_choices, gen_narr_choices, rvcgen_narr_choices, gen_speed_interactive, gen_pitch_interactive, gen_temperature_interactive, gen_repetition_interactive, gen_lang_interactive, model_choices_gr_interactive, engine_choices_interactive = at_update_dropdowns()
        return ("TTS Engine changed successfully!",  # This is your output message
        gen_stream_value, gen_char_choices, rvcgen_char_choices, gen_narr_choices, rvcgen_narr_choices, gen_speed_interactive, gen_pitch_interactive, gen_temperature_interactive, gen_repetition_interactive, gen_lang_interactive, model_choices_gr_interactive, engine_choices_interactive
        )
        
    ###############################
    # Sends voice2rvc request off #
    ###############################
    def voice2rvc(audio, rvc_voice):
        # Save the uploaded or recorded audio to a file
        input_tts_path = this_dir / "outputs" / "voice2rvcInput.wav"
        if rvc_voice == "Disabled":
            print(f"[{branding}ENG] Voice2RVC Convert: No RVC voice was selected")
            return
        if audio == None:
            print(f"[{branding}ENG] Voice2RVC Convert: No recorded audio was provided")
            return       
        if isinstance(audio, tuple):
            sample_rate, audio_data = audio
            # Save the numpy array as a wav file
            sf.write(input_tts_path, audio_data, sample_rate)
        else:
            # It's a file path
            os.rename(audio, input_tts_path)
        
        # Define the output path for the processed audio
        output_rvc_path = this_dir / "outputs" / "voice2rvcOutput.wav"
        url = f"{alltalk_protocol}{alltalk_ip_port}/api/voice2rvc"

        # Submit the paths to the API endpoint
        response = requests.post(url, data={
            "input_tts_path": str(input_tts_path),
            "output_rvc_path": str(output_rvc_path),
            "pth_name": rvc_voice
        })

        if response.status_code == 200:
            result = response.json()
            if result["status"] == "success":
                result_path = result["output_path"]
            else:
                result_path = None
        else:
            result_path = None
        
        return result_path       
            
    ##################################################################################################
    # Sends request to reload the current TTS engine model & set the default TTS engine model loaded #
    ##################################################################################################
    def change_model_loaded(engine_name, selected_model):
        global at_update_dropdowns
        try:
            print(f"[{branding}ENG]")
            print(f"[{branding}ENG] \033[94mChanging model loaded. Please wait.\033[00m")
            print(f"[{branding}ENG]")
            url = f"{alltalk_protocol}{alltalk_ip_port}/api/reload"
            payload = {"tts_method": selected_model}
            response = requests.post(url, params=payload)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            # Update the tts_engines.json file
            tts_engines_file = os.path.join(this_dir, "system", "tts_engines", "tts_engines.json")
            with open(tts_engines_file, "r") as f:
                tts_engines_data = json.load(f)
            tts_engines_data["selected_model"] = f"{selected_model}"
            for engine in tts_engines_data["engines_available"]:
                if engine["name"] == engine_name:
                    engine["selected_model"] = f"{selected_model}"
                    break
            with open(tts_engines_file, "w") as f:
                json.dump(tts_engines_data, f)
            print(f"[{branding}ENG]")
            print(f"[{branding}ENG] {branding}Server Ready")
            gen_stream_value, gen_char_choices, rvcgen_char_choices, gen_narr_choices, rvcgen_narr_choices, gen_speed_interactive, gen_pitch_interactive, gen_temperature_interactive, gen_repetition_interactive, gen_lang_interactive, model_choices_gr_interactive, engine_choices_interactive = at_update_dropdowns() 
            return ("TTS Model changed successfully!",  # This is your output message
            gen_stream_value, gen_char_choices, rvcgen_char_choices, gen_narr_choices, rvcgen_narr_choices, gen_speed_interactive, gen_pitch_interactive, gen_temperature_interactive, gen_repetition_interactive, gen_lang_interactive, model_choices_gr_interactive, engine_choices_interactive
            )
        except requests.exceptions.RequestException as e:
            # Handle the HTTP request error
            print(f"[{branding}TTS] \033[91mWarning\033[0m Error during request to webserver process: Status code:\n{e}")
            return {"status": "error", "message": str(e)}

    def update_settings_tg(activate, autoplay, show_text, language, narrator_enabled):
        # Update the config_data dictionary with the new values
        params["activate"] = activate == "Enabled"
        params["autoplay"] = autoplay == "Enabled"
        params["show_text"] = show_text == "Enabled"
        params["language"] = language
        params["narrator_enabled"] = narrator_enabled
        print(f"[{branding}TTS] Default Text-gen-webui Settings Saved")
        save_config_data()
        return "Settings updated successfully!"

    debugging_options = params['debugging']
    debugging_choices = list(debugging_options.keys())
    default_values = [key for key, value in debugging_options.items() if value]

    def update_settings_at(delete_output_wavs, gradio_interface, gradio_port_number, output_folder, api_port_number, gr_debug_tts, transcode_audio_format):
        # Update the config_data dictionary with the new values
        params["delete_output_wavs"] = delete_output_wavs
        params["gradio_interface"] = gradio_interface == "Enabled"
        params["output_folder"] = output_folder
        params['api_def']['api_port_number'] = api_port_number
        params["gradio_port_number"] = gradio_port_number
        for key in debugging_options.keys():
            params['debugging'][key] = key in gr_debug_tts
        params["transcode_audio_format"] = transcode_audio_format
        output_message = "Settings updated successfully!"
        print(f"[{branding}TTS] Default Settings Saved")
        save_config_data()
        return output_message
    
    def update_settings_api(api_length_stripping, api_legacy_ip_address, api_allowed_filter, api_max_characters, api_use_legacy_api, api_text_filtering, api_narrator_enabled, api_text_not_inside, api_language, api_output_file_name, api_output_file_timestamp, api_autoplay, api_autoplay_volume):
        # Update the params dictionary with the new values
        params["api_def"]["api_length_stripping"] = api_length_stripping
        params["api_def"]["api_allowed_filter"] = api_allowed_filter
        params["api_def"]["api_max_characters"] = api_max_characters
        params["api_def"]["api_use_legacy_api"] = api_use_legacy_api == "Legacy API"
        params["api_def"]["api_legacy_ip_address"] = api_legacy_ip_address
        params["api_def"]["api_text_filtering"] = api_text_filtering
        params["api_def"]["api_narrator_enabled"] = api_narrator_enabled
        params["api_def"]["api_text_not_inside"] = api_text_not_inside
        params["api_def"]["api_language"] = api_language
        params["api_def"]["api_output_file_name"] = api_output_file_name
        params["api_def"]["api_output_file_timestamp"] = api_output_file_timestamp == "Timestamp files"
        params["api_def"]["api_autoplay"] = api_autoplay == "Play remotely"
        params["api_def"]["api_autoplay_volume"] = api_autoplay_volume
        output_message = "Default API settings updated successfully!"
        print(f"[{branding}TTS] API Settings Saved")
        save_config_data()
        return output_message

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def load_file_urls(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def download_file(url, dest_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def update_rvc_settings(rvc_enabled, rvc_char_model_file, rvc_narr_model_file, split_audio, autotune, pitch,
                            filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method, embedder_model, training_data_size, progress=gr.Progress(track_tqdm=True)):
        params["rvc_settings"]["rvc_enabled"] = rvc_enabled
        params["rvc_settings"]["rvc_char_model_file"] = rvc_char_model_file
        params["rvc_settings"]["rvc_narr_model_file"] = rvc_narr_model_file
        params["rvc_settings"]["split_audio"] = split_audio
        params["rvc_settings"]["autotune"] = autotune
        params["rvc_settings"]["pitch"] = pitch
        params["rvc_settings"]["filter_radius"] = filter_radius
        params["rvc_settings"]["index_rate"] = index_rate
        params["rvc_settings"]["rms_mix_rate"] = rms_mix_rate
        params["rvc_settings"]["protect"] = protect
        params["rvc_settings"]["hop_length"] = hop_length
        params["rvc_settings"]["f0method"] = f0method
        params["rvc_settings"]["embedder_model"] = embedder_model
        params["rvc_settings"]["training_data_size"] = training_data_size
        if rvc_enabled:
            base_dir = os.path.join(this_dir, "models", "rvc_base")
            rvc_voices_dir = os.path.join(this_dir, "models", "rvc_voices")
            ensure_directory_exists(base_dir)
            ensure_directory_exists(rvc_voices_dir)
            json_path = os.path.join(this_dir, "system", "tts_engines", "rvc_files.json")
            file_urls = load_file_urls(json_path)
            for idx, file in enumerate(file_urls):
                if not os.path.exists(os.path.join(base_dir, file)):
                    progress(idx / len(file_urls), desc=f"Downloading Required Files: {file}...")
                    print(f"[{branding}TTS] Downloading {file}...")  # Print statement for terminal
                    download_file(file_urls[file], os.path.join(base_dir, file))
            result = "Files downloaded successfully." if len(file_urls) > 0 else "All files are present."
            print(f"[{branding}TTS] {result}")
        print(f"[{branding}TTS] RVC Settings Saved")
        save_config_data()
        return "RVC settings updated successfully!"

    def generate_tts(gen_text, gen_char, rvcgen_char, gen_narr, rvcgen_narr, gen_narren, gen_textni, gen_repetition, gen_lang, gen_filter, gen_speed, gen_pitch, gen_autopl, gen_autoplvol, gen_filen, gen_temperature, gen_filetime, gen_stream, gen_stopcurrentgen):
        api_url = f"http://{my_current_url}/api/tts-generate"
        if gen_text == "":
            print(f"[{branding}TTS] No Text was sent to generate as TTS")
            return None, str("No Text was sent to generate as TTS")
        if gen_stopcurrentgen:
            stop_generate_tts()
        if gen_stream == "true":
            api_url = f"http://{my_current_url}/api/tts-generate-streaming"
            encoded_text = requests.utils.quote(gen_text)
            streaming_url = f"{api_url}?text={encoded_text}&voice={gen_char}&language={gen_lang}&output_file={gen_filen}"
            return streaming_url, str("TTS Streaming Audio Generated")
        else:
            data = {
                "text_input": gen_text,
                "text_filtering": gen_filter,
                "character_voice_gen": gen_char,
                "rvccharacter_voice_gen": rvcgen_char,
                "narrator_enabled": str(gen_narren).lower(),
                "narrator_voice_gen": gen_narr,
                "rvcnarrator_voice_gen": rvcgen_narr,
                "text_not_inside": gen_textni,
                "language": gen_lang,
                "output_file_name": gen_filen,
                "output_file_timestamp": str(gen_filetime).lower(),
                "autoplay": str(gen_autopl).lower(),
                "autoplay_volume": str(gen_autoplvol),
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
                if gen_autopl == "true":
                    return None, str("TTS Audio Generated (Played remotely)")
                else:
                    if params["api_def"]["api_use_legacy_api"]:
                        return result['output_file_url'], str("TTS Audio Generated")
                    else:
                        # Set the protocol type
                        protocol = "http://"  # or "https://" if using HTTPS
                        # Prepend the URL and PORT to the output_file_url
                        output_file_url = f"{protocol}{my_current_url}{result['output_file_url']}"
                        return output_file_url, str("TTS Audio Generated")
            except requests.exceptions.RequestException as e:
                error_message = "An error occurred. Please see console output."
                print(f"[{branding}TTS] \033[91mError:\033[0m {error_message}")
                print(f"[{branding}TTS] \033[91mError Details:\033[0m {str(e)}")
                # Handle the error or return an appropriate error message
                return None, str(error_message)

    def alltalk_gradio():
        global languages_list, at_settings
        # Get the URL IP or domain name
        def get_domain_name(request: gr.Request):
            global my_current_url
            if request:
                host = request.headers.get("host", "Unknown")
                my_current_url = host.split(":")[0]  # Split the host by ":" and take the first part
                my_current_url = my_current_url + ":" + str(params['api_def']['api_port_number'])
                return None
            else:
                return "Unable to retrieve the domain name."
            
        # Get the list of languages from languages
        languages = list(languages_list.keys())
        with gr.Blocks(theme=selected_theme, title="AllTalk") as app:
            with gr.Row():
                gr.Markdown("## AllTalk TTS")
                gr.Markdown("")
                gr.Markdown("")
                dark_mode_btn = gr.Button("Light/Dark Mode", variant="primary", size="sm")
                dark_mode_btn.click(None, None, None,
                js="""() => {
                    if (document.querySelectorAll('.dark').length) {
                        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                        // localStorage.setItem('darkMode', 'disabled');
                    } else {
                        document.querySelector('body').classList.add('dark');
                        // localStorage.setItem('darkMode', 'enabled');
                    }
                }""", show_api=False)
            if params["firstrun_splash"]:
                alltalk_welcome()
            with gr.Tab("Generate TTS"):
                with gr.Tab("Generate"):
                    with gr.Row():
                        gen_text = gr.Textbox(label="Text Input", lines=10)
                    with gr.Row():
                        with gr.Group():
                            with gr.Row():
                                engine_choices = gr.Dropdown(choices=engines_available, label="TTS Engine", value=engine_loaded, show_label=False)
                                engine_btn = gr.Button("Swap TTS Engine")
                        with gr.Group():
                            with gr.Row():
                                model_choices_gr = gr.Dropdown(choices=models_available, label="TTS Models", value=current_model_loaded, interactive=True, show_label=False, allow_custom_value=True)
                                model_btn_gr = gr.Button("Load Different Model")
                    with gr.Group():
                        at_available_voices_gr = at_settings["voices"]
                        rvcat_available_voices_gr = at_settings["rvcvoices"]
                        with gr.Row():
                            at_default_voice_gr = params["tgwui"]["tgwui_character_voice"]
                            if at_default_voice_gr not in at_available_voices_gr:
                                at_default_voice_gr = at_available_voices_gr[0] if at_available_voices_gr else ""
                            gen_char = gr.Dropdown(choices=at_available_voices_gr, label="Character Voice", value=at_default_voice_gr, allow_custom_value=True,)
                            rvcat_default_voice_gr = params["rvc_settings"]["rvc_char_model_file"]
                            if rvcat_default_voice_gr not in rvcat_available_voices_gr:
                                rvcat_default_voice_gr = rvcat_available_voices_gr[0] if rvcat_available_voices_gr else ""
                            rvcgen_char = gr.Dropdown(choices=rvcat_available_voices_gr, label="RVC Character Voice", value=rvcat_default_voice_gr, allow_custom_value=True,)    
                            at_narrator_voice_gr = params["tgwui"]["tgwui_narrator_voice"]
                            if at_narrator_voice_gr not in at_available_voices_gr:
                                at_narrator_voice_gr = at_available_voices_gr[0] if at_available_voices_gr else ""
                            gen_narr = gr.Dropdown(choices=at_available_voices_gr, label="Narrator Voice", value=at_narrator_voice_gr, allow_custom_value=True,)         
                            rvcat_narrator_voice_gr = params["rvc_settings"]["rvc_narr_model_file"]
                            if rvcat_narrator_voice_gr not in rvcat_available_voices_gr:
                                rvcat_narrator_voice_gr = rvcat_available_voices_gr[0] if rvcat_available_voices_gr else ""
                            rvcgen_narr = gr.Dropdown(choices=rvcat_available_voices_gr, label="RVC Narrator Voice", value=rvcat_narrator_voice_gr, allow_custom_value=True,)                   
                        with gr.Row():
                            #Get the current URL from the page
                            domain_name_output = gr.Textbox(label="Domain Name", visible=False)
                            def on_load(request: gr.Request):
                                domain_name = get_domain_name(request)
                                domain_name_output.value = domain_name
                            app.load(on_load, inputs=None, outputs=None)
                            if at_settings["streaming_capable"]:
                                gen_choices = [("Standard", "false"), ("Streaming (Disable Narrator)", "true")]
                            else:
                                gen_choices = [("Standard", "false")]
                            #Continue on with the Gradio interface
                            gen_stream = gr.Dropdown(choices=gen_choices, label="Generation Mode", value="false")
                            gen_lang = gr.Dropdown(value=params['api_def']['api_language'], choices=["ar", "zh", "cs", "nl", "en", "fr", "de", "hi", "hu", "it", "ja", "ko", "pl", "pt", "ru", "es", "tr"], label="Languages" if at_settings["languages_capable"] else "Model not multi language", interactive=at_settings["languages_capable"])
                            gen_narren = gr.Dropdown(choices=[("Enabled", "true"), ("Disabled", "false"), ("Enabled (Silent)", "silent")], label="Narrator Enabled/Disabled", value="true" if params['api_def']['api_narrator_enabled'] == "true" else ("silent" if params['api_def']['api_narrator_enabled'] == "silent" else "false"))                   
                            gen_textni = gr.Dropdown(choices=[("Character", "character"), ("Narrator", "narrator"), ("Silent", "silent")], label="Narrator Text-not-inside", value=params['api_def']['api_text_not_inside'])
                            gen_stopcurrentgen = gr.Dropdown(choices={("Stop", "true"), ("Dont stop", "false")}, label="Auto-Stop current generation", value="true")  
                        with gr.Row():
                            gen_filter = gr.Dropdown(value=params['api_def']['api_text_filtering'], label="Text filtering", choices=["none", "standard", "html"])
                            gen_filetime = gr.Dropdown(choices={("Timestamp files", "true"), ("Dont Timestamp (Over-write)", "false")}, label="Include Timestamp", value="true" if params['api_def']['api_output_file_timestamp'] else "false")                   
                            gen_autopl = gr.Dropdown(choices={("Play locally", "false"), ("Play remotely", "true")}, label="Play Locally or Remotely", value="true" if params['api_def']['api_autoplay'] else "false")
                            gen_autoplvol = gr.Dropdown(choices=[str(i / 10) for i in range(11)], value=str(params['api_def']['api_autoplay_volume']), label="Remote play volume", allow_custom_value=True)                   
                            gen_filen = gr.Textbox(value=params['api_def']['api_output_file_name'], label="Output File Name")
                        with gr.Row():                            
                            gen_speed = gr.Slider(minimum=0.25, maximum=2.00, step=0.25, label="Speed", value="1.00", interactive=at_settings["generationspeed_capable"])
                            gen_pitch = gr.Slider(minimum=-10, maximum=10, step=1, label="Pitch", value="1", interactive=at_settings["pitch_capable"])
                            gen_temperature = gr.Slider(minimum=0.05, maximum=1.0, step=0.05, label="Temperature", value=0.75, interactive=at_settings["temperature_capable"])
                            gen_repetition = gr.Slider(minimum=1.0, maximum=20.0, step=1.0, label="Repetition Penalty", value=10, interactive=at_settings["repetitionpenalty_capable"])                        
                    #Toggle narrator selection on Streaming select
                    def update_narren_and_autopl(gen_stream):
                        if gen_stream == "true":
                            return "false", "false"
                        else:
                            return gen_narren.value, gen_autopl.value
                    gen_stream.change(update_narren_and_autopl, inputs=[gen_stream], outputs=[gen_narren, gen_autopl])
                    with gr.Row():
                        output_audio = gr.Audio(show_label=False, label="Generated Audio", autoplay=True, scale=3)
                        output_message = gr.Textbox( label="TTS Result", lines=5, scale=1) 
                    with gr.Row():
                        dark_mode_btn = gr.Button("Light/Dark Mode", variant="primary")
                        refresh_button = gr.Button("Refresh Server Settings", elem_id="refresh_button")
                        stop_button = gr.Button("Interupt TTS Generation")
                        submit_button = gr.Button("Generate TTS")

                    model_btn_gr.click(fn=change_model_loaded, inputs=[engine_choices, model_choices_gr], outputs=[output_message, gen_stream, gen_char, rvcgen_char, gen_narr, rvcgen_narr, gen_speed, gen_pitch, gen_temperature, gen_repetition, gen_lang, model_choices_gr, engine_choices])
                    engine_btn.click(fn=set_engine_loaded, inputs=[engine_choices], outputs=[output_message, gen_stream, gen_char, rvcgen_char, gen_narr, rvcgen_narr, gen_speed, gen_pitch, gen_temperature, gen_repetition, gen_lang, model_choices_gr, engine_choices])
                    
                    dark_mode_btn.click(None, None, None,
                        js="""() => {
                            if (document.querySelectorAll('.dark').length) {
                                document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                                // localStorage.setItem('darkMode', 'disabled');
                            } else {
                                document.querySelector('body').classList.add('dark');
                                // localStorage.setItem('darkMode', 'enabled');
                            }
                        }""", show_api=False)
                    refresh_button.click(at_update_dropdowns, None, [gen_stream, gen_char, rvcgen_char, gen_narr, rvcgen_narr, gen_speed, gen_pitch, gen_temperature, gen_repetition, gen_lang, model_choices_gr, engine_choices])
                    stop_button.click(stop_generate_tts, inputs=[], outputs=[output_message])
                    submit_button.click(generate_tts, inputs=[gen_text, gen_char, rvcgen_char, gen_narr, rvcgen_narr, gen_narren, gen_textni, gen_repetition, gen_lang, gen_filter, gen_speed, gen_pitch, gen_autopl, gen_autoplvol, gen_filen, gen_temperature, gen_filetime, gen_stream, gen_stopcurrentgen], outputs=[output_audio, output_message])

                with gr.Tab("Generate Help"):
                    help_content = alltalk_generation_help()
                    gr.Markdown(help_content)

            with gr.Tab("Voice2RVC"):
                gr.Markdown("""Voice2RVC allows you to convert your spoken audio files into synthesized speech using advanced RVC (Retrieval-based Voice Conversion) models. You can either record your own speech or upload a pre-recorded audio file for processing. The tool offers features trim your input audio and undo changes if necessary. Simply record or upload your audio, select an RVC voice model, and submit it for processing. Once completed, you can download your synthesized speech.""")
                with gr.Row():
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record audio or Upload a spoken audio file")
                with gr.Row():
                    rvc_voices_dropdown = gr.Dropdown(choices=at_settings["rvcvoices"], label="Select RVC Voice to generate as", value=at_settings["rvcvoices"][0])
                    submit_button = gr.Button("Submit to RVC")
                audio_output = gr.Audio(label="Converted Audio")
                
                submit_button.click(fn=voice2rvc, inputs=[audio_input, rvc_voices_dropdown], outputs=audio_output)
                
            with gr.Tab("TTS Generator"):
                gr.Markdown("""With the TTS Generator you can create incredibly long audio e.g. entire books. Yet to be migrated into Gradio""")
                gr.Markdown("""Please find it on the web address http://127.0.0.1:7851/static/tts_generator/tts_generator.html (Assuming you have not changed your IP Address)""")
                
            with gr.Tab("Global Settings"):
                with gr.Tab("AllTalk Settings"):
                    with gr.Row():
                        delete_output_wavs = gr.Dropdown(value=params["delete_output_wavs"], label="Del WAV's older than", choices=["Disabled", "1 Day", "2 Days", "3 Days", "4 Days", "5 Days", "6 Days", "7 Days", "14 Days", "21 Days", "28 Days"])
                        api_port_number = gr.Number(value=int(params['api_def']['api_port_number']), label="API Port Number", precision=0)
                        gradio_port_number = gr.Number(value=int(params["gradio_port_number"]), label="Gradio Port Number", precision=0)
                        output_folder = gr.Textbox(value=params["output_folder"], label=f"Output Folder name (sub {branding})")
                    with gr.Row():
                        transcode_audio_format = gr.Dropdown(choices={"Disabled": "disabled", "aac": "aac", "flac": "flac", "mp3": "mp3", "opus": "opus", "wav": "wav"}, label="Audio Transcoding", value=params["transcode_audio_format"])
                        with gr.Row():
                            themes_select = gr.Dropdown(loadThemes.get_list(), value=loadThemes.read_json(), label="Gradio Theme Selection", visible=True,)
                            themes_select.change(fn=loadThemes.select_theme, inputs=[themes_select], outputs=[gr.Textbox(label="Gradio Selection Result")],)
                    with gr.Row():
                        with gr.Column():
                            gr_debug_tts = gr.CheckboxGroup(choices=debugging_choices, label="Debugging Options list", value=default_values)
                        with gr.Column():
                            gradio_interface = gr.Dropdown(choices={"Enabled": "true", "Disabled": "false"}, label="Gradio Interface", value="Enabled" if params["gradio_interface"] else "Disabled", info="**WARNING**: This will disable the AllTalk Gradio interface from loading. To re-enable the interface, go to the API address in a web browser and enable it there. http://127.0.0.1:7851/")
                    with gr.Row():
                        submit_button = gr.Button("Update Settings")
                        output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)

                    submit_button.click(update_settings_at, inputs=[delete_output_wavs, gradio_interface, gradio_port_number, output_folder, api_port_number, gr_debug_tts, transcode_audio_format], outputs=output_message)

                with gr.Tab("AllTalk API Defaults"):
                    gr.Markdown("""## &nbsp;&nbsp;API Version Settings""")
                    with gr.Group():
                        with gr.Row():
                            api_use_legacy_api = gr.Dropdown(choices=["AllTalk v2 API", "AllTalk v1 API (Legacy)"], label=f"{branding} API version", value="AllTalk v1 API (Legacy)" if params['api_def']['api_use_legacy_api'] else "AllTalk v2 API", scale=1)
                            gr.Textbox(value="Determines the API version to use. The legacy API includes the full URL (protocol, IP address, and port) in the output responses, while the new API returns only the relative path of the output file. Default: AllTalk v2 API", interactive=False, show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_legacy_ip_address = gr.Textbox(value=params['api_def']['api_legacy_ip_address'], label="AllTalk v1 API IP address")
                            gr.Textbox(value="Specifies the IP address to be included in the output responses when using the legacy API. Default: 127.0.0.1", interactive=False,show_label=False, lines=2, scale=4) 
                    gr.Markdown("""## &nbsp;&nbsp;API Default Settings""")
                    with gr.Group():                 
                        with gr.Row():
                            api_length_stripping =  gr.Slider(minimum=1, maximum=20, step=1, value=int(params['api_def']['api_length_stripping']), label="Strip sentences shorter than", scale=1)
                            gr.Textbox(value="Defines the minimum length of a sentence (in characters) that will be processed for text-to-speech. Sentences shorter than the X characters value will be filtered out by the Narrator to remove unwanted text characters. Default: 3", interactive=False,show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_max_characters = gr.Slider(minimum=50, maximum=10000, step=50, value=int(params['api_def']['api_max_characters']), label="Maximum amount of characters")
                            gr.Textbox(value="Sets the maximum number of characters allowed in a single text-to-speech generation request. Requests exceeding this limit will be rejected. Default: 2000", interactive=False,show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_text_filtering = gr.Dropdown(value=params['api_def']['api_text_filtering'], label="Text filtering", choices=["none", "standard", "html"])
                            gr.Textbox(value="Determines the text filtering method applied to the input text before processing. Available options are 'none' (no filtering), 'standard' (basic filtering), and 'html' (HTML-specific filtering). Default: Standard", interactive=False,show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_language = gr.Dropdown(value=params['api_def']['api_language'], label="Language", choices=["ar", "zh", "cs", "nl", "en", "fr", "de", "hi", "hu", "it", "ja", "ko", "pl", "pt", "ru", "es", "tr"])
                            gr.Textbox(value="Sets the default language for text-to-speech if no language is explicitly provided in the request. Default: en", interactive=False,show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_narrator_enabled = gr.Dropdown(
                                choices={("Enabled", "true"), ("Disabled", "false"), ("Enabled (Silent)", "silent")},
                                label="Narrator Enabled/Disable/Silent",
                                value=("true" if params['api_def']['api_narrator_enabled'] == "true"
                                    else "false" if params['api_def']['api_narrator_enabled'] == "false"
                                    else "silent")
                            )
                            gr.Textbox(value="Determines whether the narrator functionality is enabled by default when not explicitly specified in the request. Please note, if you set `Enabled` or `Enabled (silent)` as the APi defaults, then all text will go into the narrator function unless `disabled` is sent as part of the TTS generation request, possibly resulting in silenced TTS. Default: Disabled", interactive=False,show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_text_not_inside = gr.Dropdown(choices={"character", "narrator", "silent"}, label="Narrator Text-not-inside", value="character" if params['api_def']['api_text_not_inside'] else "narrator")
                            gr.Textbox(value="Defines how narrated text is split and processed when not explicitly specified in the request. The available options are 'character' (text is associated with the character) and 'narrator' (text is associated with the narrator). Default: Narrator", interactive=False,show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_output_file_name = gr.Textbox(value=params['api_def']['api_output_file_name'], label="Output file name")
                            gr.Textbox(value="Specifies the default name for the output file when no filename is provided in the request. Default: myoutputfile", interactive=False,show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_output_file_timestamp = gr.Dropdown(choices={"Timestamp files", "Dont Timestamp (Over-write)"}, label="Include Timestamp", value="Timestamp files" if params['api_def']['api_output_file_timestamp'] else "Dont Timestamp (Over-write)")
                            gr.Textbox(value="Determines whether a unique identifier (UUID) timestamp is appended to the generated text-to-speech output file. When enabled, each output file will have a unique timestamp, preventing overwriting of files. When disabled, files with the same name will be overwritten. Default: Timestamp files", interactive=False,show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_autoplay = gr.Dropdown(choices={"Play locally", "Play remotely"}, label="Play Locally or Remotely", value="Play remotely" if params['api_def']['api_autoplay'] else "Play locally")
                            gr.Textbox(value="Specifies whether the generated audio should be played locally on the client-side or remotely on the server-side console/terminal. Default: Play locally", interactive=False,show_label=False, lines=2, scale=4)
                        with gr.Row():
                            api_autoplay_volume = gr.Slider(minimum=0.1, maximum=0.9, step=0.1, label="Remote play volume", value=float(params['api_def']['api_autoplay_volume']))
                            gr.Textbox(value="Adjusts the volume level for audio playback when the 'Play Remotely' option is selected. The value ranges from 0.1 (lowest) to 0.9 (highest). Default: 0.9", interactive=False,show_label=False, lines=2, scale=4)
                    gr.Markdown("""## &nbsp;&nbsp;API Allowed Text Filtering/Passthrough Settings""")
                    with gr.Row():
                        api_allowed_filter = gr.Textbox(value=params['api_def']['api_allowed_filter'], label="Allowed text filter", show_label=False, lines=6, scale=1)
                        gr.Textbox(value="""Defines the set of characters and Unicode ranges that are permitted to be processed by the AllTalk TTS system. This filter ensures that only valid and supported characters are passed to the TTS engine or AI model for generation. The allowed characters include ASCII letters and digits, punctuation, whitespace, and various Unicode ranges for different languages and scripts.          

                        a-zA-Z0-9: ASCII letters and digits
                        .,;:!?: Punctuation characters
                        '": Single and double quotes
                        \s: Whitespace characters
                        \-: Hyphen/dash
                        $: Dollar sign
                        \\u00C0-\\u00FF: Latin characters with diacritics (À-ÿ)
                        \\u0400-\\u04FF: Cyrillic characters
                        \\u0900-\\u097F: Devanagari characters
                        \\u4E00-\\u9FFF: Chinese characters (CJK Unified Ideographs)
                        \\u3400-\\u4DBF: Chinese characters (CJK Unified Ideographs Extension A)
                        \\uF900-\\uFAFF: Chinese characters (CJK Compatibility Ideographs)
                        \\u0600-\\u06FF: Arabic characters (Arabic)
                        \\u0750-\\u077F: Arabic characters (Arabic Supplement)
                        \\uFB50-\\uFDFF: Arabic characters (Arabic Presentation Forms-A)
                        \\uFE70-\\uFEFF: Arabic characters (Arabic Presentation Forms-B)
                        \\u3040-\\u309F: Hiragana characters (Japanese)
                        \\u30A0-\\u30FF: Katakana characters (Japanese)
                        \\uAC00-\\uD7A3: Hangul Syllables (Korean)
                        \\u1100-\\u11FF: Hangul Jamo (Korean)
                        \\u3130-\\u318F: Hangul Compatibility Jamo (Korean)
                        \\u0150\\u0151\\u0170\\u0171: Hungarian characters
                        \\u2018\\u2019: Left and right single quotation marks
                        \\u201C\\u201D: Left and right double quotation marks
                        \\u3001\\u3002: Ideographic comma and full stop
                        \\uFF01\\uFF0C\\uFF1A\\uFF1B\\uFF1F: Fullwidth exclamation, comma, colon, semicolon & question mark
                        """, interactive=False,show_label=False, lines=32, scale=1)
                    with gr.Row():
                        submit_button = gr.Button("Update Settings")
                        output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False) 

                    submit_button.click(update_settings_api, inputs=[api_length_stripping, api_legacy_ip_address, api_allowed_filter, api_max_characters, api_use_legacy_api, api_text_filtering, api_narrator_enabled, api_text_not_inside, api_language, api_output_file_name, api_output_file_timestamp, api_autoplay, api_autoplay_volume], outputs=output_message)

                def rvc_update_dropdowns():
                    global at_settings
                    at_settings = get_alltalk_settings()   # Pull all the current settings from the AllTalk server, if its online.
                    current_voices = at_settings["rvcvoices"]
                    current_char = params["rvc_settings"]["rvc_char_model_file"]
                    current_narr = params["rvc_settings"]["rvc_narr_model_file"]
                    if current_char not in current_voices:
                        current_char = current_voices[0] if current_voices else ""
                    if current_narr not in current_voices:
                        current_narr = current_voices[0] if current_voices else ""
                    return (
                        gr.Dropdown(choices=current_voices, value=current_char, interactive=True),
                        gr.Dropdown(choices=current_voices, value=current_narr, interactive=True),
                    )

                def gr_update_rvc_settings(rvc_enabled, rvc_char_model_file, rvc_narr_model_file, split_audio, autotune, pitch,
                                        filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method, embedder_model, training_data_size):
                    progress = gr.Progress(track_tqdm=True)
                    return update_rvc_settings(rvc_enabled, rvc_char_model_file, rvc_narr_model_file, split_audio, autotune, pitch,
                                            filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method, embedder_model, training_data_size, progress)

                with gr.Tab("RVC Settings"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Row():
                                rvc_model_file_available = at_settings["rvcvoices"]
                                rvc_char_model_file_default = params["rvc_settings"]["rvc_char_model_file"]
                                if rvc_char_model_file_default not in rvc_model_file_available:
                                    rvc_char_model_file_default = rvc_model_file_available[0] if rvc_model_file_available else ""
                                rvc_char_model_file_gr = gr.Dropdown(choices=rvc_model_file_available, label="Default Character Voice Model", info="Select the Character voice model used for conversion.", value=rvc_char_model_file_default, allow_custom_value=True,)
                                rvc_narr_model_file_default = params["rvc_settings"]["rvc_narr_model_file"]
                                if rvc_narr_model_file_default not in rvc_model_file_available:
                                    rvc_narr_model_file_default = rvc_model_file_available[0] if rvc_model_file_available else ""
                                rvc_narr_model_file_gr = gr.Dropdown(choices=rvc_model_file_available, label="Default Narrator Voice Model", info="Select the Narrator voice model used for conversion.", value=rvc_narr_model_file_default, allow_custom_value=True,)        
                                rvc_refresh_button = gr.Button("Refresh Model Choices")
                        with gr.Column(scale=0):
                            rvc_enabled = gr.Checkbox(label="Enable RVC", info="RVC (Real-Time Voice Cloning) enhances TTS by replicating voice characteristics for characters or narrators, adding depth to synthesized speech.", value=params["rvc_settings"].get("rvc_enabled", False), interactive=True)
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            pitch = gr.Slider(minimum=-24, maximum=24, step=1, label="Pitch", info="Set the pitch of the audio, the higher the value, the higher the pitch.", value=params["rvc_settings"].get("pitch", 0), interactive=True,)
                        with gr.Column():
                            hop_length = gr.Slider(minimum=1, maximum=512, step=1, label="Hop Length", info="Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy.", value=params["rvc_settings"].get("hop_length", 128), interactive=True,)
                    with gr.Row(equal_height=True):       
                        with gr.Column():                     
                            training_data_size = gr.Slider(minimum=10000, maximum=100000, step=5000, label="Training Data Size",  info="Determines the number of training data points used to train the FAISS index. Increasing the size may improve the quality but can also increase computation time.", value=params["rvc_settings"].get("training_data_size", 25000), interactive=True)
                        with gr.Column():                     
                            index_rate = gr.Slider(minimum=0, maximum=1, label="Index Influence Ratio",  info="Sets the influence exerted by the index file on the final output. A higher value increases the impact of the index, potentially enhancing detail but also increasing the risk of artifacts.",  value=params["rvc_settings"].get("index_rate", 0.75), interactive=True)
                    with gr.Row(equal_height=True):    
                        with gr.Column():                            
                            rms_mix_rate = gr.Slider(minimum=0, maximum=1, label="Volume Envelope", info="Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed.", value=params["rvc_settings"].get("rms_mix_rate", 1), interactive=True,)
                        with gr.Column():
                            protect = gr.Slider(minimum=0, maximum=0.5, label="Protect Voiceless Consonants/Breath sounds", info="Prevents sound artifacts. Higher values (up to 0.5) provide stronger protection but may affect indexing.", value=params["rvc_settings"].get("protect", 0.5), interactive=True,)
                        with gr.Column():                            
                            filter_radius = gr.Slider(minimum=0, maximum=7, label="Filter Radius", info="If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration.", value=params["rvc_settings"].get("filter_radius", 3), step=1, interactive=True,)
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                embedder_model = gr.Radio(label="Embedder Model", info="Model used for learning speaker embedding.", choices=["hubert", "contentvec"], value=params["rvc_settings"].get("embedder_model", "hubert"), interactive=True,)
                            with gr.Row(): 
                                split_audio = gr.Checkbox(label="Split Audio", info="Split the audio into chunks for inference to obtain better results in some cases.", value=params["rvc_settings"].get("split_audio", False), interactive=True,)
                                autotune = gr.Checkbox(label="Autotune", info="Apply a soft autotune to your inferences, recommended for singing conversions.", value=params["rvc_settings"].get("autotune", False), interactive=True,)
                        with gr.Column():
                            f0method = gr.Radio(label="Pitch Extraction Algorithm", info="Select the algorithm to be used for extracting the pitch (F0) during audio conversion. The default algorithm is rmvpe, which is generally recommended for most cases due to its balance of accuracy and performance.", choices=["crepe", "crepe-tiny", "dio", "fcpe", "harvest", "hybrid[rmvpe+fcpe]", "pm", "rmvpe"], value=params["rvc_settings"].get("f0method", "rmvpe"), interactive=True,)
                    with gr.Row():
                        update_button = gr.Button("Update RVC Settings")
                        update_output = gr.Textbox(label="Update Status", show_label=False)
                    rvc_refresh_button.click(rvc_update_dropdowns, None, [rvc_char_model_file_gr, rvc_narr_model_file_gr])
                    update_button.click(fn=gr_update_rvc_settings, inputs=[rvc_enabled, rvc_char_model_file_gr, rvc_narr_model_file_gr, split_audio, autotune, pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method, embedder_model, training_data_size], outputs=[update_output])
        
                with gr.Tab("Text-generation-webui Settings"):
                    with gr.Row():
                        activate = gr.Dropdown(choices={"Enabled": "true", "Disabled": "false"}, label="Activate TTS", value="Enabled" if params["tgwui"]["tgwui_activate_tts"] else "Disabled")
                        autoplay = gr.Dropdown(choices={"Enabled": "true", "Disabled": "false"}, label="Autoplay TTS", value="Enabled" if params["tgwui"]["tgwui_autoplay_tts"] else "Disabled")
                        show_text = gr.Dropdown(choices={"Enabled": "true", "Disabled": "false"}, label="Show Text", value="Enabled" if params["tgwui"]["tgwui_show_text"] else "Disabled")
                        narrator_enabled = gr.Dropdown(choices=[("Enabled", "true"), ("Disabled", "false"), ("Enabled (Silent)", "silent")], label="Narrator enabled", value="true" if params["tgwui"]["tgwui_narrator_enabled"] == "true" else ("silent" if params["tgwui"]["tgwui_narrator_enabled"] == "silent" else "false"))
                        language = gr.Dropdown(value=params["tgwui"]["tgwui_language"], label="Default Language", choices=languages_list)
                    with gr.Row():
                        submit_button = gr.Button("Update Settings")
                        output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
                        
                    submit_button.click(update_settings_tg, inputs=[activate, autoplay, show_text, language, narrator_enabled], outputs=output_message)

                disk_space_page = get_disk_interface()
                disk_space_page()
                    

            with gr.Tab("TTS Engines Settings"):
                    with gr.Tabs():
                        for engine_name in engines_available:
                            with gr.Tab(f"{engine_name.capitalize()} TTS"):
                                gr.Markdown(f"### &nbsp;&nbsp;{engine_name.capitalize()} TTS")
                                globals()[f"{engine_name}_at_gradio_settings_page"](globals()[f"{engine_name}_model_config_data"])
                        
            alltalk_documentation()
            api_documentation()
            with gr.Tab("About this project"):
                alltalk_about()
        return app

    if __name__ == "__main__":
        app = alltalk_gradio().queue()
        app.launch(server_port=params['gradio_port_number'], prevent_thread_lock=True, quiet=True)

    if not running_in_standalone:
        app = alltalk_gradio().queue()
        app.launch(server_port=params['gradio_port_number'], prevent_thread_lock=True, quiet=True)

#########################################
# START-UP # Final Splash before Gradio #
#########################################
print(f"[{branding}TTS] Please use \033[91mCtrl+C\033[0m when exiting AllTalk otherwise a")
print(f"[{branding}TTS] subprocess may continue running in the background.")
print(f"[{branding}TTS]")
print(f"[{branding}TTS] {branding}Server Ready")

###############################################################################################
# START-UP # Loop to keep the script from exiting out if its being run as a standalone script #
###############################################################################################
if running_in_standalone:
    while True:
        try:
            time.sleep(1)  # Add a small delay to avoid high CPU usage
        except KeyboardInterrupt:
            break  # Allow graceful exit on Ctrl+C
