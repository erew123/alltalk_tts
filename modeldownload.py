import os
import sys
from pathlib import Path
import requests
from requests.exceptions import ConnectionError
from tqdm import tqdm
import importlib.metadata as metadata
import json
from packaging import version
from datetime import datetime

#################################################################
#### LOAD PARAMS FROM confignew.json - REQUIRED FOR BRANDING ####
#################################################################
# STARTUP VARIABLE - Create "this_dir" variable as the current script directory
this_dir = Path(__file__).parent.resolve()

# load standard config file in and get settings
def load_config(file_path):
    with open(file_path, "r") as config_file:
        config = json.load(config_file)
    return config

config_file_path = this_dir / "confignew.json"
params = load_config(config_file_path)

# Define the path to the modeldownload config file file
modeldownload_config_file_path = this_dir / "modeldownload.json"

# Check if the JSON file exists
if modeldownload_config_file_path.exists():
    with open(modeldownload_config_file_path, "r") as config_file:
        settings = json.load(config_file)

    # Extract settings from the loaded JSON
    base_path = Path(settings.get("base_path", ""))
    model_path = Path(settings.get("model_path", ""))
    files_to_download = settings.get("files_to_download", {})
else:
    # Default settings if the JSON file doesn't exist or is empty
    print(f"[{params['branding']}Startup] \033[91mWarning\033[0m modeldownload.json is missing so please re-download it and save it in the coquii_tts main folder.")
    print(f"[{params['branding']}Startup] \033[91mWarning\033[0m API Local and XTTSv2 Local will error unless this is corrected.")

# Read the version specifier from requirements_nvidia.txt
with open(this_dir / "system" / "requirements" / "requirements_standalone.txt", "r") as req_file:
    requirements = req_file.readlines()

tts_version_required = None
for req in requirements:
    if req.startswith("TTS=="):
        tts_version_required = req.strip().split("==")[1]
        break

if tts_version_required is None:
    raise ValueError(f"[{params['branding']}Startup] \033[91mWarning\033[0m Could not find TTS version specifier in requirements file")

def create_directory_if_not_exists(directory):
    if not directory.exists():
        directory.mkdir(parents=True)

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(destination, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()


def check_tts_version():
    try:
        tts_version = metadata.version("tts")
        print(f"[{params['branding']}Startup] \033[92mCurrent TTS Version    : \033[93m"+tts_version+"\033[0m")

        if version.parse(tts_version) < version.parse(tts_version_required):
            print(f"[{params['branding']}Startup] \033[91mWarning\033[0m TTS version is too old. Please upgrade to version \033[93m"+tts_version_required+"\033[0m or later.\033[0m")
            print(f"[{params['branding']}Startup] \033[91mWarning\033[0m At your terminal/command prompt \033[94mpip install --upgrade tts\033[0m")
        else:
            print(f"[{params['branding']}Startup] \033[92mCurrent TTS Version is :\033[93m Up to date\033[0m")
    except metadata.PackageNotFoundError:
        print(f"[{params['branding']}Startup] \033[91mWarning\033[0m TTS is not installed")

def check_torch_version():
    import torch
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda
    major, minor, micro = sys.version_info[:3]
    python_version = f"{major}.{minor}.{micro}"
    print(f"[{params['branding']}Startup] \033[92mCurrent Python Version :\033[93m {python_version}\033[0m")
    print(f"[{params['branding']}Startup] \033[92mCurrent PyTorch Version:\033[93m {pytorch_version}\033[0m")
    if cuda_version is None:
        print(f"[{params['branding']}Startup] \033[92mCurrent CUDA Version   :\033[91m CUDA is not available\033[0m")
    else:
        print(f"[{params['branding']}Startup] \033[92mCurrent CUDA Version   :\033[93m {cuda_version}\033[0m")

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
            print(f"[{params['branding']}Startup] \033[92m{params['branding']}Github updated :\033[91m Failed to fetch the latest commits due to an unexpected response from GitHub")
            return None, None
    except ConnectionError:
        # This block is executed when a connection error occurs
        print(f"[{params['branding']}Startup] \033[92m{params['branding']}Github updated :\033[91m Could not reach GitHub to check for updates\033[0m")
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

# Check and create directories
if str(base_path) == "models":
    create_directory_if_not_exists(this_dir / base_path / model_path)
else:
    create_directory_if_not_exists(base_path / model_path)
    print(f"[{params['branding']}Startup] \033[94mInfo\033[0m Custom path set in \033[93mmodeldownload.json\033[0m. Using the following settings:")
    print(f"[{params['branding']}Startup] \033[94mInfo\033[0m Base folder Path:\033[93m",base_path,"\033[0m",)
    print(f"[{params['branding']}Startup] \033[94mInfo\033[0m Model folder Path:\033[93m",model_path,"\033[0m",)
    print(f"[{params['branding']}Startup] \033[94mInfo\033[0m Full Path:\033[93m",base_path / model_path,"\033[0m",)

# Download files if they don't exist
print(f"[{params['branding']}Startup] \033[92mModel is available     :\033[93m Checking\033[0m")
for filename, url in files_to_download.items():
    if str(base_path) == "models":
        destination = this_dir / base_path / model_path / filename
    else:
        destination = Path(base_path) / model_path / filename
    if not destination.exists():
        print(f"[{params['branding']}Startup] \033[92mModel Downloading file : \033[93m" + filename + "\033[0m")
        download_file(url, destination)
print(f"[{params['branding']}Startup] \033[92mModel is available     :\033[93m Checked\033[0m")

github_site = "erew123"
github_repository = "alltalk_tts"

check_torch_version()
check_tts_version()
# Define the file path based on your directory structure
sha_file_path = this_dir / "system" / "config" / "at_github_sha.json"

# Read or initialize the SHA (adjusted for handling both SHA and date)
last_known_commit_sha = read_or_initialize_sha(sha_file_path)  # Assuming adjustment for date

# Assuming you have fetched the latest commit SHA and date
latest_commit_sha, latest_commit_date = fetch_latest_commit_sha_and_date(github_site, github_repository)

formatted_date = format_datetime(latest_commit_date) if latest_commit_date else "an unknown date"

if latest_commit_sha and latest_commit_sha != last_known_commit_sha:
    #print(f"There's an update available for alltalk_tts.")
    print(f"[{params['branding']}Startup] \033[92m{params['branding']}Github updated :\033[93m {formatted_date}\033[0m")
    # Update the file with the new SHA
    update_sha_file(sha_file_path, latest_commit_sha)
elif latest_commit_sha == last_known_commit_sha:
    #print(f"Your alltalk_tts software is up to date.")
    print(f"[{params['branding']}Startup] \033[92m{params['branding']}Github updated :\033[93m {formatted_date}\033[0m")