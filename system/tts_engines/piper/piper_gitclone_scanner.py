import os
import json
from urllib.parse import urljoin

REPO_PATH = "E:/alltalk_tts/models/piper/"
DISK_PATH_PREFIX = "/piper"
print(f"Current Working Directory: {os.getcwd()}")
print(f"Contents of {REPO_PATH}:")
for root, dirs, files in os.walk(REPO_PATH):
    print(f"Directory: {root}")
    for file in files:
        print(f"  File: {os.path.join(root, file)}")

HF_REPO_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"

def get_file_info(onnx_path, repo_url):
    base_name = os.path.splitext(os.path.basename(onnx_path))[0]
    rel_onnx_path = os.path.relpath(onnx_path, REPO_PATH).replace("\\", "/")
    download_url_onnx = urljoin(repo_url, rel_onnx_path) + "?download=true"
    json_path = None
    for root, dirs, files in os.walk(os.path.dirname(onnx_path)):
        for file in files:
            if file.endswith(".json") and base_name in file:
                json_path = os.path.join(root, file)
                rel_json_path = os.path.relpath(json_path, REPO_PATH).replace("\\", "/")
                download_url_json = urljoin(repo_url, rel_json_path) + "?download=true"
                break
    if json_path:
        return {
            "name": base_name,
            "files_download": [download_url_onnx, download_url_json],
        }
    else:
        return None

voice_files = []
for root, dirs, files in os.walk(REPO_PATH):
    for file in files:
        if file.endswith(".onnx"):
            onnx_path = os.path.join(root, file)
            print(f"Found ONNX file: {onnx_path}")
            file_info = get_file_info(onnx_path, HF_REPO_URL)
            if file_info:
                voice_files.append(file_info)
            else:
                print(f"No associated JSON file found for {onnx_path}")

if not voice_files:
    print("No ONNX/JSON file pairs found.")
else:
    print(f"Found {len(voice_files)} ONNX/JSON file pairs.")
    with open("voice_files.json", "w") as f:
        json.dump(voice_files, f, indent=2)