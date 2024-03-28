import json
import argparse
from pathlib import Path
import os
import glob
import platform
import site
import subprocess
import sys
# version 0.8

def find_files_in_path_with_wildcard(pattern):
    # Get the site-packages directory of the current Python environment
    site_packages_path = site.getsitepackages()
    found_paths = []
    # Adjust the sub-directory based on the operating system
    sub_directory = "nvidia/cublas"
    if platform.system() == "Linux":
        sub_directory = os.path.join(sub_directory, "lib")
    else:
        sub_directory = os.path.join(sub_directory, "bin")
    
    # Iterate over each site-packages directory (there can be more than one)
    for directory in site_packages_path:
        # Construct the search directory path
        search_directory = os.path.join(directory, sub_directory)
        # Use glob to find all files matching the pattern in this directory
        for file_path in glob.glob(os.path.join(search_directory, pattern)):
            if os.path.isfile(file_path):  # Ensure it's a file
                found_paths.append(file_path)
    return found_paths

def detect_cublas():
    # Use different file patterns based on the operating system
    file_name_pattern = 'cublas64_11.*' if platform.system() == "Windows" else 'libcublas.so.11*'
    found_paths = find_files_in_path_with_wildcard(file_name_pattern)
    if found_paths:
        print("[AllTalk TTSDiff] \033[94mCublas:\033[0m \033[92mDetected\033[0m", found_paths)
    else:
        print("[AllTalk TTSDiff] \033[94mCublas:\033[0m \033[91mERROR Not Detected\033[0m")
        print("[AllTalk TTSDiff] \033[94mPlease install Cublas from the Nvidia CUDA Toolkit \033[0mhttps://developer.nvidia.com/cuda-downloads")
        exit(1)
        

parser = argparse.ArgumentParser(description="Compare TTS output with the original text using detailed comparison.")
parser.add_argument("--threshold", type=int, default=98, help="Similarity threshold for considering a match (default: 98)")
parser.add_argument("--ttslistpath", help="Path to the ttsList.json file")
parser.add_argument("--wavfilespath", help="Path to the wav outputs folder")
args = parser.parse_args()

if not args.ttslistpath:
    parser.error("[AllTalk TTSDiff] Please specify the path to the ttslist.json file using the --ttslistpath argument.")
if not args.wavfilespath:
    parser.error("[AllTalk TTSDiff] Please specify the path to the wavs output folder file using the --wavfilespath argument.")

json_file_path = Path(args.ttslistpath).resolve()
wav_file_path = Path(args.wavfilespath).resolve()
accuracy_threshold = args.threshold

def disclaimer_text():
    print(f"\nDESCRIPTION: ")
    print(f"  This script is designed to assist in identifying bad TTS generation ID's from the TTS Generator.\n")
    print(f"  This will download the Whisper Large-v2 model (if not already downloaded), and use it to compare your text to")
    print(f"  to the generated TTS. If it finds an issue, it will flag the ID number that has the issue. This does NOT 100%")
    print(f"  ensure that all TTS generated is perfect, sounds good etc, but it should flag up glaring issues and help with")
    print(f"  the process of identifying bad TTS generations.")
    print(f"\nNOTES ON USE: ")
    print(f"  - Ensure AllTalk is running in its own command prompt/terminal.")
    print(f"  - This script should be running in its own command prompt/terminal window in the same Python Environment that")
    print(f"    AllTalk is running in.")
    print(f"  - After generating your text in the TTS Generator, you need to `Export List to JSON` and save the file in the")
    print(f"    same folder/directory this script is running from.")
    print(f"  - You can use `--threshold` with a numerical value from 1 to 100 to set accuracy detection. For example it may")
    print(f"    detect/transcribe `their` as `there`, which of course, both words sound the same. If you want 100% accuracy")
    print(f"    you will set 100 as the threshold. You will more than likely want a number in the high 90's. Default is set")
    print(f"    at 98, which gives a little flexibility.")
    print(f"  - The JSON file must be named \033[93mttsList.json\033[0m")
    print(f"  - When you have your ID list, go back into the TTS Generator, correct any lines and regenerate them. If you")
    print(f"    want to re-test everthing again after re-generating, you will need to export your list again and re-run this")
    print(f"    script again, against your newly exported JSON list.")
    print(f"  - This requires access to \033[93mcublas64_11\033[0m, the same as Finetuning.")
    print(f"    https://github.com/erew123/alltalk_tts/tree/main?#-important-requirements-cuda-118\n")
    return

try:
    from faster_whisper import WhisperModel
    import re
    import string
    import spacy
    import torch
    from fuzzywuzzy import fuzz
except ImportError as e:
    print(f"[AllTalk TTSDiff] ERROR STARTING SCRIPT:")
    print("[AllTalk TTSDiff] An error occurred importing one or more required libraries.")
    print("[AllTalk TTSDiff] Please ensure you have \033[94mactivated your Python environment \033[0mand \033[94minstalled all requirements.\033[0m")
    print("[AllTalk TTSDiff] \033[94mMissing library:\033[93m", str(e), "\033[0m")
    exit(1)

# Attempt to load the spaCy model
try:
    # Load a medium-sized spaCy model, adjust to "en_core_web_md" or "en_core_web_lg" as needed
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    print(f"[AllTalk TTSDiff] ERROR STARTING SCRIPT:")
    print("[AllTalk TTSDiff] Failed to load the spaCy language model.")
    
    # Attempt to download the spaCy model automatically
    try:
        print("[AllTalk TTSDiff] Attempting to automatically download the spaCy language model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"], check=True)
        
        # Try to load the model again after downloading
        nlp = spacy.load("en_core_web_md")
        print("[AllTalk TTSDiff] Model downloaded and loaded successfully.")
    except Exception as download_exception:
        print("[AllTalk TTSDiff] Automatic download failed.")
        print("[AllTalk TTSDiff] Please manually install the spaCy language model by running:")
        print("[AllTalk TTSDiff] \033[93mpython -m spacy download en_core_web_md\033[0m")
        exit(1)

def texts_are_similar(text1, text2, threshold=0.8):
    # Preliminary check for short texts
    if len(text1) < 10 or len(text2) < 10:
        ratio = fuzz.ratio(text1, text2) / 100.0  # Use FuzzyWuzzy for short texts
        return ratio > threshold

    # Process the texts through the NLP model for longer texts
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    # Compute semantic similarity
    similarity = doc1.similarity(doc2)
    
    return similarity >= threshold

def normalize_text(text):
    # Normalize or remove CRLF and other non-standard whitespaces
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    
    # Convert to lowercase
    text = text.lower()
    
    # Standardize and then remove quotation marks
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.translate(str.maketrans('', '', '"\''))
    
    # Remove all other punctuation except hyphens to preserve compound words
    text = text.translate(str.maketrans('', '', string.punctuation.replace("-", "")))
    
    # Collapse any sequence of whitespace (including spaces) into a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

spoken_punctuation_mapping = {
    "dot": ".",
    "comma": ",",
    # Add more mappings as needed
}

def contains_spoken_punctuation(transcribed_text):
    for spoken, punctuation in spoken_punctuation_mapping.items():
        if spoken in transcribed_text:
            return True
    return False

def detailed_comparison(original_text, transcribed_text, threshold=98):
    original_clean = normalize_text(original_text)
    transcribed_clean = normalize_text(transcribed_text)
    ratio = fuzz.ratio(original_clean, transcribed_clean)
    return ratio > threshold

def access_local_file(url):
    # Extract the file name from the URL
    file_name = url.split('/')[-1]
    # Construct the full path to the file on disk
    file_path = wav_file_path / file_name
    # Check if the file exists
    if not file_path.exists():
        print(f"[AllTalk TTSDiff] File does not exist: {file_path}")
        return None
    return file_path

def transcribe_and_compare(file_url, original_text, model, item_id, flagged_ids):
    audio_file_path = access_local_file(file_url)
    if audio_file_path is None:
        print(f"[AllTalk TTSDiff] Could not access local file for URL: {file_url}")
        return  # Skip this file if it can't be accessed

    try:
        segments, info = model.transcribe(str(audio_file_path), beam_size=5)
        transcribed_text = " ".join([segment.text for segment in segments])
    except Exception as e:
        print(f"[AllTalk TTSDiff] Error transcribing file {audio_file_path}: {e}")
        return  # Skip this file if transcription fails
    
    segments, info = model.transcribe(str(audio_file_path), beam_size=5)
    transcribed_text = " ".join([segment.text for segment in segments])
    
    # Normalize texts for comparison
    original_text_normalized = normalize_text(original_text)
    transcribed_text_normalized = normalize_text(transcribed_text)
    
    # Enhanced comparison using detailed fuzzy matching
    is_detailed_match = detailed_comparison(original_text_normalized, transcribed_text_normalized, args.threshold)
    has_spoken_punctuation = contains_spoken_punctuation(transcribed_text_normalized) and not contains_spoken_punctuation(original_text_normalized)
    
    # Adjust is_match based on detailed comparison and spoken punctuation check
    is_match = is_detailed_match and not has_spoken_punctuation
    
    # Only log and flag IDs for review if there's a mismatch or detected issues
    if not is_match:
        #print(f"\033[93mMismatch Detected - ID:\033[0m {item_id}")
        print(f"[AllTalk TTSDiff] \033[93mText:\033[0m {original_text}")
        print(f"[AllTalk TTSDiff] \033[91mTTS :\033[0m{transcribed_text}")
        if has_spoken_punctuation:
            print(f"[AllTalk TTSDiff] Note: Potential incorrect spoken punctuation detected.")
        flagged_ids.append(item_id)  # Track the ID for review

def main():
    print(f"[AllTalk TTSDiff]")
    print(f"[AllTalk TTSDiff]")
    print("[AllTalk TTSDiff] \033[92mStarting Compare of Text vs TTS:\033[0m")
    print("[AllTalk TTSDiff]")
    print("[AllTalk TTSDiff] \033[94mJSON file  :\033[92m", json_file_path, "\033[0m")
    print("[AllTalk TTSDiff] \033[94mWAV files  :\033[92m", wav_file_path, "\033[0m")
    print("[AllTalk TTSDiff] \033[94mAccuracy   :\033[92m", accuracy_threshold, "%", "\033[0m")
    detect_cublas()
    model_size = "large-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_size, device=device, compute_type="float32")
    
    flagged_ids = []  # Initialize the list to track IDs needing review

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            tts_list = json.load(f)
    except FileNotFoundError:
        print(f"[AllTalk TTSDiff] ERROR STARTING SCRIPT:")
        print("[AllTalk TTSDiff] Error: \033[93mttsList.json\033[0m file not found.")
        print("[AllTalk TTSDiff] Please ensure you follow these steps:")
        print("[AllTalk TTSDiff] After generating your text in the TTS Generator, you need to `Export List to JSON` and save the file in the")
        print("[AllTalk TTSDiff] same folder/directory this script is running from.")
        print("[AllTalk TTSDiff] The JSON file must be named \033[93mttsList.json\033[0m")
        exit(1)

    print(f"[AllTalk TTSDiff]")

    for item in tts_list:
        print(f"[AllTalk TTSDiff] Processing ID: {item['id']}")
        transcribe_and_compare(item['fileUrl'], item['text'], model, item['id'], flagged_ids)
    
    print(f"[AllTalk TTSDiff]")
    # Print summary information at the end
    if flagged_ids:
        print(f"[AllTalk TTSDiff] \033[94mIDs needing review:\033[0m", ', '.join(map(str, flagged_ids)))
        print(f"[AllTalk TTSDiff]")
        print(f"[AllTalk TTSDiff] Review ID's and correct any lines by editing & regenerating as necessary.")
    else:
        print("[AllTalk TTSDiff] \033[94mIDs needing review:\033[0m No issues detected.")
    summary_data = {
        "flagged_ids": flagged_ids
    }
    with open(wav_file_path / "analysis_summary.json", "w") as summary_file:
        json.dump(summary_data, summary_file)
    print("[AllTalk TTSDiff]")

if __name__ == "__main__":
    main()
