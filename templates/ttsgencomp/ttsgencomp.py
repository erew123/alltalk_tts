import json
import requests
from pathlib import Path
# version 0.1

def disclaimer_text():
    print(f"\nDISCLAIMER:")
    print(f"  \033[92mThis is a work in progress, aka a BETA test at best. It is not refined or production ready,")
    print(f"  just a proof of concept and should therefore be viewed and used as such.")
    print(f"\n  There is no support on this while its in BETA.\033[0m")
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
    print(f"  - The JSON file must be named \033[93mttsList.json\033[0m")
    print(f"  - When you have your ID list, go back into the TTS Generator, correct any lines and regenerate them. If you")
    print(f"    want to re-test everthing again after re-generating, you will need to export your list again and re-run this")
    print(f"    script again, against your newly exported JSON list.")
    print(f"  - This requires access to \033[93mcublas64_11\033[0m, the same as Finetuning.")
    print(f"    https://github.com/erew123/alltalk_tts/tree/main?#-important-requirements-cuda-118")
    return

try:
    from faster_whisper import WhisperModel
    import re
    import string
    import spacy
    import torch
    from fuzzywuzzy import fuzz
except ImportError as e:
    disclaimer_text()
    print(f"\nERROR STARTING SCRIPT:")
    print("  An error occurred importing one or more required libraries.")
    print("  Please ensure you have \033[94mactivated your Python environment \033[0mand \033[94minstalled all requirements.\033[0m")
    print("\n  \033[94mMissing library:\033[93m", str(e), "\033[0m")
    exit(1)

# Attempt to load the spaCy model
try:
    # Load a medium-sized spaCy model, adjust to "en_core_web_md" or "en_core_web_lg" as needed
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    disclaimer_text()
    print(f"\nERROR STARTING SCRIPT:")
    print("  Failed to load the spaCy language model.")
    print("  Please ensure the spaCy language model is installed by running:")
    print("  \033[93mpython -m spacy download en_core_web_md\033[0m")
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

def download_file(url, destination):
    response = requests.get(url)
    response.raise_for_status()
    with open(destination, 'wb') as f:
        f.write(response.content)

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

def transcribe_and_compare(file_url, original_text, model, item_id, flagged_ids):
    audio_file = Path(file_url.split('/')[-1])
    download_file(file_url, audio_file)
    
    segments, info = model.transcribe(str(audio_file), beam_size=5)
    transcribed_text = " ".join([segment.text for segment in segments])
    
    original_text_normalized = normalize_text(original_text)
    transcribed_text_normalized = normalize_text(transcribed_text)
    
    is_match = texts_are_similar(original_text_normalized, transcribed_text_normalized)
    has_spoken_punctuation = contains_spoken_punctuation(transcribed_text_normalized) and not contains_spoken_punctuation(original_text_normalized)
    
    if has_spoken_punctuation:
        is_match = False  # Consider as mismatch if unexpected spoken punctuation is detected
    
    if not is_match or has_spoken_punctuation:
        print(f"\033[93mOriginal:\033[0m {original_text}")
        print(f"\033[91mTranscribed:\033[0m {transcribed_text}")
        print(f"Match: {is_match}\n")
        if has_spoken_punctuation:
            print(f"Note: Potential incorrect spoken punctuation detected in ID: {item_id}.")
        flagged_ids.append(item_id)  # Track the ID for review
    
    audio_file.unlink()  # Remove the downloaded file after processing

def main():
    model_size = "large-v3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disclaimer_text()
    model = WhisperModel(model_size, device=device, compute_type="float32")
    
    flagged_ids = []  # Initialize the list to track IDs needing review

    try:
        with open("ttsList.json", "r", encoding="utf-8") as f:
            tts_list = json.load(f)
    except FileNotFoundError:
        print(f"\nERROR STARTING SCRIPT:")
        print("  Error: \033[93mttsList.json\033[0m file not found.")
        print("\n  Please ensure you follow these steps:")
        print("  - After generating your text in the TTS Generator, you need to `Export List to JSON` and save the file in the")
        print("    same folder/directory this script is running from.")
        print("  - The JSON file must be named \033[93mttsList.json\033[0m")
        exit(1)

    print(f"\nPROCESSING LIST:\n")

    for item in tts_list:
        print(f"Processing ID: {item['id']}")
        transcribe_and_compare(item['fileUrl'], item['text'], model, item['id'], flagged_ids)
    
    # Print summary information at the end
    if flagged_ids:
        print(f"  For ID's showm, in the TTS Generator, review and correct any lines by editing & regenerating them. If you")
        print(f"  want to re-test everthing again after re-generating, you will need to export the JSON list again and re-run")
        print(f"  the script again, against the newly exported JSON list.")
        print("\nSUMMARY: IDs needing review:", ', '.join(map(str, flagged_ids)))
    else:
        print("\nSUMMARY: No issues detected.")

if __name__ == "__main__":
    main()