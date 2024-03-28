import json
import argparse
import soundfile as sf
from pathlib import Path

parser = argparse.ArgumentParser(description="Compare TTS output with the original text using detailed comparison.")
parser.add_argument("--ttslistpath", help="Path to the ttsList.json file")
parser.add_argument("--wavfilespath", help="Path to the wav outputs folder")
args = parser.parse_args()

if not args.ttslistpath:
    parser.error("[AllTalk TTSSRT] Please specify the path to the ttslist.json file using the --ttslistpath argument.")
if not args.wavfilespath:
    parser.error("[AllTalk TTSSRT] Please specify the path to the wavs output folder file using the --wavfilespath argument.")

json_file_path = Path(args.ttslistpath).resolve()
audio_files_base_path = Path(args.wavfilespath).resolve()

print(f"[AllTalk TTSSRT]")
print("[AllTalk TTSSRT] \033[92mStarting SRT Subtitle generation:\033[0m")
print("[AllTalk TTSSRT]")
print("[AllTalk TTSSRT] \033[94mJSON file  :\033[92m", json_file_path, "\033[0m")
print("[AllTalk TTSSRT] \033[94mWAV files  :\033[92m", audio_files_base_path, "\033[0m")
print("[AllTalk TTSSRT]")

# Function to convert time in seconds to SRT time format
def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

# Load the JSON data
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Initialize start time and SRT lines
start_time = 0.0
srt_lines = []

# Loop through the JSON data to build SRT entries
for entry in data:
    file_name = Path(entry['fileUrl']).name  # Get the file name from the URL
    file_path = audio_files_base_path / file_name
    print(f"[AllTalk TTSSRT] Processing file: {file_name}")

    if not file_path.exists():
        print(f"[AllTalk TTSSRT] Audio file does not exist: {file_path}")
        continue

    try:
        # Use soundfile to read the duration of the WAV file
        with sf.SoundFile(str(file_path), 'r') as f:
            duration = len(f) / f.samplerate
            end_time = start_time + duration
            
            # Format the SRT entry
            start_timestamp = format_srt_time(start_time)
            end_timestamp = format_srt_time(end_time)
            srt_lines.append(f"{len(srt_lines)+1}\n{start_timestamp} --> {end_timestamp}\n{entry['text']}\n")
            
            # Update the start time for the next entry
            start_time = end_time
    except RuntimeError as e:
        print(f"[AllTalk TTSSRT] Error reading {file_path}: {e}")

# Join all SRT lines into the final SRT content
srt_content = "\n".join(srt_lines)

# Save the SRT content to a file
srt_file_path = audio_files_base_path / "subtitles.srt"
with open(srt_file_path, "w") as srt_file:
    srt_file.write(srt_content)

print("[AllTalk TTSSRT]")
print(f"[AllTalk TTSSRT] \033[94mSRT file created at \033[0m{srt_file_path}")
