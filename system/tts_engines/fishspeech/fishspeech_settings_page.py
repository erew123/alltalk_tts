import os
import json
import requests
import gradio as gr
import whisper
from typing import List, Tuple
from tqdm import tqdm
from pathlib import Path
import gc
import torch
import warnings
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention")
from .help_content import AllTalkHelpContent
this_dir = Path(__file__).parent.resolve()
main_dir = Path(__file__).parent.parent.parent.parent.resolve()

##########################################################################
# Populate the voices list for Fish Speech                               #
##########################################################################
def fishspeech_voices_file_list():
    directory = main_dir / "voices"
    voices = []

    def has_reference_text(wav_path):
        text_path = wav_path.with_suffix('.reference.txt')
        return text_path.exists()

    # Check files in main voices directory
    for f in directory.glob("*.wav"):
        if has_reference_text(f):
            voices.append(f.name)

    # Check subdirectories
    for folder in directory.iterdir():
        if folder.is_dir():
            for wav_file in folder.glob("*.wav"):
                if has_reference_text(wav_file):
                    rel_path = wav_file.relative_to(directory)
                    voices.append(str(rel_path))

    if not voices:
        return ["No Voices Found"]

    return sorted(voices)

def get_files_needing_transcription(directory: Path) -> List[Tuple[Path, str]]:
    """Get all audio files that need transcription."""
    files_to_transcribe = []

    for ext in ['.wav', '.mp3', '.flac']:
        # Check main directory
        for f in directory.glob(f"*{ext}"):
            if not f.with_suffix('.reference.txt').exists():
                files_to_transcribe.append((f, ext[1:]))

        # Check subdirectories
        for folder in directory.iterdir():
            if folder.is_dir():
                for f in folder.glob(f"*{ext}"):
                    if not f.with_suffix('.reference.txt').exists():
                        files_to_transcribe.append((f, ext[1:]))

    return files_to_transcribe

def transcribe_files(model_name: str, use_cpu: bool = False, progress=gr.Progress()) -> str:
    """Transcribe all files without reference text."""
    files = get_files_needing_transcription(Path(main_dir)/'voices')

    if not files:
        return "No files need transcription!"

    model = None
    try:
        progress(0, desc="Loading Whisper model...")

        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        model = whisper.load_model(model_name).to(device)

        progress(0, desc=f"Transcribing files using {device.upper()}...")

        for idx, (file_path, _) in enumerate(progress.tqdm(files, desc="Transcribing files")):
            result = model.transcribe(str(file_path))

            with open(file_path.with_suffix('.reference.txt'), 'w', encoding='utf-8') as f:
                f.write(result["text"].strip())

            progress((idx + 1) / len(files))

        return f"Successfully transcribed {len(files)} files using {device.upper()}!"

    except Exception as e:
        return f"Error during transcription: {str(e)}"

    finally:
        if model is not None:
            if torch.cuda.is_available() and not use_cpu:
                model.cpu()
                torch.cuda.empty_cache()

            del model
            model = None
            gc.collect()

######################################################
# Update settings function                           #
######################################################
def fishspeech_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
    with open(os.path.join(this_dir, "model_settings.json"), "r") as f:
        model_config_data = json.load(f)

    model_config_data["settings"]["def_character_voice"] = def_character_voice_gr
    model_config_data["settings"]["def_narrator_voice"] = def_narrator_voice_gr
    model_config_data["openai_voices"]["alloy"] = alloy_gr
    model_config_data["openai_voices"]["echo"] = echo_gr
    model_config_data["openai_voices"]["fable"] = fable_gr
    model_config_data["openai_voices"]["nova"] = nova_gr
    model_config_data["openai_voices"]["onyx"] = onyx_gr
    model_config_data["openai_voices"]["shimmer"] = shimmer_gr
    model_config_data["settings"]["lowvram_enabled"] = lowvram_enabled_gr == "Enabled"
    model_config_data["settings"]["deepspeed_enabled"] = deepspeed_enabled_gr == "Enabled"
    model_config_data["settings"]["temperature_set"] = temperature_set_gr
    model_config_data["settings"]["repetitionpenalty_set"] = repetitionpenalty_set_gr
    model_config_data["settings"]["pitch_set"] = pitch_set_gr
    model_config_data["settings"]["generationspeed_set"] = generationspeed_set_gr

    with open(os.path.join(this_dir, "model_settings.json"), "w") as f:
        json.dump(model_config_data, f, indent=4)
    return "Settings updated successfully!"

#######################################################
# Main settings page function                         #
#######################################################
def fishspeech_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = fishspeech_voices_file_list()
    with gr.Blocks(title="Fish Speech TTS", analytics_enabled=False) as app:
        with gr.Tab("Default Settings"):
            with gr.Row():
                lowvram_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="Low VRAM" if model_config_data["model_capabilties"]["lowvram_capable"] else "Low VRAM N/A", value="Enabled" if model_config_data["settings"]["lowvram_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["lowvram_capable"])
                deepspeed_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="DeepSpeed Activate" if model_config_data["model_capabilties"]["deepspeed_capable"] else "DeepSpeed N/A", value="Enabled" if model_config_data["settings"]["deepspeed_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["deepspeed_capable"])
                temperature_set_gr = gr.Slider(value=float(model_config_data["settings"]["temperature_set"]), minimum=0, maximum=1, step=0.05, label="Temperature" if model_config_data["model_capabilties"]["temperature_capable"] else "Temperature N/A", interactive=model_config_data["model_capabilties"]["temperature_capable"])
                repetitionpenalty_set_gr = gr.Slider(value=float(model_config_data["settings"]["repetitionpenalty_set"]), minimum=1, maximum=2, step=0.1, label="Repetition Penalty" if model_config_data["model_capabilties"]["repetitionpenalty_capable"] else "Repetition N/A", interactive=model_config_data["model_capabilties"]["repetitionpenalty_capable"])
                pitch_set_gr = gr.Slider(value=float(model_config_data["settings"]["pitch_set"]), minimum=-10, maximum=10, step=1, label="Pitch" if model_config_data["model_capabilties"]["pitch_capable"] else "Pitch N/A", interactive=model_config_data["model_capabilties"]["pitch_capable"])
                generationspeed_set_gr = gr.Slider(value=float(model_config_data["settings"]["generationspeed_set"]), minimum=0.30, maximum=2.00, step=0.10, label="Speed" if model_config_data["model_capabilties"]["generationspeed_capable"] else "Speed N/A", interactive=model_config_data["model_capabilties"]["generationspeed_capable"])
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### OpenAI Voice Mappings")
                    with gr.Group():
                        with gr.Row():
                            alloy_gr = gr.Dropdown(value=model_config_data["openai_voices"]["alloy"], label="Alloy", choices=voice_list, allow_custom_value=True)
                            echo_gr = gr.Dropdown(value=model_config_data["openai_voices"]["echo"], label="Echo", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            fable_gr = gr.Dropdown(value=model_config_data["openai_voices"]["fable"], label="Fable", choices=voice_list, allow_custom_value=True)
                            nova_gr = gr.Dropdown(value=model_config_data["openai_voices"]["nova"], label="Nova", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            onyx_gr = gr.Dropdown(value=model_config_data["openai_voices"]["onyx"], label="Onyx", choices=voice_list, allow_custom_value=True)
                            shimmer_gr = gr.Dropdown(value=model_config_data["openai_voices"]["shimmer"], label="Shimmer", choices=voice_list, allow_custom_value=True)
                with gr.Column():
                    gr.Markdown("### Default Voices")
                    with gr.Row():
                        def_character_voice_gr = gr.Dropdown(value=model_config_data["settings"]["def_character_voice"], label="Default/Character Voice", choices=voice_list, allow_custom_value=True)
                        def_narrator_voice_gr = gr.Dropdown(value=model_config_data["settings"]["def_narrator_voice"], label="Narrator Voice", choices=voice_list, allow_custom_value=True)
                    with gr.Group():
                        with gr.Row():
                            details_text = gr.Textbox(label="Details", show_label=False, lines=5, interactive=False, value="Configure default settings and voice mappings for Fish Speech. Temperature controls generation randomness (0.5-0.8 recommended). Repetition penalty helps prevent loops (1.1-1.2 recommended). Low VRAM mode moves models between CPU/GPU as needed.")
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            with gr.Accordion("HELP - Fish Speech Settings", open=False):
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION, elem_classes="custom-markdown")
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION1, elem_classes="custom-markdown")
                    gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION2, elem_classes="custom-markdown")
            submit_button.click(fishspeech_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr], outputs=output_message)

        ###########################################################################################
        # Engine Information Tab                                                                   #
        ###########################################################################################
        with gr.Tab("Engine Information"):
            with gr.Row():
                with gr.Group():
                    gr.Textbox(label="Manufacturer Name", value=model_config_data['model_details']['manufacturer_name'], interactive=False)
                    gr.Textbox(label="Manufacturer Website/TTS Engine Support", value=model_config_data['model_details']['manufacturer_website'], interactive=False)
                    gr.Textbox(label="Engine/Model Description", value=model_config_data['model_details']['model_description'], interactive=False, lines=13)
                with gr.Column():
                    with gr.Row():
                        gr.Textbox(label="DeepSpeed Capable", value='Yes' if features_list['deepspeed_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Pitch Capable", value='Yes' if features_list['pitch_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Generation Speed Capable", value='Yes' if features_list['generationspeed_capable'] else 'No', interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Repetition Penalty Capable", value='Yes' if features_list['repetitionpenalty_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Multi Languages Capable", value='Yes' if features_list['languages_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Streaming Capable", value='Yes' if features_list['streaming_capable'] else 'No', interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Low VRAM Capable", value='Yes' if features_list['lowvram_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Temperature Capable", value='Yes' if features_list['temperature_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Multi Model Capable Engine", value='Yes' if features_list['multimodel_capable'] else 'No', interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Multi Voice Capable Models", value='Yes' if features_list['multivoice_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Default Audio output format", value=model_config_data['model_capabilties']['audio_format'], interactive=False)
                        gr.Textbox(label="TTS Engine Name", value="fishspeech", interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Windows Support", value='Yes' if features_list['windows_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Linux Support", value='Yes' if features_list['linux_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Mac Support", value='Yes' if features_list['mac_capable'] else 'No', interactive=False)
            with gr.Row():
                with gr.Accordion("HELP - Understanding TTS Engine Capabilities", open=False):
                    with gr.Row():
                        gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION, elem_classes="custom-markdown")
                    with gr.Row():
                        gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION1, elem_classes="custom-markdown")
                        gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION2, elem_classes="custom-markdown")

        #######################################################################################################################################################################################################
        # Models/Voices Download Tab                                                                                                                                                                          #
        #######################################################################################################################################################################################################
        with gr.Tab("Models/Voices Download"):
            with gr.Row():
                gr.Markdown("""
                ## Fish Speech Model Download

                **Note:** Fish Speech models are hosted on HuggingFace and may require authentication.
                If download fails, you may need to:
                1. Create a HuggingFace account at https://huggingface.co
                2. Accept the model license terms
                3. Set up a HuggingFace token (see HuggingFace docs)

                **Model sizes:**
                - OpenAudio S1-mini: ~3.6GB (recommended for most users)
                - OpenAudio S1: ~16GB (higher quality, requires more VRAM)
                """)
            with gr.Row():
                with open(os.path.join(this_dir, "available_models.json"), "r") as f:
                    available_models = json.load(f)
                model_names = [model["model_name"] for model in available_models["models"]]
                model_dropdown = gr.Dropdown(choices=sorted(model_names), label="Select Model", value=model_names[0])
                download_button = gr.Button("Download Model/Missing Files")

            def download_model(model_name, force_download=False):
                selected_model = next(model for model in available_models["models"] if model["model_name"] == model_name)

                base_folder_path = os.path.join(main_dir, "models", "fishspeech")
                model_folder_path = os.path.join(base_folder_path, selected_model["folder_path"])

                files_to_download = selected_model["files_to_download"]

                all_files_exists = all(
                    os.path.exists(os.path.join(model_folder_path, file_name))
                    for file_name in files_to_download
                )

                if all_files_exists and not force_download:
                    return "All files are already downloaded. No need to download again."
                else:
                    for file_name, url in files_to_download.items():
                        file_path = os.path.join(model_folder_path, file_name)

                        os.makedirs(os.path.dirname(file_path), exist_ok=True)

                        if not os.path.exists(file_path) or force_download:
                            print(f"Downloading {file_name} to {file_path}...")
                            try:
                                response = requests.get(url, stream=True)
                                response.raise_for_status()
                                total_size_in_bytes = int(response.headers.get("content-length", 0))
                                block_size = 1024
                                progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

                                with open(file_path, "wb") as file:
                                    for data in response.iter_content(block_size):
                                        progress_bar.update(len(data))
                                        file.write(data)
                                progress_bar.close()
                            except requests.exceptions.HTTPError as e:
                                if e.response.status_code == 401:
                                    return f"Authentication required. Please set up HuggingFace authentication. Error: {str(e)}"
                                elif e.response.status_code == 403:
                                    return f"Access denied. Please accept the model license on HuggingFace. Error: {str(e)}"
                                else:
                                    return f"Download failed: {str(e)}"

                    return "Model downloaded successfully!"

            with gr.Row():
                download_status = gr.Textbox(label="Download Status")

            download_button.click(download_model, inputs=model_dropdown, outputs=download_status)

            def show_confirm_cancel(model_name):
                selected_model = next(model for model in available_models["models"] if model["model_name"] == model_name)
                model_folder_path = os.path.join(main_dir, "models", "fishspeech", selected_model["folder_path"])
                all_files_exists = all(
                    os.path.exists(os.path.join(model_folder_path, file_name))
                    for file_name in selected_model["files_to_download"]
                )
                if all_files_exists:
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]
                else:
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)]

            def confirm_download(model_name):
                download_status_text = download_model(model_name, force_download=True)
                return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), download_status_text]

            def cancel_download():
                return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), ""]

            with gr.Row():
                confirm_button = gr.Button("Download Anyway", visible=False)
                cancel_button = gr.Button("Cancel", visible=False)

            download_button.click(show_confirm_cancel, inputs=model_dropdown, outputs=[confirm_button, download_button, cancel_button])
            confirm_button.click(confirm_download, inputs=model_dropdown, outputs=[confirm_button, download_button, cancel_button, download_status])
            cancel_button.click(cancel_download, inputs=None, outputs=[confirm_button, download_button, cancel_button, download_status])

        #######################################################################################################################################################################################################
        # Reference Text/Sample Manager Tab                                                                                                                                                                    #
        #######################################################################################################################################################################################################
        def get_voice_files():
            """Get list of voice files with their status"""
            files = [
                f"🔴 {f}" if not Path(main_dir/'voices'/f).with_suffix('.reference.txt').exists() else f"🟢 {f}"
                for f in os.listdir(Path(main_dir/'voices'))
                if f.endswith('.wav')
            ]
            return sorted(files)

        with gr.Tab("Reference Text/Sample Manager"):
            with gr.Row():
                file_list = gr.Dropdown(
                    label="Voice Files",
                    choices=get_voice_files(),
                    value=None
                )
                refresh_btn = gr.Button("Refresh List")
                model_selector = gr.Dropdown(
                    label="Whisper Model",
                    choices=["base", "medium", "large-v3"],
                    value="base"
                )
                transcribe_button = gr.Button("Auto-Transcribe Missing Files")
            with gr.Row():
                audio_player = gr.Audio(
                    label="Preview Voice Sample",
                    type="filepath",
                    interactive=False,
                    elem_classes="small-audio-player"
                )
            with gr.Row():
                current_text = gr.Textbox(
                    label="Manually Edit Reference Text",
                    placeholder="Enter the exact text that was spoken in the recording...",
                    lines=2,
                    scale=2
                )
                status_text = gr.Textbox(label="Status", interactive=False, scale=1)
            with gr.Row():
                save_btn = gr.Button("Save Reference Text", variant="primary")
                delete_btn = gr.Button("Delete Reference Text", variant="secondary")
            with gr.Accordion("HELP - Reference Text/Sample Manager", open=False):
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.VOICE_SETUP_HELP, elem_classes="custom-markdown")

            def load_file_data(file_name):
                if not file_name:
                    return "", "", ""

                clean_name = file_name.lstrip("🔴🟢 ")
                wav_path = Path(main_dir/'voices'/clean_name)
                txt_path = wav_path.with_suffix('.reference.txt')

                current_text = ""
                if txt_path.exists():
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        current_text = f.read()

                return str(wav_path), current_text, ""

            def save_reference_text(file_name, text):
                if not file_name:
                    return "Please select a file first"

                clean_name = file_name.lstrip("🔴🟢 ")
                txt_path = Path(main_dir/'voices'/clean_name).with_suffix('.reference.txt')

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                return f"Saved reference text for {clean_name}"

            def delete_reference_text(file_name):
                if not file_name:
                    return "Please select a file first", ""

                clean_name = file_name.lstrip("🔴🟢 ")
                txt_path = Path(main_dir/'voices'/clean_name).with_suffix('.reference.txt')

                if txt_path.exists():
                    os.remove(txt_path)
                    return f"Deleted reference text for {clean_name}", ""
                return "No reference text file to delete", ""

            file_list.change(load_file_data, inputs=[file_list], outputs=[audio_player, current_text, status_text])
            save_btn.click(save_reference_text, inputs=[file_list, current_text], outputs=[status_text])
            delete_btn.click(delete_reference_text, inputs=[file_list], outputs=[status_text, current_text])
            refresh_btn.click(lambda: gr.update(choices=get_voice_files()), outputs=[file_list])
            transcribe_button.click(transcribe_files, inputs=[model_selector], outputs=[status_text])

    return app

# Alias for compatibility with script.py dynamic loading
fishspeech_at_gradio_settings_page = fishspeech_model_alltalk_settings
