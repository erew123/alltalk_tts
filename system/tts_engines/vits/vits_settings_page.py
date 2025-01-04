import os
import json
import shutil
import zipfile
import requests
import gradio as gr
from tqdm import tqdm
from pathlib import Path
from .help_content import AllTalkHelpContent
this_dir = Path(__file__).parent.resolve()                         # Sets up self.this_dir as a variable for the folder THIS script is running in.
main_dir = Path(__file__).parent.parent.parent.parent.resolve()    # Sets up self.main_dir as a variable for the folder AllTalk is running in

##########################################################################
# REQUIRED CHANGE                                                        #
# Populate the voices list, using the method specific to your TTS engine #
##########################################################################
# This function is responsible for populating the list of available voices for your TTS engine.
# You need to modify this function to use the appropriate method for your engine to retrieve the voice list.
#
# The current implementation lists all the WAV files in a "voices" directory, which may not be suitable for your engine.
# You should replace the `xxxx_voices_file_list` function name to match your engine name. For example, if your engine 
# is named "mytts", the function should be named `mytts_voices_file_list`. 
# 
# You will also neef to update the code with your own implementation that retrieves the voice list according to your 
# engine's specific requirements. Typically this is the same code as will be in your model_engine.py file.
#
# For example, if your engine has a dedicated API or configuration file for managing voices, you should modify this
# function to interact with that API or read from that configuration file.
#
# After making the necessary changes, this function should return a list of available voices that can be used
# in your TTS engine's settings page.
# Load the available models from the JSON file
with open(os.path.join(this_dir, "available_models.json"), "r") as f:
    available_models = json.load(f)
# Extract the model names for the dropdown
model_names = [model["model_name"] for model in available_models["models"]]

def vits_voices_file_list():
    models_folder = main_dir / "models" / "vits"
    # Create the folder if it doesn't exist
    if not models_folder.exists():
        models_folder.mkdir(parents=True, exist_ok=True)
    voices = []
    available_models = {}
    
    # Define possible model files
    model_files = ["model_file.pth", "model.pth"]
    
    for subfolder in models_folder.iterdir():
        if subfolder.is_dir():
            model_name = subfolder.name
            
            # Check if config.json exists (minimum requirement)
            if not (subfolder / "config.json").exists():
                continue

            # Load config to check model type
            try:
                with open(subfolder / "config.json", "r") as f:
                    config = json.load(f)
                use_speaker_embedding = config["model_args"].get("use_speaker_embedding", False)
                num_speakers = config["model_args"].get("num_speakers", 0)
            except Exception:
                continue

            # Check for model file
            has_model_file = any(subfolder.joinpath(model_file).exists() for model_file in model_files)
            has_model_file_tar = subfolder.joinpath("model_file.pth.tar").exists()
            
            if not (has_model_file or has_model_file_tar):
                continue

            # Determine model type
            if (subfolder / "language_ids.json").exists() and (subfolder / "speaker_ids.json").exists():
                available_models[f"VITS - {model_name}"] = "VITS_multi_language"
            elif (subfolder / "speaker_ids.json").exists() or (use_speaker_embedding or num_speakers > 0):
                available_models[f"VITS - {model_name}"] = "VITS_multi_speaker"
            else:
                available_models[f"VITS - {model_name}"] = "VITS_single_speaker"

            # Get voices based on model type
            model_type = available_models[f"VITS - {model_name}"]
            if model_type in ["VITS_multi_speaker", "VITS_multi_language"]:
                speaker_ids_path = subfolder / "speaker_ids.json"
                if speaker_ids_path.exists():
                    with open(speaker_ids_path, "r") as f:
                        speaker_ids = json.load(f)
                    speakers = [speaker for speaker in speaker_ids.keys() if speaker != "ED\n"]
                    voices.extend(speakers)
                elif use_speaker_embedding or num_speakers > 0:
                    # If no speaker_ids.json but config indicates multiple speakers
                    voices.extend([f"Speaker{i}" for i in range(num_speakers)])
            elif model_type == "VITS_single_speaker":
                voices.append("default")

    # Remove duplicate speaker names
    voices = list(set(voices))
    # Sort the voices alphabetically
    voices.sort()
    # Move the "default" voice to the top of the list
    if "default" in voices:
        voices.remove("default")
        voices.insert(0, "default")
    return voices

def group_vits_models_by_language():
    """Group available VITS models by language code"""
    language_groups = {}
    
    for model in available_models["models"]:
        model_name = model["model_name"]
        # Extract language code from model name (format: tts_models--{lang}--...)
        parts = model_name.split('--')
        if len(parts) >= 2:
            lang_code = parts[1]
            
            if lang_code not in language_groups:
                language_groups[lang_code] = {
                    "name": get_language_name(lang_code),
                    "models": []
                }
            
            language_groups[lang_code]["models"].append(model)
    
    return language_groups

def get_language_name(lang_code):
    """Convert VITS language code to full name"""
    language_names = {
        "en": "English",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "ru": "Russian",
        "cs": "Czech",
        "hu": "Hungarian",
        "bn": "Bengali",
        "ca": "Catalan",
        "el": "Greek",
        "fi": "Finnish",
        "hr": "Croatian",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mt": "Maltese",
        "ro": "Romanian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "sv": "Swedish",
        "bg": "Bulgarian",
        "da": "Danish",
        "et": "Estonian",
        "ga": "Irish",
        "uk": "Ukrainian",
        "ewe": "Ewe",
        "hau": "Hausa",
        "lin": "Lingala",
        "tw": "Twi",
        "yor": "Yoruba"
    }
    return language_names.get(lang_code, f"Unknown ({lang_code})")

def download_language_pack(lang_code, progress=gr.Progress()):
    """Download all models for a specific language"""
    language_groups = group_vits_models_by_language()
    if lang_code not in language_groups:
        return f"Error: Language pack {lang_code} not found"
    
    models = language_groups[lang_code]["models"]
    total_models = len(models)
    
    progress(0, desc=f"Downloading {get_language_name(lang_code)} voice pack")
    
    downloaded_files = []
    errors = []
    
    for idx, model in enumerate(models, 1):
        progress(idx/total_models, desc=f"Downloading model {idx}/{total_models}")
        try:
            # Create the model folder
            vits_folder = os.path.join(main_dir, "models", "vits")
            folder_path = os.path.join(vits_folder, model["folder_path"])
            os.makedirs(folder_path, exist_ok=True)
            
            # Download and extract the model
            zip_url = model["github_rls_url"]
            zip_file_path = os.path.join(vits_folder, f"{model['model_name']}.zip")
            
            response = requests.get(zip_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            
            with open(zip_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(vits_folder)
            
            # Cleanup
            os.remove(zip_file_path)
            macosx_folder = os.path.join(vits_folder, "__MACOSX")
            if os.path.exists(macosx_folder):
                shutil.rmtree(macosx_folder)
                
            downloaded_files.append(model["model_name"])
            
        except Exception as e:
            errors.append(f"Error downloading {model['model_name']}: {str(e)}")
    
    if errors:
        return f"Downloaded {len(downloaded_files)} models with {len(errors)} errors:\n" + "\n".join(errors)
    else:
        return f"Successfully downloaded {len(downloaded_files)} models for {get_language_name(lang_code)}"

######################################################
# REQUIRED CHANGE                                    #
# Imports and saves the TTS engine-specific settings #
######################################################
# This function is responsible for importing and saving the settings specific to your TTS engine.
# You need to make the following change:
#
# 1. Change the name of the function `xxxx_model_update_settings` to match your engine's name.
#    For example, if your engine is named "mytts", the function should be named `mytts_model_update_settings`.
#
# After making this change, the function will load the model settings from a JSON file, update the settings and voice
# dictionaries with the values provided as arguments, and save the updated settings back to the JSON file.
#
# You do not need to modify the function's logic or any other part of the code.
def vits_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, streaming_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr,  alloy_gr, ash_gr, coral_gr, echo_gr, fable_gr, nova_gr, onyx_gr, sage_gr, shimmer_gr):
    # Load the model_config_data from the JSON file
    with open(os.path.join(this_dir, "model_settings.json"), "r") as f:
        model_config_data = json.load(f)
    # Update the settings and openai_voices dictionaries with the new values
    model_config_data["settings"]["def_character_voice"] = def_character_voice_gr
    model_config_data["settings"]["def_narrator_voice"] = def_narrator_voice_gr
    model_config_data["openai_voices"]["alloy"] = alloy_gr
    model_config_data["openai_voices"]["ash"] = ash_gr
    model_config_data["openai_voices"]["coral"] = coral_gr
    model_config_data["openai_voices"]["echo"] = echo_gr
    model_config_data["openai_voices"]["fable"] = fable_gr
    model_config_data["openai_voices"]["nova"] = nova_gr
    model_config_data["openai_voices"]["onyx"] = onyx_gr
    model_config_data["openai_voices"]["sage"] = sage_gr
    model_config_data["openai_voices"]["shimmer"] = shimmer_gr
    model_config_data["settings"]["lowvram_enabled"] = lowvram_enabled_gr == "Enabled"
    model_config_data["settings"]["deepspeed_enabled"] = deepspeed_enabled_gr == "Enabled"
    model_config_data["settings"]["streaming_enabled"] = streaming_enabled_gr == "Enabled"
    model_config_data["settings"]["temperature_set"] = temperature_set_gr
    model_config_data["settings"]["repetitionpenalty_set"] = repetitionpenalty_set_gr
    model_config_data["settings"]["pitch_set"] = pitch_set_gr
    model_config_data["settings"]["generationspeed_set"] = generationspeed_set_gr
    # Save the updated model_config_data to the JSON file
    with open(os.path.join(this_dir, "model_settings.json"), "w") as f:
        json.dump(model_config_data, f, indent=4)
    return "Settings updated successfully!"

#######################################################
# REQUIRED CHANGE                                     #
# Sets up the engine-specific settings page in Gradio #
#######################################################
# This function sets up the Gradio interface for the settings page specific to your TTS engine.
# You need to make the following changes:
#
# 1. Change the name of the function `xxxx_model_alltalk_settings` to match your engine's name.
#    For example, if your engine is named "mytts", the function should be named `mytts_model_alltalk_settings`.
#
# 2. Change the name of the `submit_button.click` function call to match the name you gave to the function
#    that imports and saves your engine's settings (the function you modified above).
#
# 3. Change the name of the `voice_list` function call to match the name of the function that lists
#    the available voices for your TTS engine.
#
# 4. Change the 'title' of the `gr.Blocks` to match your engine's name e.g. title="mytts TTS"
#
# After making these changes, this function will create and return the Gradio interface for your TTS engine's
# settings page, allowing users to configure various options and voice selections.
def vits_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = vits_voices_file_list()
    with gr.Blocks(title="vits TTS", analytics_enabled=False) as app:
        with gr.Tab("Default Settings"):
            with gr.Row():
                lowvram_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="Low VRAM" if model_config_data["model_capabilties"]["lowvram_capable"] else "Low VRAM N/A", value="Enabled" if model_config_data["settings"]["lowvram_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["lowvram_capable"])
                deepspeed_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="DeepSpeed Activate" if model_config_data["model_capabilties"]["deepspeed_capable"] else "DeepSpeed N/A", value="Enabled" if model_config_data["settings"]["deepspeed_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["deepspeed_capable"])
                streaming_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="Streaming" if model_config_data["model_capabilties"]["streaming_capable"] else "Streaming N/A", value="Enabled" if model_config_data["settings"]["streaming_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["streaming_capable"])
                temperature_set_gr = gr.Slider(value=float(model_config_data["settings"]["temperature_set"]), minimum=0, maximum=1, step=0.05, label="Temperature" if model_config_data["model_capabilties"]["temperature_capable"] else "Temperature N/A", interactive=model_config_data["model_capabilties"]["temperature_capable"])
                repetitionpenalty_set_gr = gr.Slider(value=float(model_config_data["settings"]["repetitionpenalty_set"]), minimum=1, maximum=20, step=1, label="Repetition Penalty" if model_config_data["model_capabilties"]["repetitionpenalty_capable"] else "Repetition N/A", interactive=model_config_data["model_capabilties"]["repetitionpenalty_capable"])
                pitch_set_gr = gr.Slider(value=float(model_config_data["settings"]["pitch_set"]), minimum=-10, maximum=10, step=1, label="Pitch" if model_config_data["model_capabilties"]["pitch_capable"] else "Pitch N/A", interactive=model_config_data["model_capabilties"]["pitch_capable"])
                generationspeed_set_gr = gr.Slider(value=float(model_config_data["settings"]["generationspeed_set"]), minimum=0.25, maximum=2.00, step=0.25, label="Speed" if model_config_data["model_capabilties"]["generationspeed_capable"] else "Speed N/A", interactive=model_config_data["model_capabilties"]["generationspeed_capable"])
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### OpenAI Voice Mappings")
                    with gr.Group():
                        with gr.Row():
                            alloy_gr = gr.Dropdown(value=model_config_data["openai_voices"]["alloy"], label="Alloy", choices=voice_list, allow_custom_value=True)
                            ash_gr = gr.Dropdown(value=model_config_data["openai_voices"]["ash"], label="Ash", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            coral_gr = gr.Dropdown(value=model_config_data["openai_voices"]["coral"], label="Coral", choices=voice_list, allow_custom_value=True)
                            echo_gr = gr.Dropdown(value=model_config_data["openai_voices"]["echo"], label="Echo", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            fable_gr = gr.Dropdown(value=model_config_data["openai_voices"]["fable"], label="Fable", choices=voice_list, allow_custom_value=True)
                            nova_gr = gr.Dropdown(value=model_config_data["openai_voices"]["nova"], label="Nova", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            onyx_gr = gr.Dropdown(value=model_config_data["openai_voices"]["onyx"], label="Onyx", choices=voice_list, allow_custom_value=True)
                            sage_gr = gr.Dropdown(value=model_config_data["openai_voices"]["sage"], label="Sage", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            shimmer_gr = gr.Dropdown(value=model_config_data["openai_voices"]["shimmer"], label="Shimmer", choices=voice_list, allow_custom_value=True)
                with gr.Column():
                    gr.Markdown("### Default Voices")         
                    with gr.Row():
                        def_character_voice_gr = gr.Dropdown(value=model_config_data["settings"]["def_character_voice"], label="Default/Character Voice", choices=voice_list, allow_custom_value=True)
                        def_narrator_voice_gr = gr.Dropdown(value=model_config_data["settings"]["def_narrator_voice"], label="Narrator Voice", choices=voice_list, allow_custom_value=True)
                    with gr.Group():
                        with gr.Row():
                            details_text = gr.Textbox(label="Details", show_label=False, lines=5, interactive=False, value="Configure default settings and voice mappings for the selected TTS engine. Unavailable options are grayed out based on engine capabilities. See the Help section below for detailed information about each setting.")
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            with gr.Accordion("HELP - 🔊 Understanding TTS Engine Default Settings Page", open=False):
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS, elem_classes="custom-markdown")                               
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS1, elem_classes="custom-markdown")
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS2, elem_classes="custom-markdown")                 
            submit_button.click(vits_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, streaming_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, ash_gr, coral_gr, echo_gr, fable_gr, nova_gr, onyx_gr, sage_gr, shimmer_gr], outputs=output_message)

        ###########################################################################################
        # Do not change this section apart from "TTS Engine Name" value to match your engine name #
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
                        gr.Textbox(label="TTS Engine Name", value="VITS", interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Windows Support", value='Yes' if features_list['windows_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Linux Support", value='Yes' if features_list['linux_capable'] else 'No', interactive=False)
                        gr.Textbox(label="Mac Support", value='Yes' if features_list['mac_capable'] else 'No', interactive=False)
            with gr.Row():
                with gr.Accordion("HELP - 🔊 Understanding TTS Engine Capabilities", open=False):
                    with gr.Row():
                        gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION, elem_classes="custom-markdown")                               
                    with gr.Row():
                        gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION1, elem_classes="custom-markdown")
                        gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION2, elem_classes="custom-markdown")

        #######################################################################################################################################################################################################
        # REQUIRED CHANGE                                                                                                                                                                                     #
        # You will need to build a custom method to identify if your models are installed and download them. Store your models list in the available_models.json file, alongside your model_settings.json etc #
        #######################################################################################################################################################################################################
        with gr.Tab("Models/Voices Download"):
            with gr.Tab("Language Pack Download"):
                with gr.Row():
                    language_groups = group_vits_models_by_language()
                    language_choices = [
                        f"{code} - {info['name']} ({len(info['models'])} models)" 
                        for code, info in sorted(language_groups.items())
                    ]
                    
                    lang_pack_dropdown = gr.Dropdown(
                        choices=language_choices,
                        label="Select Language Pack",
                        value=language_choices[0]
                    )
                    lang_pack_download_button = gr.Button("Download Language Pack")
                
                with gr.Row():
                    lang_pack_status = gr.Textbox(label="Language Pack Download Status")
                
                def download_selected_language_pack(selected_pack):
                    lang_code = selected_pack.split(" - ")[0]
                    return download_language_pack(lang_code)
                    
                lang_pack_download_button.click(
                    download_selected_language_pack,
                    inputs=lang_pack_dropdown,
                    outputs=lang_pack_status
                )

            with gr.Tab("Individual Voice Download"):
                with gr.Row():
                    # Create the dropdown
                    model_dropdown = gr.Dropdown(choices=sorted(model_names), label="Select Model", value=model_names[0])
                    download_button = gr.Button("Download Model/Missing Files")

                with gr.Row():
                    download_status = gr.Textbox(label="Download Status")

            def download_model(model_name):
                # Find the selected model in the available models
                selected_model = next(model for model in available_models["models"] if model["model_name"] == model_name)
                # Get the folder path and ZIP file URL
                vits_folder = os.path.join(main_dir, "models", "vits")
                folder_path = os.path.join(vits_folder, selected_model["folder_path"])
                zip_url = selected_model["github_rls_url"]
                # Check if the model is already downloaded based on the file structure
                required_files = ["config.json"]
                if os.path.exists(os.path.join(folder_path, "model_file.pth.tar")):
                    required_files.append("model_file.pth.tar")
                else:
                    required_files.append("model_file.pth")
                if os.path.exists(os.path.join(folder_path, "speaker_ids.json")):
                    required_files.append("speaker_ids.json")
                if os.path.exists(os.path.join(folder_path, "language_ids.json")):
                    required_files.append("language_ids.json")
                if all(os.path.exists(os.path.join(folder_path, file)) for file in required_files):
                    return "The model is already downloaded. No need to download again."
                else:
                    # Download the ZIP file to the "vits" folder
                    print(f"Downloading {model_name}...")
                    response = requests.get(zip_url, stream=True)
                    total_size_in_bytes = int(response.headers.get("content-length", 0))
                    block_size = 1024  # 1 Kibibyte
                    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                    zip_file_path = os.path.join(vits_folder, f"{model_name}.zip")
                    with open(zip_file_path, "wb") as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)
                    progress_bar.close()
                    # Extract the ZIP file contents to the desired folder
                    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                        zip_ref.extractall(vits_folder)
                    # Remove the ZIP file
                    os.remove(zip_file_path)
                    # Delete the __MACOSX folder if it exists
                    macosx_folder = os.path.join(vits_folder, "__MACOSX")
                    if os.path.exists(macosx_folder):
                        shutil.rmtree(macosx_folder)
                    return "Model downloaded successfully!"

            def show_confirm_cancel(model_name):
                selected_model = next(model for model in available_models["models"] if model["model_name"] == model_name)
                folder_path = os.path.join(main_dir, "models", "vits", selected_model["folder_path"])
                required_files = ["config.json"]
                if os.path.exists(os.path.join(folder_path, "model_file.pth.tar")):
                    required_files.append("model_file.pth.tar")
                else:
                    required_files.append("model_file.pth")
                if os.path.exists(os.path.join(folder_path, "speaker_ids.json")):
                    required_files.append("speaker_ids.json")
                if os.path.exists(os.path.join(folder_path, "language_ids.json")):
                    required_files.append("language_ids.json")
                if all(os.path.exists(os.path.join(folder_path, file)) for file in required_files):
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]
                else:
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)]

            def confirm_download(model_name):
                # Find the selected model in the available models
                selected_model = next(model for model in available_models["models"] if model["model_name"] == model_name)
                # Get the folder path and ZIP file URL
                vits_folder = os.path.join(main_dir, "models", "vits")
                folder_path = os.path.join(vits_folder, selected_model["folder_path"])
                zip_url = selected_model["github_rls_url"]
                # Download the ZIP file to the "vits" folder
                print(f"Downloading {model_name}...")
                response = requests.get(zip_url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                zip_file_path = os.path.join(vits_folder, f"{model_name}.zip")
                with open(zip_file_path, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
               # Extract the ZIP file contents to the desired folder
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(vits_folder)
                # Remove the ZIP file
                os.remove(zip_file_path)
                # Delete the __MACOSX folder if it exists
                macosx_folder = os.path.join(vits_folder, "__MACOSX")
                if os.path.exists(macosx_folder):
                    shutil.rmtree(macosx_folder)
                return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)]
            
            download_button.click(download_model, inputs=model_dropdown, outputs=download_status)
            
            with gr.Row():
                confirm_button = gr.Button("Download Anyway", visible=False)
                cancel_button = gr.Button("Cancel", visible=False)
            download_button.click(show_confirm_cancel, model_dropdown, [confirm_button, download_button, cancel_button])
            confirm_button.click(confirm_download, model_dropdown, [confirm_button, download_button, cancel_button])
            cancel_button.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, [confirm_button, download_button, cancel_button])

        ###################################################################################################
        # REQUIRED CHANGE                                                                                 #
        # Add any engine specific help, bugs, issues, operating system specifc requirements/setup in here #
        # Please use Markdown format, so gr.Markdown() with your markdown inside it.                      #
        ###################################################################################################
        with gr.Tab("Engine Help"):
            with gr.Row():
                gr.Markdown(AllTalkHelpContent.HELP_PAGE, elem_classes="custom-markdown")                               
            with gr.Row():
                gr.Markdown(AllTalkHelpContent.HELP_PAGE1, elem_classes="custom-markdown")
                gr.Markdown(AllTalkHelpContent.HELP_PAGE2, elem_classes="custom-markdown")

    return app

################################
# REQUIRED CHANGE              #
# Sets up the Gradio interface #
################################
# This function sets up the Gradio interface for your TTS engine's settings page.
# You need to change the name of the function calls to match the names you set in the functions above.
#
# Specifically, you need to update the following:
#
# 1. The name of the function `xxxx_at_gradio_settings_page` to match your engine's name.
#    For example, if your engine is named "mytts", the function should be named `mytts_at_gradio_settings_page`.
#
# 2. The name of the function call `xxxx_model_alltalk_settings(model_config_data)`.
#    This should match the name you gave to the function that sets up the engine-specific settings page in Gradio.
#    If you named that function `mytts_model_alltalk_settings`, then the call should be:
#    `mytts_model_alltalk_settings(model_config_data)`
#
# After making these changes, this function will create and return the Gradio app for your TTS engine's settings page.
def vits_at_gradio_settings_page(model_config_data):
    app = vits_model_alltalk_settings(model_config_data)
    return app
def vits_at_gradio_settings_page(model_config_data):
    app = vits_model_alltalk_settings(model_config_data)
    return app
