import os
import json
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
def parler_voices_file_list():
    voices_file = os.path.join(main_dir, "system", "tts_engines", "parler", "parler_voices.json")
    if os.path.exists(voices_file):
        with open(voices_file, "r") as f:
            voices_data = json.load(f)
            voices_list = [voice["voice_name"] for voice in voices_data["voices"]]
            return sorted(voices_list)
    else:
        return []


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
def parler_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, streaming_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr,  alloy_gr, ash_gr, coral_gr, echo_gr, fable_gr, nova_gr, onyx_gr, sage_gr, shimmer_gr):
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
def parler_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = parler_voices_file_list()
    with gr.Blocks(title="parler TTS", analytics_enabled=False) as app:
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
            submit_button.click(parler_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, streaming_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, ash_gr, coral_gr, echo_gr, fable_gr, nova_gr, onyx_gr, sage_gr, shimmer_gr], outputs=output_message)
            with gr.Accordion("HELP - 🔊 Understanding TTS Engine Default Settings Page", open=False):
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS, elem_classes="custom-markdown")                               
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS1, elem_classes="custom-markdown")
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS2, elem_classes="custom-markdown")   

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
                        gr.Textbox(label="TTS Engine Name", value="parler", interactive=False)
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
            with gr.Row():
                # Load the available models from the JSON file
                with open(os.path.join(this_dir, "available_models.json"), "r") as f:
                    available_models = json.load(f)
                # Extract the model names for the dropdown
                model_names = [model["model_name"] for model in available_models["models"]]
                # Create the dropdown
                model_dropdown = gr.Dropdown(choices=sorted(model_names), label="Select Model", value=model_names[0])
                download_button = gr.Button("Download Model/Missing Files")

            def download_model(model_name, force_download=False):
                # Find the selected model in the available models
                selected_model = next(model for model in available_models["models"] if model["model_name"] == model_name)
                # Get the folder path and files to download
                folder_path = os.path.join(main_dir, "models", "parler", model_name)
                files_to_download = selected_model["files_to_download"]
                # Check if all files are already downloaded
                all_files_exists = all(os.path.exists(os.path.join(folder_path, os.path.basename(url.split('?')[0]))) for url in files_to_download)
                if all_files_exists and not force_download:
                    return "All files are already downloaded. No need to download again."
                else:
                    # Create the folder if it doesn't exist
                    os.makedirs(folder_path, exist_ok=True)
                    # Download the missing files
                    for url in files_to_download:
                        file_name = os.path.basename(url.split('?')[0])
                        file_path = os.path.join(folder_path, file_name)
                        if not os.path.exists(file_path) or force_download:
                            print(f"Downloading {file_name}...")
                            response = requests.get(url, stream=True)
                            total_size_in_bytes = int(response.headers.get("content-length", 0))
                            block_size = 1024  # 1 Kibibyte
                            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                            with open(file_path, "wb") as file:
                                for data in response.iter_content(block_size):
                                    progress_bar.update(len(data))
                                    file.write(data)
                            progress_bar.close()
                    return "Model downloaded successfully!"

            with gr.Row():
                download_status = gr.Textbox(label="Download Status")

            download_button.click(download_model, inputs=model_dropdown, outputs=download_status)

            def show_confirm_cancel(model_name):
                all_files_exists = all(os.path.exists(os.path.join(main_dir, "models", "parler", os.path.basename(file.split('?')[0]))) for model in available_models["models"] if model["model_name"] == model_name for file in model["files_to_download"])
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

        #####################################
        # Voices Editor - Unique to Parler #
        #####################################
        # Load voices from the JSON file
        def load_voices():
            voices_file = os.path.join(main_dir, "system", "tts_engines", "parler", "parler_voices.json")
            if os.path.exists(voices_file):
                with open(voices_file, "r") as f:
                    return json.load(f)
            else:
                return {"voices": []}

        # Save voices to the JSON file
        def save_voices(voices):
            voices_file = os.path.join(main_dir, "system", "tts_engines", "parler", "parler_voices.json")
            with open(voices_file, "w") as f:
                json.dump(voices, f, indent=2)

        # Convert voices to a list for display
        def voices_to_list(voices):
            return [f"{i+1}. {voice['voice_name']} - {voice['description']}" for i, voice in enumerate(voices["voices"])]

        # Populate text boxes with selected voice data
        def populate_voice_data(selected):
            if selected:
                index = int(selected.split(".")[0]) - 1
                voices = load_voices()
                selected_voice = voices["voices"][index]
                return selected_voice["voice_name"], selected_voice["description"], index, gr.update(interactive=False), "Save Voice"
            return "", "", -1, gr.update(interactive=True), "Add/Save Voice"

        # Add or update voice
        def add_update_voice(voice_name, description, selected_index, button_label):
            voices = load_voices()
            if selected_index >= 0:
                voices["voices"][selected_index]["description"] = description
            else:
                voices["voices"].append({"voice_name": voice_name, "description": description})
            save_voices(voices)
            return (gr.Dropdown(choices=voices_to_list(voices), value=voices_to_list(voices)[0] if voices["voices"] else None),
                    "", "", -1, gr.update(interactive=True), "Add/Save Voice")

        # Delete selected voice
        def delete_selected_voice(selected_index):
            voices = load_voices()
            if selected_index >= 0:
                del voices["voices"][selected_index]
                save_voices(voices)
            return (gr.Dropdown(choices=voices_to_list(voices), value=voices_to_list(voices)[0] if voices["voices"] else None),
                    "", "", -1, gr.update(interactive=True), "Add/Save Voice")

        # Clear text boxes
        def clear_text_boxes():
            return "", "", -1, gr.update(interactive=True), "Add/Save Voice"

        # Load initial voices
        voices = load_voices()

        with gr.Tab("Voice Editor"):
            with gr.Row():
                voice_name = gr.Textbox(label="Voice Name", scale=1)
                description = gr.Textbox(label="Voice Description", lines=4, scale=3)
            with gr.Row():
                add_update_button = gr.Button("Add/Save Voice")
                delete_button = gr.Button("Delete Selected Voice")
                clear_button = gr.Button("Clear Text Boxes")
            with gr.Row():
                voice_list = gr.Dropdown(
                    choices=voices_to_list(voices),
                    label="Voices",
                    interactive=True,
                    value=voices_to_list(voices)[0] if voices["voices"] else None
                )
            # Hidden state for selected index and button label
            selected_index = gr.Number(value=-1, visible=False)
            button_label = gr.State(value="Add/Save Voice")

            # Button click events
            voice_list.change(populate_voice_data, inputs=voice_list, outputs=[voice_name, description, selected_index, voice_name, button_label])
            add_update_button.click(add_update_voice, inputs=[voice_name, description, selected_index, button_label], outputs=[voice_list, voice_name, description, selected_index, voice_name, button_label])
            delete_button.click(delete_selected_voice, inputs=selected_index, outputs=[voice_list, voice_name, description, selected_index, voice_name, button_label])
            clear_button.click(clear_text_boxes, outputs=[voice_name, description, selected_index, voice_name, button_label])
            with gr.Accordion("HELP - 🎙️ Parler Voice Editor Help", open=False):
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.VOICE_EDITOR, elem_classes="custom-markdown")                               
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.VOICE_EDITOR1, elem_classes="custom-markdown")
                    gr.Markdown(AllTalkHelpContent.VOICE_EDITOR2, elem_classes="custom-markdown")            

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
def parler_at_gradio_settings_page(model_config_data):
    app = parler_model_alltalk_settings(model_config_data)
    return app
def parler_at_gradio_settings_page(model_config_data):
    app = parler_model_alltalk_settings(model_config_data)
    return app
