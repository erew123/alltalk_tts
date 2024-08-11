import os
import json
import requests
import gradio as gr
from tqdm import tqdm
from pathlib import Path
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
def parler_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr,  alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
    # Load the model_config_data from the JSON file
    with open(os.path.join(this_dir, "model_settings.json"), "r") as f:
        model_config_data = json.load(f)
    # Update the settings and openai_voices dictionaries with the new values
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
    with gr.Blocks(title="parler TTS") as app:
        with gr.Tab("Default Settings"):
            with gr.Row():
                lowvram_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="Low VRAM" if model_config_data["model_capabilties"]["lowvram_capable"] else "Low VRAM N/A", value="Enabled" if model_config_data["settings"]["lowvram_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["lowvram_capable"])
                deepspeed_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="DeepSpeed Activate" if model_config_data["model_capabilties"]["deepspeed_capable"] else "DeepSpeed N/A", value="Enabled" if model_config_data["settings"]["deepspeed_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["deepspeed_capable"])
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
                            details_text = gr.Textbox(label="Details", show_label=False, lines=5, interactive=False, value="In this section, you can set the default settings for this TTS engine. Settings that are not supported by the current engine will be greyed out and cannot be selected. Default voices specified here will be used when no specific voice is provided in the TTS generation request. If a voice is specified in the request, it will override these default settings. When using the OpenAI API compatable API with this TTS engine, the voice mappings will be applied. As the OpenAI API has a limited set of 6 voices, these mappings ensure compatibility by mapping the OpenAI voices to the available voices in this TTS engine.")
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            submit_button.click(parler_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr], outputs=output_message)

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
                gr.Markdown("""
                ####  🟧 DeepSpeed Capable
                DeepSpeed is a deep learning optimization library that can significantly speed up model training and inference. If a model is DeepSpeed capable, it means it can utilize DeepSpeed to accelerate the generation of text-to-speech output. This requires the model to be loaded into CUDA/VRAM on an Nvidia GPU and the model's inference method to support DeepSpeed.
                ####  🟧 Pitch Capable
                Pitch refers to the highness or lowness of a sound. If a model is pitch capable, it can adjust the pitch of the generated speech, allowing for more expressive and varied output.
                #### 🟧 Generation Speed Capable
                Generation speed refers to the rate at which the model can generate text-to-speech output. If a model is generation speed capable, it means the speed of the generated speech can be adjusted, making it faster or slower depending on the desired output.
                #### 🟧 Repetition Penalty Capable
                Repetition penalty is a technique used to discourage the model from repeating the same words or phrases multiple times in the same sounding way. If a model is repetition penalty capable, it can apply this penalty during generation to improve the diversity and naturalness of the output.
                #### 🟧 Multi-Languages Capable
                Multi-language capable models can generate speech in multiple languages. This means that the model has been trained on data from different languages and can switch between them during generation. Some models are language-specific.
                #### 🟧 Multi-Voice Capable
                Multi-voice capable models generate speech in multiple voices or speaking styles. This means that the model has been trained on data from different speakers and can mimic their voices during generation, or is a voice cloning model that can generate speech based on the input sample.
                """)
                gr.Markdown("""
                #### 🟧 Streaming Capable
                Streaming refers to the ability to generate speech output in real-time, without the need to generate the entire output before playback. If a model is streaming capable, it can generate speech on-the-fly, allowing for faster response times and more interactive applications.
                #### 🟧 Low VRAM Capable
                VRAM (Video Random Access Memory) is a type of memory used by GPUs to store and process data. If a model is low VRAM capable, it can efficiently utilize the available VRAM by moving data between CPU and GPU memory as needed, allowing for generation even on systems with limited VRAM where it may be competing with an LLM model for VRAM.
                #### 🟧 Temperature Capable
                Temperature is a hyperparameter that controls the randomness of the generated output. If a model is temperature capable, the temperature can be adjusted to make the output more or less random, affecting the creativity and variability of the generated speech.
                #### 🟧 Multi-Model Capable Engine
                If an engine is multi-model capable, it means that it can support and utilize multiple models for text-to-speech generation. This allows for greater flexibility and the ability to switch between different models depending on the desired output. Different models may be capable of different languages, specific languages, voices, etc.
                #### 🟧 Default Audio Output Format
                Specifies the file format in which the generated speech will be saved. Common audio formats include WAV, MP3, FLAC, Opus, AAC, and PCM. If you want different outputs, you can set the transcode function to change the output audio, though transcoding will add a little time to the generation and is not available for streaming generation.
                #### 🟧 Windows/Linux/Mac Support
                These indicators show whether the model and engine are compatible with Windows, Linux, or macOS. However, additional setup or requirements may be necessary to ensure full functionality on your operating system. Please note that full platform support has not been extensively tested.
                """)

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

        ###################################################################################################
        # REQUIRED CHANGE                                                                                 #
        # Add any engine specific help, bugs, issues, operating system specifc requirements/setup in here #
        # Please use Markdown format, so gr.Markdown() with your markdown inside it.                      #
        ###################################################################################################
        with gr.Tab("Engine Help"):
            with gr.Row():
                gr.Markdown("""
                    ### 🟧 Where are the Parler models stored?
                    This extension will download the models to `/alltalk_tts/models/parler/` folder.<br>
                    
                    ### 🟧 Where are the voices stored for parler models?
                    Parler is unlike any other TTS engine. It is actually mode like Stable Diffusion, where you write a text based description of what you want your image to look like, but in this case, you describe what you want the voice to sound like. For example `A female speaker with an enthusiastic and lively voice. Her tone is bright and energetic, with a fast pace and a lot of inflection. The audio is clear and vibrant.` 
                    
                    This means you can have pretty much any voice you want, however, the downside to this is, there is little consistency of the voices, meaning that no 2x TTS generations will sound exactly like the same voice. You can however use one of the `Native` built in voices to give a level of consistency. Please refer to the Parler website for more infomation https://github.com/huggingface/parler-tts.<br>
                   
                    ### 🟧 So how do I create my voices for Parler?
                    On the settings page for Parler, there is a Voice Editor tab. In there you can add/remove/ammend voices as you wish, or edit the `parler_voices.json` file stored in `system/tts_engines/parler/`.
                   
                    ### 🟧 Where are the outputs stored & Automatic output wav file deletion
                    Voice outputs are stored in `/alltalk_tts/outputs/`. You can configure automatic maintenance deletion of old wav files by setting `Del WAV's older than` in the global settings.<br>
                    
                    > When `Disabled`, your output wav files will be left untouched.<br>
                    > When set to a setting `1 Day` or greater, your output wav files older than that time period will be automatically deleted on start-up of AllTalk.<br>
                    
                    ### 🟧 Skipped text/speech
                    At time of writing, this model is 1x day old and I could see reports of issues with some TTS not being generated, some longer text missing the middle portion of the text etc. You can confirm what AllTalk sent to the Parler TTS engine by looking at the command prompt/terminal window. If your text shows up there, it was sent to the Parler TTS engine. Please refer to the Parler website for more details https://github.com/huggingface/parler-tts. 
                    """)
                gr.Markdown("""
                    ### 🟧 How do I use Parler's voice generation system?
                    To create a voice with Parler:

                    Describe the voice characteristics in your prompt. For example: `Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.`
                    
                    Use one of the 34 inbuilt `Native` speaker names for a level of consistency: `Aaron, Alisa, Anna, Barbara, Bill, Brenda, Bruce, Carol, David, Eileen, Emily, Eric, Gary, James, Jason, Jenna, Jerry, Jon, Jordan, Joy, Karen, Laura, Lauren, Lea, Mike, Naomie, Patrick, Rebecca, Rick, Rose, Tina, Tom, Will, Yann`
                    
                    Control audio quality in the voice characteristics prompt:

                    > Include `very clear audio` for highest quality<br>
                    > Use `very noisy audio` for high levels of background noise

                    Adjust other features in your voice characteristics prompt:

                    > Gender<br>
                    > Speaking rate<br>
                    > Pitch<br>
                    > Reverberation
                    
                    Remember, while you can create diverse voices, each generation may sound slightly different even with the same description.

                    In the text sent for TTS generation, you can use punctuation to control speech rhythm (e.g. use `,` commas for small breaks)
                    
                    ### 🟧 UserWarning: 1Torch was not compiled with flash attention
                    This is a limitation of Windows Pytorch not yet having full flash attention support on later builds of Pytorch (is my understanding at this time).
                    """)

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