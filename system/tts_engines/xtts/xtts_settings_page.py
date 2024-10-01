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

def xtts_voices_file_list():
    directory = main_dir / "voices"
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".wav")]
    return files

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

def xtts_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr,  alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
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

def xtts_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = xtts_voices_file_list()
    with gr.Blocks(title="Xtts TTS") as app:
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

            submit_button.click(xtts_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr], outputs=output_message)

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
                        gr.Textbox(label="TTS Engine Name", value="XTTS", interactive=False)
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

            with gr.Row():
                download_status = gr.Textbox(label="Download Status")

            def download_model(model_name):
                # Find the selected model in the available models
                selected_model = next(model for model in available_models["models"] if model["model_name"] == model_name)

                # Get the folder path and files to download
                folder_path = os.path.join(main_dir, "models", "xtts", selected_model["folder_path"])
                files_to_download = selected_model["files_to_download"]

                # Check if all files are already downloaded
                all_files_exists = all(os.path.exists(os.path.join(folder_path, file)) for file in files_to_download)

                if all_files_exists:
                    return "All files are already downloaded. No need to download again."
                else:
                    # Create the folder if it doesn't exist
                    os.makedirs(folder_path, exist_ok=True)

                    # Download the missing files
                    for file, url in files_to_download.items():
                        file_path = os.path.join(folder_path, file)
                        if not os.path.exists(file_path):
                            print(f"Downloading {file}...")

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

            download_button.click(download_model, inputs=model_dropdown, outputs=download_status)

            def show_confirm_cancel(model_name):
                all_files_exists = all(os.path.exists(os.path.join(main_dir, "models", "xtts", model["folder_path"], file)) for model in available_models["models"] if model["model_name"] == model_name for file in model["files_to_download"])

                if all_files_exists:
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]
                else:
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)]

            def confirm_download(model_name):
                download_model(model_name)
                return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)]

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
                gr.Markdown("""
                    ### 🟧 Using my own Finetuned models
                    Please your Finetuned models within their own folder in `/alltalk_tts/models/xtts/`. Once they are in here they will become availabe within the interface after reloading the XTTS engine.<br>
                            
                    ### 🟧 Using Single Voice Samples
                    Voice samples are stored in `/alltalk_tts/voices/` and should be named using the following format `name.wav`. These files will be listed as `name.wav` in the available voices list.<br>
                    
                    ### 🟧 Using Multiple Voice Samples
                    If you have multiple voice samples for a single voice, you can organize them into subfolders within the `/alltalk_tts/voices/` directory. Each subfolder should be named according to the voice it contains, up to 5 voice samples will be randomly selected for use.<br>
                                      
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Each subfolder should reflect the name or type of the voice it contains (e.g., `female_voice`, `male_voice`).<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• The voice samples inside each subfolder should follow the standard `.wav` format.<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• An example folder path would be `/alltalk_tts/voices/mynewvoice/` and this would be listed in the available voices list as `mynewvoice/`.<br>

                    This organization allows for easy selection and management of multiple voice samples while ensuring the system can correctly identify and utilize each voice. Manual CURL API requests would send the folder in the format `mynewvoice/`.
                    
                    ### 🟧 Where are the outputs stored & Automatic output wav file deletion
                    Voice outputs are stored in `/alltalk_tts/outputs/`. You can configure automatic maintenance deletion of old wav files by setting `Del WAV's older than` in the global settings.<br>
                    
                    > When `Disabled`, your output wav files will be left untouched.<br>
                    > When set to a setting `1 Day` or greater, your output wav files older than that time period will be automatically deleted on start-up of AllTalk.<br>
                    
                    ### 🟧 Where are the models stored?
                    This extension will download the models to `/alltalk_tts/models/xtts/` folder.<br>
                    
                    ### 🟧 API Local, XTTSv2 Generation methods & Speed
                    These two methods both produce sound output in slightly different ways. XTTSv2 is the perferable method a it supports DeepSpeed, which, if you have a system capable of DeepSpeed genereation, can result in a 2-3x speed gain in generation.
                    
                    ### 🟧 Hindi Support on XTTS
                    Currently Hindi only works on XTTS model 2.0.3 and it has to be loaded as the API Local method.
                    """)
                gr.Markdown("""
                    ### 🟧 How do I create a new voice sample?
                    To create a new voice sample, you need to make a wav file that is `22050Hz`, `Mono`, `16 bit` and between 6 to 30 seconds long, though 8 to 10 seconds is usually good enough. The model can handle up to 30 second samples, however I've not noticed any improvement in voice output from much longer clips.<br><br>
                    You want to find a nice clear selection of audio, so lets say you wanted to clone your favourite celebrity. You may go looking for an interview where they are talking. Pay close attention to the audio you are listening to and trying to sample. Are there noises in the background, hiss on the soundtrack, a low humm, some quiet music playing or something? The better quality the audio the better the final TTS result. Don't forget, the AI that processes the sounds can hear everything in your sample and it will use them in the voice its trying to recreate.<br><br>
                    Try make your clip one of nice flowing speech, like the included example files. No big pauses, gaps or other sounds. Preferably a sample that the person you are trying to copy will show a little vocal range and emotion in their voice. Also, try to avoid a clip starting or ending with breathy sounds (breathing in/out etc).<br>
                            
                    ### 🟧 Editing your sample!
                    So, you've downloaded your favourite celebrity interview off YouTube, from here you need to chop it down to 6 to 30 seconds in length and resample it. If you need to clean it up, do audio processing, volume level changes etc, do this before down-sampling.<br><br>
                    Using the latest version of Audacity `select/highlight` your 6 to 30 second clip and:<br><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `Tracks` > `Resample to 22050Hz`<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `Tracks` > `Mix` > `Stereo to Mono`<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `File` > `Export Audio` saving it as a `WAV` of `22050Hz`.<br><br>
                    Save your generated wav file in the `/alltalk_tts/voices/` folder.<br>
                    
                    ### 🟧 Why doesnt it sound like XXX Person?
                    Maybe you might be interested in trying Finetuning of the XTTS model. Otherwise, the reasons can be that you:<br>
                    
                    > Didn't down-sample it as above.<br>
                    > Have a bad quality voice sample.<br>
                    > Try using the 2x different generation methods `API Local` and `XTTSv2 Local`, as they generate output in slightly different ways.<br>
                     
                    Additionally, use the RVC pipeline with a matching voice model, however, some samples just never seem to work correctly, so maybe try a different sample. Always remember though, this is an AI model attempting to re-create a voice, so you will never get a 100% match.
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

def xtts_at_gradio_settings_page(model_config_data):
    app = xtts_model_alltalk_settings(model_config_data)
    return app
def xtts_at_gradio_settings_page(model_config_data):
    app = xtts_model_alltalk_settings(model_config_data)
    return app