import os
import json
import gradio as gr
from pathlib import Path
from .help_content import AllTalkHelpContent

this_dir = Path(__file__).parent.resolve()                         # Sets up self.this_dir as a variable for the folder THIS script is running in.
main_dir = Path(__file__).parent.parent.parent.parent.resolve()    # Sets up self.main_dir as a variable for the folder AllTalk is running in

##########################################################################
# REQUIRED CHANGE                                                        #
# Populate the voices list, using the method specific to your TTS engine #
##########################################################################
def chatterbox_voices_file_list():
    """
    For Chatterbox TTS, voices are provided via audio prompts.
    This function lists available WAV files in the voices directory that can be used as prompts.
    """
    directory = main_dir / "voices"
    voices = []
    
    # Check files in main voices directory
    for f in directory.glob("*.wav"):
        voices.append(f.name)
    
    # Check subdirectories
    for folder in directory.iterdir():
        if folder.is_dir():
            for wav_file in folder.glob("*.wav"):
                # If it's in a subdirectory, add the subdirectory name
                rel_path = wav_file.relative_to(directory)
                voices.append(str(rel_path))
    
    # Add default option
    voices.insert(0, "default")
    
    if len(voices) == 1:  # Only "default" was added
        return ["default", "No Voice Samples Found"]
        
    return sorted(voices)

######################################################
# REQUIRED CHANGE                                    #
# Imports and saves the TTS engine-specific settings #
######################################################
def chatterbox_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
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
def chatterbox_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = chatterbox_voices_file_list()
    
    with gr.Blocks(title="Chatterbox TTS", analytics_enabled=False) as app:
        with gr.Tab("Default Settings"):
            with gr.Row():
                lowvram_enabled_gr = gr.Radio(
                    choices={"Enabled": "true", "Disabled": "false"}, 
                    label="Low VRAM" if model_config_data["model_capabilties"]["lowvram_capable"] else "Low VRAM N/A", 
                    value="Enabled" if model_config_data["settings"]["lowvram_enabled"] else "Disabled", 
                    interactive=model_config_data["model_capabilties"]["lowvram_capable"]
                )
                deepspeed_enabled_gr = gr.Radio(
                    choices={"Enabled": "true", "Disabled": "false"}, 
                    label="DeepSpeed Activate" if model_config_data["model_capabilties"]["deepspeed_capable"] else "DeepSpeed N/A", 
                    value="Enabled" if model_config_data["settings"]["deepspeed_enabled"] else "Disabled", 
                    interactive=model_config_data["model_capabilties"]["deepspeed_capable"]
                )
                temperature_set_gr = gr.Slider(
                    value=float(model_config_data["settings"]["temperature_set"]), 
                    minimum=0, maximum=1, step=0.05, 
                    label="Temperature" if model_config_data["model_capabilties"]["temperature_capable"] else "Temperature N/A", 
                    interactive=model_config_data["model_capabilties"]["temperature_capable"]
                )
                repetitionpenalty_set_gr = gr.Slider(
                    value=float(model_config_data["settings"]["repetitionpenalty_set"]), 
                    minimum=1, maximum=20, step=1, 
                    label="Repetition Penalty" if model_config_data["model_capabilties"]["repetitionpenalty_capable"] else "Repetition N/A", 
                    interactive=model_config_data["model_capabilties"]["repetitionpenalty_capable"]
                )
                pitch_set_gr = gr.Slider(
                    value=float(model_config_data["settings"]["pitch_set"]), 
                    minimum=-10, maximum=10, step=1, 
                    label="Pitch" if model_config_data["model_capabilties"]["pitch_capable"] else "Pitch N/A", 
                    interactive=model_config_data["model_capabilties"]["pitch_capable"]
                )
                generationspeed_set_gr = gr.Slider(
                    value=float(model_config_data["settings"]["generationspeed_set"]), 
                    minimum=0.30, maximum=2.00, step=0.10, 
                    label="Speed" if model_config_data["model_capabilties"]["generationspeed_capable"] else "Speed N/A", 
                    interactive=model_config_data["model_capabilties"]["generationspeed_capable"]
                )
            
            with gr.Row():
                with gr.Column():
                    def_character_voice_gr = gr.Dropdown(
                        choices=voice_list, 
                        label="Default/Character Voice", 
                        value=model_config_data["settings"]["def_character_voice"]
                    )
                    def_narrator_voice_gr = gr.Dropdown(
                        choices=voice_list, 
                        label="Narrator Voice", 
                        value=model_config_data["settings"]["def_narrator_voice"]
                    )
                    
                with gr.Column():
                    alloy_gr = gr.Dropdown(
                        choices=voice_list, 
                        label="OpenAI Alloy Voice", 
                        value=model_config_data["openai_voices"]["alloy"]
                    )
                    echo_gr = gr.Dropdown(
                        choices=voice_list, 
                        label="OpenAI Echo Voice", 
                        value=model_config_data["openai_voices"]["echo"]
                    )
                    fable_gr = gr.Dropdown(
                        choices=voice_list, 
                        label="OpenAI Fable Voice", 
                        value=model_config_data["openai_voices"]["fable"]
                    )
                    
                with gr.Column():
                    nova_gr = gr.Dropdown(
                        choices=voice_list, 
                        label="OpenAI Nova Voice", 
                        value=model_config_data["openai_voices"]["nova"]
                    )
                    onyx_gr = gr.Dropdown(
                        choices=voice_list, 
                        label="OpenAI Onyx Voice", 
                        value=model_config_data["openai_voices"]["onyx"]
                    )
                    shimmer_gr = gr.Dropdown(
                        choices=voice_list, 
                        label="OpenAI Shimmer Voice", 
                        value=model_config_data["openai_voices"]["shimmer"]
                    )
            
            with gr.Row():
                submit_button = gr.Button("Update Settings", variant="primary")
                result_label = gr.Label(label="Result")
            
            submit_button.click(
                chatterbox_model_update_settings,
                inputs=[
                    def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, 
                    deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, 
                    pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, 
                    nova_gr, onyx_gr, shimmer_gr
                ],
                outputs=result_label
            )
        
        with gr.Tab("Engine Capabilities"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION)
                    gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION1)
                with gr.Column():
                    gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION2)
        
        with gr.Tab("Default Settings Help"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS)
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS1)
                with gr.Column():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS2)
        
        with gr.Tab("Chatterbox TTS Help"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(AllTalkHelpContent.HELP_PAGE)
                    gr.Markdown(AllTalkHelpContent.HELP_PAGE1)
                with gr.Column():
                    gr.Markdown(AllTalkHelpContent.HELP_PAGE2)
    
    return app

def chatterbox_at_gradio_settings_page(model_config_data):
    app = chatterbox_model_alltalk_settings(model_config_data)
    return app 