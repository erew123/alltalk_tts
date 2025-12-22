import os
import json
import gradio as gr
from pathlib import Path
from .help_content import AllTalkHelpContent
this_dir = Path(__file__).parent.resolve()
main_dir = Path(__file__).parent.parent.parent.parent.resolve()

##########################################################################
# REQUIRED CHANGE                                                        #
# Populate the voices list, using the method specific to your TTS engine #
##########################################################################
def orpheus_voices_file_list():
    voices_file = os.path.join(this_dir, "orpheus_voices.json")
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
def orpheus_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
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
def orpheus_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = orpheus_voices_file_list()
    with gr.Blocks(title="Orpheus TTS", analytics_enabled=False) as app:
        with gr.Tab("Default Settings"):
            with gr.Row():
                lowvram_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="Low VRAM" if model_config_data["model_capabilties"]["lowvram_capable"] else "Low VRAM N/A", value="Enabled" if model_config_data["settings"]["lowvram_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["lowvram_capable"])
                deepspeed_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="DeepSpeed Activate" if model_config_data["model_capabilties"]["deepspeed_capable"] else "DeepSpeed N/A", value="Enabled" if model_config_data["settings"]["deepspeed_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["deepspeed_capable"])
                temperature_set_gr = gr.Slider(value=float(model_config_data["settings"]["temperature_set"]), minimum=0, maximum=1, step=0.05, label="Temperature" if model_config_data["model_capabilties"]["temperature_capable"] else "Temperature N/A", interactive=model_config_data["model_capabilties"]["temperature_capable"])
                repetitionpenalty_set_gr = gr.Slider(value=float(model_config_data["settings"]["repetitionpenalty_set"]), minimum=1, maximum=20, step=0.1, label="Repetition Penalty" if model_config_data["model_capabilties"]["repetitionpenalty_capable"] else "Repetition N/A", interactive=model_config_data["model_capabilties"]["repetitionpenalty_capable"])
                pitch_set_gr = gr.Slider(value=float(model_config_data["settings"]["pitch_set"]), minimum=-10, maximum=10, step=1, label="Pitch" if model_config_data["model_capabilties"]["pitch_capable"] else "Pitch N/A", interactive=model_config_data["model_capabilties"]["pitch_capable"])
                generationspeed_set_gr = gr.Slider(value=float(model_config_data["settings"]["generationspeed_set"]), minimum=0.5, maximum=2.0, step=0.1, label="Speed" if model_config_data["model_capabilties"]["generationspeed_capable"] else "Speed N/A", interactive=model_config_data["model_capabilties"]["generationspeed_capable"])
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
                            details_text = gr.Textbox(label="Details", show_label=False, lines=5, interactive=False, value="Configure default settings and voice mappings for Orpheus TTS. Temperature and repetition penalty can be adjusted. Use emotion tags like <laugh>, <sigh>, <cough> in your text for expressive speech. See the Emotion Tags tab for details.")
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            submit_button.click(orpheus_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr], outputs=output_message)
            with gr.Accordion("HELP - Understanding TTS Engine Default Settings Page", open=False):
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
                        gr.Textbox(label="TTS Engine Name", value="orpheus", interactive=False)
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
        # Model Info Tab - Orpheus model information and installation                                                                                                                                          #
        #######################################################################################################################################################################################################
        with gr.Tab("Model Information"):
            with gr.Row():
                gr.Markdown("""
                ## Orpheus Model Management

                Orpheus TTS requires the appropriate Python packages to be installed based on your platform.

                ### Installation

                **macOS (Apple Silicon with Metal):**
                ```bash
                pip install orpheus-cpp
                pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
                ```

                **Windows/Linux (NVIDIA CUDA):**
                ```bash
                pip install orpheus-speech
                pip install vllm==0.7.3
                ```

                ### Available Models

                | Model | HuggingFace ID | Description |
                |-------|----------------|-------------|
                | Production | `canopylabs/orpheus-3b-0.1-ft` | Finetuned model optimized for everyday TTS |
                | Pretrained | `canopylabs/orpheus-3b-0.1-pretrained` | Base model trained on 100k+ hours of English |

                ### First Run
                When you first use Orpheus TTS, the model files will be automatically downloaded from HuggingFace.
                This may take several minutes depending on your internet connection (~6GB download).

                ### System Requirements

                **macOS:**
                - Apple Silicon (M1/M2/M3/M4) with Metal support
                - Minimum 8GB RAM (16GB recommended)

                **Windows/Linux:**
                - NVIDIA GPU with CUDA support
                - Minimum 8GB VRAM recommended
                - CUDA toolkit installed
                """, elem_classes="custom-markdown")

        #####################################
        # Voice Browser - View all voices   #
        #####################################
        with gr.Tab("Voice Browser"):
            def load_voices():
                voices_file = os.path.join(this_dir, "orpheus_voices.json")
                if os.path.exists(voices_file):
                    with open(voices_file, "r") as f:
                        return json.load(f)
                else:
                    return {"voices": []}

            def format_voices_table(gender_filter):
                voices = load_voices()
                if gender_filter == "All":
                    filtered = voices["voices"]
                else:
                    filtered = [v for v in voices["voices"] if v.get("gender") == gender_filter.lower()]

                if not filtered:
                    return "No voices found."
                rows = ["| Voice Name | Gender | Description |", "|---|---|---|"]
                for v in filtered:
                    rows.append(f"| {v['voice_name']} | {v['gender'].capitalize()} | {v['description']} |")
                return "\n".join(rows)

            gr.Markdown("""
            ## Available Voices

            Orpheus TTS includes 8 high-quality voices ranked by conversational realism.
            **tara** is recommended as the most natural-sounding voice.
            """)

            with gr.Row():
                gender_filter = gr.Dropdown(
                    choices=["All", "Female", "Male"],
                    value="All",
                    label="Filter by Gender"
                )
            with gr.Row():
                voices_table = gr.Markdown(format_voices_table("All"))

            gender_filter.change(format_voices_table, inputs=gender_filter, outputs=voices_table)

        #####################################
        # Emotion Tags - Reference          #
        #####################################
        with gr.Tab("Emotion Tags"):
            gr.Markdown("""
            ## Emotion Tags Reference

            Orpheus TTS supports inline emotion tags that allow you to add expressive elements to generated speech.
            Simply include these tags in your text where you want the emotion to occur.

            ### Available Tags

            | Tag | Effect | Example Usage |
            |-----|--------|---------------|
            | `<laugh>` | Laughter | "That's hilarious! <laugh>" |
            | `<chuckle>` | Light chuckle | "Well, that's funny <chuckle>" |
            | `<sigh>` | Sighing | "<sigh> I suppose you're right." |
            | `<cough>` | Coughing | "Sorry <cough> excuse me." |
            | `<sniffle>` | Sniffling | "It's so sad <sniffle>" |
            | `<groan>` | Groaning | "<groan> Not again!" |
            | `<yawn>` | Yawning | "I'm so tired <yawn>" |
            | `<gasp>` | Gasping | "<gasp> I can't believe it!" |

            ### Usage Tips

            - **Place tags where natural**: Put emotion tags where a person would naturally express that emotion
            - **Don't overuse**: Too many emotion tags can make speech sound unnatural
            - **Combine with punctuation**: Tags work well with exclamation marks and question marks
            - **Test different placements**: The timing of the emotion can change based on tag placement

            ### Examples

            **Happy announcement:**
            ```
            I just got the job! <laugh> I can't believe it actually happened!
            ```

            **Tired response:**
            ```
            <yawn> Sorry, I was up all night working on this project.
            ```

            **Surprised reaction:**
            ```
            <gasp> You're getting married? That's wonderful news!
            ```

            **Frustrated statement:**
            ```
            <groan> The deadline got moved up again. <sigh> Here we go.
            ```
            """, elem_classes="custom-markdown")

        #####################################
        # Voice Cloning - Reference         #
        #####################################
        with gr.Tab("Voice Cloning"):
            gr.Markdown("""
            ## Zero-Shot Voice Cloning

            Orpheus supports zero-shot voice cloning using the **pretrained** model.
            This allows you to clone any voice from a short audio sample without fine-tuning.

            ### Current Status

            The AllTalk integration currently uses the **finetuned** model with 8 built-in voices.
            Voice cloning requires additional setup with the pretrained model.

            ### How Voice Cloning Works

            1. **Encode Reference Audio**: Convert 5-30 seconds of speech to SNAC tokens
            2. **Provide Transcript**: Include accurate text of what's spoken
            3. **Condition Generation**: Model learns voice characteristics from examples
            4. **Generate New Speech**: Produce speech in the cloned voice

            ### Requirements

            | Component | Description |
            |-----------|-------------|
            | Model | `orpheus-3b-0.1-pretrained` (not finetuned) |
            | SNAC | `hubertsiuzdak/snac_24khz` for audio encoding |
            | Reference Audio | 5-30 seconds, clean, 24kHz |
            | Transcript | Accurate text of reference speech |

            ### Basic Code Example

            ```python
            from snac import SNAC

            # Load SNAC for audio tokenization
            snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

            # Encode reference audio to tokens
            audio_codes = snac.encode(reference_audio)

            # Structure prompt: [reference_text] + [audio_codes] + [new_text]
            # Run inference with pretrained model
            # Decode output tokens to audio
            output_audio = snac.decode(generated_codes)
            ```

            ### Tips for Best Results

            - Use **clean, noise-free** reference audio
            - Provide **accurate transcripts** matching the audio
            - Include **3-5 text-speech pairs** for better voice matching
            - Use audio with **varied intonation** for expressive cloning
            - Reference clips should be **5-30 seconds** each

            ### Resources

            - [OrpheusTTS-WebUI](https://github.com/Saganaki22/OrpheusTTS-WebUI) - Community UI with voice cloning
            - [Pretrained Model Colab](https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS)
            - [GitHub Issue #6](https://github.com/canopyai/Orpheus-TTS/issues/6) - Voice cloning discussion

            ### Future AllTalk Integration

            Full voice cloning support is planned and would include:
            - Pretrained model option in model selector
            - UI for uploading reference audio
            - Transcript input field
            - SNAC integration for audio encoding/decoding
            """, elem_classes="custom-markdown")

        ###################################################################################################
        # Engine Help Tab                                                                                 #
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
def orpheus_at_gradio_settings_page(model_config_data):
    app = orpheus_model_alltalk_settings(model_config_data)
    return app
