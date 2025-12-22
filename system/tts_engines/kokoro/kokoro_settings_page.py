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
def kokoro_voices_file_list():
    voices_file = os.path.join(this_dir, "kokoro_voices.json")
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
def kokoro_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
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
def kokoro_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = kokoro_voices_file_list()
    with gr.Blocks(title="Kokoro TTS", analytics_enabled=False) as app:
        with gr.Tab("Default Settings"):
            with gr.Row():
                lowvram_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="Low VRAM" if model_config_data["model_capabilties"]["lowvram_capable"] else "Low VRAM N/A", value="Enabled" if model_config_data["settings"]["lowvram_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["lowvram_capable"])
                deepspeed_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="DeepSpeed Activate" if model_config_data["model_capabilties"]["deepspeed_capable"] else "DeepSpeed N/A", value="Enabled" if model_config_data["settings"]["deepspeed_enabled"] else "Disabled", interactive=model_config_data["model_capabilties"]["deepspeed_capable"])
                temperature_set_gr = gr.Slider(value=float(model_config_data["settings"]["temperature_set"]), minimum=0, maximum=1, step=0.05, label="Temperature" if model_config_data["model_capabilties"]["temperature_capable"] else "Temperature N/A", interactive=model_config_data["model_capabilties"]["temperature_capable"])
                repetitionpenalty_set_gr = gr.Slider(value=float(model_config_data["settings"]["repetitionpenalty_set"]), minimum=1, maximum=20, step=1, label="Repetition Penalty" if model_config_data["model_capabilties"]["repetitionpenalty_capable"] else "Repetition N/A", interactive=model_config_data["model_capabilties"]["repetitionpenalty_capable"])
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
                            details_text = gr.Textbox(label="Details", show_label=False, lines=5, interactive=False, value="Configure default settings and voice mappings for Kokoro TTS. Speed can be adjusted from 0.5x to 2.0x. The model supports 54 voices across 9 languages. See the Help section below for detailed information.")
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            submit_button.click(kokoro_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr], outputs=output_message)
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
                        gr.Textbox(label="TTS Engine Name", value="kokoro", interactive=False)
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
        # Model Info Tab - Kokoro models are auto-managed by the pip package                                                                                                                                  #
        #######################################################################################################################################################################################################
        with gr.Tab("Model Information"):
            with gr.Row():
                gr.Markdown("""
                ## Kokoro Model Management

                The Kokoro TTS model is automatically managed by the `kokoro` Python package.

                ### First Run
                When you first use Kokoro TTS, the model files will be automatically downloaded.
                This may take a few minutes depending on your internet connection.

                ### Model Location
                Models are cached by the `kokoro` package in your system's default cache directory.

                ### Supported Languages
                - **American English** (a): 11 female, 9 male voices
                - **British English** (b): 4 female, 4 male voices
                - **Japanese** (j): 4 female, 1 male voice
                - **Mandarin Chinese** (z): 4 female, 4 male voices
                - **Spanish** (e): 1 female, 2 male voices
                - **French** (f): 1 female voice
                - **Hindi** (h): 2 female, 2 male voices
                - **Italian** (i): 1 female, 1 male voice
                - **Brazilian Portuguese** (p): 1 female, 2 male voices

                ### System Requirements
                - **espeak-ng** must be installed on your system:
                  - macOS: `brew install espeak-ng`
                  - Linux: `sudo apt-get install espeak-ng`
                  - Windows: Download from espeak-ng releases
                """, elem_classes="custom-markdown")

        #####################################
        # Voice Browser - View all voices   #
        #####################################
        with gr.Tab("Voice Browser"):
            # Load voices from the JSON file
            def load_voices():
                voices_file = os.path.join(this_dir, "kokoro_voices.json")
                if os.path.exists(voices_file):
                    with open(voices_file, "r") as f:
                        return json.load(f)
                else:
                    return {"voices": []}

            def get_voices_by_language(lang_filter):
                voices = load_voices()
                if lang_filter == "All Languages":
                    return voices["voices"]
                lang_map = {
                    "American English": "en-us",
                    "British English": "en-gb",
                    "Japanese": "ja",
                    "Mandarin Chinese": "zh",
                    "Spanish": "es",
                    "French": "fr",
                    "Hindi": "hi",
                    "Italian": "it",
                    "Portuguese": "pt"
                }
                target_lang = lang_map.get(lang_filter, "")
                return [v for v in voices["voices"] if v.get("language") == target_lang]

            def format_voices_table(lang_filter):
                filtered = get_voices_by_language(lang_filter)
                if not filtered:
                    return "No voices found for this language."
                rows = ["| Voice Name | Voice Code | Gender | Description |", "|---|---|---|---|"]
                for v in filtered:
                    rows.append(f"| {v['voice_name']} | {v['voice_code']} | {v['gender']} | {v['description']} |")
                return "\n".join(rows)

            with gr.Row():
                lang_filter = gr.Dropdown(
                    choices=["All Languages", "American English", "British English", "Japanese", "Mandarin Chinese", "Spanish", "French", "Hindi", "Italian", "Portuguese"],
                    value="All Languages",
                    label="Filter by Language"
                )
            with gr.Row():
                voices_table = gr.Markdown(format_voices_table("All Languages"))

            lang_filter.change(format_voices_table, inputs=lang_filter, outputs=voices_table)

        #####################################
        # Voice Blender - Create blends     #
        #####################################
        with gr.Tab("Voice Blender"):
            # Load voices and blends from JSON
            def load_voices_data():
                voices_file = os.path.join(this_dir, "kokoro_voices.json")
                if os.path.exists(voices_file):
                    with open(voices_file, "r") as f:
                        return json.load(f)
                return {"voices": [], "blends": []}

            def save_voices_data(data):
                voices_file = os.path.join(this_dir, "kokoro_voices.json")
                with open(voices_file, "w") as f:
                    json.dump(data, f, indent=2)

            def get_base_voices():
                """Get list of non-blend voices for selection"""
                data = load_voices_data()
                return [v["voice_code"] for v in data["voices"] if not v.get("is_blend", False)]

            def get_blends_list():
                """Get formatted list of saved blends"""
                data = load_voices_data()
                blends = [v for v in data["voices"] if v.get("is_blend", False)]
                if not blends:
                    return "No custom blends saved yet."
                rows = ["| Blend Name | Blend Code | Description |", "|---|---|---|"]
                for b in blends:
                    rows.append(f"| {b['voice_name']} | `{b['voice_code']}` | {b['description']} |")
                return "\n".join(rows)

            def create_blend(blend_name, voice1, weight1, voice2, weight2, voice3, weight3, description):
                """Create a voice blend and save it"""
                if not blend_name:
                    return "Error: Please enter a blend name.", get_blends_list()
                if not voice1:
                    return "Error: Please select at least one voice.", get_blends_list()

                # Build blend code
                parts = []
                if voice1 and weight1 > 0:
                    parts.append(f"{voice1}:{int(weight1)}")
                if voice2 and weight2 > 0:
                    parts.append(f"{voice2}:{int(weight2)}")
                if voice3 and weight3 > 0:
                    parts.append(f"{voice3}:{int(weight3)}")

                if len(parts) < 2:
                    return "Error: Please select at least 2 voices with weights > 0.", get_blends_list()

                blend_code = ",".join(parts)

                # Normalize weights to sum to 100
                total = sum([weight1 if voice1 else 0, weight2 if voice2 else 0, weight3 if voice3 else 0])
                if total != 100:
                    return f"Warning: Weights sum to {int(total)}, not 100. Blend saved anyway.", get_blends_list()

                # Save to voices JSON
                data = load_voices_data()

                # Check if blend name already exists
                for i, v in enumerate(data["voices"]):
                    if v["voice_name"] == blend_name and v.get("is_blend", False):
                        # Update existing blend
                        data["voices"][i] = {
                            "voice_name": blend_name,
                            "voice_code": blend_code,
                            "language": "en-us",  # Default to English for blends
                            "gender": "blend",
                            "description": description or f"Custom blend: {blend_code}",
                            "is_blend": True
                        }
                        save_voices_data(data)
                        return f"Updated blend '{blend_name}': {blend_code}", get_blends_list()

                # Add new blend
                data["voices"].append({
                    "voice_name": blend_name,
                    "voice_code": blend_code,
                    "language": "en-us",
                    "gender": "blend",
                    "description": description or f"Custom blend: {blend_code}",
                    "is_blend": True
                })
                save_voices_data(data)
                return f"Created blend '{blend_name}': {blend_code}", get_blends_list()

            def delete_blend(blend_name):
                """Delete a saved blend"""
                if not blend_name:
                    return "Error: Please enter a blend name to delete.", get_blends_list()

                data = load_voices_data()
                original_count = len(data["voices"])
                data["voices"] = [v for v in data["voices"] if not (v["voice_name"] == blend_name and v.get("is_blend", False))]

                if len(data["voices"]) == original_count:
                    return f"Error: Blend '{blend_name}' not found.", get_blends_list()

                save_voices_data(data)
                return f"Deleted blend '{blend_name}'", get_blends_list()

            base_voices = get_base_voices()

            gr.Markdown("""
            ## Voice Blender

            Create custom voices by blending multiple Kokoro voices together.
            Weights determine how much each voice contributes to the final sound (should sum to 100).

            **Example:** `af_sarah:60,am_adam:40` creates a voice that's 60% Sarah, 40% Adam.
            """)

            with gr.Row():
                with gr.Column():
                    blend_name_input = gr.Textbox(label="Blend Name", placeholder="e.g., my_custom_voice")
                    blend_description = gr.Textbox(label="Description (optional)", placeholder="e.g., Warm female with slight male depth")

            with gr.Row():
                with gr.Column():
                    voice1_select = gr.Dropdown(choices=base_voices, label="Voice 1", value=base_voices[0] if base_voices else None)
                    weight1_slider = gr.Slider(minimum=0, maximum=100, value=50, step=5, label="Weight 1 (%)")
                with gr.Column():
                    voice2_select = gr.Dropdown(choices=base_voices, label="Voice 2", value=base_voices[1] if len(base_voices) > 1 else None)
                    weight2_slider = gr.Slider(minimum=0, maximum=100, value=50, step=5, label="Weight 2 (%)")
                with gr.Column():
                    voice3_select = gr.Dropdown(choices=base_voices, label="Voice 3 (optional)", value=None)
                    weight3_slider = gr.Slider(minimum=0, maximum=100, value=0, step=5, label="Weight 3 (%)")

            with gr.Row():
                create_button = gr.Button("Create/Update Blend", variant="primary")
                delete_button = gr.Button("Delete Blend", variant="stop")

            with gr.Row():
                status_output = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Saved Blends")
            with gr.Row():
                blends_table = gr.Markdown(get_blends_list())

            create_button.click(
                create_blend,
                inputs=[blend_name_input, voice1_select, weight1_slider, voice2_select, weight2_slider, voice3_select, weight3_slider, blend_description],
                outputs=[status_output, blends_table]
            )

            delete_button.click(
                delete_blend,
                inputs=[blend_name_input],
                outputs=[status_output, blends_table]
            )

            with gr.Accordion("Voice Blending Tips", open=False):
                gr.Markdown("""
                ## Tips for Creating Voice Blends

                ### Weight Distribution
                - Weights should sum to **100** for predictable results
                - Use **60:40** for a subtle blend with one dominant voice
                - Use **50:50** for an equal mix of two voices
                - Use **50:30:20** for a three-voice blend

                ### Best Practices
                - Blend voices of the **same language** for best results
                - Mixing male and female voices creates unique androgynous tones
                - Start with small weight differences and adjust

                ### Using Blends
                - Saved blends appear in the voice selection dropdown
                - You can also use blend syntax directly: `af_sarah:60,am_adam:40`
                - Blends work with the API by passing the blend code as the voice parameter

                ### Limitations
                - Cross-language blends may produce unexpected results
                - Very unequal weights (e.g., 95:5) may not produce noticeable blending
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
def kokoro_at_gradio_settings_page(model_config_data):
    app = kokoro_model_alltalk_settings(model_config_data)
    return app
