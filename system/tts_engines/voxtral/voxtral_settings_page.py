import os
import json
import requests
import gradio as gr
from pathlib import Path
from .help_content import AllTalkHelpContent
this_dir = Path(__file__).parent.resolve()
main_dir = Path(__file__).parent.parent.parent.parent.resolve()

##########################################################################
# REQUIRED CHANGE                                                        #
# Populate the voices list, using the method specific to your TTS engine #
##########################################################################
def voxtral_voices_file_list():
    voices_file = os.path.join(this_dir, "voxtral_voices.json")
    if os.path.exists(voices_file):
        with open(voices_file, "r") as f:
            voices_data = json.load(f)
            voices_list = [voice["voice_code"] for voice in voices_data["voices"]]
            return sorted(voices_list)
    else:
        return []


######################################################
# REQUIRED CHANGE                                    #
# Imports and saves the TTS engine-specific settings #
######################################################
def voxtral_model_update_settings(def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
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
    # Note: API URL, API key, and response format are saved separately via the Voxtral API tab
    # Save the updated model_config_data to the JSON file
    with open(os.path.join(this_dir, "model_settings.json"), "w") as f:
        json.dump(model_config_data, f, indent=4)
    return "Settings updated successfully!"


def voxtral_test_connection(api_url_gr, api_key_gr):
    """Test connectivity to the Voxtral API endpoint."""
    try:
        url = api_url_gr.rstrip("/")
        headers = {}
        if api_key_gr:
            headers["Authorization"] = f"Bearer {api_key_gr}"
        # Try the models endpoint first (vllm-omni), fall back to health check
        for endpoint in [f"{url}/v1/models", f"{url}/health", url]:
            try:
                resp = requests.get(endpoint, headers=headers, timeout=5)
                if resp.status_code < 500:
                    return f"Connected successfully to {endpoint} (HTTP {resp.status_code})"
            except requests.exceptions.RequestException:
                continue
        return f"Failed to connect to {url}. Is the server running?"
    except Exception as e:
        return f"Connection error: {str(e)}"


#######################################################
# REQUIRED CHANGE                                     #
# Sets up the engine-specific settings page in Gradio #
#######################################################
def voxtral_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    voice_list = voxtral_voices_file_list()
    with gr.Blocks(title="Voxtral TTS", analytics_enabled=False) as app:
        with gr.Tab("Default Settings"):
            with gr.Row():
                lowvram_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="Low VRAM N/A", value="Disabled", interactive=False)
                deepspeed_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="DeepSpeed N/A", value="Disabled", interactive=False)
                temperature_set_gr = gr.Slider(value=float(model_config_data["settings"]["temperature_set"]), minimum=0, maximum=1, step=0.05, label="Temperature N/A", interactive=False)
                repetitionpenalty_set_gr = gr.Slider(value=float(model_config_data["settings"]["repetitionpenalty_set"]), minimum=1, maximum=20, step=1, label="Repetition N/A", interactive=False)
                pitch_set_gr = gr.Slider(value=float(model_config_data["settings"]["pitch_set"]), minimum=-10, maximum=10, step=1, label="Pitch N/A", interactive=False)
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
                            details_text = gr.Textbox(label="Details", show_label=False, lines=5, interactive=False, value="Configure default settings and voice mappings for Voxtral TTS. This is an HTTP-client engine that connects to a local vllm-omni server or the Mistral Cloud API. Configure the API connection in the Voxtral API tab. See the Help section below for detailed information.")
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            submit_button.click(voxtral_model_update_settings, inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr], outputs=output_message)
            with gr.Accordion("HELP - Understanding TTS Engine Default Settings Page", open=False):
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS, elem_classes="custom-markdown")
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS1, elem_classes="custom-markdown")
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS2, elem_classes="custom-markdown")

        #####################################
        # Voxtral API Settings Tab          #
        #####################################
        with gr.Tab("Voxtral API"):
            gr.Markdown("### API Connection Settings")
            gr.Markdown("Configure the backend and connection for Voxtral TTS.")
            with gr.Row():
                backend_type_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("backend_type", "mlx-audio"),
                    label="Backend",
                    choices=["mlx-audio", "vllm-omni", "mistral-cloud"],
                    info="mlx-audio: macOS/Apple Silicon | vllm-omni: NVIDIA GPU | mistral-cloud: hosted API"
                )
                mlx_model_variant_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("mlx_model_variant", "bf16"),
                    label="MLX Model Variant",
                    choices=["bf16", "6bit", "4bit"],
                    info="bf16: best quality (~3GB) | 6bit: balanced (~1GB) | 4bit: smallest (~0.8GB). mlx-audio backend only."
                )
                api_url_gr = gr.Textbox(
                    value=model_config_data["settings"]["api_url"],
                    label="API URL",
                    placeholder="http://localhost:8000",
                    info="mlx-audio/vllm-omni: http://localhost:8000 | Mistral Cloud: https://api.mistral.ai"
                )
            with gr.Row():
                api_key_gr = gr.Textbox(
                    value=model_config_data["settings"]["api_key"],
                    label="API Key",
                    placeholder="Leave empty for local server",
                    type="password",
                    info="Required for Mistral Cloud. Leave empty for mlx-audio / vllm-omni."
                )
                response_format_gr = gr.Dropdown(
                    value=model_config_data["settings"]["response_format"],
                    label="Response Format",
                    choices=["wav", "mp3", "flac", "opus", "aac", "pcm"],
                    info="Audio output format. WAV is recommended for quality."
                )
            with gr.Row():
                auto_start_server_gr = gr.Checkbox(
                    value=model_config_data["settings"].get("auto_start_server", True),
                    label="Auto-Start Server (mlx-audio only)",
                    info="Automatically launch the mlx-audio server when the Voxtral engine loads. Only applies to the mlx-audio backend."
                )
                server_port_gr = gr.Number(
                    value=model_config_data["settings"].get("server_port", 7852),
                    label="Server Port",
                    precision=0,
                    info="Port for the auto-started mlx-audio server. Default: 7852"
                )
            with gr.Row():
                test_button = gr.Button("Test Connection", variant="primary")
                test_output = gr.Textbox(label="Connection Status", interactive=False, show_label=False)
            test_button.click(voxtral_test_connection, inputs=[api_url_gr, api_key_gr], outputs=test_output)
            with gr.Row():
                api_submit_button = gr.Button("Save API Settings")
                api_output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)

            def save_api_settings(backend_type, mlx_model_variant, api_url, api_key, response_format, auto_start_server, server_port):
                with open(os.path.join(this_dir, "model_settings.json"), "r") as f:
                    data = json.load(f)
                data["settings"]["backend_type"] = backend_type
                data["settings"]["mlx_model_variant"] = mlx_model_variant
                data["settings"]["api_url"] = api_url
                data["settings"]["api_key"] = api_key
                data["settings"]["response_format"] = response_format
                data["settings"]["auto_start_server"] = bool(auto_start_server)
                data["settings"]["server_port"] = int(server_port)
                with open(os.path.join(this_dir, "model_settings.json"), "w") as f:
                    json.dump(data, f, indent=4)
                return "API settings saved successfully!"

            api_submit_button.click(save_api_settings, inputs=[backend_type_gr, mlx_model_variant_gr, api_url_gr, api_key_gr, response_format_gr, auto_start_server_gr, server_port_gr], outputs=api_output_message)

            with gr.Accordion("Setup Instructions", open=True):
                gr.Markdown("""
                ## Quick Setup

                ### Option A: mlx-audio (macOS / Apple Silicon)
                ```bash
                pip install mlx-audio
                ```
                Set **Backend** to `mlx-audio`, leave **API Key** empty.

                **Auto-Start (recommended):** Enable **Auto-Start Server** above and AllTalk will automatically launch
                and stop the mlx-audio server when the Voxtral engine loads/unloads. The server runs on the configured
                **Server Port** (default 7852). The first launch may take a few minutes as the model (~3GB) downloads from HuggingFace.

                **Manual start:** Disable Auto-Start and run the server yourself:
                ```bash
                python -m mlx_audio.server --model mlx-community/Voxtral-4B-TTS-2603-mlx-bf16 --port 7852
                ```
                Then set **API URL** to `http://localhost:7852`.

                ### Option B: vllm-omni (Linux / NVIDIA GPU, 16GB+ VRAM)
                ```bash
                pip install -U vllm
                pip install git+https://github.com/vllm-project/vllm-omni.git --upgrade
                vllm serve mistralai/Voxtral-4B-TTS-2603 --omni
                ```
                Set **Backend** to `vllm-omni`, **API URL** to `http://localhost:8000`, leave **API Key** empty.
                vllm-omni also exposes `/v1/chat/completions`, so it can double as the LLM for narrative emotion detection.

                ### Option C: Mistral Cloud API (any platform, no GPU)
                1. Get an API key at https://console.mistral.ai
                2. Set **Backend** to `mistral-cloud`
                3. Set **API URL** to `https://api.mistral.ai`
                4. Enter your API key
                """, elem_classes="custom-markdown")

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
                        gr.Textbox(label="TTS Engine Name", value="voxtral", interactive=False)
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

        #####################################
        # Voice Browser - View all voices   #
        #####################################
        with gr.Tab("Voice Browser"):
            def load_voices():
                voices_file = os.path.join(this_dir, "voxtral_voices.json")
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
                    "English": "en",
                    "French": "fr",
                    "German": "de",
                    "Spanish": "es",
                    "Italian": "it",
                    "Portuguese": "pt",
                    "Dutch": "nl",
                    "Arabic": "ar",
                    "Hindi": "hi",
                }
                target_lang = lang_map.get(lang_filter, "")
                return [v for v in voices["voices"] if v.get("language") == target_lang]

            def format_voices_table(lang_filter):
                filtered = get_voices_by_language(lang_filter)
                if not filtered:
                    return "No voices found for this language."
                rows = ["| Voice Name | Voice Code | Gender | Style | Description |", "|---|---|---|---|---|"]
                for v in filtered:
                    rows.append(f"| {v['voice_name']} | {v['voice_code']} | {v['gender']} | {v.get('style', '')} | {v['description']} |")
                return "\n".join(rows)

            with gr.Row():
                lang_filter = gr.Dropdown(
                    choices=["All Languages", "English", "French", "German", "Spanish", "Italian", "Portuguese", "Dutch", "Arabic", "Hindi"],
                    value="All Languages",
                    label="Filter by Language"
                )
            with gr.Row():
                voices_table = gr.Markdown(format_voices_table("All Languages"))

            lang_filter.change(format_voices_table, inputs=lang_filter, outputs=voices_table)

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
def voxtral_at_gradio_settings_page(model_config_data):
    app = voxtral_model_alltalk_settings(model_config_data)
    return app
