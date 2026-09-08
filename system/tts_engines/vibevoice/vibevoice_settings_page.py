import os
import json
import requests
import gradio as gr
from pathlib import Path
from .help_content import AllTalkHelpContent

this_dir = Path(__file__).parent.resolve()
main_dir = Path(__file__).parent.parent.parent.parent.resolve()


def _is_realtime(variant):
    s = (variant or "").lower()
    return "realtime" in s or "streaming" in s


def vibevoice_voices_file_list(variant=None):
    """List voices for the given variant. Falls back to whichever folder has files."""
    if variant is None:
        try:
            with open(os.path.join(this_dir, "model_settings.json")) as f:
                variant = json.load(f)["settings"].get("model_variant", "")
        except Exception:
            variant = ""
    if _is_realtime(variant):
        voices_dir, ext = this_dir / "voices_realtime", ".pt"
    else:
        voices_dir, ext = this_dir / "voices", ".wav"
    voices = []
    if voices_dir.is_dir():
        voices = sorted({f.stem for f in voices_dir.glob(f"**/*{ext}")})
    if not voices:
        return ["No Voices Found"]
    return voices


def vibevoice_model_update_settings(def_character_voice_gr, def_narrator_voice_gr,
                                    lowvram_enabled_gr, deepspeed_enabled_gr,
                                    temperature_set_gr, repetitionpenalty_set_gr,
                                    pitch_set_gr, generationspeed_set_gr,
                                    alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
    with open(os.path.join(this_dir, "model_settings.json"), "r") as f:
        cfg = json.load(f)
    # Route default voices to the per-variant fields so 1.5B and 0.5B keep separate defaults
    variant = cfg["settings"].get("model_variant", "")
    if _is_realtime(variant):
        cfg["settings"]["def_character_voice_realtime"] = def_character_voice_gr
        cfg["settings"]["def_narrator_voice_realtime"] = def_narrator_voice_gr
    else:
        cfg["settings"]["def_character_voice"] = def_character_voice_gr
        cfg["settings"]["def_narrator_voice"] = def_narrator_voice_gr
    cfg["openai_voices"]["alloy"] = alloy_gr
    cfg["openai_voices"]["echo"] = echo_gr
    cfg["openai_voices"]["fable"] = fable_gr
    cfg["openai_voices"]["nova"] = nova_gr
    cfg["openai_voices"]["onyx"] = onyx_gr
    cfg["openai_voices"]["shimmer"] = shimmer_gr
    cfg["settings"]["lowvram_enabled"] = lowvram_enabled_gr == "Enabled"
    cfg["settings"]["deepspeed_enabled"] = deepspeed_enabled_gr == "Enabled"
    cfg["settings"]["temperature_set"] = temperature_set_gr
    cfg["settings"]["repetitionpenalty_set"] = repetitionpenalty_set_gr
    cfg["settings"]["pitch_set"] = pitch_set_gr
    cfg["settings"]["generationspeed_set"] = generationspeed_set_gr
    with open(os.path.join(this_dir, "model_settings.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    return "Settings updated successfully!"


def vibevoice_save_engine_settings(cfg_scale_gr, ddpm_steps_gr, device_gr, attn_gr, model_variant_gr,
                                   speaker1_gr, speaker2_gr, speaker3_gr, speaker4_gr):
    with open(os.path.join(this_dir, "model_settings.json"), "r") as f:
        cfg = json.load(f)
    cfg["settings"]["cfg_scale"] = float(cfg_scale_gr)
    cfg["settings"]["ddpm_inference_steps"] = int(ddpm_steps_gr)
    cfg["settings"]["device"] = device_gr
    cfg["settings"]["attn_implementation"] = attn_gr
    cfg["settings"]["model_variant"] = model_variant_gr
    cfg["settings"]["speaker_voice_map"] = {
        "Speaker 1": speaker1_gr,
        "Speaker 2": speaker2_gr,
        "Speaker 3": speaker3_gr,
        "Speaker 4": speaker4_gr,
    }
    with open(os.path.join(this_dir, "model_settings.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    return "VibeVoice settings saved. Engine reload required for device/attention/variant changes."


def vibevoice_test_connection(api_url_gr, api_key_gr):
    """Test connectivity to a network-backend VibeVoice server."""
    try:
        url = api_url_gr.rstrip("/")
        headers = {}
        if api_key_gr:
            headers["Authorization"] = f"Bearer {api_key_gr}"
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


def vibevoice_save_network_settings(backend_type, api_url, api_key, response_format, api_timeout):
    with open(os.path.join(this_dir, "model_settings.json"), "r") as f:
        cfg = json.load(f)
    cfg["settings"]["backend_type"] = backend_type
    cfg["settings"]["api_url"] = api_url
    cfg["settings"]["api_key"] = api_key
    cfg["settings"]["api_response_format"] = response_format
    cfg["settings"]["api_timeout"] = int(api_timeout)
    with open(os.path.join(this_dir, "model_settings.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    return "Network backend settings saved. Engine reload required to take effect."


def vibevoice_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    current_variant = model_config_data["settings"].get("model_variant", "microsoft/VibeVoice-1.5b")
    voice_list = vibevoice_voices_file_list(current_variant)
    speaker_map = model_config_data["settings"].get("speaker_voice_map", {})

    with gr.Blocks(title="VibeVoice TTS", analytics_enabled=False) as app:
        with gr.Tab("Default Settings"):
            with gr.Row():
                lowvram_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="Low VRAM N/A", value="Disabled", interactive=False)
                deepspeed_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="DeepSpeed N/A", value="Disabled", interactive=False)
                temperature_set_gr = gr.Slider(value=float(model_config_data["settings"]["temperature_set"]), minimum=0, maximum=1, step=0.05, label="Temperature N/A", interactive=False)
                repetitionpenalty_set_gr = gr.Slider(value=float(model_config_data["settings"]["repetitionpenalty_set"]), minimum=1, maximum=20, step=1, label="Repetition N/A", interactive=False)
                pitch_set_gr = gr.Slider(value=float(model_config_data["settings"]["pitch_set"]), minimum=-10, maximum=10, step=1, label="Pitch N/A", interactive=False)
                generationspeed_set_gr = gr.Slider(value=float(model_config_data["settings"]["generationspeed_set"]), minimum=0.5, maximum=2.0, step=0.1, label="Speed N/A", interactive=False)
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
                    # Seed from per-variant default so 1.5B shows Alice and 0.5B shows Grace
                    if _is_realtime(current_variant):
                        _default_char = model_config_data["settings"].get("def_character_voice_realtime", model_config_data["settings"]["def_character_voice"])
                        _default_narr = model_config_data["settings"].get("def_narrator_voice_realtime", model_config_data["settings"]["def_narrator_voice"])
                    else:
                        _default_char = model_config_data["settings"]["def_character_voice"]
                        _default_narr = model_config_data["settings"]["def_narrator_voice"]
                    with gr.Row():
                        def_character_voice_gr = gr.Dropdown(value=_default_char, label="Default/Character Voice", choices=voice_list, allow_custom_value=True)
                        def_narrator_voice_gr = gr.Dropdown(value=_default_narr, label="Narrator Voice", choices=voice_list, allow_custom_value=True)
                    with gr.Group():
                        details_text = gr.Textbox(
                            label="Details", show_label=False, lines=5, interactive=False,
                            value=("Voices are auto-discovered from the engine's voices/ subdirectory — "
                                   "drop any .wav file in there to add a voice. For multi-speaker dialogue, "
                                   "configure the Speaker → Voice mapping on the VibeVoice tab and use "
                                   "'Speaker 1:', 'Speaker 2:' lines in your input text.")
                        )
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            submit_button.click(
                vibevoice_model_update_settings,
                inputs=[def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr,
                        temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr,
                        alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr],
                outputs=output_message,
            )
            with gr.Accordion("HELP - Understanding TTS Engine Default Settings Page", open=False):
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS, elem_classes="custom-markdown")
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS1, elem_classes="custom-markdown")
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS2, elem_classes="custom-markdown")

        with gr.Tab("VibeVoice"):
            gr.Markdown("### Model & Inference")
            with gr.Row():
                model_variant_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("model_variant", "microsoft/VibeVoice-1.5b"),
                    label="Model Variant",
                    choices=[
                        "microsoft/VibeVoice-1.5b",
                        "microsoft/VibeVoice-Realtime-0.5B",
                        "vibevoice-community/VibeVoice-Large-pt",
                    ],
                    info="1.5B: long-form multi-speaker, ~5 GB, .wav voices. Realtime-0.5B: single-speaker streaming, ~2 GB, .pt voices in voices_realtime/. Large (~7B): ~18 GB VRAM.",
                    allow_custom_value=True,
                )
                device_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("device", "auto"),
                    label="Device",
                    choices=["auto", "cuda", "mps", "cpu"],
                    info="auto: pick best available. mps requires Apple Silicon.",
                )
                attn_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("attn_implementation", "auto"),
                    label="Attention Implementation",
                    choices=["auto", "flash_attention_2", "sdpa"],
                    info="auto: flash_attention_2 on CUDA, sdpa elsewhere. Falls back to sdpa if flash-attn isn't installed.",
                )
            with gr.Row():
                cfg_scale_gr = gr.Slider(
                    value=float(model_config_data["settings"].get("cfg_scale", 1.3)),
                    minimum=1.0, maximum=3.0, step=0.05,
                    label="CFG Scale",
                    info="Classifier-Free Guidance. 1.3 = recommended. Higher = stricter reference adherence.",
                )
                ddpm_steps_gr = gr.Slider(
                    value=int(model_config_data["settings"].get("ddpm_inference_steps", 10)),
                    minimum=3, maximum=50, step=1,
                    label="DDPM Inference Steps",
                    info="Diffusion denoising steps per audio token. 10 = balanced; 5 = faster.",
                )

            gr.Markdown("### Speaker → Voice Mapping")
            gr.Markdown(
                "Used when input text contains `Speaker 1:`, `Speaker 2:`, etc. lines. "
                "VibeVoice supports up to 4 distinct speakers per generation."
            )
            with gr.Row():
                speaker1_gr = gr.Dropdown(value=speaker_map.get("Speaker 1", model_config_data["settings"]["def_character_voice"]), label="Speaker 1", choices=voice_list, allow_custom_value=True)
                speaker2_gr = gr.Dropdown(value=speaker_map.get("Speaker 2", model_config_data["settings"]["def_narrator_voice"]), label="Speaker 2", choices=voice_list, allow_custom_value=True)
            with gr.Row():
                speaker3_gr = gr.Dropdown(value=speaker_map.get("Speaker 3", model_config_data["settings"]["def_character_voice"]), label="Speaker 3", choices=voice_list, allow_custom_value=True)
                speaker4_gr = gr.Dropdown(value=speaker_map.get("Speaker 4", model_config_data["settings"]["def_narrator_voice"]), label="Speaker 4", choices=voice_list, allow_custom_value=True)

            with gr.Row():
                engine_submit_button = gr.Button("Save VibeVoice Settings")
                engine_output = gr.Textbox(label="Output", interactive=False, show_label=False)
            engine_submit_button.click(
                vibevoice_save_engine_settings,
                inputs=[cfg_scale_gr, ddpm_steps_gr, device_gr, attn_gr, model_variant_gr,
                        speaker1_gr, speaker2_gr, speaker3_gr, speaker4_gr],
                outputs=engine_output,
            )

            # Live-refresh all voice dropdowns when variant changes.
            # Default character/narrator dropdowns ALSO reset their value to the per-variant default.
            def _refresh_voice_dropdowns(variant):
                voices = vibevoice_voices_file_list(variant)
                with open(os.path.join(this_dir, "model_settings.json"), "r") as f:
                    cfg = json.load(f)
                s = cfg["settings"]
                if _is_realtime(variant):
                    char_val = s.get("def_character_voice_realtime", s["def_character_voice"])
                    narr_val = s.get("def_narrator_voice_realtime", s["def_narrator_voice"])
                else:
                    char_val = s["def_character_voice"]
                    narr_val = s["def_narrator_voice"]
                choices_only = gr.update(choices=voices)
                return (
                    gr.update(choices=voices, value=char_val),  # def_character
                    gr.update(choices=voices, value=narr_val),  # def_narrator
                    choices_only, choices_only, choices_only, choices_only, choices_only, choices_only,  # 6 openai
                    choices_only, choices_only, choices_only, choices_only,  # 4 speakers
                )
            model_variant_gr.change(
                _refresh_voice_dropdowns,
                inputs=[model_variant_gr],
                outputs=[def_character_voice_gr, def_narrator_voice_gr,
                         alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr,
                         speaker1_gr, speaker2_gr, speaker3_gr, speaker4_gr],
            )

            with gr.Accordion("Setup Instructions", open=True):
                gr.Markdown(
                    """
## Quick Setup

### 1. Install the community VibeVoice package
```bash
pip install git+https://github.com/vibevoice-community/VibeVoice.git
```
(The engine will also auto-install on first model load and restart AllTalk.)

### 2. Pre-download model weights (optional — happens on first use otherwise)
```bash
pip install "huggingface_hub[cli]"
hf download microsoft/VibeVoice-1.5b
```

### 3. Add reference voice WAVs
Copy `.wav` files into:
```
system/tts_engines/vibevoice/voices/
```
Any WAV in that folder becomes a usable voice. Each file's basename is the voice name. 10–30s of clean speech works best as a reference.

The community fork ships starter voices at `demo/voices/` in their repo —
download a couple to get going.

### 4. Multi-speaker scripts
Pass text like:
```
Speaker 1: Welcome back.
Speaker 2: Thanks for having me.
```
…and the engine routes each speaker via the mapping above.
                    """,
                    elem_classes="custom-markdown",
                )

        with gr.Tab("Network Backend"):
            gr.Markdown("### Run inference on a remote server instead of locally")
            gr.Markdown(
                "By default VibeVoice loads the model into local PyTorch (CUDA/MPS/CPU). Switching to "
                "**network** makes this engine a thin HTTP client instead — it POSTs text + reference "
                "voices to a remote server's `/v1/audio/speech` endpoint and gets audio back. Use this to "
                "run inference on another machine (e.g. a bespoke tinygrad-based VibeVoice server, or any "
                "server implementing the same contract). Only the long-form (.wav voice) variants are "
                "supported over the network backend — the Realtime variant's `.pt` cached prompts are "
                "PyTorch-specific and can't be sent remotely."
            )
            with gr.Row():
                net_backend_type_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("backend_type", "local"),
                    label="Backend",
                    choices=["local", "network"],
                    info="local: run the model in this process. network: call a remote server instead.",
                )
                net_api_url_gr = gr.Textbox(
                    value=model_config_data["settings"].get("api_url", "http://localhost:8100"),
                    label="API URL",
                    placeholder="http://localhost:8100",
                    info="Base URL of the remote VibeVoice server.",
                )
            with gr.Row():
                net_api_key_gr = gr.Textbox(
                    value=model_config_data["settings"].get("api_key", ""),
                    label="API Key",
                    placeholder="Leave empty if the server doesn't require auth",
                    type="password",
                )
                net_response_format_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("api_response_format", "wav"),
                    label="Response Format",
                    choices=["wav", "mp3", "flac", "opus", "aac", "pcm"],
                )
                net_timeout_gr = gr.Number(
                    value=model_config_data["settings"].get("api_timeout", 120),
                    label="Request Timeout (seconds)",
                    precision=0,
                )
            with gr.Row():
                net_test_button = gr.Button("Test Connection", variant="primary")
                net_test_output = gr.Textbox(label="Connection Status", interactive=False, show_label=False)
            net_test_button.click(vibevoice_test_connection, inputs=[net_api_url_gr, net_api_key_gr], outputs=net_test_output)
            with gr.Row():
                net_submit_button = gr.Button("Save Network Backend Settings")
                net_output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            net_submit_button.click(
                vibevoice_save_network_settings,
                inputs=[net_backend_type_gr, net_api_url_gr, net_api_key_gr, net_response_format_gr, net_timeout_gr],
                outputs=net_output_message,
            )
            with gr.Accordion("Remote server contract", open=False):
                gr.Markdown(
                    """
The remote server must expose `POST {api_url}/v1/audio/speech` accepting JSON:

```json
{
  "model": "microsoft/VibeVoice-1.5b",
  "text": "Speaker 1: Welcome back.\\nSpeaker 2: Thanks for having me.",
  "voices": {"Speaker 1": "<base64 wav>", "Speaker 2": "<base64 wav>"},
  "cfg_scale": 1.5,
  "ddpm_inference_steps": 10,
  "response_format": "wav"
}
```

...and respond `200 OK` with the raw audio bytes in the requested format (like OpenAI's
`/v1/audio/speech`). `voices` is base64-encoded reference WAV bytes, one entry per distinct
`Speaker N:` label found in `text`, keyed by that same label. Optional `Authorization: Bearer <api_key>`
header is sent if an API key is configured. `GET {api_url}/v1/models` or `/health` is used for the
connection test and startup ping.
                    """,
                    elem_classes="custom-markdown",
                )

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
                        gr.Textbox(label="TTS Engine Name", value="vibevoice", interactive=False)
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

        with gr.Tab("Voice Browser"):
            def load_voices():
                voices_file = os.path.join(this_dir, "vibevoice_voices.json")
                if os.path.exists(voices_file):
                    with open(voices_file, "r") as f:
                        return json.load(f)
                return {"voices": []}

            def format_voices_table(lang_filter):
                voices = load_voices().get("voices", [])
                lang_map = {"All Languages": None, "English": "en", "Indian English": "in", "Chinese": "zh"}
                target = lang_map.get(lang_filter)
                filtered = voices if target is None else [v for v in voices if v.get("language") == target]
                if not filtered:
                    return "No voices found for this language."
                rows = ["| Voice Name | Voice Code | Gender | Style | Description |", "|---|---|---|---|---|"]
                for v in filtered:
                    rows.append(f"| {v['voice_name']} | {v['voice_code']} | {v['gender']} | {v.get('style', '')} | {v['description']} |")
                return "\n".join(rows)

            with gr.Row():
                lang_filter = gr.Dropdown(choices=["All Languages", "English", "Indian English", "Chinese"], value="All Languages", label="Filter by Language")
            with gr.Row():
                voices_table = gr.Markdown(format_voices_table("All Languages"))
            lang_filter.change(format_voices_table, inputs=lang_filter, outputs=voices_table)

        with gr.Tab("Engine Help"):
            with gr.Row():
                gr.Markdown(AllTalkHelpContent.HELP_PAGE, elem_classes="custom-markdown")
            with gr.Row():
                gr.Markdown(AllTalkHelpContent.HELP_PAGE1, elem_classes="custom-markdown")
                gr.Markdown(AllTalkHelpContent.HELP_PAGE2, elem_classes="custom-markdown")

    return app


def vibevoice_at_gradio_settings_page(model_config_data):
    return vibevoice_model_alltalk_settings(model_config_data)
