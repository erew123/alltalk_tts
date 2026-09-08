import os
import json
import requests
import gradio as gr
from pathlib import Path
from .help_content import AllTalkHelpContent

this_dir = Path(__file__).parent.resolve()
main_dir = Path(__file__).parent.parent.parent.parent.resolve()

BUILTIN_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
LANGUAGE_CHOICES = ["Auto", "English", "Chinese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]
KNOWN_MODEL_VARIANTS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]


def _model_family(variant):
    s = (variant or "").lower()
    if "customvoice" in s:
        return "customvoice"
    if "voicedesign" in s:
        return "voicedesign"
    return "base"


def _load_voices_json():
    voices_file = this_dir / "qwen3tts_voices.json"
    if voices_file.exists():
        with open(voices_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"custom_voice_presets": [], "voice_design_presets": []}


def _save_voices_json(data):
    with open(this_dir / "qwen3tts_voices.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _base_voices_file_list():
    voices_dir = this_dir / "voices"
    voices = []
    if voices_dir.is_dir():
        for f in voices_dir.glob("*.wav"):
            if f.with_suffix(".reference.txt").is_file():
                voices.append(f.stem)
    return sorted(set(voices))


def qwen3tts_voices_file_list(variant=None):
    """List voices for the given model variant (presets for CustomVoice/VoiceDesign, wav+txt pairs for Base)."""
    if variant is None:
        try:
            with open(this_dir / "model_settings.json") as f:
                variant = json.load(f)["settings"].get("model_variant", "")
        except Exception:
            variant = ""
    family = _model_family(variant)
    data = _load_voices_json()
    if family == "customvoice":
        voices = sorted({p["preset_name"] for p in data.get("custom_voice_presets", []) if p.get("preset_name")})
    elif family == "voicedesign":
        voices = sorted({p["preset_name"] for p in data.get("voice_design_presets", []) if p.get("preset_name")})
    else:
        voices = _base_voices_file_list()
    return voices if voices else ["No Voices Found"]


def _dataframe_to_rows(df):
    """Normalize a Gradio Dataframe value (pandas DataFrame or list-of-lists) into list-of-lists."""
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            return df.fillna("").values.tolist()
    except ImportError:
        pass
    return df or []


def qwen3tts_model_update_settings(def_character_voice_gr, def_narrator_voice_gr,
                                    lowvram_enabled_gr, deepspeed_enabled_gr,
                                    temperature_set_gr, repetitionpenalty_set_gr,
                                    pitch_set_gr, generationspeed_set_gr,
                                    alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
    with open(this_dir / "model_settings.json", "r") as f:
        cfg = json.load(f)
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
    with open(this_dir / "model_settings.json", "w") as f:
        json.dump(cfg, f, indent=4)
    return "Settings updated successfully!"


def qwen3tts_save_engine_settings(model_variant_gr, device_gr, attn_gr, default_language_gr):
    with open(this_dir / "model_settings.json", "r") as f:
        cfg = json.load(f)
    cfg["settings"]["model_variant"] = model_variant_gr
    cfg["settings"]["device"] = device_gr
    cfg["settings"]["attn_implementation"] = attn_gr
    cfg["settings"]["default_language"] = default_language_gr
    with open(this_dir / "model_settings.json", "w") as f:
        json.dump(cfg, f, indent=4)
    return "Qwen3-TTS settings saved. Engine reload required for model/device/attention changes to take effect."


def qwen3tts_save_custom_voice_presets(df):
    rows = _dataframe_to_rows(df)
    data = _load_voices_json()
    presets = []
    for row in rows:
        if not row or not str(row[0]).strip():
            continue
        presets.append({
            "preset_name": str(row[0]).strip(),
            "speaker": str(row[1]).strip() if len(row) > 1 else "",
            "instruct": str(row[2]).strip() if len(row) > 2 else "",
            "language": str(row[3]).strip() if len(row) > 3 and str(row[3]).strip() else "Auto",
        })
    data["custom_voice_presets"] = presets
    _save_voices_json(data)
    return f"Saved {len(presets)} Custom Voice preset(s)."


def qwen3tts_save_voice_design_presets(df):
    rows = _dataframe_to_rows(df)
    data = _load_voices_json()
    presets = []
    for row in rows:
        if not row or not str(row[0]).strip():
            continue
        presets.append({
            "preset_name": str(row[0]).strip(),
            "instruct": str(row[1]).strip() if len(row) > 1 else "",
            "language": str(row[2]).strip() if len(row) > 2 and str(row[2]).strip() else "Auto",
        })
    data["voice_design_presets"] = presets
    _save_voices_json(data)
    return f"Saved {len(presets)} Voice Design preset(s)."


def qwen3tts_test_connection(api_url_gr, api_key_gr):
    """Test connectivity to a network-backend Qwen3-TTS server."""
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


def qwen3tts_save_network_settings(backend_type, api_url, api_key, response_format, api_timeout):
    with open(this_dir / "model_settings.json", "r") as f:
        cfg = json.load(f)
    cfg["settings"]["backend_type"] = backend_type
    cfg["settings"]["api_url"] = api_url
    cfg["settings"]["api_key"] = api_key
    cfg["settings"]["api_response_format"] = response_format
    cfg["settings"]["api_timeout"] = int(api_timeout)
    with open(this_dir / "model_settings.json", "w") as f:
        json.dump(cfg, f, indent=4)
    return "Network backend settings saved. Engine reload required to take effect."


def qwen3tts_model_alltalk_settings(model_config_data):
    features_list = model_config_data['model_capabilties']
    current_variant = model_config_data["settings"].get("model_variant", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    voice_list = qwen3tts_voices_file_list(current_variant)
    voices_data = _load_voices_json()

    custom_rows = [[p.get("preset_name", ""), p.get("speaker", ""), p.get("instruct", ""), p.get("language", "Auto")]
                   for p in voices_data.get("custom_voice_presets", [])]
    design_rows = [[p.get("preset_name", ""), p.get("instruct", ""), p.get("language", "Auto")]
                   for p in voices_data.get("voice_design_presets", [])]

    with gr.Blocks(title="Qwen3-TTS", analytics_enabled=False) as app:
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
                    with gr.Row():
                        def_character_voice_gr = gr.Dropdown(value=model_config_data["settings"]["def_character_voice"], label="Default/Character Voice", choices=voice_list, allow_custom_value=True)
                        def_narrator_voice_gr = gr.Dropdown(value=model_config_data["settings"]["def_narrator_voice"], label="Narrator Voice", choices=voice_list, allow_custom_value=True)
                    with gr.Group():
                        gr.Textbox(
                            label="Details", show_label=False, lines=5, interactive=False,
                            value=("Voices shown here depend on the currently loaded model variant: Custom Voice "
                                   "Presets (CustomVoice models), Voice Design Presets (VoiceDesign model), or "
                                   "voices/ wav+reference.txt pairs (Base models). Configure presets on their own "
                                   "tabs, and the model variant on the Qwen3-TTS tab."),
                        )
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            submit_button.click(
                qwen3tts_model_update_settings,
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

        with gr.Tab("Qwen3-TTS"):
            gr.Markdown("### Model & Inference")
            with gr.Row():
                model_variant_gr = gr.Dropdown(
                    value=current_variant,
                    label="Model Variant",
                    choices=KNOWN_MODEL_VARIANTS,
                    info="0.6B: smaller/faster. 1.7B: higher quality. Base = voice cloning, CustomVoice = built-in speakers + instruct, VoiceDesign = free-form voice description (1.7B only). A matching local folder under models/qwen3tts/<name>/ is used automatically if present, otherwise the Hub id is downloaded on first load.",
                    allow_custom_value=True,
                )
                device_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("device", "auto"),
                    label="Device",
                    choices=["auto", "cuda", "mps", "cpu"],
                    info="auto: pick best available. mps requires Apple Silicon and falls back to cpu if loading fails.",
                )
                attn_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("attn_implementation", "auto"),
                    label="Attention Implementation",
                    choices=["auto", "flash_attention_2", "sdpa"],
                    info="auto: flash_attention_2 on CUDA, sdpa elsewhere. Falls back to sdpa if flash-attn isn't installed.",
                )
                default_language_gr = gr.Dropdown(
                    value=model_config_data["settings"].get("default_language", "Auto"),
                    label="Default Language",
                    choices=LANGUAGE_CHOICES,
                    info="Used when a preset's language is 'Auto' and no language is passed in the API call.",
                )
            with gr.Row():
                engine_submit_button = gr.Button("Save Qwen3-TTS Settings")
                engine_output = gr.Textbox(label="Output", interactive=False, show_label=False)
            engine_submit_button.click(
                qwen3tts_save_engine_settings,
                inputs=[model_variant_gr, device_gr, attn_gr, default_language_gr],
                outputs=engine_output,
            )

            def _refresh_voice_dropdowns(variant):
                voices = qwen3tts_voices_file_list(variant)
                with open(this_dir / "model_settings.json", "r") as f:
                    s = json.load(f)["settings"]
                choices_only = gr.update(choices=voices)
                return (
                    gr.update(choices=voices, value=s["def_character_voice"]),
                    gr.update(choices=voices, value=s["def_narrator_voice"]),
                    choices_only, choices_only, choices_only, choices_only, choices_only, choices_only,
                )
            model_variant_gr.change(
                _refresh_voice_dropdowns,
                inputs=[model_variant_gr],
                outputs=[def_character_voice_gr, def_narrator_voice_gr,
                         alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr],
            )

            with gr.Accordion("Setup Instructions", open=True):
                gr.Markdown(
                    """
## Quick Setup

### 1. Install the qwen-tts package
```bash
pip install -U qwen-tts
```
(The engine will also auto-install on first model load.)

### 2. Pre-download model weights (optional — happens on first use otherwise)
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
    --local-dir models/qwen3tts/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

### 3. Configure voices
- **CustomVoice**: edit presets on the **Custom Voice Presets** tab
- **VoiceDesign**: edit presets on the **Voice Design Presets** tab
- **Base**: drop `.wav` + matching `.reference.txt` pairs into `system/tts_engines/qwen3tts/voices/`
                    """,
                    elem_classes="custom-markdown",
                )

        with gr.Tab("Custom Voice Presets"):
            gr.Markdown(
                "### Custom voice properties for CustomVoice models\n"
                "Each row is a selectable AllTalk voice. `speaker` must be one of the built-in "
                f"Qwen3-TTS speakers: {', '.join(BUILTIN_SPEAKERS)}. `instruct` is a free-form "
                "natural-language emotion/style instruction (leave blank for the speaker's neutral "
                "delivery). Set `language` to a fixed language or `Auto`."
            )
            custom_voice_df = gr.Dataframe(
                headers=["preset_name", "speaker", "instruct", "language"],
                datatype=["str", "str", "str", "str"],
                value=custom_rows,
                row_count=(1, "dynamic"),
                col_count=(4, "fixed"),
                interactive=True,
                wrap=True,
            )
            with gr.Row():
                save_custom_button = gr.Button("Save Custom Voice Presets")
                save_custom_output = gr.Textbox(label="Output", interactive=False, show_label=False)
            save_custom_button.click(qwen3tts_save_custom_voice_presets, inputs=[custom_voice_df], outputs=save_custom_output)

        with gr.Tab("Voice Design Presets"):
            gr.Markdown(
                "### Voice Design presets (VoiceDesign model, 1.7B only)\n"
                "There is no `speaker` field here — `instruct` alone describes the entire voice "
                "from scratch (pitch, tone, pacing, character)."
            )
            voice_design_df = gr.Dataframe(
                headers=["preset_name", "instruct", "language"],
                datatype=["str", "str", "str"],
                value=design_rows,
                row_count=(1, "dynamic"),
                col_count=(3, "fixed"),
                interactive=True,
                wrap=True,
            )
            with gr.Row():
                save_design_button = gr.Button("Save Voice Design Presets")
                save_design_output = gr.Textbox(label="Output", interactive=False, show_label=False)
            save_design_button.click(qwen3tts_save_voice_design_presets, inputs=[voice_design_df], outputs=save_design_output)

        with gr.Tab("Network Backend"):
            gr.Markdown("### Run inference on a remote server instead of locally")
            gr.Markdown(
                "By default Qwen3-TTS loads the model into local PyTorch (CUDA/MPS/CPU). Switching to "
                "**network** makes this engine a thin HTTP client instead — it POSTs text plus the resolved "
                "speaker/instruct/reference-audio fields to a remote server's `/v1/audio/speech` endpoint and "
                "gets audio back. Use this to run inference on another machine (e.g. a bespoke tinygrad-based "
                "Qwen3-TTS server, or any server implementing the same contract). Custom Voice / Voice Design "
                "preset resolution still happens locally in AllTalk — only the resolved fields are sent."
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
                    info="Base URL of the remote Qwen3-TTS server.",
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
            net_test_button.click(qwen3tts_test_connection, inputs=[net_api_url_gr, net_api_key_gr], outputs=net_test_output)
            with gr.Row():
                net_submit_button = gr.Button("Save Network Backend Settings")
                net_output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            net_submit_button.click(
                qwen3tts_save_network_settings,
                inputs=[net_backend_type_gr, net_api_url_gr, net_api_key_gr, net_response_format_gr, net_timeout_gr],
                outputs=net_output_message,
            )
            with gr.Accordion("Remote server contract", open=False):
                gr.Markdown(
                    """
The remote server must expose `POST {api_url}/v1/audio/speech` accepting JSON. The `mode` field
matches the loaded model variant's family and determines which other fields are present:

```json
// mode: "customvoice"
{"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "mode": "customvoice", "text": "...",
 "language": "English", "speaker": "Vivian", "instruct": "", "response_format": "wav"}

// mode: "voicedesign"
{"model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "mode": "voicedesign", "text": "...",
 "language": "English", "instruct": "A deep, calm male narrator voice...", "response_format": "wav"}

// mode: "base" (voice cloning)
{"model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "mode": "base", "text": "...", "language": "English",
 "ref_audio": "<base64 wav>", "ref_text": "transcript of ref_audio", "response_format": "wav"}
```

...and respond `200 OK` with the raw audio bytes in the requested format (like OpenAI's
`/v1/audio/speech`). Optional `Authorization: Bearer <api_key>` header is sent if an API key is
configured. `GET {api_url}/v1/models` or `/health` is used for the connection test and startup ping.
This is the same contract shape as the VibeVoice engine's network backend, so one server can serve
both by branching on `model`/`mode`.
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
                        gr.Textbox(label="TTS Engine Name", value="qwen3tts", interactive=False)
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
            def format_voices_table():
                data = _load_voices_json()
                rows = ["| Preset Name | Speaker | Instruct | Language | Description |", "|---|---|---|---|---|"]
                for p in data.get("custom_voice_presets", []):
                    rows.append(f"| {p.get('preset_name','')} | {p.get('speaker','')} | {p.get('instruct','') or '-'} | {p.get('language','Auto')} | {p.get('description','')} |")
                design_rows_md = ["| Preset Name | Instruct | Language | Description |", "|---|---|---|---|"]
                for p in data.get("voice_design_presets", []):
                    design_rows_md.append(f"| {p.get('preset_name','')} | {p.get('instruct','')} | {p.get('language','Auto')} | {p.get('description','')} |")
                base_voices = _base_voices_file_list()
                base_md = "\n".join(f"- {v}" for v in base_voices) if base_voices else "_None found in voices/_"
                return (
                    "#### Custom Voice Presets (CustomVoice models)\n" + "\n".join(rows) +
                    "\n\n#### Voice Design Presets (VoiceDesign model)\n" + "\n".join(design_rows_md) +
                    "\n\n#### Base Voice Cloning (voices/*.wav + *.reference.txt)\n" + base_md
                )
            gr.Markdown(format_voices_table())

        with gr.Tab("Engine Help"):
            with gr.Row():
                gr.Markdown(AllTalkHelpContent.HELP_PAGE, elem_classes="custom-markdown")
            with gr.Row():
                gr.Markdown(AllTalkHelpContent.HELP_PAGE1, elem_classes="custom-markdown")
                gr.Markdown(AllTalkHelpContent.HELP_PAGE2, elem_classes="custom-markdown")

    return app


def qwen3tts_at_gradio_settings_page(model_config_data):
    return qwen3tts_model_alltalk_settings(model_config_data)
