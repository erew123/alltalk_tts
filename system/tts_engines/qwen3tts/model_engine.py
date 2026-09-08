###############################################
# DONT CHANGE # These are base imports needed #
###############################################
import os
import sys
import json
import time
import torch
import logging
import traceback
from pathlib import Path
from fastapi import HTTPException
logging.disable(logging.WARNING)
#################################################################
# DONT CHANGE # Get Pytorch & Python versions & setup DeepSpeed #
#################################################################
pytorch_version = torch.__version__
cuda_version = torch.version.cuda
major, minor, micro = sys.version_info[:3]
python_version = f"{major}.{minor}.{micro}"
try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    deepspeed_available = False

#############################################################################################################
# CHANGE ME # Qwen3-TTS-specific imports                                                                    #
#############################################################################################################
import subprocess
import base64
import requests
import soundfile as sf

QWEN3TTS_AVAILABLE = False
QWEN3TTS_IMPORT_ERROR = ""
Qwen3TTSModel = None


def _try_import_qwen3tts():
    global QWEN3TTS_AVAILABLE, QWEN3TTS_IMPORT_ERROR, Qwen3TTSModel
    try:
        from qwen_tts import Qwen3TTSModel as _M
        Qwen3TTSModel = _M
        QWEN3TTS_AVAILABLE = True
        QWEN3TTS_IMPORT_ERROR = ""
        return True
    except ImportError as e:
        QWEN3TTS_AVAILABLE = False
        QWEN3TTS_IMPORT_ERROR = str(e)
        return False


_try_import_qwen3tts()


def _install_qwen3tts():
    """Install the 'qwen-tts' package (official Qwen3-TTS python package)."""
    print("####################################################################")
    print("[Qwen3-TTS] Installing 'qwen-tts' package. This may take a while...")
    print("####################################################################")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "qwen-tts"])
    except subprocess.CalledProcessError as e:
        print(f"[Qwen3-TTS] \033[91mError\033[0m: Failed to install qwen-tts: {e}")
        raise ImportError("Could not install qwen-tts")
    print("##############################################################")
    print("[Qwen3-TTS] Installed. Restarting application...")
    print("##############################################################")
    os.execv(sys.executable, ["python"] + sys.argv)


# Known Qwen3-TTS checkpoints this engine understands: display name -> HuggingFace repo id.
# A matching local folder under models/qwen3tts/<name>/ is preferred over the Hub id if present.
KNOWN_MODELS = {
    "Qwen3-TTS-12Hz-0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen3-TTS-12Hz-1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# AllTalk language codes/names -> the language strings Qwen3-TTS expects.
LANGUAGE_MAP = {
    "auto": "Auto",
    "en": "English", "english": "English",
    "zh": "Chinese", "zh-cn": "Chinese", "chinese": "Chinese", "cmn": "Chinese",
    "ja": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "korean": "Korean",
    "de": "German", "german": "German",
    "fr": "French", "french": "French",
    "ru": "Russian", "russian": "Russian",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "es": "Spanish", "spanish": "Spanish",
    "it": "Italian", "italian": "Italian",
}


def _model_family(model_id_or_name):
    """Return 'customvoice', 'voicedesign' or 'base' for a model id/display-name string."""
    s = (model_id_or_name or "").lower()
    if "customvoice" in s:
        return "customvoice"
    if "voicedesign" in s:
        return "voicedesign"
    return "base"


#################################################################################################################################
# DONT CHANGE # Do not change the Class name from tts_class as this is what will be imported into the main tts_server.py script #
#################################################################################################################################
class tts_class:
    def __init__(self):
        ########################################################################
        # DONT CHANGE # Sets up the base variables required for any tts engine #
        ########################################################################
        self.branding = None
        self.this_dir = Path(__file__).parent.resolve()
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_is_available = torch.cuda.is_available()
        self.tts_generating_lock = False
        self.tts_stop_generation = False
        self.tts_narrator_generatingtts = False
        self.model = None
        self.is_tts_model_loaded = False
        self.current_model_loaded = None
        self.available_models = None
        self.setup_has_run = False
        ##############################################################################################
        # DONT CHANGE # Load in a list of the available TTS engines and the currently set TTS engine #
        ##############################################################################################
        tts_engines_file = os.path.join(self.main_dir, "system", "tts_engines", "tts_engines.json")
        with open(tts_engines_file, "r") as f:
            tts_engines_data = json.load(f)
        self.engines_available = [engine["name"] for engine in tts_engines_data["engines_available"]]
        self.engine_loaded = tts_engines_data["engine_loaded"]
        self.selected_model = tts_engines_data["selected_model"]
        ############################################################################
        # DONT CHANGE # Pull out all the settings for the currently set TTS engine #
        ############################################################################
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            tts_model_loaded = json.load(f)
        self.manufacturer_name = tts_model_loaded["model_details"]["manufacturer_name"]
        self.manufacturer_website = tts_model_loaded["model_details"]["manufacturer_website"]
        self.audio_format = tts_model_loaded["model_capabilties"]["audio_format"]
        self.deepspeed_capable = tts_model_loaded["model_capabilties"]["deepspeed_capable"]
        self.deepspeed_available = 'deepspeed' in globals() and deepspeed_available
        self.generationspeed_capable = tts_model_loaded["model_capabilties"]["generationspeed_capable"]
        self.languages_capable = tts_model_loaded["model_capabilties"]["languages_capable"]
        self.lowvram_capable = tts_model_loaded["model_capabilties"]["lowvram_capable"]
        self.multimodel_capable = tts_model_loaded["model_capabilties"]["multimodel_capable"]
        self.repetitionpenalty_capable = tts_model_loaded["model_capabilties"]["repetitionpenalty_capable"]
        self.streaming_capable = tts_model_loaded["model_capabilties"]["streaming_capable"]
        self.temperature_capable = tts_model_loaded["model_capabilties"]["temperature_capable"]
        self.multivoice_capable = tts_model_loaded["model_capabilties"]["multivoice_capable"]
        self.pitch_capable = tts_model_loaded["model_capabilties"]["pitch_capable"]
        self.def_character_voice = tts_model_loaded["settings"]["def_character_voice"]
        self.def_narrator_voice = tts_model_loaded["settings"]["def_narrator_voice"]
        self.deepspeed_enabled = tts_model_loaded["settings"]["deepspeed_enabled"]
        self.engine_installed = tts_model_loaded["settings"]["engine_installed"]
        self.generationspeed_set = tts_model_loaded["settings"]["generationspeed_set"]
        self.lowvram_enabled = tts_model_loaded["settings"]["lowvram_enabled"]
        self.lowvram_enabled = False if not torch.cuda.is_available() else self.lowvram_enabled
        self.repetitionpenalty_set = tts_model_loaded["settings"]["repetitionpenalty_set"]
        self.temperature_set = tts_model_loaded["settings"]["temperature_set"]
        self.pitch_set = tts_model_loaded["settings"]["pitch_set"]
        self.openai_alloy = tts_model_loaded["openai_voices"]["alloy"]
        self.openai_echo = tts_model_loaded["openai_voices"]["echo"]
        self.openai_fable = tts_model_loaded["openai_voices"]["fable"]
        self.openai_nova = tts_model_loaded["openai_voices"]["nova"]
        self.openai_onyx = tts_model_loaded["openai_voices"]["onyx"]
        self.openai_shimmer = tts_model_loaded["openai_voices"]["shimmer"]
        ###################################################################
        # DONT CHANGE #  Load params and api_defaults from confignew.json #
        ###################################################################
        configfile_path = self.main_dir / "confignew.json"
        with open(configfile_path, "r") as configfile:
            configfile_data = json.load(configfile)
        self.branding = configfile_data.get("branding", "")
        self.params = configfile_data
        self.debug_tts = configfile_data.get("debugging").get("debug_tts")
        self.debug_tts_variables = configfile_data.get("debugging").get("debug_tts_variables")
        #####################################
        # Qwen3-TTS-specific settings       #
        #####################################
        self.requested_device = tts_model_loaded["settings"].get("device", "auto")
        self.requested_attn = tts_model_loaded["settings"].get("attn_implementation", "auto")
        self.model_variant = tts_model_loaded["settings"].get("model_variant", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        self.default_language = tts_model_loaded["settings"].get("default_language", "Auto")

        #####################################################################
        # Network backend settings — run inference on a remote server       #
        # (e.g. a tinygrad-based Qwen3-TTS server) instead of local PyTorch #
        #####################################################################
        self.backend_type = tts_model_loaded["settings"].get("backend_type", "local")  # "local" or "network"
        self.api_url = tts_model_loaded["settings"].get("api_url", "http://localhost:8100")
        self.api_key = tts_model_loaded["settings"].get("api_key", "")
        self.api_response_format = tts_model_loaded["settings"].get("api_response_format", "wav")
        self.api_timeout = int(tts_model_loaded["settings"].get("api_timeout", 120))

        self.resolved_device = self._resolve_device(self.requested_device)
        self.resolved_attn = self._resolve_attn(self.requested_attn, self.resolved_device)
        self.resolved_dtype = self._resolve_dtype(self.resolved_device)

        self.model_family = _model_family(self.model_variant)
        self.custom_voice_presets = {}
        self.voice_design_presets = {}
        self._load_voice_presets()

    ################################################################
    # DONT CHANGE #  Print out Python, CUDA, DeepSpeed versions ####
    ################################################################
    def printout_versions(self):
        if deepspeed_available:
            print(f"[{self.branding}ENG] \033[92mDeepSpeed version :\033[93m", deepspeed.__version__, "\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mDeepSpeed version :\033[93m Not available\033[0m")
        print(f"[{self.branding}ENG] \033[92mPython Version    :\033[93m {python_version}\033[0m")
        print(f"[{self.branding}ENG] \033[92mPyTorch Version   :\033[93m {pytorch_version}\033[0m")
        if cuda_version is None:
            print(f"[{self.branding}ENG] \033[92mCUDA Version      :\033[91m Not available\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mCUDA Version      :\033[93m {cuda_version}\033[0m")
        print(f"[{self.branding}ENG]")

    ############################################################
    # Device / dtype / attention resolution                    #
    ############################################################
    def _resolve_device(self, requested):
        if requested and requested != "auto":
            if requested == "cuda" and not torch.cuda.is_available():
                print(f"[{self.branding}ENG] \033[93mCUDA requested but not available, falling back.\033[0m")
            elif requested == "mps" and not torch.backends.mps.is_available():
                print(f"[{self.branding}ENG] \033[93mMPS requested but not available, falling back.\033[0m")
            else:
                return requested
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_attn(self, requested, device):
        if requested and requested != "auto":
            return requested
        if device == "cuda":
            return "flash_attention_2"
        return "sdpa"

    def _resolve_dtype(self, device):
        if device == "cuda":
            return torch.bfloat16
        return torch.float32

    ############################################################
    # Custom voice properties — CustomVoice / VoiceDesign presets
    ############################################################
    def _load_voice_presets(self):
        try:
            with open(self.this_dir / "qwen3tts_voices.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        self.custom_voice_presets = {
            p["preset_name"]: p for p in data.get("custom_voice_presets", []) if p.get("preset_name")
        }
        self.voice_design_presets = {
            p["preset_name"]: p for p in data.get("voice_design_presets", []) if p.get("preset_name")
        }

    def _map_language(self, language):
        if not language:
            return self.default_language or "Auto"
        return LANGUAGE_MAP.get(str(language).strip().lower(), self.default_language or "Auto")

    ###################################################################################
    # CHANGE ME # Inital setup of the model and engine. Called when the script starts #
    ###################################################################################
    async def setup(self):
        self.printout_versions()
        self.available_models = self.scan_models_folder()
        print(f"[{self.branding}ENG] \033[92mQwen3-TTS backend :\033[93m {self.backend_type}\033[0m")
        if self.backend_type == "network":
            print(f"[{self.branding}ENG] \033[92mQwen3-TTS API URL :\033[93m {self.api_url}\033[0m")
            print(f"[{self.branding}ENG] \033[92mQwen3-TTS API Key :\033[93m {'Set' if self.api_key else 'Not set'}\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mQwen3-TTS device  :\033[93m {self.resolved_device}\033[0m")
            print(f"[{self.branding}ENG] \033[92mQwen3-TTS attn    :\033[93m {self.resolved_attn}\033[0m")
        print(f"[{self.branding}ENG] \033[92mQwen3-TTS variant :\033[93m {self.model_variant}\033[0m")
        if self.backend_type != "network" and not QWEN3TTS_AVAILABLE:
            print(f"[{self.branding}ENG] \033[93mqwen-tts not installed. Will attempt install on first model load.\033[0m")
        if self.selected_model:
            tts_model = f"{self.selected_model}"
            if tts_model in self.available_models:
                await self.handle_tts_method_change(tts_model)
                self.current_model_loaded = tts_model
            else:
                self.current_model_loaded = "No Models Available"
                print(f"[{self.branding}ENG] \033[91mError\033[0m: Selected model '{self.selected_model}' not found.")
        self.setup_has_run = True

    ##################################
    # CHANGE ME #  Low VRAM Swapping #
    ##################################
    async def handle_lowvram_change(self):
        return

    ########################################
    # CHANGE ME #  DeepSpeed model loading #
    ########################################
    async def handle_deepspeed_change(self, value):
        return value

    #####################################################################################
    # CHANGE ME # scan for available models/voices that are relevant to this TTS engine #
    #####################################################################################
    def scan_models_folder(self):
        self.available_models = {}
        models_root = self.main_dir / "models" / "qwen3tts"

        # Known canonical variants — prefer a local download (e.g. via `huggingface-cli download
        # Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/qwen3tts/Qwen3-TTS-12Hz-1.7B-CustomVoice`)
        # over the Hub id. If no local copy exists, the bare Hub id is used and qwen-tts downloads/caches
        # it automatically on first load.
        for name, hf_id in KNOWN_MODELS.items():
            local_dir = models_root / name
            if local_dir.is_dir() and (local_dir / "config.json").is_file():
                self.available_models[f"qwen3tts - {name}"] = str(local_dir)
            else:
                self.available_models[f"qwen3tts - {name}"] = hf_id

        # Pick up any extra local folders the user has dropped in under a different name
        if models_root.is_dir():
            for sub in models_root.iterdir():
                if sub.is_dir() and sub.name not in KNOWN_MODELS and (sub / "config.json").is_file():
                    self.available_models[f"qwen3tts - {sub.name}"] = str(sub)

        if not self.available_models:
            self.available_models = {"No Models Found": "No Models Found"}
        return self.available_models

    #############################################################
    # CHANGE ME #  POPULATE FILES LIST FROM VOICES DIRECTORY ####
    #############################################################
    def _voices_dir(self):
        return self.this_dir / "voices"

    def voices_file_list(self):
        try:
            if self.model_family == "customvoice":
                names = sorted(self.custom_voice_presets.keys())
                if not names:
                    return ["No Voices Found"]
                return names

            if self.model_family == "voicedesign":
                names = sorted(self.voice_design_presets.keys())
                if not names:
                    return ["No Voices Found"]
                return names

            # base — reference-audio voice cloning: needs a .wav + matching .reference.txt pair
            voices_dir = self._voices_dir()
            voices = []
            if voices_dir.is_dir():
                for f in voices_dir.glob("*.wav"):
                    if f.with_suffix(".reference.txt").is_file():
                        voices.append(f.stem)
            if not voices:
                print(f"[{self.branding}ENG] \033[93mNo .wav+.reference.txt pairs in {voices_dir}. Drop a reference voice there.\033[0m")
                return ["No Voices Found"]
            return sorted(set(voices))
        except Exception:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Could not enumerate voices.")
            return ["No Voices Found"]

    def _voice_ref_pair(self, voice_name):
        """Resolve a Base-model voice name to (wav_path, ref_text). Returns (None, None) if not found."""
        if not voice_name or voice_name == "No Voices Found":
            return None, None
        voices_dir = self._voices_dir()
        candidate = voices_dir / f"{voice_name}.wav"
        if not candidate.is_file() and voices_dir.is_dir():
            for f in voices_dir.glob("*.wav"):
                if f.stem.lower() == voice_name.lower():
                    candidate = f
                    break
        if not candidate.is_file():
            return None, None
        ref_text_path = candidate.with_suffix(".reference.txt")
        if not ref_text_path.is_file():
            return None, None
        return str(candidate), ref_text_path.read_text(encoding="utf-8").strip()

    ############################################################
    # Network backend — connectivity check against remote server
    ############################################################
    def _test_api_connection(self):
        try:
            url = self.api_url.rstrip("/")
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            for endpoint in [f"{url}/v1/models", f"{url}/health", url]:
                try:
                    resp = requests.get(endpoint, headers=headers, timeout=5)
                    if resp.status_code < 500:
                        return True
                except requests.exceptions.RequestException:
                    continue
            return False
        except Exception:
            return False

    #############################
    # CHANGE ME # Model loading #
    #############################
    async def api_manual_load_model(self, model_name):
        if model_name == "No Models Found":
            raise HTTPException(status_code=400, detail="No models for this TTS engine were found to load.")

        if self.backend_type == "network":
            self.model = None
            self.model_family = _model_family(model_name)
            if self._test_api_connection():
                print(f"[{self.branding}ENG] \033[92mQwen3-TTS network backend connected:\033[93m {self.api_url}\033[0m")
            else:
                print(f"[{self.branding}ENG] \033[93mQwen3-TTS network backend not reachable at {self.api_url} (will retry on generation).\033[0m")
            self.is_tts_model_loaded = True
            return None

        global QWEN3TTS_AVAILABLE
        if not QWEN3TTS_AVAILABLE:
            print(f"[{self.branding}ENG] qwen-tts not installed. Auto-installing...")
            _install_qwen3tts()
            raise HTTPException(status_code=500, detail=f"qwen-tts install failed: {QWEN3TTS_IMPORT_ERROR}")

        model_path = self.available_models.get(model_name, self.model_variant)
        self.model_family = _model_family(model_name)

        device = self.resolved_device
        attn = self.resolved_attn
        dtype = self.resolved_dtype

        print(f"[{self.branding}ENG] \033[94mLoading\033[93m {model_path}\033[94m ({self.model_family}) on\033[93m {device}\033[94m ({dtype}, {attn})\033[0m")
        try:
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path, device_map=device, dtype=dtype, attn_implementation=attn,
                )
            except Exception as e:
                if attn == "flash_attention_2":
                    print(f"[{self.branding}ENG] \033[93mflash_attention_2 failed ({type(e).__name__}: {e}). Retrying with sdpa.\033[0m")
                    attn = "sdpa"
                    self.resolved_attn = "sdpa"
                    self.model = Qwen3TTSModel.from_pretrained(
                        model_path, device_map=device, dtype=dtype, attn_implementation=attn,
                    )
                elif device == "mps":
                    print(f"[{self.branding}ENG] \033[93mmps load failed ({type(e).__name__}: {e}). Retrying on cpu.\033[0m")
                    device = "cpu"
                    dtype = torch.float32
                    self.resolved_device = "cpu"
                    self.resolved_dtype = torch.float32
                    self.model = Qwen3TTSModel.from_pretrained(
                        model_path, device_map=device, dtype=dtype, attn_implementation="sdpa",
                    )
                else:
                    raise
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError loading Qwen3-TTS:\033[0m {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load Qwen3-TTS: {e}")

        self.is_tts_model_loaded = True
        return None

    ###############################
    # CHANGE ME # Model unloading #
    ###############################
    async def unload_model(self):
        if self.backend_type == "network":
            self.model = None
            self.is_tts_model_loaded = False
            return None
        try:
            if self.model is not None:
                del self.model
        except Exception:
            pass
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        self.is_tts_model_loaded = False
        return None

    ###################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one                                                          #
    ###################################################################################################################################
    async def handle_tts_method_change(self, tts_method):
        generate_start_time = time.time()
        if "No Models Available" in self.available_models or "No Models Found" in self.available_models:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load.")
            return False
        await self.unload_model()
        await self.api_manual_load_model(tts_method)
        self.current_model_loaded = tts_method
        generate_elapsed_time = time.time() - generate_start_time
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m")
        return True

    ##########################################################################################################################################
    # CHANGE ME # TTS Generation                                                                                                            #
    ##########################################################################################################################################
    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        if voice == "No Voices Found":
            raise HTTPException(status_code=400, detail="No voices found for the currently loaded Qwen3-TTS model.")

        # Reload backend_type every call so a settings-page change takes effect without reload
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            backend_type = json.load(f)["settings"].get("backend_type", self.backend_type)

        if backend_type == "network":
            async for chunk in self._generate_tts_network(text, voice, language, output_file, streaming):
                yield chunk
            return

        if not self.is_tts_model_loaded or self.model is None:
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")

        self.tts_generating_lock = True
        generate_start_time = time.time()

        # Reload presets every call so edits made on the settings page apply without an engine reload
        self._load_voice_presets()

        try:
            lang = self._map_language(language)

            if self.model_family == "customvoice":
                preset = self.custom_voice_presets.get(voice)
                if preset is None:
                    # Allow a raw built-in speaker name to be used directly, with no instruct
                    preset = {"speaker": voice, "instruct": "", "language": "Auto"}
                speaker = preset.get("speaker", voice)
                instruct = preset.get("instruct") or ""
                preset_lang = preset.get("language", "Auto")
                final_lang = lang if preset_lang in (None, "", "Auto") else preset_lang

                if self.debug_tts:
                    print(f"[{self.branding}GEN] \033[94mQwen3-TTS CustomVoice:\033[93m speaker={speaker} lang={final_lang} instruct={instruct!r}\033[0m")

                wavs, sr = self.model.generate_custom_voice(
                    text=text, language=final_lang, speaker=speaker, instruct=instruct,
                )

            elif self.model_family == "voicedesign":
                preset = self.voice_design_presets.get(voice)
                if preset is None:
                    raise HTTPException(status_code=400, detail=f"Voice design preset '{voice}' not found.")
                instruct = preset.get("instruct") or ""
                preset_lang = preset.get("language", "Auto")
                final_lang = lang if preset_lang in (None, "", "Auto") else preset_lang

                if self.debug_tts:
                    print(f"[{self.branding}GEN] \033[94mQwen3-TTS VoiceDesign:\033[93m lang={final_lang} instruct={instruct!r}\033[0m")

                wavs, sr = self.model.generate_voice_design(
                    text=text, language=final_lang, instruct=instruct,
                )

            else:  # base — reference-audio voice cloning
                ref_audio, ref_text = self._voice_ref_pair(voice)
                if ref_audio is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Voice '{voice}' needs a matching '{voice}.wav' + '{voice}.reference.txt' pair in {self._voices_dir()}",
                    )
                if self.debug_tts:
                    print(f"[{self.branding}GEN] \033[94mQwen3-TTS voice clone:\033[93m ref={os.path.basename(ref_audio)} lang={lang}\033[0m")

                wavs, sr = self.model.generate_voice_clone(
                    text=text, language=lang, ref_audio=ref_audio, ref_text=ref_text,
                )

            wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            sf.write(output_file, wav, sr)

            # Fake streaming generator — Qwen3-TTS generation here is non-streaming
            if streaming:
                with open(output_file, "rb") as f:
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        yield chunk

        except HTTPException:
            self.tts_generating_lock = False
            raise
        except Exception as e:
            self.tts_generating_lock = False
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Qwen3-TTS generation failed: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Qwen3-TTS generation failed: {e}")

        generate_elapsed_time = time.time() - generate_start_time
        print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mDevice: \033[33m{self.resolved_device}\033[0m")
        self.tts_generating_lock = False
        return

    ##########################################################################################################################################
    # Network backend — POST to a remote /v1/audio/speech endpoint (e.g. a tinygrad-based Qwen3-TTS server) instead of running locally       #
    ##########################################################################################################################################
    async def _generate_tts_network(self, text, voice, language, output_file, streaming):
        self.tts_generating_lock = True
        generate_start_time = time.time()
        try:
            with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
                cur = json.load(f)
            api_url = cur["settings"].get("api_url", self.api_url).rstrip("/")
            api_key = cur["settings"].get("api_key", self.api_key)
            response_format = cur["settings"].get("api_response_format", self.api_response_format)
            timeout = int(cur["settings"].get("api_timeout", self.api_timeout))
            self.default_language = cur["settings"].get("default_language", self.default_language)
            self._load_voice_presets()

            lang = self._map_language(language)
            model_family = _model_family(self.model_variant)

            payload = {
                "model": self.model_variant,
                "mode": model_family,
                "text": text,
                "language": lang,
                "response_format": response_format,
            }

            if model_family == "customvoice":
                preset = self.custom_voice_presets.get(voice) or {"speaker": voice, "instruct": "", "language": "Auto"}
                preset_lang = preset.get("language", "Auto")
                payload["speaker"] = preset.get("speaker", voice)
                payload["instruct"] = preset.get("instruct") or ""
                payload["language"] = lang if preset_lang in (None, "", "Auto") else preset_lang
            elif model_family == "voicedesign":
                preset = self.voice_design_presets.get(voice)
                if preset is None:
                    raise HTTPException(status_code=400, detail=f"Voice design preset '{voice}' not found.")
                preset_lang = preset.get("language", "Auto")
                payload["instruct"] = preset.get("instruct") or ""
                payload["language"] = lang if preset_lang in (None, "", "Auto") else preset_lang
            else:  # base — reference-audio voice cloning
                ref_audio, ref_text = self._voice_ref_pair(voice)
                if ref_audio is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Voice '{voice}' needs a matching '{voice}.wav' + '{voice}.reference.txt' pair in {self._voices_dir()}",
                    )
                with open(ref_audio, "rb") as af:
                    payload["ref_audio"] = base64.b64encode(af.read()).decode("utf-8")
                payload["ref_text"] = ref_text

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            endpoint = f"{api_url}/v1/audio/speech"

            if self.debug_tts:
                print(f"[{self.branding}GEN] \033[94mQwen3-TTS network request:\033[93m {endpoint} mode={model_family}\033[0m")

            if streaming:
                resp = requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=timeout)
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=f"Qwen3-TTS network API error: {resp.text[:500]}")
                with open(output_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=4096):
                        if self.tts_stop_generation:
                            self.tts_stop_generation = False
                            break
                        if chunk:
                            f.write(chunk)
                            yield chunk
            else:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=f"Qwen3-TTS network API error: {resp.text[:500]}")
                with open(output_file, "wb") as f:
                    f.write(resp.content)

        except requests.exceptions.Timeout:
            self.tts_generating_lock = False
            raise HTTPException(status_code=504, detail="Qwen3-TTS network API request timed out.")
        except requests.exceptions.ConnectionError:
            self.tts_generating_lock = False
            raise HTTPException(status_code=503, detail=f"Cannot connect to Qwen3-TTS network backend at {self.api_url}. Is the server running?")
        except HTTPException:
            self.tts_generating_lock = False
            raise
        except Exception as e:
            self.tts_generating_lock = False
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Qwen3-TTS network generation failed: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Qwen3-TTS network generation failed: {e}")

        generate_elapsed_time = time.time() - generate_start_time
        print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mBackend: \033[33mnetwork ({self.api_url})\033[0m")
        self.tts_generating_lock = False
        return
