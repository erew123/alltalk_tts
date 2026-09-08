###############################################
# DONT CHANGE # These are base imports needed #
###############################################
import os
import re
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
# CHANGE ME # VibeVoice-specific imports                                                                    #
#############################################################################################################
import subprocess
import base64
import requests

VIBEVOICE_AVAILABLE = False
VIBEVOICE_IMPORT_ERROR = ""
VibeVoiceForConditionalGenerationInference = None
VibeVoiceProcessor = None
VibeVoiceStreamingForConditionalGenerationInference = None
VibeVoiceStreamingProcessor = None

def _try_import_vibevoice():
    global VIBEVOICE_AVAILABLE, VIBEVOICE_IMPORT_ERROR
    global VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor
    global VibeVoiceStreamingForConditionalGenerationInference, VibeVoiceStreamingProcessor
    try:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference as _M,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor as _P
        VibeVoiceForConditionalGenerationInference = _M
        VibeVoiceProcessor = _P
        # Streaming/realtime classes (optional — only present in newer community pkg)
        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference as _SM,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor as _SP,
            )
            VibeVoiceStreamingForConditionalGenerationInference = _SM
            VibeVoiceStreamingProcessor = _SP
        except ImportError:
            pass
        VIBEVOICE_AVAILABLE = True
        VIBEVOICE_IMPORT_ERROR = ""
        return True
    except ImportError as e:
        VIBEVOICE_AVAILABLE = False
        VIBEVOICE_IMPORT_ERROR = str(e)
        return False

_try_import_vibevoice()


def _is_realtime_variant(variant_or_method):
    """Return True if the model variant/method id refers to the Realtime streaming model."""
    s = (variant_or_method or "").lower()
    return "realtime" in s or "streaming" in s

VIBEVOICE_GIT_URL = "git+https://github.com/vibevoice-community/VibeVoice.git"


def _install_vibevoice():
    """Install VibeVoice from the community fork."""
    print("####################################################################")
    print("[VibeVoice] Installing from community fork. This may take a while...")
    print("####################################################################")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", VIBEVOICE_GIT_URL])
    except subprocess.CalledProcessError as e:
        print(f"[VibeVoice] \033[91mError\033[0m: Failed to install VibeVoice: {e}")
        raise ImportError("Could not install VibeVoice from " + VIBEVOICE_GIT_URL)
    print("##############################################################")
    print("[VibeVoice] Installed. Restarting application...")
    print("##############################################################")
    os.execv(sys.executable, ["python"] + sys.argv)


SPEAKER_LINE_RE = re.compile(r"^\s*Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


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
        self.processor = None
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
        self.def_character_voice_realtime = tts_model_loaded["settings"].get("def_character_voice_realtime", self.def_character_voice)
        self.def_narrator_voice_realtime = tts_model_loaded["settings"].get("def_narrator_voice_realtime", self.def_narrator_voice)
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
        # VibeVoice-specific settings       #
        #####################################
        self.cfg_scale = float(tts_model_loaded["settings"].get("cfg_scale", 1.3))
        self.ddpm_inference_steps = int(tts_model_loaded["settings"].get("ddpm_inference_steps", 10))
        self.requested_device = tts_model_loaded["settings"].get("device", "auto")
        self.requested_attn = tts_model_loaded["settings"].get("attn_implementation", "auto")
        self.model_variant = tts_model_loaded["settings"].get("model_variant", "microsoft/VibeVoice-1.5b")
        self.speaker_voice_map = tts_model_loaded["settings"].get("speaker_voice_map", {})

        self.resolved_device = self._resolve_device(self.requested_device)
        self.resolved_attn = self._resolve_attn(self.requested_attn, self.resolved_device)
        self.resolved_dtype = self._resolve_dtype(self.resolved_device)

        #####################################################################
        # Network backend settings — run inference on a remote server       #
        # (e.g. a tinygrad-based VibeVoice server) instead of local PyTorch #
        #####################################################################
        self.backend_type = tts_model_loaded["settings"].get("backend_type", "local")  # "local" or "network"
        self.api_url = tts_model_loaded["settings"].get("api_url", "http://localhost:8100")
        self.api_key = tts_model_loaded["settings"].get("api_key", "")
        self.api_response_format = tts_model_loaded["settings"].get("api_response_format", "wav")
        self.api_timeout = int(tts_model_loaded["settings"].get("api_timeout", 120))

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

    ###################################################################################
    # CHANGE ME # Inital setup of the model and engine. Called when the script starts #
    ###################################################################################
    async def setup(self):
        self.printout_versions()
        self.available_models = self.scan_models_folder()
        print(f"[{self.branding}ENG] \033[92mVibeVoice backend :\033[93m {self.backend_type}\033[0m")
        if self.backend_type == "network":
            print(f"[{self.branding}ENG] \033[92mVibeVoice API URL :\033[93m {self.api_url}\033[0m")
            print(f"[{self.branding}ENG] \033[92mVibeVoice API Key :\033[93m {'Set' if self.api_key else 'Not set'}\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mVibeVoice device  :\033[93m {self.resolved_device}\033[0m")
            print(f"[{self.branding}ENG] \033[92mVibeVoice attn    :\033[93m {self.resolved_attn}\033[0m")
        print(f"[{self.branding}ENG] \033[92mVibeVoice variant :\033[93m {self.model_variant}\033[0m")
        if self.backend_type != "network" and not VIBEVOICE_AVAILABLE:
            print(f"[{self.branding}ENG] \033[93mVibeVoice not installed. Will attempt install on first model load.\033[0m")
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
        # VibeVoice models are pulled from HuggingFace by id, not from a local folder.
        # We surface the supported variants as selectable "models".
        self.available_models = {
            "vibevoice - VibeVoice-1.5b": "microsoft/VibeVoice-1.5b",
            "vibevoice - VibeVoice-Realtime-0.5B": "microsoft/VibeVoice-Realtime-0.5B",
            "vibevoice - VibeVoice-Large": "vibevoice-community/VibeVoice-Large-pt",
        }
        return self.available_models

    def _current_model_type(self, tts_method=None):
        """Return 'realtime' or 'longform' for the active model."""
        return "realtime" if _is_realtime_variant(tts_method or self.current_model_loaded or self.model_variant) else "longform"

    def _active_default_voice(self):
        """Default character voice for the currently loaded variant."""
        return self.def_character_voice_realtime if self._current_model_type() == "realtime" else self.def_character_voice

    #############################################################
    # CHANGE ME #  POPULATE FILES LIST FROM VOICES DIRECTORY ####
    #############################################################
    def _voices_dir(self):
        """Return the voices directory and file extension for the active variant."""
        if self._current_model_type() == "realtime":
            return self.this_dir / "voices_realtime", ".pt"
        return self.this_dir / "voices", ".wav"

    def voices_file_list(self):
        try:
            voices_dir, ext = self._voices_dir()
            voices = []
            if voices_dir.is_dir():
                for f in voices_dir.glob(f"**/*{ext}"):
                    voices.append(f.stem)
            if not voices:
                print(f"[{self.branding}ENG] \033[93mNo {ext} files in {voices_dir}. Drop voice samples there.\033[0m")
                return ["No Voices Found"]
            return sorted(set(voices))
        except Exception:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Could not enumerate voices.")
            return ["No Voices Found"]

    def _voice_path(self, voice_name):
        """Resolve a voice name to a file path. Returns None if not found."""
        if not voice_name or voice_name == "No Voices Found":
            return None
        voices_dir, ext = self._voices_dir()
        if os.path.isabs(voice_name) and os.path.isfile(voice_name):
            return voice_name
        candidate = voices_dir / f"{voice_name}{ext}"
        if candidate.is_file():
            return str(candidate)
        if voices_dir.is_dir():
            for f in voices_dir.glob(f"**/*{ext}"):
                if f.stem.lower() == voice_name.lower():
                    return str(f)
        return None

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
            self.processor = None
            if self._test_api_connection():
                print(f"[{self.branding}ENG] \033[92mVibeVoice network backend connected:\033[93m {self.api_url}\033[0m")
            else:
                print(f"[{self.branding}ENG] \033[93mVibeVoice network backend not reachable at {self.api_url} (will retry on generation).\033[0m")
            self.is_tts_model_loaded = True
            return None

        global VIBEVOICE_AVAILABLE
        if not VIBEVOICE_AVAILABLE:
            print(f"[{self.branding}ENG] VibeVoice not installed. Auto-installing from community fork...")
            _install_vibevoice()
            raise HTTPException(status_code=500, detail=f"VibeVoice install failed: {VIBEVOICE_IMPORT_ERROR}")

        hf_id = self.available_models.get(model_name, self.model_variant)
        is_realtime = _is_realtime_variant(model_name) or _is_realtime_variant(hf_id)

        if is_realtime and VibeVoiceStreamingForConditionalGenerationInference is None:
            raise HTTPException(
                status_code=500,
                detail="Realtime variant requested but vibevoice streaming classes are missing. Reinstall: pip install git+https://github.com/vibevoice-community/VibeVoice.git",
            )

        model_cls = VibeVoiceStreamingForConditionalGenerationInference if is_realtime else VibeVoiceForConditionalGenerationInference
        proc_cls = VibeVoiceStreamingProcessor if is_realtime else VibeVoiceProcessor
        # Realtime model recommends 5 DDPM steps per its demo; longform default is the user's setting
        ddpm_steps = 5 if is_realtime else self.ddpm_inference_steps

        print(f"[{self.branding}ENG] \033[94mLoading\033[93m {hf_id}\033[94m ({'realtime' if is_realtime else 'longform'}) on\033[93m {self.resolved_device}\033[94m ({self.resolved_dtype}, {self.resolved_attn})\033[0m")
        try:
            self.processor = proc_cls.from_pretrained(hf_id)
            try:
                if self.resolved_device == "mps":
                    self.model = model_cls.from_pretrained(
                        hf_id,
                        torch_dtype=self.resolved_dtype,
                        attn_implementation=self.resolved_attn,
                        device_map=None,
                    )
                    self.model.to("mps")
                elif self.resolved_device == "cuda":
                    self.model = model_cls.from_pretrained(
                        hf_id,
                        torch_dtype=self.resolved_dtype,
                        device_map="cuda",
                        attn_implementation=self.resolved_attn,
                    )
                else:
                    self.model = model_cls.from_pretrained(
                        hf_id,
                        torch_dtype=self.resolved_dtype,
                        device_map="cpu",
                        attn_implementation=self.resolved_attn,
                    )
            except Exception as e:
                if self.resolved_attn == "flash_attention_2":
                    print(f"[{self.branding}ENG] \033[93mflash_attention_2 failed ({type(e).__name__}: {e}). Retrying with sdpa.\033[0m")
                    self.resolved_attn = "sdpa"
                    self.model = model_cls.from_pretrained(
                        hf_id,
                        torch_dtype=self.resolved_dtype,
                        device_map=(self.resolved_device if self.resolved_device in ("cuda", "cpu") else None),
                        attn_implementation="sdpa",
                    )
                    if self.resolved_device == "mps":
                        self.model.to("mps")
                else:
                    raise

            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=ddpm_steps)
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError loading VibeVoice:\033[0m {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load VibeVoice: {e}")

        self.is_tts_model_loaded = True
        return None

    ###############################
    # CHANGE ME # Model unloading #
    ###############################
    async def unload_model(self):
        if self.backend_type == "network":
            self.model = None
            self.processor = None
            self.is_tts_model_loaded = False
            return None
        try:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
        except Exception:
            pass
        self.model = None
        self.processor = None
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
        if "No Models Available" in self.available_models:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load.")
            return False
        await self.unload_model()
        await self.api_manual_load_model(tts_method)
        self.current_model_loaded = tts_method
        generate_elapsed_time = time.time() - generate_start_time
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m")
        return True

    ############################################################
    # Script parsing — turn user input into Speaker N: lines   #
    ############################################################
    def _parse_script(self, text, fallback_voice):
        """
        Returns (full_script: str, voice_paths_in_speaker_order: list[str]).
        If text already has "Speaker N:" lines, parse them and resolve each unique
        speaker via the configured speaker_voice_map. Otherwise wrap text as
        "Speaker 1: <text>" using fallback_voice for Speaker 1.
        """
        lines = text.strip().splitlines()
        has_speakers = any(SPEAKER_LINE_RE.match(line) for line in lines)

        if not has_speakers:
            voice_path = self._voice_path(fallback_voice) or self._voice_path(self._active_default_voice())
            if voice_path is None:
                raise HTTPException(status_code=400, detail=f"Voice '{fallback_voice}' not found in {self.this_dir / 'voices'}")
            return f"Speaker 1: {text.strip()}", [voice_path]

        # Normalize and reconstruct the multi-speaker script
        scripts = []
        speaker_order = []
        seen = set()
        current_speaker = None
        current_text = ""
        for line in lines:
            m = SPEAKER_LINE_RE.match(line)
            if m:
                if current_speaker is not None and current_text:
                    scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                    if current_speaker not in seen:
                        speaker_order.append(current_speaker)
                        seen.add(current_speaker)
                current_speaker = m.group(1).strip()
                current_text = m.group(2).strip()
            else:
                if line.strip():
                    current_text = (current_text + " " + line.strip()).strip()
        if current_speaker is not None and current_text:
            scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
            if current_speaker not in seen:
                speaker_order.append(current_speaker)
                seen.add(current_speaker)

        voice_paths = []
        for spk in speaker_order:
            mapped = self.speaker_voice_map.get(f"Speaker {spk}") or self.speaker_voice_map.get(spk)
            if not mapped:
                # Fall back to the API-provided voice for the first speaker, then defaults
                if not voice_paths:
                    mapped = fallback_voice
                else:
                    mapped = self._active_default_voice()
            voice_path = self._voice_path(mapped)
            if voice_path is None:
                raise HTTPException(status_code=400, detail=f"Voice '{mapped}' for Speaker {spk} not found in {self.this_dir / 'voices'}")
            voice_paths.append(voice_path)

        return "\n".join(scripts), voice_paths

    ##########################################################################################################################################
    # CHANGE ME # TTS Generation                                                                                                            #
    ##########################################################################################################################################
    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        if voice == "No Voices Found":
            voices_dir, ext = self._voices_dir()
            raise HTTPException(status_code=400, detail=f"No voices found. Drop {ext} samples into {voices_dir}.")

        # Reload backend_type every call so a settings-page change takes effect without reload
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            backend_type = json.load(f)["settings"].get("backend_type", self.backend_type)

        if backend_type == "network":
            async for chunk in self._generate_tts_network(text, voice, output_file, streaming):
                yield chunk
            return

        if not self.is_tts_model_loaded or self.model is None or self.processor is None:
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")
        self.tts_generating_lock = True
        generate_start_time = time.time()
        is_realtime = self._current_model_type() == "realtime"

        try:
            # Reload settings every call so changes from the settings page take effect without reload
            with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
                cur = json.load(f)
            cfg_scale = float(cur["settings"].get("cfg_scale", self.cfg_scale))
            self.speaker_voice_map = cur["settings"].get("speaker_voice_map", self.speaker_voice_map)
            self.def_character_voice = cur["settings"].get("def_character_voice", self.def_character_voice)
            self.def_narrator_voice = cur["settings"].get("def_narrator_voice", self.def_narrator_voice)
            self.def_character_voice_realtime = cur["settings"].get("def_character_voice_realtime", self.def_character_voice_realtime)
            self.def_narrator_voice_realtime = cur["settings"].get("def_narrator_voice_realtime", self.def_narrator_voice_realtime)

            target_device = self.resolved_device if self.resolved_device != "cpu" else "cpu"

            if is_realtime:
                # Single-speaker model. Strip Speaker N: prefixes if present and use the passed voice.
                clean_text = "\n".join(
                    (SPEAKER_LINE_RE.match(line).group(2) if SPEAKER_LINE_RE.match(line) else line)
                    for line in text.strip().splitlines() if line.strip()
                ).strip()
                clean_text = clean_text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')

                voice_path = self._voice_path(voice) or self._voice_path(self._active_default_voice())
                if voice_path is None:
                    voices_dir, ext = self._voices_dir()
                    raise HTTPException(status_code=400, detail=f"Voice '{voice}' not found in {voices_dir}")

                if self.debug_tts:
                    print(f"[{self.branding}GEN] \033[94mVibeVoice-Realtime text:\033[93m {clean_text[:160]}\033[0m")
                    print(f"[{self.branding}GEN] \033[94mVoice prompt (.pt):\033[93m {os.path.basename(voice_path)}\033[0m")

                # Load cached voice prompt
                from transformers.cache_utils import DynamicCache
                from transformers.modeling_outputs import BaseModelOutputWithPast
                import copy
                with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
                    all_prefilled = torch.load(voice_path, map_location=target_device, weights_only=True)

                inputs = self.processor.process_input_with_cached_prompt(
                    text=clean_text,
                    cached_prompt=all_prefilled,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(target_device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(all_prefilled),
                )
            else:
                # Long-form multi-speaker path
                full_script, voice_paths = self._parse_script(text, voice)
                full_script = full_script.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')

                if self.debug_tts:
                    print(f"[{self.branding}GEN] \033[94mVibeVoice script:\033[0m")
                    for line in full_script.splitlines():
                        print(f"[{self.branding}GEN]   {line[:120]}")
                    print(f"[{self.branding}GEN] \033[94mVoices:\033[93m {[os.path.basename(p) for p in voice_paths]}\033[0m")

                inputs = self.processor(
                    text=[full_script],
                    voice_samples=[voice_paths],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(target_device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    is_prefill=True,
                )

            if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
                raise HTTPException(status_code=500, detail="VibeVoice produced no audio.")

            self.processor.save_audio(outputs.speech_outputs[0], output_path=output_file)

            # Fake streaming generator — VibeVoice is non-streaming
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
            print(f"[{self.branding}ENG] \033[91mError\033[0m: VibeVoice generation failed: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"VibeVoice generation failed: {e}")

        generate_elapsed_time = time.time() - generate_start_time
        print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mDevice: \033[33m{self.resolved_device}\033[0m")
        self.tts_generating_lock = False
        return

    ##########################################################################################################################################
    # Network backend — POST to a remote /v1/audio/speech endpoint (e.g. a tinygrad-based VibeVoice server) instead of running locally      #
    ##########################################################################################################################################
    async def _generate_tts_network(self, text, voice, output_file, streaming):
        if _is_realtime_variant(self.model_variant):
            raise HTTPException(
                status_code=400,
                detail="The network backend only supports the long-form (.wav voice) VibeVoice variants. "
                       "The Realtime variant's cached .pt voice prompts are PyTorch-specific and can't be sent to a remote server.",
            )

        self.tts_generating_lock = True
        generate_start_time = time.time()
        try:
            with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
                cur = json.load(f)
            cfg_scale = float(cur["settings"].get("cfg_scale", self.cfg_scale))
            ddpm_steps = int(cur["settings"].get("ddpm_inference_steps", self.ddpm_inference_steps))
            self.speaker_voice_map = cur["settings"].get("speaker_voice_map", self.speaker_voice_map)
            api_url = cur["settings"].get("api_url", self.api_url).rstrip("/")
            api_key = cur["settings"].get("api_key", self.api_key)
            response_format = cur["settings"].get("api_response_format", self.api_response_format)
            timeout = int(cur["settings"].get("api_timeout", self.api_timeout))

            full_script, voice_paths = self._parse_script(text, voice)
            full_script = full_script.replace("’", "'").replace("“", '"').replace("”", '"')

            # voice_paths is ordered by first appearance of each distinct "Speaker N:" label in
            # full_script (see _parse_script) — recover those labels so the remote server can map
            # each reference voice to the right speaker tag in the script.
            speaker_labels = []
            seen_labels = set()
            for line in full_script.splitlines():
                m = SPEAKER_LINE_RE.match(line)
                if m and m.group(1) not in seen_labels:
                    seen_labels.add(m.group(1))
                    speaker_labels.append(m.group(1))

            voices_b64 = {}
            for label, path in zip(speaker_labels, voice_paths):
                with open(path, "rb") as vf:
                    voices_b64[f"Speaker {label}"] = base64.b64encode(vf.read()).decode("utf-8")

            payload = {
                "model": self.model_variant,
                "text": full_script,
                "voices": voices_b64,
                "cfg_scale": cfg_scale,
                "ddpm_inference_steps": ddpm_steps,
                "response_format": response_format,
            }

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            endpoint = f"{api_url}/v1/audio/speech"

            if self.debug_tts:
                print(f"[{self.branding}GEN] \033[94mVibeVoice network request:\033[93m {endpoint}\033[0m")
                for line in full_script.splitlines():
                    print(f"[{self.branding}GEN]   {line[:120]}")

            if streaming:
                resp = requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=timeout)
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=f"VibeVoice network API error: {resp.text[:500]}")
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
                    raise HTTPException(status_code=resp.status_code, detail=f"VibeVoice network API error: {resp.text[:500]}")
                with open(output_file, "wb") as f:
                    f.write(resp.content)

        except requests.exceptions.Timeout:
            self.tts_generating_lock = False
            raise HTTPException(status_code=504, detail="VibeVoice network API request timed out.")
        except requests.exceptions.ConnectionError:
            self.tts_generating_lock = False
            raise HTTPException(status_code=503, detail=f"Cannot connect to VibeVoice network backend at {self.api_url}. Is the server running?")
        except HTTPException:
            self.tts_generating_lock = False
            raise
        except Exception as e:
            self.tts_generating_lock = False
            print(f"[{self.branding}ENG] \033[91mError\033[0m: VibeVoice network generation failed: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"VibeVoice network generation failed: {e}")

        generate_elapsed_time = time.time() - generate_start_time
        print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mBackend: \033[33mnetwork ({self.api_url})\033[0m")
        self.tts_generating_lock = False
        return
