import os
import json
import inspect
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass

@dataclass
class AlltalkConfigTheme:
    file: str | None = None
    clazz = ""

@dataclass
class AlltalkConfigRvcSettings:
    rvc_enabled = False
    rvc_char_model_file = ""
    rvc_narr_model_file = ""
    split_audio = False
    autotune = False
    pitch = 0
    filter_radius = 0
    index_rate = 0.0
    rms_mix_rate = 0
    protect = 0
    hop_length = 0
    f0method = ""
    embedder_model = ""
    training_data_size = 0

@dataclass
class AlltalkConfigTgwUi:
    tgwui_activate_tts = False
    tgwui_autoplay_tts = False
    tgwui_narrator_enabled = ""
    tgwui_non_quoted_text_is = ""
    tgwui_deepspeed_enabled = False
    tgwui_language = ""
    tgwui_lowvram_enabled = False
    tgwui_pitch_set = 0
    tgwui_temperature_set = 0.0
    tgwui_repetitionpenalty_set = 0
    tgwui_generationspeed_set = 0
    tgwui_narrator_voice = ""
    tgwui_show_text = False
    tgwui_character_voice = ""
    tgwui_rvc_char_voice = ""
    tgwui_rvc_narr_voice = ""

@dataclass
class AlltalkConfigApiDef:
    api_port_number = 0
    api_allowed_filter = ""
    api_length_stripping = 0
    api_max_characters = 0
    api_use_legacy_api = False
    api_legacy_ip_address = ""
    api_text_filtering = ""
    api_narrator_enabled = ""
    api_text_not_inside = ""
    api_language = ""
    api_output_file_name = ""
    api_output_file_timestamp = False
    api_autoplay = False
    api_autoplay_volume = 0.0

@dataclass
class AlltalkConfigDebug:
    debug_transcode = False
    debug_tts = False
    debug_openai = False
    debug_concat = False
    debug_tts_variables = False
    debug_rvc = False

class AlltalkTTSEnginesConfig:
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self):
        self.engines_available = []
        self.engine_loaded = ""
        self.selected_model = ""


    @staticmethod
    def get_instance():
        if AlltalkTTSEnginesConfig.__instance is None:
            AlltalkTTSEnginesConfig.__instance = AlltalkTTSEnginesConfig()
            AlltalkTTSEnginesConfig.__instance.__load_config()
        return AlltalkTTSEnginesConfig.__instance

    def reload(self):
        self.__load_config()

    def __load_config(self):
        tts_engines_file = os.path.join(self.__this_dir, "system", "tts_engines", "tts_engines.json")
        with open(tts_engines_file, "r") as f:
            tts_engines_data = json.load(f)

        # List of the available TTS engines from tts_engines.json
        self.engines_available = [engine["name"] for engine in tts_engines_data["engines_available"]]

        # The currently set TTS engine from tts_engines.json
        self.engine_loaded = tts_engines_data["engine_loaded"]
        self.selected_model = tts_engines_data["selected_model"]


class AlltalkConfig:
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self):
        self.branding = ""
        self.delete_output_wavs = ""
        self.gradio_interface = False
        self.output_folder = ""
        self.gradio_port_number = 0
        self.firstrun_model = False
        self.firstrun_splash = False
        self.launch_gradio = False
        self.transcode_audio_format = ""
        self.theme = AlltalkConfigTheme()
        self.rvc_settings = AlltalkConfigRvcSettings()
        self.tgwui = AlltalkConfigTgwUi()
        self.api_def = AlltalkConfigApiDef()
        self.debugging = AlltalkConfigDebug()

    @staticmethod
    def get_instance():
        if AlltalkConfig.__instance is None:
            AlltalkConfig.__instance = AlltalkConfig()
            AlltalkConfig.__instance.__load_config()
        return AlltalkConfig.__instance

    def reload(self):
        self.__load_config()

    def __load_config(self):
        configfile_path = self.__this_dir / "confignew.json"
        with open(configfile_path, "r") as configfile:
            configfile_data = json.load(configfile, object_hook=lambda d: SimpleNamespace(**d))

        # Copy those fields for which there are members in this class:
        for n, v in inspect.getmembers(configfile_data):
            if hasattr(self, n) and not n.startswith("__"):
                setattr(self, n, v)

        # Special properties (cannot use 'class' as property name):
        self.theme.clazz = configfile_data.theme.__dict__["class"]

        # As a side effect, create the output directory
        output_directory = self.__this_dir / self.output_folder
        output_directory.mkdir(parents=True, exist_ok=True)
