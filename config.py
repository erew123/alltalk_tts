import os
import json
import time
import inspect
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Any


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

class AbstractConfig(ABC):

    def __init__(self, config_path: Path | str, file_check_interval: int):
        self.__config_path = Path(config_path) if type(config_path) is str else config_path
        self.__last_read_time = 0  # Track when we last read the file
        self.__file_check_interval = file_check_interval

    def get_config_path(self):
        return self.__config_path

    def reload(self):
        self._load_config()
        return self

    def _reload_on_change(self):
        # Check if config file has been modified and reload if needed
        if time.time() - self.__last_read_time >= self.__file_check_interval:
            try:
                most_recent_modification = self.get_config_path().stat().st_mtime
                if most_recent_modification > self.__last_read_time:
                    self.reload()
            except Exception as e:
                print(f"Error checking config file: {e}")
        return self

    def _load_config(self):
        self.__last_read_time = self.get_config_path().stat().st_mtime
        with open(self.get_config_path(), "r") as configfile:
            data = json.load(configfile, object_hook=self._object_hook())

        self._handle_loaded_config(data)

    def _object_hook(self) -> Callable[[dict[Any, Any]], Any] | None:
        return None

    @abstractmethod
    def _handle_loaded_config(self, data):
        pass

class AlltalkTTSEnginesConfig(AbstractConfig):
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self, config_path: Path | str = os.path.join(__this_dir, "system", "tts_engines", "tts_engines.json")):
        super().__init__(config_path, 5)
        self.engines_available = []
        self.engine_loaded = ""
        self.selected_model = ""
        self._load_config()

    @staticmethod
    def get_instance():
        if AlltalkTTSEnginesConfig.__instance is None:
            AlltalkTTSEnginesConfig.__instance = AlltalkTTSEnginesConfig()
        return AlltalkTTSEnginesConfig.__instance._reload_on_change()

    def _handle_loaded_config(self, data):
        # List of the available TTS engines from tts_engines.json
        self.engines_available = [engine["name"] for engine in data["engines_available"]]

        # The currently set TTS engine from tts_engines.json
        self.engine_loaded = data["engine_loaded"]
        self.selected_model = data["selected_model"]

class AlltalkConfig(AbstractConfig):
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self, config_path: Path | str = __this_dir / "confignew.json"):
        super().__init__(config_path, 5)
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
        self._load_config()

    @staticmethod
    def get_instance():
        if AlltalkConfig.__instance is None:
            AlltalkConfig.__instance = AlltalkConfig()
        return AlltalkConfig.__instance._reload_on_change()

    def save(self, path: Path | None = None):
        configfile_path = path if path is not None else self.get_config_path()

        # Remove private fields:
        without_privates = {}
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                without_privates[attr] = value

        with open(configfile_path, "w") as json_file:
            json.dump(without_privates, json_file, indent=4,  default=lambda o: o.__dict__)

    def _object_hook(self) -> Callable[[dict[Any, Any]], Any] | None:
        return lambda d: SimpleNamespace(**d)

    def _handle_loaded_config(self, data):
        # Copy those fields for which there are members in this class:
        for n, v in inspect.getmembers(data):
            if hasattr(self, n) and not n.startswith("__"):
                setattr(self, n, v)

        # Special properties (cannot use 'class' as property name):
        self.theme.clazz = data.theme.__dict__["class"]

        # As a side effect, create the output directory
        output_directory = self.__this_dir / self.output_folder
        output_directory.mkdir(parents=True, exist_ok=True)
