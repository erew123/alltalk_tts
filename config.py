import os
import json
import time
import inspect
import shutil
from pathlib import Path
from filelock import FileLock
from types import SimpleNamespace
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Any, MutableSequence


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

@dataclass
class AlltalkConfigGradioPages:
    Generate_Help_page = False
    Voice2RVC_page = False
    TTS_Generator_page = False
    TTS_Engines_Settings_page = False
    alltalk_documentation_page = False
    api_documentation_page = False


@dataclass
class AlltalkAvailableEngine:
    name = ""
    selected_model = ""


class AbstractJsonConfig(ABC):

    def __init__(self, config_path: Path | str, file_check_interval: int):
        self.__config_path = Path(config_path) if type(config_path) is str else config_path
        self.__last_read_time = 0  # Track when we last read the file
        self.__file_check_interval = file_check_interval

    def get_config_path(self):
        return self.__config_path

    def reload(self):
        self._load_config()
        return self

    def to_dict(self):
        # Remove private fields:
        without_private_fields = {}
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                without_private_fields[attr] = value
        return without_private_fields

    def _reload_on_change(self):
        # Check if config file has been modified and reload if needed
        if time.time() - self.__last_read_time >= self.__file_check_interval:
            try:
                most_recent_modification = self.get_config_path().stat().st_mtime
                if most_recent_modification > self.__last_read_time:
                    self.reload()
            except Exception as e:
                print(f"Error checking config file: {e}")

    def _load_config(self):
        self.__last_read_time = self.get_config_path().stat().st_mtime
        def __load():
            with open(self.get_config_path(), "r") as configfile:
                data = json.load(configfile, object_hook=self._object_hook())
            self._handle_loaded_config(data)
        self.__with_lock_and_backup(self.get_config_path(), False, __load)

    def _object_hook(self) -> Callable[[dict[Any, Any]], Any] | None:
        return lambda d: SimpleNamespace(**d)

    def _save_file(self, path: Path | None | str, default = lambda o: o.__dict__, indent = 4):
        file_path = (Path(path) if type(path) is str else path) if path is not None else self.get_config_path()

        def __save():
            with open(file_path, "w") as file:
                json.dump(self.to_dict(), file, indent=indent, default=default)
        self.__with_lock_and_backup(file_path, True, __save)

    def __with_lock_and_backup(self, path: Path, backup: bool, callable: Callable[[], None]):
        lock_path = path.with_suffix('.lock')
        try:
            with FileLock(lock_path):
                # Create backup:
                if path.exists() and backup:
                    backup_path = path.with_suffix('.backup')
                    shutil.copy(path, backup_path)

                try:
                    callable()
                except Exception as e:
                    if backup_path.exists():
                        shutil.copy(backup_path, path)
                    raise Exception(f"Failed to save config: {e}")
        finally:
            # Cleanup lock and backup files:
            lock_path.unlink()
            if backup and backup_path.exists():
                backup_path.unlink()


    @abstractmethod
    def _handle_loaded_config(self, data):
        pass

class AlltalkNewEnginesConfig(AbstractJsonConfig):
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self, config_path: Path | str = os.path.join(__this_dir, "system", "tts_engines", "new_engines.json")):
        super().__init__(config_path, 5)
        self.engines_available: MutableSequence[AlltalkAvailableEngine] = []
        self._load_config()

    def get_engine_names_available(self):
        return [engine.name for engine in self.engines_available]

    @staticmethod
    def get_instance():
        if AlltalkNewEnginesConfig.__instance is None:
            AlltalkNewEnginesConfig.__instance = AlltalkNewEnginesConfig()
        AlltalkNewEnginesConfig.__instance._reload_on_change()
        return AlltalkNewEnginesConfig.__instance

    def _handle_loaded_config(self, data):
        self.engines_available = data.engines_available

    def get_engines_matching(self, condition: Callable[[AlltalkAvailableEngine], bool]):
        return [x for x in self.engines_available if condition(x)]


class AlltalkTTSEnginesConfig(AbstractJsonConfig):
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self, config_path: Path | str = os.path.join(__this_dir, "system", "tts_engines", "tts_engines.json")):
        super().__init__(config_path, 5)
        self.engines_available: MutableSequence[AlltalkAvailableEngine] = []
        self.engine_loaded = ""
        self.selected_model = ""
        self._load_config()

    def get_engine_names_available(self):
        return [engine.name for engine in self.engines_available]

    @staticmethod
    def get_instance(force_reload = False):
        if AlltalkTTSEnginesConfig.__instance is None:
            force_reload = False
            AlltalkTTSEnginesConfig.__instance = AlltalkTTSEnginesConfig()

        if force_reload:
            AlltalkTTSEnginesConfig.__instance.reload()
        else:
            AlltalkTTSEnginesConfig.__instance._reload_on_change()
        return AlltalkTTSEnginesConfig.__instance

    def _handle_loaded_config(self, data):
        # List of the available TTS engines:
        self.engines_available = self.__handle_loaded_config_engines(data)

        # The currently set TTS engine from tts_engines.json
        self.engine_loaded = data.engine_loaded
        self.selected_model = data.selected_model

    def __handle_loaded_config_engines(self, data):
        available_engines = data.engines_available
        available_engine_names = [engine.name for engine in available_engines]

        # Getting the engines that are not already part of the available engines:
        new_engines_config = AlltalkNewEnginesConfig.get_instance()
        new_engines = new_engines_config.get_engines_matching(lambda eng: eng.name not in available_engine_names)

        # Merge engines:
        return available_engines + new_engines

    def save(self, path: Path | str | None = None):
        self._save_file(path)

    def is_valid_engine(self, engine_name):
        return engine_name in self.get_engine_names_available()

    def change_engine(self, requested_engine):
        if requested_engine == self.engine_loaded:
            return self
        for engine in self.engines_available:
            if engine.name == requested_engine:
                self.engine_loaded = requested_engine
                self.selected_model = engine.selected_model
                return self
        return self


class AlltalkConfig(AbstractJsonConfig):
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
        self.gradio_pages = AlltalkConfigGradioPages()
        self._load_config()

    @staticmethod
    def default_config_path():
        return AlltalkConfig.__this_dir / "confignew.json"

    @staticmethod
    def get_instance(force_reload = False):
        if AlltalkConfig.__instance is None:
            force_reload = False
            AlltalkConfig.__instance = AlltalkConfig()

        if force_reload:
            AlltalkConfig.__instance.reload()
        else:
            AlltalkConfig.__instance._reload_on_change()
        return AlltalkConfig.__instance

    def get_output_directory(self):
        return self.__this_dir / self.output_folder

    def save(self, path: Path | str | None = None):
        self._save_file(path)

    def _handle_loaded_config(self, data):
        # Copy those fields for which there are members in this class:
        for n, v in inspect.getmembers(data):
            if hasattr(self, n) and not n.startswith("__"):
                setattr(self, n, v)

        # Special properties (cannot use 'class' as property name):
        self.theme.clazz = data.theme.__dict__["class"]

        # As a side effect, create the output directory:
        self.get_output_directory().mkdir(parents=True, exist_ok=True)
