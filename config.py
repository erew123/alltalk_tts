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
    rvc_enabled: bool = False
    rvc_char_model_file: str = "Disabled"
    rvc_narr_model_file: str = "Disabled" 
    split_audio: bool = True
    autotune: bool = False
    pitch: int = 0
    filter_radius: int = 3
    index_rate: float = 0.75
    rms_mix_rate: int = 1
    protect: float = 0.5
    hop_length: int = 130
    f0method: str = "fcpe"
    embedder_model: str = "hubert"
    training_data_size: int = 45000

@dataclass 
class AlltalkConfigTgwUi:
    tgwui_activate_tts: bool = True
    tgwui_autoplay_tts: bool = True
    tgwui_narrator_enabled: str = "false"
    tgwui_non_quoted_text_is: str = "character"
    tgwui_deepspeed_enabled: bool = False
    tgwui_language: str = "English"
    tgwui_lowvram_enabled: bool = False
    tgwui_pitch_set: int = 0
    tgwui_temperature_set: float = 0.75
    tgwui_repetitionpenalty_set: int = 10
    tgwui_generationspeed_set: int = 1
    tgwui_narrator_voice: str = "female_01.wav"
    tgwui_show_text: bool = True
    tgwui_character_voice: str = "female_01.wav"
    tgwui_rvc_char_voice: str = "Disabled"
    tgwui_rvc_char_pitch: int = 0
    tgwui_rvc_narr_voice: str = "Disabled"
    tgwui_rvc_narr_pitch: int = 0

@dataclass
class AlltalkConfigApiDef:
    api_port_number: int = 7851
    api_allowed_filter: str = "[^a-zA-Z0-9\\s.,;:!?\\-\\'\"$\\u0400-\\u04FF\\u00C0-\\u017F\\u0150\\u0151\\u0170\\u0171\\u011E\\u011F\\u0130\\u0131\\u0900-\\u097F\\u2018\\u2019\\u201C\\u201D\\u3001\\u3002\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FFF\\u3400-\\u4DBF\\uF900-\\uFAFF\\u0600-\\u06FF\\u0750-\\u077F\\uFB50-\\uFDFF\\uFE70-\\uFEFF\\uAC00-\\uD7A3\\u1100-\\u11FF\\u3130-\\u318F\\uFF01\\uFF0c\\uFF1A\\uFF1B\\uFF1F]"
    api_length_stripping: int = 3
    api_max_characters: int = 2000
    api_use_legacy_api: bool = False
    api_legacy_ip_address: str = "127.0.0.1"
    api_text_filtering: str = "standard"
    api_narrator_enabled: str = "false"
    api_text_not_inside: str = "character"
    api_language: str = "en"
    api_output_file_name: str = "myoutputfile"
    api_output_file_timestamp: bool = True
    api_autoplay: bool = False
    api_autoplay_volume: float = 0.5

@dataclass
class AlltalkConfigDebug:
    debug_transcode: bool = False
    debug_tts: bool = False
    debug_openai: bool = False
    debug_concat: bool = False
    debug_tts_variables: bool = False
    debug_rvc: bool = False
    debug_func: bool = False
    debug_api: bool = False
    debug_fullttstext: bool = False
    debug_narrator: bool = False
    debug_gradio_IP: bool = False

@dataclass
class AlltalkConfigGradioPages:
    Generate_Help_page: bool = True
    Voice2RVC_page: bool = True
    TTS_Generator_page: bool = True
    TTS_Engines_Settings_page: bool = True
    alltalk_documentation_page: bool = True
    api_documentation_page: bool = True


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

    def _save_file(self, path: Path | None | str, default=None, indent=4):
        file_path = (Path(path) if type(path) is str else path) if path is not None else self.get_config_path()
        
        def custom_default(o):
            if isinstance(o, Path):
                return str(o)  # Convert Path objects to strings
            elif hasattr(o, '__dict__'):
                return o.__dict__  # Use the object's __dict__ if it exists
            else:
                raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
        
        default = default or custom_default
        
        def __save():
            with open(file_path, 'w') as file:
                json.dump(self.to_dict(), file, indent=indent, default=default)
        
        self.__with_lock_and_backup(file_path, True, __save)


    def __with_lock_and_backup(self, path: Path, backup: bool, callable: Callable[[], None]):
        lock_path = path.with_suffix('.lock')
        backup_path = None
        try:
            with FileLock(lock_path):
                # Create backup:
                if path.exists() and backup:
                    backup_path = path.with_suffix('.backup')
                    shutil.copy(path, backup_path)

                try:
                    callable()
                except Exception as e:
                    if backup_path and backup_path.exists():
                        shutil.copy(backup_path, path)
                    raise Exception(f"Failed to save config: {e}")
        finally:
            # Cleanup lock and backup files:
            if lock_path.exists():  # Only try to delete if it exists
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass  # Ignore if file doesn't exist
            
            if backup and backup_path and backup_path.exists():
                try:
                    backup_path.unlink()
                except FileNotFoundError:
                    pass  # Ignore if file doesn't exist

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

@dataclass
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
        from dataclasses import fields, is_dataclass, asdict
        debug_me =False
        if debug_me:
            print("=== Loading Config ===")
            print(f"Initial data state: {vars(data)}")

        # Create new instances with defaults
        default_instances = {
            'debugging': AlltalkConfigDebug(),
            'rvc_settings': AlltalkConfigRvcSettings(),
            'tgwui': AlltalkConfigTgwUi(),
            'api_def': AlltalkConfigApiDef(),
            'theme': AlltalkConfigTheme(),
            'gradio_pages': AlltalkConfigGradioPages()
        }
        if debug_me:
            print("\nDefault values for each class:")
        for name, instance in default_instances.items():
            if debug_me:
                print(f"{name}: {asdict(instance)}")        
                # Show actual default values from dataclass
                print(f"Default values: {[(f.name, getattr(instance, f.name)) for f in fields(instance)]}")
            if hasattr(data, name):
                source = getattr(data, name)
                if debug_me:
                    print(f"Source data: {vars(source) if hasattr(source, '__dict__') else source}")

                for field in fields(instance):
                    if hasattr(source, field.name):
                        setattr(instance, field.name, getattr(source, field.name))
                    if debug_me:
                        print(f"Field {field.name}: {getattr(instance, field.name)}")

            setattr(self, name, instance)

        # Handle non-dataclass fields
        for n, v in inspect.getmembers(data):
            if hasattr(self, n) and not n.startswith("__") and not is_dataclass(type(getattr(self, n))):
                setattr(self, n, v)

        self.theme.clazz = data.theme.__dict__.get("class", data.theme.__dict__.get("clazz", ""))
        self.get_output_directory().mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        from dataclasses import is_dataclass, asdict
        debug_me =False
        if debug_me:
            print("=== Converting to dict ===")
        result = {}

        for key, value in vars(self).items():
            if not key.startswith('_'):
                # print(f"\nProcessing {key}:")
                if is_dataclass(value):
                    # print(f"Dataclass value before conversion: {vars(value)}")
                    result[key] = asdict(value)
                    # print(f"Converted to dict: {result[key]}")
                elif isinstance(value, SimpleNamespace):
                    # print(f"SimpleNamespace value: {value.__dict__}")
                    result[key] = value.__dict__
                else:
                    # print(f"Regular value: {value}")
                    result[key] = value

        if 'theme' in result:
            if debug_me:
                print("\nProcessing theme:")
                print(f"Before class handling: {result['theme']}")
            result['theme']['class'] = self.theme.clazz
            result['theme'].pop('clazz', None)
            if debug_me:
                print(f"After class handling: {result['theme']}")
        if debug_me:
            print(f"\nFinal dict: {result}")           
        return result



