import os
import time
import shutil
from pathlib import Path

from filelock import FileLock
from abc import ABC, abstractmethod
from typing import Callable, MutableSequence

from pydantic import BaseModel, ConfigDict, AliasGenerator, AliasChoices, Field

class AlltalkConfigTheme(BaseModel):
    # Map 'class' to 'clazz' and vice versa
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name:
                {
                    "clazz": AliasChoices("clazz", "class"),
                }.get(field_name, None),
            serialization_alias=lambda field_name: "class" if field_name == "clazz" else field_name,
        )
    )
    file: str | None = None
    clazz: str = "gradio/base"

class AlltalkConfigRvcSettings(BaseModel):
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

class AlltalkConfigTgwUi(BaseModel):
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

class AlltalkConfigApiDef(BaseModel):
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

class AlltalkConfigDebug(BaseModel):
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
    debug_transcribe: bool = False
    debug_proxy: bool = False

class AlltalkConfigGradioPages(BaseModel):
    Generate_Help_page: bool = True
    Voice2RVC_page: bool = True
    TTS_Generator_page: bool = True
    TTS_Engines_Settings_page: bool = True
    alltalk_documentation_page: bool = True
    api_documentation_page: bool = True

class AlltalkAvailableEngine(BaseModel):
    name: str = ""
    selected_model: str = ""

class AlltalkConfigProxyEndpoint(BaseModel):
    enabled: bool = False
    external_port: int = 0
    external_ip: str = "0.0.0.0"
    cert_name: str = ""

class AlltalkConfigProxySettings(BaseModel):
    proxy_enabled: bool = False
    start_on_startup: bool = False
    gradio_endpoint: AlltalkConfigProxyEndpoint = AlltalkConfigProxyEndpoint(external_port=444)
    api_endpoint: AlltalkConfigProxyEndpoint = AlltalkConfigProxyEndpoint(external_port=443)
    cert_validation: bool = True
    logging_enabled: bool = True
    log_level: str = "INFO"

class AbstractJsonConfig(ABC):
    def __init__(self, config_path: Path | str, file_check_interval: int):
        super().__init__()
        self.__delegate = None
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
        if self.__file_check_interval != 0 and time.time() - self.__last_read_time >= self.__file_check_interval:
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
                json_string = configfile.read()
            # The delegate is the actual config loaded:
            self.__delegate = self._handle_loaded_config(json_string)
        self.__with_lock_and_backup(self.get_config_path(), False, __load)

    def _save_file(self, path: Path | None | str, default=None, indent=4):
        file_path = (Path(path) if type(path) is str else path) if path is not None else self.get_config_path()

        def __save():
            with open(file_path, 'w') as file:
                file.write(self.__delegate.model_dump_json(indent=indent, by_alias=True))

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
            if lock_path.exists():
                lock_path.unlink()

            if backup and backup_path and backup_path.exists():
                backup_path.unlink()

    @abstractmethod
    def _handle_loaded_config(self, json_string: str):
        pass

    def to_dict(self):
        return self.__delegate.model_dump(by_alias=True)

    # Delegation to the loaded config data
    @property
    def _xtra(self): return [o for o in dir(self.__delegate) if not o.startswith('_')]

    def __getattr__(self, key):
        if key in self._xtra: return getattr(self.__delegate, key)
        raise AttributeError(key)

    def __getattribute__(self, key):
        if key.startswith("_"):
            # Private and protected attributes always are on self:
            return super().__getattribute__(key)
        if key in self._xtra:
            # Delegate attribute
            return getattr(self.__delegate, key)
        # Get own attribute:
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            # Private and protected attributes always are on self:
            super().__setattr__(key, value)
        elif key in dir(self.__delegate):
            # Delegate attribute
            setattr(self.__delegate, key, value)
        else:
            # Set own attribute:
            super().__setattr__(key, value)

    def __dir__(self):
        def custom_dir(c, add): return dir(type(c)) + list(c.__dict__.keys()) + add
        return custom_dir(self, self._xtra)

class AlltalkNewEnginesConfigFields:
    engines_available: MutableSequence[AlltalkAvailableEngine] = Field(default_factory=list)

class AlltalkNewEnginesConfigModel(BaseModel, AlltalkNewEnginesConfigFields):

    def get_engine_names_available(self):
        return [engine.name for engine in self.engines_available]

    def get_engines_matching(self, condition: Callable[[AlltalkAvailableEngine], bool]):
        return [x for x in self.engines_available if condition(x)]

class AlltalkNewEnginesConfig(AbstractJsonConfig, AlltalkNewEnginesConfigFields):
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self, config_path: Path | str = os.path.join(__this_dir, "system", "tts_engines", "new_engines.json")):
        super().__init__(config_path, 5)
        self._load_config()

    @staticmethod
    def get_instance():
        if AlltalkNewEnginesConfig.__instance is None:
            AlltalkNewEnginesConfig.__instance = AlltalkNewEnginesConfig()
        AlltalkNewEnginesConfig.__instance._reload_on_change()
        return AlltalkNewEnginesConfig.__instance

    def _handle_loaded_config(self, json_string: str):
        return AlltalkNewEnginesConfigModel.model_validate_json(json_string)

class AlltalkTTSEnginesConfigFields:
    engines_available: MutableSequence[AlltalkAvailableEngine] = Field(default_factory=list)
    engine_loaded: str = ""
    selected_model: str = ""

class AlltalkTTSEnginesConfigModel(BaseModel, AlltalkTTSEnginesConfigFields):

    def get_engine_names_available(self):
        return [engine.name for engine in self.engines_available]

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

class AlltalkTTSEnginesConfig(AbstractJsonConfig, AlltalkTTSEnginesConfigFields):
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self, config_path: Path | str = os.path.join(__this_dir, "system", "tts_engines", "tts_engines.json")):
        super().__init__(config_path, 5)
        self._load_config()

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

    def _handle_loaded_config(self, json_string: str):
        cfg = AlltalkTTSEnginesConfigModel.model_validate_json(json_string)
        cfg.engines_available = self.__handle_loaded_config_engines(cfg.engines_available)
        return cfg

    def __handle_loaded_config_engines(self, available_engines):
        available_engine_names = [engine.name for engine in available_engines]

        # Getting the engines that are not already part of the available engines:
        new_engines_config = AlltalkNewEnginesConfig.get_instance()
        new_engines = new_engines_config.get_engines_matching(lambda eng: eng.name not in available_engine_names)

        # Merge engines:
        return available_engines + new_engines

    def save(self, path: Path | str | None = None):
        self._save_file(path)

class AlltalkConfigFields:
    branding: str = "AllTalk "
    delete_output_wavs: str = "Disabled"
    gradio_interface: bool = True
    output_folder: str = "outputs"
    gradio_port_number: int = 7852
    firstrun_model: bool = True
    firstrun_splash: bool = True
    launch_gradio: bool = True
    transcode_audio_format: str = "Disabled"
    theme: AlltalkConfigTheme = AlltalkConfigTheme()
    rvc_settings: AlltalkConfigRvcSettings = AlltalkConfigRvcSettings()
    tgwui: AlltalkConfigTgwUi = AlltalkConfigTgwUi()
    api_def: AlltalkConfigApiDef = AlltalkConfigApiDef()
    debugging: AlltalkConfigDebug = AlltalkConfigDebug()
    gradio_pages: AlltalkConfigGradioPages = AlltalkConfigGradioPages()
    proxy_settings: AlltalkConfigProxySettings = AlltalkConfigProxySettings()

class AlltalkConfigModel(BaseModel, AlltalkConfigFields):
    __this_dir = Path(__file__).parent.resolve()

    def get_output_directory(self):
        return self.__this_dir / self.output_folder

class AlltalkConfig(AbstractJsonConfig, AlltalkConfigFields):
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self, config_path: Path | str = __this_dir / "confignew.json"):
        super().__init__(config_path, 5)
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

    def save(self, path: Path | str | None = None):
        self._save_file(path)

    def _handle_loaded_config(self, json_string: str):
        model = AlltalkConfigModel.model_validate_json(json_string)
        model.get_output_directory().mkdir(parents=True, exist_ok=True)
        return model

class AlltalkMultiEngineManagerConfigFields:
    base_port: int = 7001
    api_server_port: int = 7851
    auto_start_engines: int = 0
    max_instances: int = 8
    gradio_interface_port: int = 7500
    max_retries: int = 12
    initial_wait: float = 2
    backoff_factor: float = 1.2
    debug_mode: bool = False
    max_queue_time: int = 60,  # Maximum time a request can wait in the queue (in seconds)
    queue_check_interval: float = 0.1,  # Time between checks for available instances (in seconds)
    tts_request_timeout: int = 30,  # Timeout for individual TTS requests (in seconds)
    text_length_factor: float = 0.2,  # Increase timeout by 20% per 100 characters
    concurrent_request_factor: float = 0.5,  # Increase timeout by 50% per concurrent request
    diminishing_factor: float = 0.5,  # Reduce additional time for long-running requests by 50%
    queue_position_factor: float = 1.0  # Add 100% of base timeout for each queue position

class AlltalkMultiEngineManagerConfigModel(BaseModel, AlltalkMultiEngineManagerConfigFields):
    pass

class AlltalkMultiEngineManagerConfig(AbstractJsonConfig, AlltalkMultiEngineManagerConfigFields):
    __instance = None
    __this_dir = Path(__file__).parent.resolve()

    def __init__(self, config_path: Path | str = os.path.join(__this_dir, "mem_config.json")):
        super().__init__(config_path, 0)
        self._load_config()

    @staticmethod
    def get_instance():
        if AlltalkMultiEngineManagerConfig.__instance is None:
            AlltalkMultiEngineManagerConfig.__instance = AlltalkMultiEngineManagerConfig()
        AlltalkMultiEngineManagerConfig.__instance._reload_on_change()
        return AlltalkMultiEngineManagerConfig.__instance

    def _handle_loaded_config(self, json_string: str):
        return AlltalkMultiEngineManagerConfigModel.model_validate_json(json_string)

    def save(self, path: Path | str | None = None):
        self._save_file(path)