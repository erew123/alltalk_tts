"""
XTTS finetune module for training and customizing text-to-speech models. 
Provides functionality for dataset creation, model training, and inference.
"""
# Standard Library Imports
import argparse
import datetime
import gc
import glob
import logging
import math
import os
import re
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# Third-Party Imports
import warnings
from importlib import metadata
import gradio as gr
import pandas as pd
import psutil
import torchaudio
import torchaudio.transforms as T
import torch
from packaging import version
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from word2number import w2n
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# TTS Package Imports
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
    XttsAudioConfig,
)

# Trainer Imports
from trainer_alltalk.trainer import TrainerArgs, Trainer

# Local Module Imports
from trainer_alltalk.metrics_logger import MetricsLogger
from system.ft_tokenizer.tokenizer import multilingual_cleaners

# Help documentation
from trainer_alltalk.finetune_content import FinetuneContent

# Suppress Warnings
warnings.filterwarnings(
    "ignore",
    message="1Torch was not compiled with flash attention")
warnings.filterwarnings(
    "ignore",
    message="Failed to launch Triton kernels, likely due to missing CUDA toolkit")

# Disable Gradio Analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Try to Import Whisper, Install if Not Found
try:
    import whisper
except ImportError:
    print("[FINETUNE] OpenAI Whisper not found. Attempting to install...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "openai-whisper"])
        import whisper

        print("[FINETUNE] Successfully installed OpenAI Whisper! Continuing.")
    except Exception as e:
        print("[FINETUNE] Failed to install OpenAI Whisper:")
        print(f"[FINETUNE] Error: {str(e)}")
        print(
            "[FINETUNE] Please try manually installing with: pip install openai-whisper")
        sys.exit(1)

# STARTUP VARIABLES

# Paths and Directories
this_dir = Path(__file__).parent.resolve()
audio_folder = this_dir / "finetune" / "put-voice-samples-in-here"
default_path = this_dir / "finetune" / "tmp-trn"
base_path = this_dir / "models" / "xtts"

# Gradio Configuration
theme = gr.themes.Default()
gradio_temp_dir = this_dir / "finetune" / "gradio_temp"
os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)

# Environment Variables
os.environ["TRAINER_TELEMETRY"] = "0"
out_path = default_path

# Progress Tracking
progress = 0

# Validation Globals (Set to None initially)
VALIDATE_TRAIN_METADATA_PATH = None
VALIDATE_EVAL_METADATA_PATH = None
VALIDATE_AUDIO_FOLDER = None
VALIDATE_WHISPER_MODEL = None
VALIDATE_TARGET_LANGUAGE = None
XTTS_MODEL = None

#################################
#### Define the Logger class ####
#################################


class Logger:
    """
    Singleton class to handle logging output to both the terminal and a log file.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """
        Ensure a single instance of the Logger class is created.
        Initializes the log file and other necessary attributes.
        """
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.log_file = "finetune.log"
            cls._instance.terminal = sys.stdout
            cls._instance.current_model_path = None  # To store current training path
            
            # Open in append mode
            cls._instance.log = open(cls._instance.log_file, "a", encoding="utf-8")
        return cls._instance

    def __init__(self, *args, **kwargs):
        """Initialize logger instance."""        
        pass

    def set_model_path(self, path):
        """
        Set the current model training path for logging context.
        :param path: Path to the current model directory.
        """        
        self.current_model_path = path

    def write(self, message):
        """
        Write a message to both the terminal and the log file.
        Filters out non-printable characters.
        :param message: The message to be logged.
        """        
        filtered_message = ''.join(char for char in message 
                                 if char.isprintable() or char in '\n\r\t')
        self.terminal.write(filtered_message)
        try:
            self.log.write(filtered_message)
            self.log.flush()
        except:
            pass

    def flush(self):
        """
        Flush any buffered log content to the log file and terminal.
        """        
        self.terminal.flush()
        try:
            self.log.flush()
        except:
            pass

    def isatty(self):
        """
        Mimic the isatty method to comply with terminal-like behavior.
        :return: Always returns False.
        """        
        return False
    
    def clear_log(self):
        """
        Delete the existing log file and recreate it to clear its contents.
        """
        try:
            # Close the current append-only file handle
            self.log.close()
            # Delete the log file
            os.remove(self.log_file)
            # Reopen the file in append mode for further logging
            self.log = open(self.log_file, "a", encoding="utf-8")
        except Exception as e:
            print(f"Failed to delete and recreate log file: {e}")

_logging_setup_done = False

def setup_logging():
    global _logging_setup_done
    if _logging_setup_done:
        return
        
    # redirect stdout and stderr to a file
    sys.stdout = Logger()
    sys.stderr = sys.stdout

    logging.basicConfig(
        level=logging.INFO,
        format="[FINETUNE] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    _logging_setup_done = True

# Call setup at module level
setup_logging()

c_logger = MetricsLogger()

def load_metrics():
    return c_logger.plot_metrics(), f"Running Time: {c_logger.format_duration(c_logger.total_duration)} - Estimated Completion: {c_logger.format_duration(c_logger.estimated_duration)}"

def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r", encoding="utf-8") as f:
        content = f.read()
        # Additional filtering when reading the file
        return ''.join(char for char in content 
                      if char.isprintable() or char in '\n\r\t')

##############################
#### Debugging management ####
##############################


class DebugLevels:
    """
    Defines debug levels for Stage 1 Dataset Creation

    Each debug level is a boolean flag that controls whether debug information
    for a specific area of functionality is logged.
    """

    GPU_MEMORY = False  # GPU memory and CUDA related debugging
    MODEL_OPS = True  # Model loading, transcription, cleanup operations
    DATA_PROCESS = False  # Data processing, words, sentences
    GENERAL = True  # General flow, file operations, metadata
    AUDIO = False  # Audio processing statistics and info
    SEGMENTS = False  # Detailed segment information
    DUPLICATES = False  # Duplicate handling information
    VALIDATION = False  # DataSet validation handling


def debug_print(
        message,
        level,
        is_error=False,
        is_warning=False,
        is_info=False):
    """Enhanced debug printing with categorization and formatting"""
    prefix = "[FINETUNE]"
    if is_error:
        prefix += " ERROR:"
    elif is_warning:
        prefix += " WARNING:"
    elif is_info:
        prefix += ""

    if level == "GPU_MEMORY" and DebugLevels.GPU_MEMORY:
        print(f"{prefix} [GPU] {message}")
    elif level == "MODEL_OPS" and DebugLevels.MODEL_OPS:
        print(f"{prefix} [MODEL] {message}")
    elif level == "DATA_PROCESS" and DebugLevels.DATA_PROCESS:
        print(f"{prefix} [DATA] {message}")
    elif level == "GENERAL" and DebugLevels.GENERAL:
        print(f"{prefix} [INFO] {message}")
    elif level == "AUDIO" and DebugLevels.AUDIO:
        print(f"{prefix} [AUDIO] {message}")
    elif level == "SEGMENTS" and DebugLevels.SEGMENTS:
        print(f"{prefix} [SEG] {message}")
    elif level == "DUPLICATES" and DebugLevels.DUPLICATES:
        print(f"{prefix} [DUP] {message}")
    elif level == "VALIDATION" and DebugLevels.VALIDATION:
        print(f"{prefix} [VAL] {message}")        


class AudioStats:
    """Track audio processing statistics"""

    def __init__(self):
        self.total_segments = 0
        self.segments_under_min = 0
        self.segments_over_max = 0
        self.total_duration = 0
        self.segment_durations = []

    def add_segment(self, duration):
        """
        Adds a new audio segment duration to tracking statistics.

        Args:
            duration (float): Duration of audio segment in seconds
        """
        self.total_segments += 1
        self.total_duration += duration
        self.segment_durations.append(duration)

    def print_stats(self):
        """
        Prints summary statistics for all processed audio segments.
        Shows total segments, average duration, segments under minimum length,
        and segments over maximum length.
        """
        if not self.segment_durations:
            return

        avg_duration = self.total_duration / self.total_segments
        debug_print("Audio Processing Statistics:", "AUDIO")
        debug_print(f"Total segments: {self.total_segments}", "AUDIO")
        debug_print(f"Average duration: {avg_duration:.2f}s", "AUDIO")
        debug_print(
            f"Segments under minimum: {self.segments_under_min}",
            "AUDIO")
        debug_print(
            f"Segments over maximum: {self.segments_over_max}",
            "AUDIO")

        if self.segments_under_min > 0:
            debug_print(
                f"{self.segments_under_min} segments are under minimum duration!",
                "AUDIO",
                is_warning=True,
            )


def get_gpu_memory():
    """Enhanced GPU memory reporting"""
    if DebugLevels.GPU_MEMORY:
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            debug_print("GPU Memory Status:", "GPU_MEMORY")
            debug_print(f"Total: {info.total / 1024**2:.2f} MB", "GPU_MEMORY")
            debug_print(f"Used:  {info.used / 1024**2:.2f} MB", "GPU_MEMORY")
            debug_print(f"Free:  {info.free / 1024**2:.2f} MB", "GPU_MEMORY")
            # Add warning if memory is low
            if info.free / info.total < 0.1:  # Less than 10% free
                debug_print(
                    "Low GPU memory available!",
                    "GPU_MEMORY",
                    is_warning=True)
        except Exception as e:
            debug_print(
                f"NVML not available: {e}",
                "GPU_MEMORY",
                is_error=True)


########################
#### Find All Model ####
########################
BASE_MODEL_DETECTED = False


def scan_models_folder():
    """
    Scans the models folder to detect available XTTS models and verifies required files.

    This function checks each subfolder within the base models directory to see if it contains
    all the required files for a valid model. It updates the `BASE_MODEL_DETECTED` global variable
    based on whether at least one valid model is found.

    Returns:
        dict: A dictionary where keys are model names (subfolder names) and values are booleans
        indicating whether the model folder contains all the required files.

    Globals:
        BASE_MODEL_DETECTED (bool): Set to `True` if at least one valid model is detected,
        otherwise `False`.

    Raises:
        FileNotFoundError: If the models folder is not found, prints an error message and exits
        the script.
    """
    global BASE_MODEL_DETECTED
    models_folder = base_path
    scan_available_models = {}
    scan_required_files = [
        "config.json",
        "model.pth",
        "mel_stats.pth",
        "speakers_xtts.pth",
        "vocab.json",
        "dvae.pth",
    ]
    try:
        for subfolder in models_folder.iterdir():
            if subfolder.is_dir():
                model_name = subfolder.name
                if all(subfolder.joinpath(file).exists()
                       for file in scan_required_files):
                    scan_available_models[model_name] = True
                    BASE_MODEL_DETECTED = True
                else:
                    debug_print(
                        f"Model folder '{model_name}' is missing required files",
                        level="GENERAL",
                        is_warning=True)
        if not scan_available_models:
            scan_available_models["No Model Available"] = False
            BASE_MODEL_DETECTED = False
    except FileNotFoundError:
        debug_print(
            "No XTTS models folder found. You have not yet downloaded any models or no XTTS",
            "GENERAL",
            is_error=True)
        debug_print(
            "models can be found. Please run AllTalk and download an XTTS model that can be",
            "GENERAL",
            is_error=True)
        debug_print(
            "used for training. Or place a full model in the following location",
            "GENERAL",
            is_error=True)
        debug_print(
            "\\models\\xtts\\{modelfolderhere}",
            "GENERAL",
            is_error=True)
        sys.exit(1)  # Exit the script with an error status
    return scan_available_models


# Get available models
available_models = scan_models_folder()

#################################################
#### Pre-Flight Checklist Functions & Gradio ####
#################################################


class SystemChecks:
    """Centralized system check management"""

    def __init__(self):
        self.status = {
            "overall": True,
            "disk_space": True,
            "ram": True,
            "gpu": True,
            "cuda": True,
            "pytorch": True,
            "tts": True,
            "base_model": True,
        }
        self.results = {}

    def check_disk_space(self, required_gb=18):
        """Check available disk space"""
        try:
            disk_usage = shutil.disk_usage(os.getcwd())
            free_space_gb = disk_usage.free / (1 << 30)

            status = free_space_gb > required_gb
            self.status["disk_space"] = status

            self.results["disk_space"] = {
                "status": "✅ Pass" if status else "❌ Fail",
                "details": f"{free_space_gb:.2f} GB available",
                "icon": "✅" if status else "❌",
                "message": (
                    "Sufficient disk space available"
                    if status
                    else f"Insufficient disk space. Need {required_gb}GB, have {free_space_gb:.2f}GB"
                ),
            }
        except Exception as e:
            self.handle_check_error("disk_space", str(e))

    def check_system_ram(self):
        """Check system RAM and GPU memory with refined thresholds"""
        try:
            vm = psutil.virtual_memory()
            total_ram_gb = vm.total / (1024**3)
            available_ram_gb = vm.available / (1024**3)

            # RAM check
            ram_status = available_ram_gb >= 8
            self.status["ram"] = ram_status

            # GPU memory check
            if torch.cuda.is_available():
                gpu_id = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(gpu_id)
                total_gpu_mem = torch.cuda.get_device_properties(
                    gpu_id).total_memory / (1024**3)
                used_gpu_mem = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                available_gpu_mem = total_gpu_mem - used_gpu_mem

                # New GPU status logic with better thresholds
                if total_gpu_mem >= 13:
                    gpu_status = "pass"  # More than 13GB - Clear pass
                elif total_gpu_mem >= 11.5:
                    gpu_status = "warning"  # Between 11.5GB and 13GB - Warning
                else:
                    gpu_status = "fail"  # Less than 11.5GB - Fail

                # Both pass and warning are considered "okay"
                self.status["gpu"] = gpu_status in ["pass", "warning"]
            else:
                gpu_name = "No GPU detected"
                total_gpu_mem = 0
                available_gpu_mem = 0
                gpu_status = "fail"
                self.status["gpu"] = False

            self.results["memory"] = {
                "ram_status": "✅ Pass" if ram_status else "❌ Fail",
                "ram_details": f"{available_ram_gb:.2f}GB available of {total_ram_gb:.2f}GB",
                "gpu_status": {
                    "pass": "✅ Pass",
                    "warning": "⚠️ Warning",
                    "fail": "❌ Fail"}[gpu_status],
                "gpu_details": f"{gpu_name}: {available_gpu_mem:.2f}GB available of {total_gpu_mem:.2f}GB",
            }
        except Exception as e:
            self.handle_check_error("memory", str(e))

    def check_cuda_pytorch(self):
        """Check CUDA and PyTorch setup"""
        try:
            cuda_available = torch.cuda.is_available()
            pytorch_version = torch.__version__

            if cuda_available:
                cuda_version = torch.version.cuda

                # Check if CUDA is actually working
                try:
                    torch.tensor([1.0, 2.0]).cuda()
                    cuda_working = True
                except BaseException:
                    cuda_working = False

                pytorch_cuda_status = cuda_version in ["11.8", "12.1"]
            else:
                cuda_working = False
                cuda_version = "N/A"
                pytorch_cuda_status = False

            self.status["cuda"] = cuda_working
            self.status["pytorch"] = pytorch_cuda_status

            # Separate results for CUDA and PyTorch
            self.results["cuda"] = {
                "status": "✅ Pass" if cuda_working else "❌ Fail",
                "details": f"CUDA {cuda_version}" if cuda_working else "CUDA not working",
            }

            self.results["pytorch"] = {
                "status": "✅ Pass" if pytorch_cuda_status else "❌ Fail",
                "details": f"PyTorch {pytorch_version} with CUDA {cuda_version}",
            }
        except Exception as e:
            self.handle_check_error("cuda_pytorch", str(e))

    def check_tts_version(self, required_version="0.24.0"):
        """Check TTS version"""
        try:
            installed_version = metadata.version("coqui-tts")
            status = version.parse(
                installed_version) >= version.parse(required_version)

            self.status["tts"] = status
            self.results["tts"] = {
                "status": "✅ Pass" if status else "❌ Fail",
                "details": f"TTS version {installed_version} installed",
                "meets_requirement": status,
            }
        except Exception as e:
            self.handle_check_error("tts", str(e))

    def check_base_model(self):
        """Check XTTS base model"""
        try:
            BASE_MODEL_DETECTED = any(
                available_models.values()) if available_models else False

            self.status["base_model"] = BASE_MODEL_DETECTED
            self.results["base_model"] = {
                "status": "✅ Pass" if BASE_MODEL_DETECTED else "❌ Fail",
                "details": "Base model detected" if BASE_MODEL_DETECTED else "No base model found",
            }
        except Exception as e:
            self.handle_check_error("base_model", str(e))

    def handle_check_error(self, check_name: str, error_msg: str):
        """Handle errors in checks"""
        self.status[check_name] = False
        self.status["overall"] = False
        self.results[check_name] = {
            "status": "❌ Error",
            "details": f"Check failed: {error_msg}"}

    def run_all_checks(self):
        """Run all system checks"""
        self.check_disk_space()
        self.check_system_ram()
        self.check_cuda_pytorch()
        self.check_tts_version()
        self.check_base_model()

        # Update overall status
        self.status["overall"] = all(self.status.values())
        return self.status["overall"]

    def get_markdown_report(self):
        """Generate markdown report of all checks"""
        report = []

        # Overall Status
        overall_icon = "✅" if self.status["overall"] else "❌"
        report.append(f"## System Check Results {overall_icon}\n")

        # Individual Checks
        if "disk_space" in self.results:
            report.append(
                f"### Storage\n{self.results['disk_space']['status']} {self.results['disk_space']['details']}\n"
            )

        if "memory" in self.results:
            report.append("### Memory")
            report.append(
                f"- RAM: {self.results['memory']['ram_status']} {self.results['memory']['ram_details']}"
            )
            report.append(
                f"- GPU: {self.results['memory']['gpu_status']} {self.results['memory']['gpu_details']}\n"
            )

        if "cuda_pytorch" in self.results:
            report.append(
                f"### CUDA & PyTorch\n{self.results['cuda_pytorch']['status']} {self.results['cuda_pytorch']['details']}\n"
            )

        if "tts" in self.results:
            report.append(
                f"### TTS\n{self.results['tts']['status']} {self.results['tts']['details']}\n"
            )

        if "base_model" in self.results:
            report.append(
                f"### Base Model\n{self.results['base_model']['status']} {self.results['base_model']['details']}\n"
            )

        return "\n".join(report)

    def get_ui_updates(self, _components):
        """Map check results to UI components"""
        # Get status for all checks
        status_updates = []

        # Overall status first
        overall_status = "✅ All Systems Go!" if self.status["overall"] else "⚠️ Some Checks Failed"
        status_updates.append(overall_status)

        # Add status for each check in the order they appear in
        # CHECK_REQUIREMENTS
        for check_id in CHECK_REQUIREMENTS:
            if check_id in self.results:
                status_updates.append(self.results[check_id]["status"])
            else:
                status_updates.append("❌ Check not run")

        return status_updates


# PFC Help Content and Requirements
HELP_CONTENT = {
    "disk_space_details": """
### Disk Space Requirements

#### Minimum Requirements
- **18GB** free disk space for training
- SSD recommended for better performance

#### Space Usage Breakdown
- Model files: ~4GB
- Training data: ~5-10GB
- Temporary files: ~4GB
- Safety margin: ~2GB

#### Troubleshooting
1. **Insufficient Space**
   - Clear space on the current partition
   - Run fine-tuning on another partition with more space
   - Keep at least 20% free space on drive

2. **Performance Issues**
   - Mechanical HDDs will be significantly slower
   - Consider using an SSD for training
""",
    "gpu_requirements": """
### GPU Memory Requirements

#### System Requirements
- **NVIDIA GPU Required**
- **12GB+ VRAM** recommended for optimal performance
- **8GB+ VRAM** minimum requirement

#### Platform Differences
- **Windows Systems:**
  * Can use system RAM as extended VRAM
  * 12GB VRAM or less requires 24GB+ System RAM and may fail.
  * 12GB VRAM cards will have a warning because of the above.
  * Performance may vary when using extended VRAM

- **Linux Systems:**
  * Cannot use extended VRAM
  * Limited to physical GPU VRAM only
  * Minimum 12GB VRAM recommended
""",
    "pytorch_cuda": """
### PyTorch and CUDA

- CUDA-enabled PyTorch installation required
- Compatible with CUDA 11.8/12.1
- Current PyTorch version recommended
""",
    "model_setup": """
### Base Model Requirements

#### Model Installation
- Download XTTS v2.0.3 model (recommended)
- BPE Tokenize needs a v2.0.3 model
- Use AllTalk's main interface:
  * TTS Engine Settings > XTTS > Model/Voices Download

#### Model Location
- Place in `/models/xtts/{modelname}/`
- Required files:
  * model.pth
  * config.json
  * vocab.json
  * dvae.pth
  * mel_stats.pth
  * speakers_xtts.pth
""",
}

CHECK_REQUIREMENTS = {
    "disk_space": {
        "label": "Disk Space",
        "requirement": "18GB+ Required",
        "description": "Required for temporary files and model storage",
        "priority": "high",
        "check_function": "check_disk_space",
    },
    "ram": {
        "label": "System RAM",
        "requirement": "16GB+ Recommended",
        "description": "Affects overall processing speed and stability",
        "priority": "medium",
        "check_function": "check_system_ram",
    },
    "vram": {
        "label": "GPU VRAM",
        "requirement": "12GB+ Recommended",
        "description": "Crucial for model training performance",
        "priority": "high",
        "check_function": "check_gpu_memory",
    },
    "cuda": {
        "label": "CUDA Support",
        "requirement": "CUDA 11.8/12.1",
        "description": "Required for GPU acceleration",
        "priority": "high",
        "check_function": "check_cuda_pytorch",
    },
    "pytorch": {
        "label": "PyTorch",
        "requirement": "CUDA enabled",
        "description": "Must be CUDA compatible version",
        "priority": "high",
        "check_function": "check_cuda_pytorch",
    },
    "tts": {
        "label": "TTS Version",
        "requirement": "0.24.0+",
        "description": "Required for fine-tuning",
        "priority": "high",
        "check_function": "check_tts_version",
    },
    "base_model": {
        "label": "Base Model",
        "requirement": "Must be present",
        "description": "XTTS base model required",
        "priority": "high",
        "check_function": "check_base_model",
    },
}

# Display categories for organizing checks
CHECK_CATEGORIES = {
    "hardware": {
        "title": "💻 Hardware Requirements",
        "description": "System hardware capabilities",
        "checks": ["disk_space", "ram", "vram"],
    },
    "cuda": {
        "title": "🔧 CUDA Setup",
        "description": "CUDA and PyTorch configuration",
        "checks": ["cuda", "pytorch"],
    },
    "software": {
        "title": "🤖 Software Setup",
        "description": "Required software and models",
        "checks": ["tts", "base_model"],
    },
}


class PFCComponents:
    """Store all PFC UI components"""

    def __init__(self):
        self.overall_status = None
        self.status_boxes = {}
        self.accordions = {}

    def create_status_box(self, check_id):
        """Create a status box for a specific check"""
        check_info = CHECK_REQUIREMENTS[check_id]
        with gr.Group():
            status = gr.Label(
                label=check_info["label"],
                value="Ready to Check",
                elem_classes="status-indicator")
            gr.Markdown(f"*Required: {check_info['requirement']}*")
            self.status_boxes[check_id] = status
            return status


def create_pfc_interface():
    """
    Creates a pre-flight checklist Gradio interface for validating system requirements before XTTS model training.

    Validates:
    - Disk space (min 18GB)
    - System RAM (min 16GB recommended)
    - GPU VRAM (min 12GB recommended)
    - CUDA support
    - PyTorch configuration
    - TTS version (min 0.24.0)
    - Base model presence

    Returns:
        PFCComponents: Container with all UI component references for status updates
    """
    components = PFCComponents()
    system_checks = SystemChecks()

    with gr.Column():
        # Status and button at top
        with gr.Row():
            components.overall_status = gr.Label(
                value="Click 'Run System Checks' to begin",
                label="System Status",
                elem_classes="status-label",
                scale=2,
            )
        with gr.Row():
            refresh_btn = gr.Button("Run System Checks", scale=1)

        # Checks in 3x2 grid
        with gr.Row():
            for check_id in CHECK_REQUIREMENTS:
                with gr.Column(scale=1):
                    components.create_status_box(check_id)

        # Help sections in 2x2 grid
        with gr.Group("💡 Help & Troubleshooting"):
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("💻 Hardware Requirements", open=False):
                        gr.Markdown(HELP_CONTENT["gpu_requirements"])

                with gr.Column():
                    with gr.Accordion("🖥️ Storage Requirements", open=False):
                        gr.Markdown(HELP_CONTENT["disk_space_details"])

            with gr.Row():
                with gr.Column():
                    with gr.Accordion("🤖 PyTorch & CUDA", open=False):
                        gr.Markdown(HELP_CONTENT["pytorch_cuda"])

                with gr.Column():
                    with gr.Accordion("📦 Model Setup", open=False):
                        gr.Markdown(HELP_CONTENT["model_setup"])

            def run_checks():
                system_checks.run_all_checks()

                # Prepare updates for all components
                updates = []

                # Overall status
                updates.append(
                    "✅ All Systems Go!"
                    if system_checks.status["overall"]
                    else "⚠️ Some Checks Failed"
                )

                # Add updates for each category's checks
                for category in CHECK_CATEGORIES.values():
                    for check_id in category["checks"]:
                        if check_id == "ram":
                            updates.append(
                                system_checks.results["memory"]["ram_status"])
                        elif check_id == "vram":
                            updates.append(
                                system_checks.results["memory"]["gpu_status"])
                        elif check_id == "cuda":
                            updates.append(
                                system_checks.results["cuda"]["status"])
                        elif check_id == "pytorch":
                            updates.append(
                                system_checks.results["pytorch"]["status"])
                        else:
                            updates.append(
                                system_checks.results.get(check_id, {}).get(
                                    "status", "❌ Check not run"
                                )
                            )

                return updates

            refresh_btn.click(
                fn=run_checks,
                outputs=[
                    components.overall_status] +
                list(
                    components.status_boxes.values()),
            )

    return components


###########################################
#### STEP 1 Dataset Creation Functions ####
###########################################

def format_audio_list(
        fal_target_language,
        fal_whisper_model,
        fal_max_sample_length,
        fal_eval_split_number,
        fal_speaker_name_input,
        fal_create_bpe_tokenizer,
        fal_gradio_progress=gr.Progress(),
        fal_use_vad=True,
        fal_precision="mixed",
        fal_min_sample_length=6):
    """
    Process and format audio files for XTTS training. Handles audio segmentation, transcription,
    and metadata creation with optional VAD and precision settings.

    Returns:
        tuple: (train_metadata_path, eval_metadata_path, audio_total_size)
    """
    global validate_train_metadata_path, validate_eval_metadata_path, validate_audio_folder 
    global validate_whisper_model, validate_target_language, out_path, torch, whisper # pylint: disable=no-member

    # Clear down the finetune.log file
    Logger().clear_log()

    # Initialize statistics tracker
    stats = AudioStats()

    # Basic setup
    buffer = 0.3
    max_duration = float(fal_max_sample_length)
    min_duration = float(fal_min_sample_length)
    eval_percentage = fal_eval_split_number / 100.0
    speaker_name = fal_speaker_name_input
    audio_total_size = 0
    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    too_long_files = []

    # Initialize directories
    if speaker_name and speaker_name != 'personsname':
        out_path = this_dir / "finetune" / speaker_name
    else:
        out_path = default_path

    debug_print(f"Initializing output directory: {out_path}", "GENERAL")

    os.makedirs(out_path, exist_ok=True)
    temp_folder = os.path.join(out_path, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    audio_folder = os.path.join(out_path, "wavs")
    os.makedirs(audio_folder, exist_ok=True)
    original_samples_folder = os.path.join(
        out_path, "..", "put-voice-samples-in-here")

    # Handle language file
    lang_file_path = os.path.join(out_path, "lang.txt")
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

    if current_language != fal_target_language:
        with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
            lang_file.write(fal_target_language + '\n')
        debug_print(f"Updated language to: {fal_target_language}", "GENERAL")
    else:
        debug_print("Using existing language setting", "GENERAL")

    # Load Whisper model with specified precision
    fal_gradio_progress((1, 10), desc="Loading Whisper Model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_print(f"Using device: {device}", "MODEL_OPS")

    # Monitor GPU memory before model loading
    get_gpu_memory()

    debug_print(f"Loading Whisper model: {fal_whisper_model}", "MODEL_OPS")
    asr_model = whisper.load_model(fal_whisper_model, device=device)

    if fal_precision == "float16" and device == "cuda":
        debug_print("Using FP16 precision", "MODEL_OPS")
        asr_model = asr_model.half()
    elif fal_precision == "mixed" and device == "cuda":
        debug_print("Using mixed precision", "MODEL_OPS")
    else:
        debug_print("Using FP32 precision", "MODEL_OPS")

    # Initialize SileroVAD if requested
    vad_model = None
    if fal_use_vad:
        fal_gradio_progress((2, 10), desc="Loading VAD Model")
        debug_print("Initializing Silero VAD", "MODEL_OPS")
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False)
        vad_model = model.to(device)
        get_speech_timestamps, collect_chunks = utils[0], utils[4]

    # Monitor GPU memory after model loading
    get_gpu_memory()

    # Load existing metadata
    fal_gradio_progress((3, 10), desc="Checking for Existing Metadata")
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    existing_metadata = {'train': None, 'eval': None}

    if os.path.exists(train_metadata_path):
        existing_metadata['train'] = pd.read_csv(train_metadata_path, sep="|")
        debug_print("Loaded existing training metadata", "DATA_PROCESS")

    if os.path.exists(eval_metadata_path):
        existing_metadata['eval'] = pd.read_csv(eval_metadata_path, sep="|")
        debug_print("Loaded existing evaluation metadata", "DATA_PROCESS")

    # Get audio files list
    original_audio_files = [os.path.join(original_samples_folder, file)
                            for file in os.listdir(original_samples_folder)
                            if file.endswith(('.mp3', '.flac', '.wav'))]

    fal_gradio_progress((4, 10), desc="Scanning for Audio Files")
    if not original_audio_files:
        debug_print(
            f"No audio files found in {original_samples_folder}",
            "AUDIO",
            is_error=True)
        return None, None, 0

    debug_print(
        f"Found {len(original_audio_files)} audio files to process",
        "AUDIO")
    # Initialize processing
    whisper_words = []
    audio_steps = (0, len(original_audio_files))
    gradio_progress_duration = 0
    fal_gradio_progress(
        audio_steps,
        desc="Processing Audio Files",
        unit="files")

    for audio_path in original_audio_files:
        start = datetime.datetime.now()
        audio_file_name_without_ext, _ = os.path.splitext(
            os.path.basename(audio_path))
        temp_audio_path = os.path.join(
            temp_folder, f"{audio_file_name_without_ext}.wav")

        try:
            shutil.copy2(audio_path, temp_audio_path)
            fal_gradio_progress(
                audio_steps,
                desc=f"Processing {audio_file_name_without_ext}",
                unit="files")
            debug_print(
                f"Processing: {audio_file_name_without_ext}",
                "GENERAL")
        except Exception as e:
            debug_print(
                f"Error copying file {audio_path}: {str(e)}",
                "GENERAL",
                is_error=True)
            continue

        # Check if already processed
        prefix_check = f"wavs/{audio_file_name_without_ext}_"
        skip_processing = False
        for key in ['train', 'eval']:
            if existing_metadata[key] is not None:
                mask = existing_metadata[key]['audio_file'].str.startswith(
                    prefix_check)
                if mask.any():
                    debug_print(
                        f"Skipping previously processed file: {audio_file_name_without_ext}",
                        "GENERAL")
                    skip_processing = True
                    audio_total_size = 121
                    break

        if skip_processing:
            continue

        # Load and process audio
        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        original_duration = wav.size(-1) / sr
        debug_print(
            f"Original audio duration: {original_duration:.2f}s",
            "AUDIO")

        # Process with VAD if enabled
        if fal_use_vad and vad_model is not None:
            debug_print("Processing with VAD", "AUDIO")
            # Get VAD segments with resampling
            vad_segments = process_audio_with_vad(
                wav, sr, vad_model, get_speech_timestamps)

            # Group short segments that are close together
            merged_segments = merge_short_segments(
                vad_segments, min_duration, max_gap=0.3)
            debug_print(
                f"Merged {len(vad_segments)-len(merged_segments)} short segments",
                "SEGMENTS")

            # Convert VAD segments to audio chunks
            speech_chunks = []
            for segment in merged_segments:
                chunk = wav[segment['start']:segment['end']]
                duration = chunk.size(-1) / sr
                if duration < min_duration:
                    debug_print(
                        f"Segment too short ({duration:.2f}s), attempting to extend",
                        "SEGMENTS",
                        is_warning=True)
                    # Try to extend segment if possible
                    chunk = extend_segment(
                        wav, segment['start'], segment['end'], sr, min_duration)
                    duration = chunk.size(-1) / sr

                if chunk.numel() > 0:
                    speech_chunks.append((chunk, duration))
                    stats.add_segment(duration)

            # Process each speech chunk
            for chunk_idx, (chunk, duration) in enumerate(speech_chunks):
                if duration < min_duration:
                    stats.segments_under_min += 1
                    debug_print(
                        f"Short segment: {duration:.2f}s",
                        "SEGMENTS",
                        is_warning=True)
                elif duration > max_duration:
                    stats.segments_over_max += 1
                    debug_print(
                        f"Long segment: {duration:.2f}s",
                        "SEGMENTS",
                        is_warning=True)

                chunk_path = os.path.join(
                    temp_folder, f"{audio_file_name_without_ext}_chunk_{chunk_idx}.wav")
                torchaudio.save(chunk_path, chunk.unsqueeze(0), sr)

                # Transcribe with appropriate precision
                if fal_precision == "mixed" and device == "cuda":
                    with torch.cuda.amp.autocast():
                        fal_gradio_progress((5, 10), desc="Transcribing Audio")
                        result = asr_model.transcribe(
                            chunk_path,
                            language=fal_target_language,
                            word_timestamps=True,
                            verbose=None
                        )
                else:
                    fal_gradio_progress((5, 10), desc="Transcribing Audio")
                    result = asr_model.transcribe(
                        chunk_path,
                        language=fal_target_language,
                        word_timestamps=True,
                        verbose=None
                    )

                if not result.get("text", "").strip():
                    debug_print(
                        f"Empty transcription for chunk {chunk_idx}",
                        "DATA_PROCESS",
                        is_warning=True)
                    continue

                # Process transcription result
                process_transcription_result(
                    result,
                    chunk,
                    sr,
                    chunk_idx,
                    audio_file_name_without_ext,
                    metadata,
                    whisper_words,
                    max_duration,
                    buffer,
                    speaker_name,
                    audio_folder,
                    too_long_files,
                    fal_create_bpe_tokenizer,
                    fal_target_language)

                os.remove(chunk_path)
                debug_print(
                    f"Processed chunk {chunk_idx} ({duration:.2f}s)",
                    "SEGMENTS")

        else:
            # Regular processing without VAD
            debug_print("Processing without VAD", "AUDIO")
            if fal_precision == "mixed" and device == "cuda":
                with torch.cuda.amp.autocast():
                    result = asr_model.transcribe(
                        audio_path,
                        language=fal_target_language,
                        word_timestamps=True,
                        verbose=None
                    )
            else:
                result = asr_model.transcribe(
                    audio_path,
                    language=fal_target_language,
                    word_timestamps=True,
                    verbose=None
                )

            # Process transcription result
            process_transcription_result(
                result,
                wav,
                sr,
                0,
                audio_file_name_without_ext,
                metadata,
                whisper_words,
                max_duration,
                buffer,
                speaker_name,
                audio_folder,
                too_long_files,
                fal_create_bpe_tokenizer,
                fal_target_language)

        os.remove(temp_audio_path)

        # Update progress
        end = datetime.datetime.now()
        gradio_progress_duration += (end - start).total_seconds()
        audio_steps = (audio_steps[0] + 1, audio_steps[1])
        additional_data_points_needed = audio_steps[1] - audio_steps[0]
        avg_duration = gradio_progress_duration / audio_steps[0]
        gradio_estimated_duration = (
            avg_duration * additional_data_points_needed)
        fal_gradio_progress(
            audio_steps,
            desc=f"Processing. Estimated Completion: {c_logger.format_duration(gradio_estimated_duration)}",
            unit="files")

    # Print final statistics
    stats.print_stats()

    # Verify processed files exist
    audio_files = [os.path.join(audio_folder, file) for file in os.listdir(
        audio_folder) if file.endswith('.wav')]
    if not audio_files:
        debug_print("No processed audio files found", "AUDIO", is_error=True)
        return None, None, 0

    # Final statistics before metadata handling
    stats.print_stats()

    # Handle existing metadata case
    if os.path.exists(train_metadata_path) and os.path.exists(
            eval_metadata_path):
        debug_print("Using existing metadata files", "DATA_PROCESS")
        _set_validation_paths(
            train_metadata_path, eval_metadata_path, audio_folder,
            fal_whisper_model, fal_target_language
        )
        _cleanup_resources(asr_model, existing_metadata)
        return train_metadata_path, eval_metadata_path, audio_total_size

    # Check for new metadata
    if not metadata["audio_file"]:
        debug_print(
            "No new audio files to process",
            "DATA_PROCESS",
            is_warning=True)
        _set_validation_paths(
            train_metadata_path, eval_metadata_path, audio_folder,
            fal_whisper_model, fal_target_language
        )
        _cleanup_resources(asr_model, existing_metadata)
        return train_metadata_path, eval_metadata_path, audio_total_size

    # Process metadata and handle duplicates
    debug_print("Processing metadata and handling duplicates", "DATA_PROCESS")
    new_data_df = pd.DataFrame(metadata)

    # Duplicate detection and handling
    duplicate_files = new_data_df['audio_file'].value_counts()
    duplicates_found = duplicate_files[duplicate_files > 1]

    fal_gradio_progress((6, 10), desc="Handling Duplicate Transcriptions")
    if not duplicates_found.empty:
        debug_print(
            f"Found {len(duplicates_found)} files with multiple transcriptions",
            "DUPLICATES")
        for file, count in duplicates_found.items():
            debug_print(f"{file}: {count} occurrences", "DUPLICATES")

        # Re-transcribe duplicates
        best_transcriptions = handle_duplicates(
            duplicates_found.index,
            audio_folder,
            fal_target_language,
            fal_whisper_model)

        # Update transcriptions and remove duplicates
        for file_path, trans_info in best_transcriptions.items():
            new_data_df.loc[new_data_df['audio_file'] ==
                            file_path, 'text'] = trans_info['text']
            debug_print(f"Updated transcription for {file_path}", "DUPLICATES")

        new_data_df = new_data_df.drop_duplicates(
            subset='audio_file', keep='first')
        debug_print(
            f"Cleaned up {len(duplicates_found)} duplicate entries",
            "DUPLICATES")

    # Handle evaluation split
    debug_print("Creating train/eval split", "DATA_PROCESS")
    eval_percentage = _adjust_eval_percentage(fal_eval_split_number)

    # Create and validate splits
    fal_gradio_progress((7, 10), desc="Creating Train/Eval Split")
    train_eval_split = create_dataset_splits(
        new_data_df,
        eval_percentage,
        random_seed=42
    )

    if train_eval_split is None:
        debug_print(
            "Failed to create valid dataset splits",
            "DATA_PROCESS",
            is_error=True)
        return None, None, 0

    final_training_set, final_eval_set = train_eval_split

    # Write metadata files
    debug_print(
        f"Writing {len(final_training_set)} training and {len(final_eval_set)} eval samples",
        "DATA_PROCESS")
    try:
        fal_gradio_progress((8, 10), desc="Saving Metadata Files")
        _write_metadata_files(
            final_training_set, final_eval_set,
            train_metadata_path, eval_metadata_path
        )
    except Exception as e:
        debug_print(
            f"Error writing metadata: {str(e)}",
            "DATA_PROCESS",
            is_error=True)
        raise

    # Handle BPE tokenizer
    if fal_create_bpe_tokenizer:
        fal_gradio_progress((9, 10), desc="Training BPE Tokenizer")
        debug_print("Training BPE Tokenizer", "MODEL_OPS")
        _create_bpe_tokenizer(whisper_words, out_path, base_path)

    debug_print("Finalizing processing", "GENERAL")
    fal_gradio_progress((10, 10), desc="Finalizing Process")

    # Cleanup and set validation paths
    _cleanup_resources(
        asr_model,
        final_eval_set,
        final_training_set,
        new_data_df,
        existing_metadata)
    _set_validation_paths(
        train_metadata_path, eval_metadata_path, audio_folder,
        fal_whisper_model, fal_target_language
    )

    # Log final statistics
    if too_long_files:
        debug_print("Files that were split due to length:", "SEGMENTS")
        for file_name, length in too_long_files:
            debug_print(f"  {file_name}: {length:.2f} seconds", "SEGMENTS")

    return train_metadata_path, eval_metadata_path, audio_total_size


def _create_bpe_tokenizer(bpe_whisper_words, bpe_out_path, bpe_base_path):
    """Create and train BPE tokenizer"""
    vocab_path = bpe_base_path / "xttsv2_2.0.3" / "vocab.json"
    if not vocab_path.exists():
        debug_print(
            "BPE tokenizer will not be created for this dataset creation run.",
            "GENERAL",
            is_warning=True,
        )
        debug_print(
            "XTTS v2.0.3 base model not found for BPE Tokenizer. Please download it first using AllTalk's",
            "GENERAL",
            is_warning=True,
        )
        debug_print(
            "main Gradio interface > TTS Engine Settings > XTTS > Model/Voices Downloads.",
            "GENERAL",
            is_warning=True,
        )
        debug_print(
            "Dataset creation will continue without the BPE tokenizer.",
            "GENERAL",
            is_warning=True)
        raise FileNotFoundError(f"Missing required file: {vocab_path}")

    try:
        debug_print("Initializing BPE tokenizer training", "MODEL_OPS")
        tokenizer = ByteLevelBPETokenizer(str(vocab_path))
        tokenizer.pre_tokenizer = Whitespace()

        # Add special tokens
        special_tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[STOP]",
            "[SPACE]"]
        tokenizer.add_tokens(special_tokens)

        debug_print(
            f"Training tokenizer on {len(bpe_whisper_words)} words",
            "MODEL_OPS")
        tokenizer.train_from_iterator(
            bpe_whisper_words,
            vocab_size=30000,
            show_progress=True,
            min_frequency=2,
            special_tokens=special_tokens,
        )

        # Save tokenizer
        tokenizer_path = str(bpe_out_path / "bpe_tokenizer-vocab.json")
        tokenizer.save(path=tokenizer_path, pretty=True)
        debug_print(f"Saved BPE tokenizer to {tokenizer_path}", "MODEL_OPS")

    except Exception as e:
        debug_print(
            f"Failed to create BPE tokenizer: {str(e)}",
            "MODEL_OPS",
            is_error=True)
        raise


def merge_short_segments(segments, min_duration, max_gap=0.5):
    """
    More aggressive merge strategy for short segments
    - Increases max_gap to 0.5s (from 0.3s)
    - Looks ahead multiple segments for potential merges
    - Considers surrounding context
    """
    if not segments:
        return segments

    merged = []
    current_group = []

    for i, segment in enumerate(segments):
        current_duration = sum(s["end"] - s["start"]
                               for s in current_group) if current_group else 0

        # If this is a continuation of current group
        if current_group and (
                segment["start"] - current_group[-1]["end"]) <= max_gap:
            current_group.append(segment)
        # If starting new group but previous was too short
        elif current_group and current_duration < min_duration:
            # Look ahead for close segments
            # Look up to 3 segments ahead
            look_ahead = min(3, len(segments) - i)
            for j in range(i, i + look_ahead):
                if (j < len(segments) and (
                        segments[j]["start"] - current_group[-1]["end"]) <= max_gap * 2):
                    current_group.append(segments[j])
                else:
                    break
        else:
            # Save previous group if it exists
            if current_group:
                merged_segment = {
                    "start": current_group[0]["start"],
                    "end": current_group[-1]["end"],
                }
                merged.append(merged_segment)
            current_group = [segment]

    # Handle last group
    if current_group:
        merged_segment = {
            "start": current_group[0]["start"], "end": current_group[-1]["end"]}
        merged.append(merged_segment)

    debug_print(
        f"Merged {len(segments) - len(merged)} segments into {len(merged)} longer segments",
        "SEGMENTS",
    )
    return merged


def extend_segment(wav, start, end, sr, min_duration, context_window=1.0):
    """
    Improved segment extension with better context handling
    - Adds context_window parameter for smoother extensions
    - More balanced extension on both sides
    - Checks audio content when extending
    """
    current_duration = (end - start) / sr
    if current_duration >= min_duration:
        return wav[start:end]

    samples_needed = int((min_duration - current_duration) * sr)

    # Try to extend equally on both sides
    extend_left = samples_needed // 2
    extend_right = samples_needed - extend_left

    # Add some context window
    context_samples = int(context_window * sr)
    new_start = max(0, start - extend_left - context_samples)
    new_end = min(wav.size(-1), end + extend_right + context_samples)

    # Check if we got enough duration
    if (new_end - new_start) / sr < min_duration:
        # If still too short, try to compensate from the other side
        if new_start == 0:
            new_end = min(wav.size(-1), end + samples_needed + context_samples)
        elif new_end == wav.size(-1):
            new_start = max(0, start - samples_needed - context_samples)

    debug_print(
        f"Extended segment from {current_duration:.2f}s to {(new_end - new_start) / sr:.2f}s",
        "SEGMENTS",
    )
    return wav[new_start:new_end]


# Helper functions for better organization

def _adjust_eval_percentage(aep_eval_split_number):
    """Adjust evaluation percentage to be within acceptable bounds"""
    eval_percentage = aep_eval_split_number / 100.0
    min_eval_percentage = 0.1
    max_eval_percentage = 0.3
    if eval_percentage < min_eval_percentage:
        debug_print(
            f"Adjusting eval split from {eval_percentage:.1%} to {min_eval_percentage:.1%}",
            "DATA_PROCESS",
            is_warning=True,
        )
        return min_eval_percentage
    if eval_percentage > max_eval_percentage:  # Changed from elif to if
        debug_print(
            f"Adjusting eval split from {eval_percentage:.1%} to {max_eval_percentage:.1%}",
            "DATA_PROCESS",
            is_warning=True,
        )
        return max_eval_percentage
    return eval_percentage


def _set_validation_paths(vp_train_path, vp_eval_path, vp_audio_folder,
                          vp_whisper_model, vp_target_language):
    """Set global validation paths"""
    global VALIDATE_TRAIN_METADATA_PATH, VALIDATE_EVAL_METADATA_PATH, VALIDATE_AUDIO_FOLDER # pylint: disable=no-member
    global VALIDATE_WHISPER_MODEL, VALIDATE_TARGET_LANGUAGE # pylint: disable=no-member

    VALIDATE_TRAIN_METADATA_PATH = vp_train_path
    VALIDATE_EVAL_METADATA_PATH = vp_eval_path
    VALIDATE_AUDIO_FOLDER = vp_audio_folder
    VALIDATE_WHISPER_MODEL = vp_whisper_model
    VALIDATE_TARGET_LANGUAGE = vp_target_language


def _cleanup_resources(*resources):
    """Clean up resources and free memory"""
    for resource in resources:
        del resource
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_metadata_files(
        wm_train_set,
        wm_eval_set,
        wm_train_path,
        wm_eval_path):
    """Write metadata files with error handling"""
    try:
        wm_train_set.sort_values("audio_file").to_csv(
            wm_train_path, sep="|", index=False)
        wm_eval_set.sort_values("audio_file").to_csv(
            wm_eval_path, sep="|", index=False)
        debug_print("Successfully wrote metadata files", "DATA_PROCESS")
    except Exception as e:
        debug_print(
            f"Failed to write metadata files: {str(e)}",
            "DATA_PROCESS",
            is_error=True)
        raise


def create_dataset_splits(df, eval_percentage, random_seed=42):
    """Create training and evaluation splits with validation"""
    if df.empty:
        debug_print(
            "No data available for splitting",
            "DATA_PROCESS",
            is_error=True)
        return None

    shuffled_df = df.sample(frac=1, random_state=random_seed)
    num_val_samples = max(1, int(len(shuffled_df) * eval_percentage))

    if num_val_samples >= len(shuffled_df):
        debug_print(
            "Not enough samples for valid split",
            "DATA_PROCESS",
            is_error=True)
        return None

    return (shuffled_df[num_val_samples:],
            shuffled_df[:num_val_samples])  # training  # eval


def save_audio_segment(
    sas_audio,
    sas_sr,
    sas_start_time,
    sas_end_time,
    sas_sentence,
    sas_audio_file_name_without_ext,
    sas_segment_idx,
    sas_speaker_name,
    sas_audio_folder,
    sas_metadata,
    sas_max_duration,
    _sas_buffer,
    sas_too_long_files,
    sas_target_language,
):
    """Helper function to save audio segments and update metadata"""
    sas_sentence = sas_sentence.strip()
    sas_sentence = multilingual_cleaners(sas_sentence, sas_target_language)
    sas_audio_file_name = f"{sas_audio_file_name_without_ext}_{str(sas_segment_idx).zfill(8)}.wav"

    sas_absolute_path = os.path.join(sas_audio_folder, sas_audio_file_name)
    os.makedirs(os.path.dirname(sas_absolute_path), exist_ok=True)

    # Extract audio segment
    sas_audio_start = int(sas_sr * sas_start_time)
    sas_audio_end = int(sas_sr * sas_end_time)
    sas_audio_segment = sas_audio[sas_audio_start:sas_audio_end].unsqueeze(0)

    # Handle long audio segments
    if sas_audio_segment.size(-1) > sas_max_duration * sas_sr:
        sas_too_long_files.append(
            (sas_audio_file_name, sas_audio_segment.size(-1) / sas_sr))

        while sas_audio_segment.size(-1) > sas_max_duration * sas_sr:
            sas_split_audio = sas_audio_segment[:, : int(
                sas_max_duration * sas_sr)]
            sas_audio_segment = sas_audio_segment[:, int(
                sas_max_duration * sas_sr):]
            sas_split_file_name = f"{sas_audio_file_name_without_ext}_{str(sas_segment_idx).zfill(8)}.wav"
            sas_split_relative_path = os.path.join(sas_split_file_name)
            sas_split_absolute_path = os.path.normpath(
                os.path.join(sas_audio_folder, sas_split_relative_path))

            os.makedirs(
                os.path.dirname(sas_split_absolute_path),
                exist_ok=True)
            torchaudio.save(sas_split_absolute_path, sas_split_audio, sas_sr)

            sas_metadata["audio_file"].append(
                f"wavs/{sas_split_relative_path}")
            sas_metadata["text"].append(sas_sentence)
            sas_metadata["speaker_name"].append(sas_speaker_name)
            sas_segment_idx += 1

    # Only save if segment is at least 1 second
    if sas_audio_segment.size(-1) >= sas_sr:
        torchaudio.save(sas_absolute_path, sas_audio_segment, sas_sr)
        sas_metadata["audio_file"].append(f"wavs/{sas_audio_file_name}")
        sas_metadata["text"].append(sas_sentence)
        sas_metadata["speaker_name"].append(sas_speaker_name)


def process_transcription_result(
    ptr_result,
    ptr_audio,
    ptr_sr,
    ptr_segment_idx,
    ptr_audio_file_name_without_ext,
    ptr_metadata,
    ptr_whisper_words,
    ptr_max_duration,
    ptr_buffer,
    ptr_speaker_name,
    ptr_audio_folder,
    ptr_too_long_files,
    ptr_create_bpe_tokenizer,
    ptr_target_language,
):
    """Helper function to process transcription results and save audio segments"""
    ptr_i = ptr_segment_idx + 1
    ptr_sentence = ""
    ptr_sentence_start = None
    ptr_first_word = True
    ptr_current_words = []

    for ptr_segment in ptr_result["segments"]:
        if "words" not in ptr_segment:
            continue

        for ptr_word_info in ptr_segment["words"]:
            ptr_word = ptr_word_info.get("word", "").strip()
            if not ptr_word:
                continue

            ptr_start_time = ptr_word_info.get("start", 0)
            ptr_end_time = ptr_word_info.get("end", 0)

            if ptr_create_bpe_tokenizer:
                ptr_whisper_words.append(ptr_word)

            if ptr_first_word:
                ptr_sentence_start = ptr_start_time
                if len(ptr_current_words) == 0:
                    ptr_sentence_start = max(
                        ptr_sentence_start - ptr_buffer, 0)
                else:
                    ptr_previous_end = ptr_current_words[-1].get(
                        "end", 0) if ptr_current_words else 0
                    ptr_sentence_start = max(
                        ptr_sentence_start - ptr_buffer,
                        (ptr_previous_end + ptr_start_time) / 2)
                ptr_sentence = ptr_word
                ptr_first_word = False
            else:
                ptr_sentence += " " + ptr_word

            ptr_current_words.append(
                {"word": ptr_word, "start": ptr_start_time, "end": ptr_end_time})

            # Handle sentence splitting and audio saving
            if ptr_word[-1] in ["!", ".",
                                "?"] or (ptr_end_time - ptr_sentence_start) > ptr_max_duration:
                save_audio_segment(
                    ptr_audio,
                    ptr_sr,
                    ptr_sentence_start,
                    ptr_end_time,
                    ptr_sentence,
                    ptr_audio_file_name_without_ext,
                    ptr_i,
                    ptr_speaker_name,
                    ptr_audio_folder,
                    ptr_metadata,
                    ptr_max_duration,
                    ptr_buffer,
                    ptr_too_long_files,
                    ptr_target_language,
                )
                ptr_i += 1
                ptr_first_word = True
                ptr_current_words = []
                ptr_sentence = ""


def process_audio_with_vad(wav, sr, vad_model, get_speech_timestamps):
    """
    Enhanced VAD processing with better end-of-speech detection
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav = wav.to(device)

    resampler = T.Resample(sr, 16000).to(device)
    wav_16k = resampler(wav)

    # Adjusted VAD parameters
    vad_segments = get_speech_timestamps(
        wav_16k,
        vad_model,
        sampling_rate=16000,
        threshold=0.2,  # Lower threshold to be more sensitive to speech
        min_speech_duration_ms=200,  # Shorter to catch brief utterances
        max_speech_duration_s=float("inf"),
        min_silence_duration_ms=300,  # Shorter silence duration
        window_size_samples=1024,  # Smaller window for more precise detection
        speech_pad_ms=300,  # Add padding to end of speech segments
    )

    # Scale timestamps back to original sample rate
    scale_factor = sr / 16000
    for segment in vad_segments:
        segment["start"] = int(segment["start"] * scale_factor)
        # Add extra padding at the end
        segment["end"] = int(segment["end"] * scale_factor) + \
            int(0.2 * sr)  # Add 200ms padding

    merged_segments = merge_short_segments(
        vad_segments, min_duration=6.0, max_gap=0.5)

    debug_print(
        f"VAD processing: {len(vad_segments)} original segments, {len(merged_segments)} after merging",
        "SEGMENTS",
    )
    return merged_segments


def handle_duplicates(
        duplicate_files,
        dup_audio_folder,
        dup_target_language,
        dup_whisper_model):
    """Re-transcribe duplicate files to get best transcription"""
    debug_print(
        "Re-transcribing duplicate files to get best transcription",
        "DUPLICATES")

    best_transcriptions = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model(dup_whisper_model, device=device)

    for file_path in duplicate_files:
        debug_print(f"Re-transcribing {file_path}", "DUPLICATES")

        # Get full path
        full_path = os.path.join(
            dup_audio_folder, os.path.basename(
                file_path.replace(
                    "wavs/", "")))

        # Re-transcribe with highest quality settings
        result = asr_model.transcribe(
            full_path,
            language=dup_target_language,
            word_timestamps=True,
            verbose=None)

        # Store the new transcription
        best_transcriptions[file_path] = {
            "text": result["text"].strip(),
            "confidence": sum(s.get("confidence", 0) for s in result["segments"])
            / len(result["segments"]),
        }

    return best_transcriptions

#############################################
#### STEP 1 Dataset Validation Functions ####
#############################################


def normalize_text(text):
    """
    Normalizes text by converting to lowercase, removing punctuation, converting written numbers
    to digits, and standardizing whitespace.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Replace multiple spaces with a single space  
    text = re.sub(r"\s+", " ", text)
    # Convert written numbers to digits
    words = text.split()
    normalized_words = []
    for word in words:
        try:
            # Try to convert word to a number  
            normalized_word = str(w2n.word_to_num(word))
        except ValueError:
            # If it fails, keep the original word
            normalized_word = word
        normalized_words.append(normalized_word)
    
    return " ".join(normalized_words)


def get_audio_file_list(mismatches):
    """Gets list of audio file paths from mismatched transcriptions DataFrame."""
    if mismatches.empty:
        return ["No bad transcriptions"]
    
    return mismatches["Audio Path"].tolist()


def load_and_display_mismatches():
    """
    Loads training/eval metadata CSVs, validates transcriptions against Whisper,
    and displays mismatches in Gradio interface.

    Returns:
        tuple: (
            mismatches_df: Full DataFrame with mismatch details,
            display_df: DataFrame with visible columns,
            message: Status message
        )
    """

    def validate_audio_transcriptions(
            vat_csv_paths,
            vat_audio_folder,
            vat_whisper_model,
            vat_target_language,
            vat_progress=None):
        # Load and combine metadata from CSV files
        metadata_dfs = []
        for csv_path in vat_csv_paths:
            debug_print(f"Reading CSV file: {csv_path}", "VALIDATION", is_info=True)
            metadata_df = pd.read_csv(csv_path, sep="|")
            debug_print(f"CSV columns: {metadata_df.columns.tolist()}", "VALIDATION", is_info=True)
            debug_print(f"Number of rows: {len(metadata_df)}", "VALIDATION", is_info=True)
            # Add source CSV tracking
            metadata_df["source_csv"] = csv_path
            metadata_df["row_index"] = metadata_df.index
            metadata_dfs.append(metadata_df)

        metadata_df = pd.concat(metadata_dfs, ignore_index=True)
        debug_print(f"Total combined rows: {len(metadata_df)}", "VALIDATION", is_info=True)

        # Load Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        asr_model = whisper.load_model(vat_whisper_model, device=device)

        mismatches = []
        missing_files = []
        total_files = metadata_df.shape[0]

        if vat_progress is not None:
            vat_progress((0, total_files), desc="Processing files")

        for index, row in tqdm(metadata_df.iterrows(
        ), total=total_files, unit="file", disable=False, leave=True):
            audio_file = row["audio_file"]
            expected_text = row["text"]
            debug_print(f"Processing file {index + 1}/{total_files}: {audio_file}", "VALIDATION", is_info=True)
            debug_print(f"Expected text length: {len(expected_text)}", "VALIDATION", is_info=True)                
            audio_file_name = audio_file.replace("wavs/", "")
            audio_path = os.path.normpath(
                os.path.join(
                    vat_audio_folder,
                    audio_file_name))

            if not os.path.exists(audio_path):
                missing_files.append(audio_file_name)
                debug_print(f"File not found: {audio_path}", "GENERAL", is_warning=True)                
                if vat_progress is not None:
                    vat_progress((index + 1, total_files),
                                 desc="Processing files")
                continue

            # Transcribe with OpenAI Whisper
            result = asr_model.transcribe(
                audio_path,
                language=vat_target_language,
                word_timestamps=True,
                verbose=None)

            # Get the full transcription from the result
            transcribed_text = result["text"].strip()
            debug_print(f"Transcribed text length: {len(transcribed_text)}", "VALIDATION", is_info=True)

            # Normalize and compare texts
            normalized_expected_text = normalize_text(expected_text)
            normalized_transcribed_text = normalize_text(transcribed_text)
            debug_print(f"Normalized expected text length: {len(normalized_expected_text)}", "VALIDATION", is_info=True)
            debug_print(f"Normalized transcribed text length: {len(normalized_transcribed_text)}", "VALIDATION", is_info=True)

            if normalized_transcribed_text != normalized_expected_text:
                debug_print("Mismatch found! Adding to mismatches list", "VALIDATION", is_info=True)
                mismatch_entry = {
                    "expected_text": row["text"],
                    "transcribed_text": transcribed_text,
                    "filename": audio_file_name,
                    "full_path": audio_path,
                    "row_index": row["row_index"],
                    "source_csv": row["source_csv"],
                }
                debug_print(f"Mismatch entry keys: {mismatch_entry.keys()}", "VALIDATION", is_info=True)
                mismatches.append(mismatch_entry)
                
            if vat_progress is not None:
                vat_progress((index + 1, total_files), desc="Processing files")

        debug_print(f"Total mismatches found: {len(mismatches)}", "GENERAL", is_info=True)
        if mismatches:
            debug_print("Sample mismatch entry:", "VALIDATION", is_info=True)
            debug_print(str(mismatches[0]), "VALIDATION", is_info=True)

        if missing_files:
            debug_print("Missing files:", "GENERAL", is_warning=True)
            for file_name in missing_files:
                debug_print(f"- {file_name}", "GENERAL", is_warning=True)

        if missing_files:
            debug_print("", "GENERAL")
            debug_print(
                "The following files are missing and should be removed from the CSV files:",
                "GENERAL")
            for file_name in missing_files:
                debug_print(f"- {file_name}", "GENERAL")
        return mismatches

    vat_progress = gr.Progress(track_tqdm=True)

    if (
        VALIDATE_TRAIN_METADATA_PATH
        and VALIDATE_EVAL_METADATA_PATH
        and VALIDATE_AUDIO_FOLDER
        and VALIDATE_WHISPER_MODEL
        and VALIDATE_TARGET_LANGUAGE
    ):
        mismatches = validate_audio_transcriptions(
            [VALIDATE_TRAIN_METADATA_PATH, VALIDATE_EVAL_METADATA_PATH],
            VALIDATE_AUDIO_FOLDER,
            VALIDATE_WHISPER_MODEL,
            VALIDATE_TARGET_LANGUAGE,
            vat_progress,
        )

        if not mismatches:
            debug_print("No transcription mismatches found!", "GENERAL", is_info=True)
            empty_df = pd.DataFrame(columns=["expected_text", "transcribed_text", "filename", 
                                           "full_path", "row_index", "source_csv"])
            display_df = pd.DataFrame(columns=["expected_text", "transcribed_text", "filename"])
            display_df.loc[0] = ["No bad transcriptions", "No bad transcriptions", "N/A"]
            return empty_df, display_df, "No transcription mismatches found - all transcriptions match!"

        # Convert mismatches list to DataFrame
        df = pd.DataFrame(mismatches)

        # Ensure all fields are single values, not series
        for col in df.columns:
            if isinstance(df[col].iloc[0], pd.Series):
                df[col] = df[col].apply(
                    lambda x: x.iloc[0] if isinstance(
                        x, pd.Series) else x)

        # Clean all text columns
        df["expected_text"] = df["expected_text"].astype(
            str).apply(lambda x: x.strip())
        df["transcribed_text"] = df["transcribed_text"].astype(
            str).apply(lambda x: x.strip())
        df["full_path"] = df["full_path"].astype(
            str).apply(lambda x: x.strip())

        # Create display version with only visible columns
        display_df = df[["expected_text",
                         "transcribed_text", "filename"]].copy()

        return df, display_df, ""
    else:
        empty_df = pd.DataFrame(
            columns=[
                "Expected Text",
                "Transcribed Text",
                "Filename"])
        return empty_df, empty_df, "Please generate your dataset first"


def save_correction_to_csv(csv_path, row_index, new_text):
    """Update the relevant parts of the CSV files"""
    try:
        # Ensure we have single values, not Series
        if isinstance(csv_path, pd.Series):
            csv_path = str(csv_path.iloc[0])
        else:
            csv_path = str(csv_path)

        if isinstance(row_index, pd.Series):
            row_index = int(row_index.iloc[0])
        else:
            row_index = int(row_index)

        if isinstance(new_text, pd.Series):
            new_text = str(new_text.iloc[0])
        else:
            new_text = str(new_text)

        # Read the CSV file
        df = pd.read_csv(csv_path, sep="|")

        # Update the text
        df.loc[row_index, "text"] = new_text

        # Save back to CSV
        df.to_csv(csv_path, sep="|", index=False)

        # Verify the save
        df_check = pd.read_csv(csv_path, sep="|")
        if not df_check.loc[row_index, "text"] == new_text:
            debug_print(
                "Save verification failed. Text mismatch.",
                "GENERAL",
                is_error=True)
            return "Error: Save verification failed"

        return f"Successfully updated transcription in {os.path.basename(csv_path)}"

    except Exception as e:
        debug_print(
            f"Error saving correction: {str(e)}",
            "GENERAL",
            is_error=True)
        debug_print(f"CSV path: {csv_path}", "GENERAL", is_error=True)
        debug_print(f"Row index: {row_index}", "GENERAL", is_error=True)
        debug_print("Full error traceback:", "GENERAL", is_error=True)
        traceback.print_exc()
        return f"Error updating CSV: {str(e)}"

def save_audio_and_correction(
        choice,
        manual_text,
        audio_data,
        df,
        current_idx):
    """Handle both audio and transcription saves"""
    if current_idx is None:
        return {
            mismatch_table: df[["expected_text", "transcribed_text", "filename"]],
            current_expected: "",
            save_status: "Please select a row first",
            audio_player: None,
        }

    try:
        # Handle current_idx coming as a list from Gradio
        if isinstance(current_idx, list):
            current_idx = current_idx[0]

        row = df.iloc[int(current_idx)]
        audio_path = str(row["full_path"]).strip()

        debug_print(f"Processing file: {audio_path}", "DATA_PROCESS")
        save_status_msg = []

        # Handle audio save if audio was edited
        if audio_data is not None and isinstance(
                audio_data, tuple) and len(audio_data) == 2:
            try:
                sr, audio = audio_data
                debug_print(
                    f"Saving edited audio: {sr}Hz, length: {len(audio)}",
                    "DATA_PROCESS")
                audio_tensor = torch.tensor(audio).unsqueeze(0)
                torchaudio.save(audio_path, audio_tensor, sr)
                save_status_msg.append("Audio saved successfully")
                debug_print(
                    f"Saved edited audio to {audio_path}",
                    "DATA_PROCESS")
            except Exception as e:
                save_status_msg.append(f"Error saving audio: {str(e)}")
                debug_print(
                    f"Error saving audio: {str(e)}",
                    "DATA_PROCESS",
                    is_error=True)

        # Handle text correction
        if choice == "Use Original":
            new_text = str(row["expected_text"])
        elif choice == "Use Whisper":
            new_text = str(row["transcribed_text"])
        elif choice == "Edit Manually":
            new_text = str(manual_text)

        # Save text correction to CSV
        result = save_correction_to_csv(
            str(row["source_csv"]), int(row["row_index"]), new_text)
        save_status_msg.append(result)

        # Update both text and expected_text in DataFrame
        if "Successfully" in result:
            df.loc[current_idx, "text"] = new_text
            df.loc[current_idx, "expected_text"] = new_text
            debug_print(
                f"Updated DataFrame with new text: {new_text}",
                "DATA_PROCESS")

        # Create updated display DataFrame
        display_df = df[["expected_text",
                         "transcribed_text", "filename"]].copy()

        return {
            mismatch_table: display_df,
            current_expected: new_text if "Successfully" in result else row["expected_text"],
            save_status: " | ".join(save_status_msg),
            audio_player: audio_path,
        }

    except Exception as e:
        error_msg = f"Error saving correction: {str(e)}"
        debug_print(error_msg, "DATA_PROCESS", is_error=True)
        traceback.print_exc()
        return {
            mismatch_table: df[["expected_text", "transcribed_text", "filename"]],
            current_expected: row["expected_text"] if "row" in locals() else "",
            save_status: error_msg,
            audio_player: None,
        }


#########################################
#### STEP 2 Model Training Functions ####
#########################################
def basemodel_or_finetunedmodel_choice(value):
    """update basemodel"""
    global basemodel_or_finetunedmodel
    if value == "Base Model":
        basemodel_or_finetunedmodel = True
    elif value == "Existing finetuned model":
        basemodel_or_finetunedmodel = False


def train_gpt(
        language,
        num_epochs,
        batch_size,
        grad_acumm,
        train_csv,
        eval_csv,
        learning_rate,
        model_to_train,
        continue_run,
        disable_shared_memory,
        learning_rate_scheduler,
        optimizer,
        num_workers,
        warm_up,
        max_audio_length=255995,
        progress=gr.Progress()):
    if "No Models Available" in model_to_train:
        print("[FINETUNE] Error: No XTTS model selected for training.")
        print("[FINETUNE] Please download a model using AllTalk's main interface > TTS Engine Settings > XTTS > Model/Voices Download")
        return

    #  Logging parameters
    project_run_name = "XTTS_FT"
    project_name = "XTTS_trainer"
    dashboard_logger = "tensorboard"
    logger_uri = None
    model_path = this_dir / "models" / "xtts" / model_to_train

    # Set here the path that the checkpoints will be saved. Default:
    # ./training/
    project_path = os.path.join(out_path, "training")
    debug_print(
        "Starting Step 2 - Fine-tuning the XTTS Model",
        level="GENERAL",
        is_info=True)
    debug_print("Configuration Summary:", level="GENERAL", is_info=True)
    debug_print(f"- Language: {language}", level="GENERAL", is_info=True)
    debug_print(
        f"- Training Epochs: {num_epochs}",
        level="GENERAL",
        is_info=True)
    debug_print(f"- Batch Size: {batch_size}", level="GENERAL", is_info=True)
    debug_print(
        f"- Gradient Accumulation Steps: {grad_acumm}",
        level="GENERAL",
        is_info=True)
    debug_print("File Paths:", level="GENERAL", is_info=True)
    debug_print(f"- Training Data: {train_csv}", level="GENERAL", is_info=True)
    debug_print(
        f"- Evaluation Data: {eval_csv}",
        level="GENERAL",
        is_info=True)
    debug_print(f"- Base Model: {model_path}", level="GENERAL", is_info=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Get the current device ID
        gpu_device_id = torch.cuda.current_device()
        gpu_available_mem_gb = (torch.cuda.get_device_properties(
            gpu_device_id).total_memory - torch.cuda.memory_allocated(gpu_device_id)) / (1024 ** 3)
        debug_print("GPU Memory Status:", level="GPU_MEMORY", is_info=True)
        debug_print(
            f"- Available VRAM: {gpu_available_mem_gb:.2f} GB",
            level="GPU_MEMORY",
            is_info=True)
        if gpu_available_mem_gb < 12:
            debug_print(
                "*** IMPORTANT MEMORY CONSIDERATION ***",
                level="GPU_MEMORY",
                is_warning=True)
            debug_print(
                "Your available VRAM is below the recommended 12GB threshold.",
                level="GPU_MEMORY",
                is_warning=True)
            # Empty line for formatting
            debug_print("", level="GPU_MEMORY", is_warning=True)
            debug_print(
                "System-Specific Considerations:",
                level="GPU_MEMORY",
                is_warning=True)
            debug_print(
                "- Windows: Will utilize system RAM as extended VRAM",
                level="GPU_MEMORY",
                is_warning=True)
            debug_print(
                "  * Ensure sufficient system RAM is available",
                level="GPU_MEMORY",
                is_warning=True)
            debug_print(
                "  * Recommended minimum: 24GB system RAM",
                level="GPU_MEMORY",
                is_warning=True)
            debug_print(
                "- Linux: Limited to physical VRAM only",
                level="GPU_MEMORY",
                is_warning=True)
            debug_print(
                "  * Training may fail with insufficient VRAM",
                level="GPU_MEMORY",
                is_warning=True)
            debug_print(
                "  * Consider reducing batch size or using gradient accumulation",
                level="GPU_MEMORY",
                is_warning=True)
            # Empty line for formatting
            debug_print("", level="GPU_MEMORY", is_warning=True)
            debug_print(
                "For detailed memory management strategies and optimization tips:",
                level="GPU_MEMORY",
                is_warning=True)
            debug_print(
                "1. Refer to the 'Memory Management' section in the Training Guide",
                level="GPU_MEMORY",
                is_warning=True)
            debug_print(
                "2. Review the Pre-flight Check tab for system requirements",
                level="GPU_MEMORY",
                is_warning=True)

    # Create the directory
    os.makedirs(project_path, exist_ok=True)

    # Training Parameters
    # For multi-GPU training, set to False
    param_optimizer_wd_only_on_weights = True
    # If True, it will start with evaluation
    param_start_with_eval = False
    param_batch_size = batch_size           # Set the batch size here
    # Set the gradient accumulation steps here
    param_grad_acumm_steps = grad_acumm

    # Define here the dataset that you want to use for the fine-tuning on.
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=train_csv,
        meta_file_val=eval_csv,
        language=language,
    )

    # Dataset Configuration list
    dataset_config_list = [config_dataset]
    dataset_tokenizer_file = str(model_path / "vocab.json")
    dataset_xtts_checkpoint = str(model_path / "model.pth")
    dataset_xtts_config_file = str(model_path / "config.json")
    dataset_dvae_checkpoint = model_path / "dvae.pth"
    dataset_mel_norm_file = model_path / "mel_stats.pth"
    dataset_speakers_file = model_path / "speakers_xtts.pth"

    training_assets = None
    if (out_path / "bpe_tokenizer-vocab.json").exists():
        debug_print(
            "Using custom BPE tokenizer",
            level="DATA_PROCESS",
            is_info=True)
        training_assets = {
            'Tokenizer': str(out_path / "bpe_tokenizer-vocab.json")
        }

    continue_path = None
    if continue_run:
        folders = glob.glob(os.path.join(project_path, '*/'))
        if folders:
            last_run = max(folders, key=os.path.getmtime)
            if last_run:
                checkpoints = glob.glob(
                    os.path.join(
                        last_run,
                        "best_model_*.pth"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
                    if latest_checkpoint:
                        dataset_xtts_checkpoint = None
                        continue_path = last_run
                        print(
                            f"[FINETUNE] Continuing previous fine tuning {latest_checkpoint}")

    # Copy the supporting files
    destination_dir = out_path / "chkptandnorm"
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        dataset_dvae_checkpoint,
        destination_dir /
        dataset_dvae_checkpoint.name)
    shutil.copy2(
        dataset_mel_norm_file,
        destination_dir /
        dataset_mel_norm_file.name)
    shutil.copy2(
        dataset_speakers_file,
        destination_dir /
        dataset_speakers_file.name)

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=dataset_mel_norm_file,
        dvae_checkpoint=dataset_dvae_checkpoint,
        xtts_checkpoint=dataset_xtts_checkpoint,
        # checkpoint path of the model that you want to fine-tune
        tokenizer_file=dataset_tokenizer_file,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000)

    # Resolve Japanese threading issue
    number_of_workers = int(num_workers)
    if language == "ja":
        number_of_workers = 0

    lr_scheduler = None
    lr_scheduler_params = {}

    if learning_rate_scheduler and learning_rate_scheduler != "None":
        lr_gamma_mapping = {
            1e-6: 0.9,
            5e-6: 0.8,
            1e-5: 0.3,
            5e-5: 0.3,
            1e-4: 0.3,
            5e-4: 0.1,
            1e-3: 0.1
        }
        lr_scheduler = learning_rate_scheduler
        if lr_scheduler == "StepLR":
            lr_scheduler_params = {
                'step_size': 30,
                'gamma': 0.1,
                'last_epoch': -1}
        elif lr_scheduler == "MultiStepLR":
            exponent = 3 - int(math.log2(num_epochs) / 2)
            base = 2
            num_milestones = min(num_epochs, int(math.pow(base, exponent)))
            milestone_interval = num_epochs // (num_milestones + 1)
            milestones = [milestone_interval *
                          (i + 1) for i in range(num_milestones)]
            lr_scheduler_params = {
                'milestones': milestones,
                'gamma': lr_gamma_mapping[learning_rate],
                'last_epoch': -1}
        elif lr_scheduler == "ExponentialLR":
            lr_scheduler_params = {'gamma': 0.5, 'last_epoch': -1}
        elif lr_scheduler == "CosineAnnealingLR":
            lr_scheduler_params = {
                'T_max': num_epochs,
                'eta_min': 1e-6,
                'last_epoch': -1}
        elif lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler_params = {
                'mode': 'min',
                'factor': 0.8,
                'patience': 1,
                'threshold': 0.0001,
                'threshold_mode': 'rel',
                'cooldown': 0,
                'min_lr': 1e-8,
                'eps': 1e-08,
            }
        elif lr_scheduler == "CyclicLR":
            lr_scheduler_params = {
                'base_lr': learning_rate,
                'max_lr': 0.1,
                'step_size_up': 2000,
                'step_size_down': None,
                'mode': 'triangular',
                'gamma': 1.0,
                'scale_fn': None,
                'scale_mode': 'cycle',
                'cycle_momentum': True,
                'base_momentum': 0.8,
                'max_momentum': 0.9,
                'last_epoch': -1}
        elif lr_scheduler == "OneCycleLR":
            lr_scheduler_params = {
                'max_lr': learning_rate,
                'total_steps': None,
                'epochs_up': None,
                'steps_per_epoch': None,
                'anneal_strategy': 'cos',
                'cycle_momentum': True,
                'base_momentum': 0.85,
                'max_momentum': 0.95,
                'div_factor': 25.0,
                'final_div_factor': 10000.0,
                'last_epoch': -1}
        elif lr_scheduler == "CosineAnnealingWarmRestarts":
            if num_epochs < 4:
                error_message = "For Cosine Annealing Warm Restarts, epochs must be at least 4. Please set a minimum of 4 epochs."
                progress(1.0, desc=f"Error: {error_message}")
                raise ValueError(error_message)
            # Set 4 learning rate restarts
            lr_scheduler_params = {
                'T_0': int(
                    num_epochs / 4),
                'T_mult': 1,
                'eta_min': 1e-6,
                'last_epoch': -1}

    optimizer_params = None

    OPTIMIZER_PARAMS = {
        "AdamW": {
            "betas": [0.9, 0.96],
            "eps": 1e-8,
            "weight_decay": 1e-2
        },
        "RMSprop": {
            "alpha": 0.99,
            "eps": 1e-8,
            "weight_decay": 1e-4
        },
        "SGD": {
            "momentum": 0.9,
            "weight_decay": 1e-4
        },
        "Adam": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4
        },
        "Adagrad": {
            "lr_decay": 0,
            "weight_decay": 1e-4,
            "eps": 1e-10
        },
        "RAdam": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-2
        },
        "stepwisegraduallr": {},
        "noamlr": {}
    }

    optimizer_params = OPTIMIZER_PARAMS.get(optimizer, {})

    print(
        f"[FINETUNE] [INFO] Learning Scheduler {lr_scheduler}, params {lr_scheduler_params}")

    # training parameters config
    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=project_path,
        model_args=model_args,
        run_name=project_run_name,
        project_name=project_name,
        run_description="GPT XTTS training",
        dashboard_logger=dashboard_logger,
        logger_uri=logger_uri,
        audio=audio_config,
        batch_size=param_batch_size,
        batch_group_size=48,
        eval_batch_size=param_batch_size,
        num_loader_workers=number_of_workers,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with
        # modifications to not apply WD to non-weight parameters.
        optimizer=optimizer,
        optimizer_wd_only_on_weights=param_optimizer_wd_only_on_weights,
        optimizer_params=optimizer_params,
        lr=learning_rate,  # learning rate
        lr_scheduler=lr_scheduler,
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params=lr_scheduler_params,
        test_sentences=[],
    )
    progress(0, desc="Model is currently training. See console for more information")
    # init the model from config
    model = GPTTrainer.init_from_config(config)
    # load training samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config_list,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    global c_logger
    c_logger = MetricsLogger()

    # init the trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            continue_path=continue_path,
            start_with_eval=param_start_with_eval,
            grad_accum_steps=param_grad_acumm_steps,
        ),
        config,
        output_path=project_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets=training_assets,
        c_logger=c_logger,
        warmup=warm_up,
    )

    c_logger.update_model_path(project_path)

    if disable_shared_memory:
        # Limit training to GPU memory instead of shared memory
        print("[FINETUNE] [INFO] Limiting GPU memory to 95% to prevent spillover")
        torch.cuda.set_per_process_memory_fraction(0.95)

    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx = samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples, config, model_args, config_dataset
    gc.collect()
    train_samples = None
    eval_samples = None
    config_dataset = None
    trainer = None
    model = None
    model_args = None
    try:
        return dataset_xtts_config_file, dataset_xtts_checkpoint, dataset_tokenizer_file, trainer_out_path, speaker_ref
    except Exception as e:
        print(f"Error returning values: {e}")
        return "Error", "Error", "Error", "Error", "Error"

##########################
#### STEP 3 AND OTHER ####
##########################


def clear_gpu_cache():
    """clear the GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_a_speaker_file(folder_path):
    """locate a speakers_xtts.pth file"""
    search_path = folder_path / "*" / "speakers_xtts.pth"
    files = glob.glob(str(search_path), recursive=True)
    latest_file = max(files, key=os.path.getctime, default=None)
    return latest_file


def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    """Load in the XTTS model for testing"""
    global XTTS_MODEL
    clear_gpu_cache()
    if not all([xtts_checkpoint, xtts_config, xtts_vocab]):
        return "No Models were selected. Click the Refresh Dropdowns button and try again."

    xtts_speakers_pth = find_a_speaker_file(this_dir / "models")
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    debug_print("Starting Step 3 - Loading XTTS model!", level="GENERAL", is_info=True)
    
    XTTS_MODEL.load_checkpoint(
        config,
        checkpoint_path=xtts_checkpoint,
        vocab_path=xtts_vocab,
        use_deepspeed=False,
        speaker_file_path=xtts_speakers_pth,
    )
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    debug_print("Model Loaded!", level="GENERAL", is_info=True)
    return "Model Loaded!"


def run_tts(lang, tts_text, speaker_audio_file):
    """Generate the TTS for testing"""
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None
        
    speaker_audio_file = str(speaker_audio_file)
    wavs_files = [speaker_audio_file]
    
    if os.path.isdir(speaker_audio_file):
        wavs_files = glob.glob(os.path.join(speaker_audio_file, "*.wav"))
        speaker_audio_file = wavs_files[0]

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=wavs_files,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )
    out = XTTS_MODEL.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=XTTS_MODEL.config.temperature,  # Add custom parameters here
        length_penalty=XTTS_MODEL.config.length_penalty,
        repetition_penalty=XTTS_MODEL.config.repetition_penalty,
        top_k=XTTS_MODEL.config.top_k,
        top_p=XTTS_MODEL.config.top_p,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file


def get_available_voices(min_duration_seconds=6, speaker_name=None):
    """Get available voice files based on minimum duration."""
    directory = this_dir / "finetune" / speaker_name if (speaker_name and speaker_name != "personsname") else out_path
    
    valid_files = []
    wav_files = Path(f"{directory}/wavs").glob("*.wav")

    for voice_file in wav_files:
        try:
            waveform, sample_rate = torchaudio.load(str(voice_file))
            duration = waveform.size(1) / sample_rate
            
            if duration >= float(min_duration_seconds):
                valid_files.append(str(voice_file))
        except Exception as e:
            debug_print(f"Error processing {voice_file}: {str(e)}", level="GENERAL", is_error=True)

    return sorted(valid_files)


def find_best_models(directory, speaker_name=None):
    """Find the best_model.pth file for the correct project name and last training run"""
    if speaker_name and speaker_name != "personsname":
        directory = this_dir / "finetune" / speaker_name

    # Look in both the base directory and the training subdirectory
    model_files = []

    # Check the training directory first
    training_dir = directory / "training"
    if training_dir.exists():
        # Look in all subdirectories of training
        for subdir in training_dir.glob("*"):
            if subdir.is_dir():
                model_path = subdir / "best_model.pth"
                if model_path.exists():
                    model_files.append(str(model_path))

    # Also check the base directory
    for model_path in directory.glob("**/best_model.pth"):
        if "training" not in str(model_path):  # Avoid duplicates
            model_files.append(str(model_path))

    return sorted(model_files)


def find_jsons(directory, filename, speaker_name=None):
    """Locate JSON's for the correct project name and last training run"""
    if speaker_name and speaker_name != "personsname":
        directory = this_dir / "finetune" / speaker_name
    return [str(file) for file in Path(directory).rglob(filename)]


# XTTS checkpoint files (best_model.pth)
xtts_checkpoint_files = find_best_models(out_path)
# XTTS config files (config.json)
xtts_config_files = find_jsons(out_path, "config.json")
# XTTS vocab files (vocab.json)
xtts_vocab_files = find_jsons(out_path, "vocab.json")

##########################
#### STEP 4 AND OTHER ####
##########################


def compact_custom_model(
        xtts_checkpoint_copy,
        folder_path,
        overwrite_existing):
    """Compact and move all the files for the correct project name and last training run"""
    this_dir = Path(__file__).parent.resolve()
    # Early validation checks
    if not xtts_checkpoint_copy:
        error_message = "No trained model was selected. Please click Refresh Dropdowns and try again."
        debug_print(error_message, level="GENERAL", is_error=True)
        return error_message

    target_dir = this_dir / "models" / "xtts" / folder_path    
    if overwrite_existing == "Do not overwrite existing files" and target_dir.exists():
        error_message = "The target folder already exists. Please change folder name or allow overwrites."
        debug_print(error_message, level="GENERAL", is_error=True)
        return error_message

    xtts_checkpoint_copy = Path(xtts_checkpoint_copy)
    # Get the source directory (either tmp-trn or custom named directory)
    source_dir = xtts_checkpoint_copy.parent.parent.parent  # Go up to the base directory
    debug_print(
        "=== File Copy Operations ===",
        level="DATA_PROCESS",
        is_info=True)
    debug_print(
        f"Source base directory: {source_dir}",
        level="DATA_PROCESS",
        is_info=True)
    debug_print(
        f"Target directory: {target_dir}",
        level="DATA_PROCESS",
        is_info=True)
    try:
        checkpoint = torch.load(
            xtts_checkpoint_copy,
            map_location=torch.device("cpu"))
    except Exception as e:
        debug_print(
            f"Error loading checkpoint: {e}",
            level="GENERAL",
            is_error=True)
        raise

    del checkpoint["optimizer"]
    target_dir.mkdir(parents=True, exist_ok=True)

    # Remove dvae-related keys from checkpoint
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    debug_print("Processing model.pth:", level="DATA_PROCESS", is_info=True)
    debug_print(
        f"  From: {xtts_checkpoint_copy}",
        level="DATA_PROCESS",
        is_info=True)
    debug_print(
        f"  To: {target_dir / 'model.pth'}",
        level="DATA_PROCESS",
        is_info=True)
    torch.save(checkpoint, target_dir / "model.pth")

    # Copy first set of files
    folder_path_new = xtts_checkpoint_copy.parent
    debug_print("Copying config files:", level="DATA_PROCESS", is_info=True)
    for file_name in ["vocab.json", "config.json"]:
        src_path = folder_path_new / file_name
        dest_path = target_dir / file_name
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            debug_print(f"  {file_name}:", level="DATA_PROCESS", is_info=True)
            debug_print(
                f"    From: {src_path}",
                level="DATA_PROCESS",
                is_info=True)
            debug_print(
                f"    To: {dest_path}",
                level="DATA_PROCESS",
                is_info=True)
        else:
            debug_print(
                f"Warning: {src_path} not found",
                level="DATA_PROCESS",
                is_warning=True)

    # Copy second set of files from chkptandnorm directory
    chkptandnorm_path = source_dir / "chkptandnorm"
    debug_print("Copying support files:", level="DATA_PROCESS", is_info=True)
    for file_name in ["speakers_xtts.pth", "mel_stats.pth", "dvae.pth"]:
        src_path = chkptandnorm_path / file_name
        dest_path = target_dir / file_name
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            debug_print(f"  {file_name}:", level="DATA_PROCESS", is_info=True)
            debug_print(
                f"    From: {src_path}",
                level="DATA_PROCESS",
                is_info=True)
            debug_print(
                f"    To: {dest_path}",
                level="DATA_PROCESS",
                is_info=True)
        else:
            debug_print(
                f"Warning: {src_path} not found",
                level="DATA_PROCESS",
                is_warning=True)

    # Create directories for different categories of WAV files
    target_wavs_dir = target_dir / "wavs"
    target_wavs_dir.mkdir(parents=True, exist_ok=True)

    too_short_dir = target_wavs_dir / "too_short"
    too_long_dir = target_wavs_dir / "too_long"
    suitable_dir = target_wavs_dir / "suitable"

    for dir_path in [too_short_dir, too_long_dir, suitable_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Process WAV files
    source_wavs_dir = source_dir / "wavs"
    debug_print("Processing WAV files:", level="DATA_PROCESS", is_info=True)
    debug_print(
        f"  From: {source_wavs_dir}",
        level="DATA_PROCESS",
        is_info=True)
    debug_print(f"  To: {target_wavs_dir}", level="DATA_PROCESS", is_info=True)

    file_stats = {"too_short": [], "too_long": [], "suitable": []}

    if not source_wavs_dir.exists():
        debug_print(
            f"Warning: Source WAV directory {source_wavs_dir} does not exist",
            level="DATA_PROCESS",
            is_warning=True)
        return f"Model files copied to '/models/xtts/{folder_path}/' but no WAV files were found to process"

    for file_path in source_wavs_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".wav":
            try:
                # Load audio file and get duration
                waveform, sample_rate = torchaudio.load(file_path)
                duration = waveform.size(
                    1) / sample_rate  # Duration in seconds

                # Determine category and target directory
                if duration < 6:
                    category = "too_short"
                    target_subdir = too_short_dir
                elif duration > 30:
                    category = "too_long"
                    target_subdir = too_long_dir
                else:
                    category = "suitable"
                    target_subdir = suitable_dir

                # Copy file to appropriate directory
                shutil.copy2(file_path, target_subdir / file_path.name)

                # Store file info
                file_stats[category].append(
                    {"name": file_path.name, "duration": round(duration, 2)}
                )

            except Exception as e:
                print(
                    f"[FINETUNE] Error processing {file_path.name}: {str(e)}")

    debug_print("WAV File Statistics:", level="DATA_PROCESS", is_info=True)
    debug_print(
        f"  Suitable files (6-30s): {len(file_stats['suitable'])}",
        level="DATA_PROCESS",
        is_info=True)
    debug_print(
        f"  Too short files (<6s): {len(file_stats['too_short'])}",
        level="DATA_PROCESS",
        is_info=True)
    debug_print(
        f"  Too long files (>30s): {len(file_stats['too_long'])}",
        level="DATA_PROCESS",
        is_info=True)
    debug_print(
        "=== File Copy Operations Complete ===",
        level="DATA_PROCESS",
        is_info=True)

    # Create report file
    report_content = FinetuneContent.report_content  # pylint: disable=no-member

    for category, files in file_stats.items():
        report_content += f"\n{category.replace('_', ' ').title()} files ({len(files)}):\n"
        for file_info in files:
            report_content += f"- {file_info['name']}: {file_info['duration']} seconds\n"

    report_content += FinetuneContent.report_content2  # pylint: disable=no-member

    with open(target_wavs_dir / "audio_report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    # Model & WAV processing log
    debug_print(
        f"Model & WAV samples processed and copied to '/models/xtts/{folder_path}/'",
        level="DATA_PROCESS",
        is_info=True)
    return f"Model & WAV samples processed and copied to '/models/xtts/{folder_path}/'"


def delete_training_data():
    """Deletes the specified project name folder"""
    # Define the folder to be deleted
    folder_to_delete = Path(out_path)

    # Check if the folder exists before deleting
    if not folder_to_delete.exists():
        debug_print(
            f"Project Name folder > {folder_to_delete} < does not exist.",
            level="GENERAL",
            is_warning=True)
        return "Specified Project Name folder could not be found."
        
    # Iterate over all files and subdirectories
    for item in folder_to_delete.iterdir():
        # Exclude trainer_0_log.txt from deletion
        if item.name != "trainer_0_log.txt":
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except PermissionError:
                debug_print(
                    f"PermissionError: Could not delete {item}. Skipping.",
                    level="GENERAL",
                    is_error=True)

    debug_print(
        f"Project Name folder > {folder_to_delete} < was deleted successfully.",
        level="GENERAL",
        is_info=True)
    return "Specified Project Name folder & tmp data was deleted successfully."


def clear_folder_contents(folder_path):
    """Deletes the contents of the supplied folder"""
    if not folder_path.exists() or not folder_path.is_dir():
        debug_print(
            f"Folder {folder_path} does not exist.",
            level="GENERAL",
            is_warning=True)
        return f"Folder '{folder_path}' does not exist."

    # List all files and subdirectories in the folder
    for item in os.listdir(folder_path):
        item_path = folder_path / item
        if item_path.is_file():
            # If it's a file, remove it
            os.remove(item_path)
        elif item_path.is_dir():
            # If it's a subdirectory, remove it recursively
            shutil.rmtree(item_path)

    debug_print(
        f"Contents of {folder_path} deleted successfully.",
        level="GENERAL",
        is_info=True)
    return f"Contents of '{folder_path}' deleted successfully."


def delete_voice_sample_contents():
    """Clears out the specfied folders"""
    # Define the folders to be cleared
    voice_samples_folder = this_dir / "finetune" / "put-voice-samples-in-here"
    gradio_temp_folder = this_dir / "finetune" / "gradio_temp"
    # Clear the contents of the gradio_temp folder
    clear_folder_contents(gradio_temp_folder)
    # Clear the contents of the voice samples folder
    voice_samples_message = clear_folder_contents(voice_samples_folder)
    return voice_samples_message

#######################
#### OTHER Generic ####
#######################


def cleanup_before_exit(_signum, _frame):
    """Handle cleanup operations before exiting the program.""" # pylint: disable=no-member
    debug_print(
        "Received interrupt signal. Cleaning up and exiting...",
        level="GENERAL",
        is_warning=True)
    # Perform cleanup operations here if necessary
    sys.exit(0)


def create_refresh_button(
        refresh_components,
        refresh_methods,
        elem_class,
        interactive=True):
    """Create a refresh button with specified components and methods."""
    def refresh(speaker_name, min_duration_seconds):
        updates = {}
        for component, method in zip(refresh_components, refresh_methods):
            # Pass both speaker_name and min_duration_seconds to the method
            args = (
                method(
                    speaker_name=speaker_name,
                    min_duration_seconds=min_duration_seconds) if callable(method) else method)
            if args and "choices" in args:
                args["value"] = args["choices"][-1] if args["choices"] else ""
            for k, v in args.items():
                setattr(component, k, v)
            updates[component] = gr.update(**(args or {}))
        return updates

    refresh_button = gr.Button(
        "Refresh Dropdowns", elem_classes=elem_class, interactive=interactive
    )
    refresh_button.click(
        fn=refresh,
        inputs=[speaker_name_input_testing, min_audio_length],
        outputs=refresh_components,
    )

    return refresh_button


def create_refresh_button_next(
        refresh_components,
        refresh_methods,
        elem_class,
        interactive=True):
    """Create a refresh button with specified components and methods."""
    def refresh_export(speaker_name):
        global out_path
        if speaker_name and speaker_name != "personsname":
            out_path = this_dir / "finetune" / speaker_name
        else:
            out_path = this_dir / "finetune" / "tmp-trn"

        updates = {}
        for component, method in zip(refresh_components, refresh_methods):
            args = method(speaker_name=speaker_name) if callable(
                method) else method
            if args and "choices" in args:
                args["value"] = args["choices"][-1] if args["choices"] else ""
            for k, v in args.items():
                setattr(component, k, v)
            updates[component] = gr.update(**(args or {}))
        return updates

    refresh_button = gr.Button(
        "Refresh Dropdowns", elem_classes=elem_class, interactive=interactive
    )
    refresh_button.click(
        fn=refresh_export,
        inputs=[speaker_name_input_export],
        outputs=refresh_components)
    return refresh_button


if __name__ == "__main__":
    # Register the signal handler
    signal.signal(signal.SIGINT, cleanup_before_exit)

    ################
    #### GRADIO ####
    ################

    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
        """
        Example runs:
        python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 10",
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default: 4",
        default=4,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 1",
        default=1,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11",
        default=11,
    )

    args = parser.parse_args()

    #####################
    #### GRADIO INFO ####
    #####################

    with gr.Blocks(theme=gr.themes.Base(), css=FinetuneContent.custom_css) as demo:
        with gr.Row():
            gr.Markdown("## XTTS Models Finetuning")
            gr.Markdown("")
            gr.Markdown("")
            dark_mode_btn = gr.Button(
                "Light/Dark Mode", variant="primary", size="sm")
            dark_mode_btn.click(
                None,
                None,
                None,
                js="""() => {
                if (document.querySelectorAll('.dark').length) {
                    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                    // localStorage.setItem('darkMode', 'disabled');
                } else {
                    document.querySelector('body').classList.add('dark');
                    // localStorage.setItem('darkMode', 'enabled');
                }
            }""",
                show_api=False,
            )
        with gr.Tab("🚀 Pre-flight Checklist"):
            create_pfc_interface()

        #######################
        #### GRADIO STEP 1 ####
        #######################
        with gr.Tab("📁 Step 1 - Generating the dataset"):
            with gr.Tab("Generate Dataset"):
                # Define directories
                this_dir = Path(__file__).parent.resolve()
                voice_samples_dir = this_dir / "finetune" / "put-voice-samples-in-here"
                training_data_dir = this_dir / "finetune" / "tmp-trn" / "wavs"
                metadata_files = [
                    this_dir / "finetune" / "tmp-trn" / "metadata_eval.csv",
                    this_dir / "finetune" / "tmp-trn" / "metadata_train.csv",
                    this_dir / "finetune" / "tmp-trn" / "lang.txt",
                ]

                # Ensure directories exist
                voice_samples_dir.mkdir(parents=True, exist_ok=True)
                training_data_dir.mkdir(parents=True, exist_ok=True)

                def upload_audio(files):
                    """Upload audio files to the voice samples directory."""
                    for file in files:
                        shutil.copy(file.name, voice_samples_dir)
                    return f"Uploaded {len(files)} files to {voice_samples_dir}"

                def delete_existing_audio():
                    """Delete all files in the voice samples directory."""
                    for file in voice_samples_dir.iterdir():
                        if file.is_file():
                            file.unlink()
                    return f"Deleted all files in {voice_samples_dir}"

                def delete_existing_training_data():
                    """Delete all files from the project name directory."""
                    if training_data_dir.exists():
                        for file in training_data_dir.iterdir():
                            if file.is_file():
                                file.unlink()
                    for metadata_file in metadata_files:
                        if metadata_file.exists():
                            metadata_file.unlink()
                    return f"Deleted all files in {training_data_dir} and related metadata files"

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_files_upload = gr.Files(
                            label="Upload Audio Files")
                    with gr.Column(scale=3):
                        with gr.Row():
                            with gr.Column(scale=1):
                                audio_upload_button = gr.Button(
                                    "Upload New Audio Samples")
                                delete_audio_button = gr.Button(
                                    "Delete Existing Audio Samples")
                                delete_dataset_button = gr.Button(
                                    "Delete Existing Training Dataset"
                                )
                            with gr.Column(scale=2):
                                gr.Markdown(
                                    """
                                You can manually copy your audio files to `/finetune/put-voice-samples-in-here/` or use the upload to the left and click "Upload New Audio Samples". Once you have uploaded files, you can start creating your dataset.

                                - If you wish to delete previously uploaded audio samples files then use 'Delete Existing Audio Samples'.
                                - If you wish to delete previously generated training datasets, please use 'Delete Existing Training Dataset'.
                                - If you wish to re-use your previously created training data, fill in the `Training Project Name` corectly and click 'Create Dataset'.
                                """
                                )
                        with gr.Row():
                            output_text = gr.Textbox(
                                label="Audio File Management Result", interactive=False)

                # Define actions for buttons
                audio_upload_button.click(
                    upload_audio,
                    inputs=audio_files_upload,
                    outputs=output_text)
                delete_audio_button.click(
                    delete_existing_audio, outputs=output_text)
                delete_dataset_button.click(
                    delete_existing_training_data, outputs=output_text)

                def update_language_options(model):
                    # English-only models
                    if model in [
                        "tiny.en",
                        "base.en",
                        "small.en",
                            "medium.en"]:
                        languages = ["en"]
                    else:
                        # Multilingual models
                        languages = [
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "pl",
                            "tr",
                            "ru",
                            "nl",
                            "cs",
                            "ar",
                            "zh",
                            "hu",
                            "ko",
                            "ja",
                        ]
                    return gr.Dropdown(choices=languages, value=languages[0])

                with gr.Row():
                    speaker_name_input = gr.Textbox(
                        label="Training Project Name",
                        value="personsname",
                        visible=True,
                        scale=2,
                    )
                    whisper_model = gr.Dropdown(
                        label="Whisper Model",
                        value="large-v3",
                        choices=[
                            ("tiny.en", "tiny.en"),
                            ("tiny", "tiny"),
                            ("base.en", "base.en"),
                            ("base", "base"),
                            ("small.en", "small.en"),
                            ("small", "small"),
                            ("medium.en", "medium.en"),
                            ("medium", "medium"),
                            ("large-v1", "large-v1"),
                            ("large-v2", "large-v2"),
                            ("large-v3", "large-v3"),
                            ("large", "large"),
                            ("large-v3-turbo", "large-v3-turbo"),
                            ("turbo", "turbo"),
                        ],
                        scale=2,
                    )
                    lang = gr.Dropdown(
                        label="Dataset Language",
                        value="en",
                        choices=[
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "pl",
                            "tr",
                            "ru",
                            "nl",
                            "cs",
                            "ar",
                            "zh",
                            "hu",
                            "ko",
                            "ja",
                        ],
                        scale=1,
                    )
                    max_sample_length = gr.Dropdown(
                        label="Max Audio Length (in seconds)",
                        value="30",
                        choices=[
                            "10",
                            "15",
                            "20",
                            "25",
                            "30",
                            "35",
                            "40",
                            "45",
                            "50",
                            "55",
                            "60",
                            "65",
                            "70",
                            "75",
                            "80",
                            "85",
                            "90",
                        ],
                        scale=2,
                    )
                    eval_split_number = gr.Number(
                        label="Evaluation Data Split",
                        value=15,  # Default value
                        minimum=5,  # Minimum value
                        maximum=95,  # Maximum value
                        step=1,  # Increment step
                        scale=1,
                    )
                    create_bpe_tokenizer = gr.Checkbox(
                        label="BPE Tokenizer", value=False, info="Custom Tokenizer for training")
                    use_vad = gr.Checkbox(
                        label="VAD",
                        value=True,
                        info="Enable Silero VAD for better speech detection",
                    )
                    precision = gr.Dropdown(
                        label="Model Precision", value="mixed", choices=[
                            ("Mixed", "mixed"), ("FP16", "float16"), ("FP32", "float32")], )

                with gr.Accordion("🔍 Dataset Creation Debug Settings", open=False):

                    def update_debug_levels(
                            gpu, model, data, validation, general, audio, segments, duplicates):
                        DebugLevels.GPU_MEMORY = gpu
                        DebugLevels.MODEL_OPS = model
                        DebugLevels.DATA_PROCESS = data
                        DebugLevels.VALIDATION = validation
                        DebugLevels.GENERAL = general
                        DebugLevels.AUDIO = audio
                        DebugLevels.SEGMENTS = segments
                        DebugLevels.DUPLICATES = duplicates
                        return "Debug settings updated"

                    def select_all_debug():
                        return {
                            debug_gpu: True,
                            debug_model: True,
                            debug_data: True,
                            debug_validation: True,
                            debug_general: True,
                            debug_audio: True,
                            debug_segments: True,
                            debug_duplicates: True,
                        }

                    def clear_all_debug():
                        return {
                            debug_gpu: False,
                            debug_model: False,
                            debug_data: False,
                            debug_validation: False,
                            debug_general: False,
                            debug_audio: False,
                            debug_segments: False,
                            debug_duplicates: False,
                        }

                    with gr.Row():
                        gr.Markdown(
                            """
                        Enable or disable different types of debug messages during dataset creation.
                        These settings will apply to the current session only.
                        """
                        )

                    with gr.Row():
                        with gr.Column(scale=1):
                            debug_gpu = gr.Checkbox(
                                label="GPU Memory",
                                value=DebugLevels.GPU_MEMORY,
                                info="GPU memory and CUDA related debugging",
                            )
                            debug_model = gr.Checkbox(
                                label="Model Operations",
                                value=DebugLevels.MODEL_OPS,
                                info="Model loading, transcription, cleanup operations",
                            )
                            debug_data = gr.Checkbox(
                                label="Data Processing",
                                value=DebugLevels.DATA_PROCESS,
                                info="Data processing, words, sentences",
                            )
                            debug_validation = gr.Checkbox(
                                label="Dataset Validation",
                                value=DebugLevels.VALIDATION,
                                info="Dataset Validation, amount, sentences, files",
                            )                            

                        with gr.Column(scale=1):
                            debug_general = gr.Checkbox(
                                label="General",
                                value=DebugLevels.GENERAL,
                                info="General flow, file operations, metadata",
                            )
                            debug_audio = gr.Checkbox(
                                label="Audio",
                                value=DebugLevels.AUDIO,
                                info="Audio processing statistics and info",
                            )
                            debug_segments = gr.Checkbox(
                                label="Segments",
                                value=DebugLevels.SEGMENTS,
                                info="Detailed segment information",
                            )
                            debug_duplicates = gr.Checkbox(
                                label="Duplicates",
                                value=DebugLevels.DUPLICATES,
                                info="Duplicate handling information",
                            )

                    with gr.Row():
                        debug_select_all = gr.Button("Select All")
                        debug_clear_all = gr.Button("Clear All")

                    # Debug update functions
                    for checkbox in [
                        debug_gpu,
                        debug_model,
                        debug_data,
                        debug_validation,
                        debug_general,
                        debug_audio,
                        debug_segments,
                        debug_duplicates,
                    ]:
                        checkbox.change(
                            fn=update_debug_levels,
                            inputs=[
                                debug_gpu,
                                debug_model,
                                debug_data,
                                debug_validation,
                                debug_general,
                                debug_audio,
                                debug_segments,
                                debug_duplicates,
                            ],
                            # Hidden status output
                            outputs=[gr.Textbox(visible=False)],
                        )

                    debug_select_all.click(
                        fn=select_all_debug,
                        inputs=[],
                        outputs=[
                            debug_gpu,
                            debug_model,
                            debug_data,
                            debug_validation,
                            debug_general,
                            debug_audio,
                            debug_segments,
                            debug_duplicates,
                        ],
                    )

                    debug_clear_all.click(
                        fn=clear_all_debug,
                        inputs=[],
                        outputs=[
                            debug_gpu,
                            debug_model,
                            debug_data,
                            debug_validation,
                            debug_general,
                            debug_audio,
                            debug_segments,
                            debug_duplicates,
                        ],
                    )

                progress_data = gr.Label(label="Progress:")
                logs = gr.Textbox(
                    label="Logs:",
                    interactive=False,
                    lines=10,
                )
                demo.load(read_logs, None, logs, every=1)
                # Update `lang` options when the `whisper_model` changes
                whisper_model.change(
                    fn=update_language_options,
                    inputs=whisper_model,
                    outputs=lang)
                prompt_compute_btn = gr.Button(value="Step 1 - Create dataset")

                def preprocess_dataset(
                    pd_language,
                    pd_whisper_model,
                    pd_max_sample_length,
                    pd_eval_split_number,
                    pd_speaker_name_input,
                    pd_create_bpe_tokenizer,
                    pd_use_vad,
                    pd_precision,
                    pd_progress=gr.Progress(),
                ):
                    """Preprocess the dataset by validating audio files, formatting data, and splitting into training and evaluation sets."""
                    clear_gpu_cache()

                    # Check for audio files in the specified folder
                    pd_test_for_audio_files = [
                        file for file in os.listdir(audio_folder) if any(
                            file.lower().endswith(ext) for ext in [
                                '.wav', '.mp3', '.flac'])]
                    if not pd_test_for_audio_files:
                        return (
                            "I cannot find any mp3, wav or flac files in the folder called 'put-voice-samples-in-here'",
                            "",
                            "",
                        )

                    try:
                        # Format audio list and split into training and
                        # evaluation datasets
                        pd_train_meta, pd_eval_meta, pd_audio_total_size = format_audio_list(
                            fal_target_language=pd_language,
                            fal_whisper_model=pd_whisper_model,
                            fal_max_sample_length=pd_max_sample_length,
                            fal_eval_split_number=pd_eval_split_number,
                            fal_speaker_name_input=pd_speaker_name_input,
                            fal_create_bpe_tokenizer=pd_create_bpe_tokenizer,
                            fal_gradio_progress=pd_progress,
                            fal_use_vad=pd_use_vad,
                            fal_precision=pd_precision,
                        )
                    except Exception:
                        traceback.print_exc()
                        pd_error = traceback.format_exc()
                        return (
                            f"The data processing was interrupted due to an error!! Please check the console to verify the full error message! \n Error summary: {pd_error}",
                            "",
                            "",
                        )

                    clear_gpu_cache()

                    # Check total audio size
                    if pd_audio_total_size < 120:
                        pd_message = (
                            "The total duration of the audio file or files you provided was less than 2 minutes in length. "
                            "Please add more audio samples.")
                        debug_print(
                            pd_message,
                            level="DATA_PROCESS",
                            is_warning=True)
                        return pd_message, "", ""

                    # Final GPU cleanup
                    get_gpu_memory()

                    debug_print(
                        "Dataset Generated. Either run Dataset Validation or move to Step 2",
                        level="DATA_PROCESS",
                        is_info=True,
                    )

                    # Return metadata and speaker name inputs for further
                    # processing in Gradio interface
                    return (
                        "Dataset Generated. Either run Dataset Validation or move to Step 2",
                        pd_train_meta,
                        pd_eval_meta,
                        pd_speaker_name_input,
                        pd_speaker_name_input,
                        pd_speaker_name_input,
                    )

            with gr.Tab("Dataset Generation Guide"):
                gr.Markdown(
                    """
                # Dataset Generation Guide
                Below you'll find comprehensive instructions and information about generating your dataset.
                Click each section to expand its contents.
                """
                )
                with gr.Accordion("🎯 Quick Start Guide", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP1_QUICKSTART)  # pylint: disable=no-member
                with gr.Accordion("📋 Detailed Instructions", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP1_DETAILED_INSTRUCTIONS)  # pylint: disable=no-member
                with gr.Accordion("🔧 Process Overview", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP1_PROCESS_OVERVIEW)  # pylint: disable=no-member
                with gr.Accordion("🔍 Whisper Model Selection", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP1_WHISPER_MODEL_SELECTION)  # pylint: disable=no-member
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP1_ADVANCED_SETTINGS)  # pylint: disable=no-member
                with gr.Accordion("🔍 Dataset Creation Debug Settings", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP1_DEBUG_SETTINGS)  # pylint: disable=no-member
                with gr.Accordion("❗ Troubleshooting", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP1_TROUBLESHOOTING)  # pylint: disable=no-member

        with gr.TabItem("📊 Dataset Validation"):
            with gr.Row():
                gr.Markdown("""# Audio Transcription Validation""")
            with gr.Row():
                with gr.Accordion("🎯 Audio Transcription Validation Help", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                FinetuneContent.DATASET_VALIDATION_1)  # pylint: disable=no-member
                        with gr.Column():
                            gr.Markdown(
                                FinetuneContent.DATASET_VALIDATION_2)  # pylint: disable=no-member

            with gr.Row():
                progress_box = gr.Textbox(label="Progress", interactive=False)
                load_button = gr.Button("Run Validation")

            with gr.Row():
                with gr.Column(scale=2):
                    # Store full DataFrame in state
                    state = gr.State()
                    # Display DataFrame shows only visible columns
                    mismatch_table = gr.DataFrame(
                        headers=["Original Text", "Whisper Text", "Filename"],
                        datatype=["str", "str", "str"],
                        interactive=False,
                        wrap=True,
                    )

                with gr.Column(scale=1):
                    audio_player = gr.Audio(
                        label="Audio Player (Edit enabled)", interactive=True)
                    current_expected = gr.Textbox(
                        label="Original Text", interactive=False)
                    current_transcribed = gr.Textbox(
                        label="Whisper Text", interactive=False)
                    text_choice = gr.Radio(
                        choices=[
                            "Use Original",
                            "Use Whisper",
                            "Edit Manually"],
                        label="Choose Transcription",
                        value="Use Original",
                    )
                    manual_edit = gr.Textbox(
                        label="Manual Edit", interactive=True, visible=False)
                    current_index = gr.Number(visible=False)
                    save_button = gr.Button("Save Audio and Correction")
                    save_status = gr.Textbox(
                        label="Save Status", interactive=False)

            def update_audio_player(evt: gr.SelectData, df):
                """Update audio player with selected file"""
                try:
                    selected_row = df.iloc[evt.index]
                    audio_path = selected_row["full_path"]

                    # If it's a Series, get the first value
                    if isinstance(audio_path, pd.Series):
                        audio_path = audio_path.iloc[0]

                    # Clean the path string
                    audio_path = str(audio_path).strip()

                    # Check if file exists
                    if not os.path.exists(audio_path):
                        debug_print(
                            f"Audio file not found: {audio_path}",
                            level="DATA_PROCESS",
                            is_warning=True)
                        return {
                            audio_player: None,
                            current_expected: "",
                            current_transcribed: "",
                            current_index: None,
                            save_status: "Error: Audio file not found",
                        }

                    # Get text values directly from the row
                    expected = (
                        str(selected_row["expected_text"]).strip()
                        if isinstance(selected_row["expected_text"], str)
                        else selected_row["expected_text"].iloc[0]
                    )
                    transcribed = (
                        str(selected_row["transcribed_text"]).strip()
                        if isinstance(selected_row["transcribed_text"], str)
                        else selected_row["transcribed_text"].iloc[0]
                    )

                    return {
                        audio_player: audio_path,
                        current_expected: expected,
                        current_transcribed: transcribed,
                        current_index: evt.index,
                        save_status: "Ready to save correction",
                    }

                except Exception as e:
                    debug_print(
                        f"Error in update_audio_player: {str(e)}",
                        level="GENERAL",
                        is_error=True)
                    debug_print(
                        f"Selected row data: {selected_row if 'selected_row' in locals() else 'Not available'}",
                        level="GENERAL",
                        is_info=True,
                    )
                    return {
                        audio_player: None,
                        current_expected: "",
                        current_transcribed: "",
                        current_index: None,
                        save_status: f"Error: {str(e)}",
                    }

            # Event handlers
            text_choice.change(
                lambda x: gr.update(
                    visible=x == "Edit Manually"),
                text_choice,
                manual_edit)

            mismatch_table.select(
                update_audio_player,
                [state],  # Use full DataFrame from state
                [audio_player,
                 current_expected,
                 current_transcribed,
                 current_index,
                 save_status],
            )

            save_button.click(
                save_audio_and_correction,
                inputs=[
                    text_choice,
                    manual_edit,
                    audio_player,
                    state,
                    current_index],
                outputs=[
                    mismatch_table,
                    current_expected,
                    save_status,
                    audio_player],
            )

            # Store both display and full DataFrame
            load_button.click(
                load_and_display_mismatches,
                # state gets full df, mismatch_table gets display_df
                outputs=[state, mismatch_table, progress_box],
            )

        #######################
        #### GRADIO STEP 2 ####
        #######################
        with gr.Tab("💻 Step 2 - Training"):
            with gr.Tab("Training the model"):
                with gr.Row():
                    speaker_name_input_training = gr.Textbox(
                        label="Project Name",
                        value="personsname",
                        visible=True,
                        scale=1,
                    )
                    with gr.Group():
                        continue_run = gr.Checkbox(
                            value=False,
                            label="Continue Previous Project",
                            scale=1,
                            visible=False,
                        )
                        # Continue Run has been set to invisible from the interface as its setting
                        # a 1000 epoch run when used. Have been unable to track down where this is
                        # coming from.
                        disable_shared_memory = gr.Checkbox(
                            value=False,
                            label="Disable Shared Memory Use",
                            scale=1,
                        )
                        warm_up = gr.Checkbox(
                            value=False,
                            label="Perform Warmup Learning",
                            scale=1,
                        )
                    train_csv = gr.Textbox(
                        label="Train CSV file path:",
                        scale=2,
                    )
                    eval_csv = gr.Textbox(
                        label="Eval CSV file path:",
                        scale=2,
                    )
                with gr.Row():
                    model_to_train_choices = list(available_models.keys())
                    model_to_train = gr.Dropdown(
                        choices=model_to_train_choices,
                        label="Select the Model to train",
                        value=model_to_train_choices[0] if model_to_train_choices else None,
                        scale=2,
                    )

                    learning_rates = gr.Dropdown(
                        value=5e-6,
                        label="Learning Rate",
                        choices=[
                            ("1e-6", 1e-6),
                            ("5e-6", 5e-6),
                            ("1e-5", 1e-5),
                            ("5e-5", 5e-5),
                            ("1e-4", 1e-4),
                            ("5e-4", 5e-4),
                            ("1e-3", 1e-3),
                        ],
                        type="value",
                        allow_custom_value=True,
                        scale=1,
                    )
                    learning_rate_scheduler = gr.Dropdown(
                        value="CosineAnnealingWarmRestarts",
                        label="Learning Rate Scheduler(s)",
                        choices=[
                            ("None", "None"),
                            ("Cosine Annealing", "CosineAnnealingLR"),
                            ("Cosine Annealing Warm Restarts",
                             "CosineAnnealingWarmRestarts"),
                            ("Cyclic", "CyclicLR"),
                            ("Exponential", "ExponentialLR"),
                            ("Multi Step", "MultiStepLR"),
                            ("Reduce on Plateau", "ReduceLROnPlateau"),
                            ("Step", "StepLR"),
                            # ("OneCycleLR", "OneCycleLR"),
                        ],
                        type="value",
                        allow_custom_value=False,
                        scale=2,
                    )
                    optimizer = gr.Dropdown(
                        value="AdamW",
                        label="Optimizer",
                        choices=[
                            ("AdamW", "AdamW"),
                            ("SGD with Momentum", "SGD"),
                            ("RMSprop", "RMSprop"),
                            ("Adagrad", "Adagrad"),
                            ("Adam", "Adam"),
                            ("Rectified Adam", "RAdam"),
                            ("Step Wise", "stepwisegraduallr"),
                            ("Noam", "noamlr"),
                        ],
                        type="value",
                        allow_custom_value=False,
                        scale=2,
                    )
                    num_workers = gr.Dropdown(
                        value="8",
                        label="Workers/Threads",
                        choices=[
                            "0",
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                            "7",
                            "8",
                            "9",
                            "10"],
                        allow_custom_value=False,
                        interactive=True,
                        scale=0,
                    )
                with gr.Row():
                    num_epochs = gr.Slider(
                        label="Number of epochs:",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=args.num_epochs,
                    )
                    batch_size = gr.Slider(
                        label="Batch size:",
                        minimum=2,
                        maximum=512,
                        step=1,
                        value=args.batch_size,
                    )
                    grad_acumm = gr.Slider(
                        label="Grad accumulation steps:",
                        minimum=1,
                        maximum=128,
                        step=1,
                        value=args.grad_acumm,
                    )
                    max_audio_length = gr.Slider(
                        label="Max permitted audio size in seconds:",
                        minimum=2,
                        maximum=20,
                        step=1,
                        value=args.max_audio_length,
                    )

                progress_train = gr.Label(label="Progress:")

                with gr.Row():
                    train_time = gr.Label(
                        "Estimated Total Training Time", show_label=False, scale=2)
                    train_btn = gr.Button(
                        value="Step 2 - Run the training", scale=1)

                with gr.Row():
                    model_data = gr.Image(
                        c_logger.plot_metrics(), show_label=False)

                logs_tts_train = gr.Textbox(
                    label="Logs:",
                    interactive=False,
                    lines=10,
                )
                demo.load(
                    load_metrics, None, [
                        model_data, train_time], every=1)
                demo.load(read_logs, None, logs_tts_train, every=1)

                def train_model(
                        language,
                        train_csv,
                        eval_csv,
                        learning_rates,
                        model_to_train,
                        num_epochs,
                        batch_size,
                        grad_acumm,
                        max_audio_length,
                        speaker_name_input_training,
                        continue_run,
                        disable_shared_memory,
                        learning_rate_scheduler,
                        optimizer,
                        num_workers,
                        warm_up,
                        progress=gr.Progress()):
                    """
                    Trains XTTS model with specified parameters and returns model artifacts.

                    Returns:
                        tuple: Status message, config path, vocab file, checkpoint path, speaker reference, speaker name
                    """
                    clear_gpu_cache()
                    global out_path
                    if speaker_name_input_training and speaker_name_input_training != 'personsname':
                        out_path = this_dir / "finetune" / speaker_name_input_training
                    else:
                        out_path = default_path

                    if not train_csv or not eval_csv:
                        if (out_path /
                            "metadata_eval.csv").exists() and (out_path /
                                                               "metadata_train.csv").exists():
                            train_csv = out_path / "metadata_train.csv"
                            eval_csv = out_path / "metadata_eval.csv"
                            debug_print(
                                "Using existing metadata and training csv.",
                                level="GENERAL",
                                is_info=True)
                        else:
                            return (
                                "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !",
                                "",
                                "",
                                "",
                                "",
                            )
                    try:
                        # convert seconds to waveform frames
                        max_audio_length = int(max_audio_length * 22050)
                        # Convert the learning rate value to a float
                        learning_rate = float(learning_rates)
                        progress(0, "Initializing training...")
                        config_path, return_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
                            language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, learning_rate, model_to_train, continue_run, disable_shared_memory, learning_rate_scheduler, optimizer, num_workers, warm_up, max_audio_length=max_audio_length, progress=gr.Progress())

                        # copy original files to avoid parameters changes
                        # issues
                        shutil.copy(config_path, exp_path)
                        shutil.copy(vocab_file, exp_path)
                        ft_xtts_checkpoint=return_xtts_checkpoint
                        ft_xtts_checkpoint = os.path.join(
                            exp_path, "best_model.pth")
                        debug_print(
                            "Model training done. Move to Step 3",
                            level="GENERAL",
                            is_info=True)
                        clear_gpu_cache()
                        return (
                            "Model training done. Move to Step 3",
                            config_path,
                            vocab_file,
                            ft_xtts_checkpoint,
                            speaker_wav,
                            speaker_name_input_training,
                        )

                    except ValueError as ve:
                        error_message = str(ve)
                        debug_print(
                            f"{error_message}",
                            level="GENERAL",
                            is_error=True)
                        return f"Training error: {error_message}", "", "", "", "", ""
                    except Exception as e:
                        # This will catch any other unexpected errors
                        error_message = f"An unexpected error occurred: {str(e)}"
                        debug_print(
                            f"{error_message}",
                            level="GENERAL",
                            is_error=True)
                        return f"Training error: {error_message}", "", "", "", "", ""

            with gr.Tab("Training Guide"):
                gr.Markdown(
                    """
                # Comprehensive Training Guide
                Below you'll find detailed explanations of all training parameters and processes.
                Each section contains both conceptual explanations and specific configuration guidance.
                """
                )
                with gr.Accordion("🎯 Quick Start Training Guide", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP2_QUICKSTART)  # pylint: disable=no-member
                with gr.Accordion("📁 Using Your Own Training Dataset", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP2_YOUR_OWN_DATASET)  # pylint: disable=no-member
                with gr.Accordion("📊 Training Metrics and Logs", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP2_TRAINING_METRICS)  # pylint: disable=no-member
                with gr.Accordion("💾 Memory Management", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP2_MEMORY_MANAGEMENT)  # pylint: disable=no-member
                with gr.Accordion("⚙️ Batch Size & Gradient Accumulation", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP2_BATCH_SIZE)  # pylint: disable=no-member
                with gr.Accordion("📊 Learning Rate & Schedulers", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP2_LEARNING_RATE)  # pylint: disable=no-member
                with gr.Accordion("🔧 Optimizers", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP2_OPTIMIZERS)  # pylint: disable=no-member
                with gr.Accordion("🔄 Training Epochs", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP2_EPOCHS)  # pylint: disable=no-member
                with gr.Accordion("📈 Max Audio Length", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP2_AUDIO_LENGTH)  # pylint: disable=no-member

        #######################
        #### GRADIO STEP 3 ####
        #######################
        with gr.Tab("✅ Step 3 - Testing"):
            with gr.Tab("Testing"):
                with gr.Row():
                    with gr.Column() as col1:
                        xtts_checkpoint = gr.Dropdown(
                            [str(file) for file in xtts_checkpoint_files],
                            label="XTTS checkpoint path (best_model.pth):",
                            value="",
                            allow_custom_value=True,
                        )

                        xtts_config = gr.Dropdown(
                            [str(file) for file in xtts_config_files],
                            label="XTTS config path (config.json):",
                            value="",
                            allow_custom_value=True,
                        )

                        xtts_vocab = gr.Dropdown(
                            [str(file) for file in xtts_vocab_files],
                            label="XTTS vocab path (vocab.json):",
                            value="",
                            allow_custom_value=True,
                        )
                        progress_load = gr.Label(label="Progress:")
                        load_btn = gr.Button(
                            value="Step 3 - Load Fine-tuned XTTS model")

                    with gr.Column() as col2:
                        with gr.Row():
                            # Gather the voice files
                            available_speaker_audios = get_available_voices()

                            # Create Dropdown for speaker reference audio
                            speaker_reference_audio = gr.Dropdown(
                                available_speaker_audios,
                                label="Speaker reference audio (Press Refresh Dropdowns):",
                                value="",  # Set the default value if needed
                                allow_custom_value=True,  # Allow custom values
                                scale=2,
                            )
                            min_audio_length = gr.Dropdown(
                                label="Min Audio Length (seconds)",
                                value="6",
                                choices=["3", "4", "5", "6", "7", "8", "9", "10"],
                                type="value",
                                allow_custom_value=False,
                                scale=1,
                            )
                        with gr.Row():
                            speaker_name_input_testing = gr.Textbox(
                                label="Project Name",
                                value="personsname",
                                visible=True,
                                scale=1,
                            )
                            tts_language = gr.Dropdown(
                                label="Language",
                                value="en",
                                choices=[
                                    "en",
                                    "es",
                                    "fr",
                                    "de",
                                    "it",
                                    "pt",
                                    "pl",
                                    "tr",
                                    "ru",
                                    "nl",
                                    "cs",
                                    "ar",
                                    "zh",
                                    "hu",
                                    "ko",
                                    "ja",
                                ],
                            )
                            # Create refresh button
                            refresh_button = create_refresh_button(
                                [
                                    xtts_checkpoint,
                                    xtts_config,
                                    xtts_vocab,
                                    speaker_reference_audio,
                                    speaker_name_input_testing,
                                ],
                                [
                                    lambda speaker_name, min_duration_seconds: {
                                        "choices": find_best_models(
                                            out_path, speaker_name=speaker_name
                                        ),
                                        "value": "",
                                    },
                                    lambda speaker_name, min_duration_seconds: {
                                        "choices": find_jsons(
                                            out_path, "config.json", speaker_name=speaker_name
                                        ),
                                        "value": "",
                                    },
                                    lambda speaker_name, min_duration_seconds: {
                                        "choices": find_jsons(
                                            out_path, "vocab.json", speaker_name=speaker_name
                                        ),
                                        "value": "",
                                    },
                                    lambda speaker_name, min_duration_seconds: {
                                        "choices": get_available_voices(
                                            min_duration_seconds=int(min_duration_seconds),
                                            speaker_name=speaker_name,
                                        ),
                                        "value": "",
                                    },
                                ],
                                elem_class="refresh-button-class",
                            )
                        tts_text = gr.Textbox(
                            label="Input Text:",
                            value="I've just fine tuned a text to speech language model and this is how it sounds. If it doesn't sound right, I will try a different Speaker Reference Audio file.",
                            lines=5,
                        )
                        tts_btn = gr.Button(
                            value="Step 4 - Inference (Generate TTS)")

                with gr.Row():
                    progress_gen = gr.Label(label="Progress:")
                with gr.Row():
                    tts_output_audio = gr.Audio(label="TTS Generated Speech.")
                    reference_audio = gr.Audio(
                        label="Speaker Reference Audio Sample.")

            with gr.Tab("Testing Guide"):
                gr.Markdown(
                    """
                # Testing Your Fine-tuned Model
                This section guides you through evaluating the quality of your fine-tuned model using various reference audios. Testing ensures your model produces high-quality TTS that matches your target speaker characteristics. Follow the steps below for effective testing and validation.
                """
                )
                with gr.Accordion("🎯 Testing Overview", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP3_TESTING_OVERVIEW)  # pylint: disable=no-member
                with gr.Accordion("⚠️ Important Notes", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP3_IMPORTANT)  # pylint: disable=no-member
                with gr.Accordion("📝 Testing Instructions", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP3_INSTRUCTIONS)  # pylint: disable=no-member
                with gr.Accordion("🔍 What the Testing Step Does", open=False):
                    gr.Markdown(
                        FinetuneContent.STEP3_WHAT_IT_DOES)  # pylint: disable=no-member

        with gr.Tab("📦 Model Export"):
            with gr.Accordion("🎯 Export Overview", open=False):
                gr.Markdown(
                    FinetuneContent.EXPORT_OVERVIEW)  # pylint: disable=no-member
            with gr.Accordion("📊 Voice Sample Organization", open=False):
                gr.Markdown(
                    FinetuneContent.EXPORT_VOICE_SAMPLES)  # pylint: disable=no-member
            with gr.Accordion("🔄 Export Options", open=False):
                gr.Markdown(
                    FinetuneContent.EXPORT_OPTIONS)  # pylint: disable=no-member
            with gr.Accordion("💾 Storage Management", open=False):
                gr.Markdown(
                    FinetuneContent.EXPORT_STORAGE)  # pylint: disable=no-member

            final_progress_data = gr.Label(label="Progress:")
            with gr.Row():
                xtts_checkpoint_copy = gr.Dropdown(
                    [str(file) for file in xtts_checkpoint_files],
                    label="XTTS checkpoint path (Click the refresh button to populate):",
                    value="",
                    allow_custom_value=True,
                    scale=2,
                )
                speaker_name_input_export = gr.Textbox(
                    label="Project Name (Refresh Dropdowns on change)",
                    value="personsname",
                    visible=True,
                    scale=1,
                )
                # Create refresh button
                refresh_button = create_refresh_button_next(
                    [xtts_checkpoint_copy, speaker_name_input_export],
                    [
                        lambda speaker_name: {
                            "choices": find_best_models(out_path, speaker_name),
                            "value": "",
                        },
                    ],
                    elem_class="refresh-button-class",
                )
            with gr.Row():
                overwrite_existing = gr.Dropdown(
                    value="Do not overwrite existing files",
                    choices=[
                        "Overwrite existing files",
                        "Do not overwrite existing files"],
                    label="File Overwrite Options",
                )
                folder_path = gr.Textbox(
                    label="Enter a new folder name (will be sub the models folder)",
                    lines=1,
                    value="mycustomfolder",
                )
                compact_custom_btn = gr.Button(
                    value="Compact and move model to a folder name of your choosing")
            with gr.Row():
                gr.Textbox(
                    value="This will DELETE your training data and the raw finetuned model from the specified Project Name (above)",
                    scale=2,
                    show_label=False,
                    interactive=False,
                )
                delete_training_btn = gr.Button(
                    value="Delete generated training data")
            with gr.Row():
                gr.Textbox(
                    value="This will DELETE your original voice samples from /finetune/put-voice-samples-in-here/.",
                    scale=2,
                    show_label=False,
                    interactive=False,
                )
                delete_voicesamples_btn = gr.Button(
                    value="Delete original voice samples")

                prompt_compute_btn.click(
                    fn=preprocess_dataset,
                    inputs=[
                        lang,
                        whisper_model,
                        max_sample_length,
                        eval_split_number,
                        speaker_name_input,
                        create_bpe_tokenizer,
                        use_vad,
                        precision,
                    ],
                    outputs=[
                        progress_data,
                        train_csv,
                        eval_csv,
                        speaker_name_input_training,
                        speaker_name_input_testing,
                        speaker_name_input_export,
                    ],
                )

            train_btn.click(
                fn=train_model,
                inputs=[
                    lang,
                    train_csv,
                    eval_csv,
                    learning_rates,
                    model_to_train,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    max_audio_length,
                    speaker_name_input_training,
                    continue_run,
                    disable_shared_memory,
                    learning_rate_scheduler,
                    optimizer,
                    num_workers,
                    warm_up,
                ],
                outputs=[
                    progress_train,
                    xtts_config,
                    xtts_vocab,
                    xtts_checkpoint,
                    speaker_reference_audio,
                    speaker_name_input_testing,
                ],
            )

            load_btn.click(
                fn=load_model,
                inputs=[xtts_checkpoint, xtts_config, xtts_vocab],
                outputs=[progress_load],
            )

            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                ],
                outputs=[progress_gen, tts_output_audio, reference_audio],
            )
            compact_custom_btn.click(
                fn=compact_custom_model,
                inputs=[xtts_checkpoint_copy, folder_path, overwrite_existing],
                outputs=[final_progress_data],
            )
            delete_training_btn.click(
                fn=delete_training_data,
                outputs=[final_progress_data],
            )
            delete_voicesamples_btn.click(
                fn=delete_voice_sample_contents,
                outputs=[final_progress_data],
            )
            model_to_train.change(
                basemodel_or_finetunedmodel_choice,
                model_to_train,
                None)

    demo.queue().launch(
        show_api=False,
        inbrowser=True,
        share=False,
        debug=False,
        server_port=7052,
        server_name="127.0.0.1",
    )
