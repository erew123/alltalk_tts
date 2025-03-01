"""
AllTalk Multi Engine Manager (MEM)

This script is part of the AllTalk project, written by Erew123.
GitHub Repository: https://github.com/erew123/alltalk_tts

AllTalk MEM (Multi Engine Manager) is a research tool designed to manage and test
multiple instances of different Text-to-Speech (TTS) engines being loaded simultaneously,
with a view to a centralized engine being able to handle multiple requests simultaneously.

Attribution:
- Original AllTalk project by Erew123 (https://github.com/erew123)
- MEM extension developed based on the AllTalk framework

Licensing:
This project is subject to the licensing terms specified in the AllTalk repository.
Please refer to https://github.com/erew123/alltalk_tts for the most up-to-date
licensing information.

Code Usage and Modifications:
Any code snippets or algorithms adapted from the original AllTalk project are used
with permission and in accordance with the project's licensing terms. Modifications
and extensions for MEM functionality have been implemented while maintaining
compatibility with the original project's structure and intent.

Note to developers:
When using or adapting code from this script, please maintain the attribution
and adhere to the licensing terms of the original AllTalk project. For any
substantial use or modification, it's recommended to reference the original
project and this MEM extension.

MEM is not intended for production use at this time and there is NO support
being offered on MEM.

For the latest updates and documentation, please visit:
https://github.com/erew123/alltalk_tts
"""

import os
import io
import sys
import time
import queue
import socket
import signal
import asyncio
import aiohttp
import requests
import threading
import subprocess
import gradio as gr
from pathlib import Path
from gradio import routes
from collections import deque
from datetime import datetime, timedelta
from requests.exceptions import RequestException
from werkzeug.serving import make_server, WSGIRequestHandler
from config import AlltalkMultiEngineManagerConfig
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify, send_from_directory, send_file

# Setup config file
CONFIG_FILE = "mem_config.json"

# Dictionary to store all processes
processes = {}
script_path = "tts_server.py"

# Global variables
queue_items = deque()
engine_statuses = {}

# Global flag to indicate if the program should exit
should_exit = threading.Event()

# Global variable to hold the monitor thread
stop_monitor = threading.Event()
monitor = None
WITH_UI = os.getenv("WITH_UI", True).lower() == 'true'

flask_app = Flask(__name__)
CORS(flask_app)
flask_app.config['OUTPUT_FOLDER'] = str(Path(os.getcwd()) / 'outputs')

# Load configuration at the start of the script
config = AlltalkMultiEngineManagerConfig.get_instance()

def signal_handler(signum, frame):
    print("[AllTalk MEM] Interrupt received, shutting down...")
    should_exit.set()

##########################
### Live Queue Monitor ###
##########################
class QueueItem:
    def __init__(self, text, start_time):
        self.text = text
        self.start_time = start_time

class EngineStatus:
    def __init__(self):
        self.current_request = None
        self.start_time = None
        self.char_count = 0

class MonitoringControl:
    def __init__(self):
        self.stop_event = asyncio.Event()

############################################
# START-UP # Display initial splash screen #
############################################
print(f"[AllTalk MEM]\033[94m     _    _ _ \033[1;35m_____     _ _     \033[0m  _____ _____ ____  ")
print(f"[AllTalk MEM]\033[94m    / \  | | |\033[1;35m_   _|_ _| | | __ \033[0m |_   _|_   _/ ___| ")
print(f"[AllTalk MEM]\033[94m   / _ \ | | |\033[1;35m | |/ _` | | |/ / \033[0m   | |   | | \___ \ ")
print(f"[AllTalk MEM]\033[94m  / ___ \| | |\033[1;35m | | (_| | |   <  \033[0m   | |   | |  ___) |")
print(f"[AllTalk MEM]\033[94m /_/   \_\_|_|\033[1;35m |_|\__,_|_|_|\_\ \033[0m   |_|   |_| |____/ ")
print(f"[AllTalk MEM]\033[94m               __  __ \033[1;35m _____   __  __\033[0m")
print(f"[AllTalk MEM]\033[94m              |  \/  |\033[1;35m| ____| |  \/  |\033[0m")
print(f"[AllTalk MEM]\033[94m              | |\/| |\033[1;35m|  _|   | |\/| |\033[0m")
print(f"[AllTalk MEM]\033[94m              | |  | |\033[1;35m| |___  | |  | |\033[0m")
print(f"[AllTalk MEM]\033[94m              |_|  |_|\033[1;35m|_____| |_|  |_|\033[0m")
print(f"[AllTalk MEM]")
print(f"[AllTalk MEM] \033[93m     MEM is not intended for production use and\033[00m")
print(f"[AllTalk MEM] \033[93m      there is NO support being offered on MEM\033[00m")
print(f"[AllTalk MEM]")
print(f"[AllTalk MEM] \033[94mAPI/Queue   :\033[00m \033[92mhttp://127.0.0.1:{config.api_server_port}/api/tts-generate\033[00m")
if WITH_UI:
    print(f"[AllTalk MEM] \033[94mGradio Light:\033[00m \033[92mhttp://127.0.0.1:{config.gradio_interface_port}\033[00m")
    print(f"[AllTalk MEM] \033[94mGradio Dark :\033[00m \033[92mhttp://127.0.0.1:{config.gradio_interface_port}?__theme=dark\033[00m")
print(f"[AllTalk MEM]")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port, num_instances):
    available_ports = []
    current_port = start_port
    while len(available_ports) < num_instances:
        if not is_port_in_use(current_port):
            available_ports.append(current_port)
        current_port += 1
    return available_ports[0], available_ports

def start_subprocess(instance_id, port):
    if instance_id in processes and processes[instance_id].poll() is None:
        return f"✔️ Running on port {port}"
    
    print(f"[AllTalk MEM] \033[94m********************************\033[00m")
    print(f"[AllTalk MEM] \033[94mStarting TTS Engine on port {port}\033[00m")
    print(f"[AllTalk MEM] \033[94m********************************\033[00m")
    print(f"[AllTalk MEM] \033[92mEngine number     :\033[93m {instance_id}\033[00m")
    print(f"[AllTalk MEM] \033[92mPort number       :\033[93m {port}\033[00m")
    processes[instance_id] = subprocess.Popen([sys.executable, script_path, "--port", str(port)])
    
    # Wait for the server to be ready
    if is_server_ready(port):
        print(f"[AllTalk MEM]")
        print(f"[AllTalk MEM] Engine {instance_id} ready on port {port}")
        return f"✔️ Running on port {port}"
    else:
        print(f"[AllTalk MEM] Warning: Engine {instance_id} may not be fully ready on port {port}")
        return f"Warning: Engine {instance_id} may not be fully ready on port {port}."

def stop_subprocess(instance_id):
    if instance_id in processes:
        print(f"[AllTalk MEM] Terminating Engine {instance_id}")
        processes[instance_id].terminate()
        processes[instance_id].wait()
        del processes[instance_id]
        return f"Engine {instance_id} stopped."
    return f"Engine {instance_id} is not running."

def restart_subprocess(instance_id, port):
    print(f"[AllTalk MEM] Re-starting Engine {instance_id}")
    stop_result = stop_subprocess(instance_id)
    if "not running" in stop_result.lower():
        start_result = start_subprocess(instance_id, port)
        print(f"[AllTalk MEM] Engine {instance_id} was not running so starting an instance")
        return f"Engine {instance_id} was not running. {start_result}"
    else:
        return start_subprocess(instance_id, port)

def check_subprocess_status(instance_id, port):
    if instance_id in processes and processes[instance_id].poll() is None:
        return f"✔️ Running on port {port}"
    return "❌ Not running"

# This code block tests if one of the engines has started in a timely fashion. You can increase the max retries and/or backoff factor
# to allow for longer startup times of the TTS engine. 
def retry_with_backoff(func):
    retries = 0
    wait_time = config.initial_wait
    while retries < config.max_retries:
        try:
            result = func()
            print(f"[AllTalk MEM] Operation successful on attempt {retries + 1}") if config.debug_mode else None
            return result
        except RequestException as e:
            retries += 1
            print(f"[AllTalk MEM] Attempt {retries}") if config.debug_mode else None
            if retries == config.max_retries:
                print(f"[AllTalk MEM] All {config.max_retries} attempts failed waiting for this instance of the Engine to load.") if config.debug_mode else None
                return False
            print(f"[AllTalk MEM] Retrying in {wait_time} seconds...") if config.debug_mode else None
            time.sleep(wait_time)
            wait_time = round(wait_time * config.backoff_factor, 1)
    return False

def is_server_ready(port, timeout=3):
    def check_ready():
        response = requests.get(f"http://127.0.0.1:{port}/api/ready", timeout=timeout)
        if response.text.strip() == "Ready":
            return True
        raise RequestException("Server not ready")
    return retry_with_backoff(check_ready)

def stop_all_instances():
    instance_ids = list(processes.keys())
    results = ["❌ Not running"] * config.max_instances
    for instance_id in instance_ids:
        stop_subprocess(instance_id)
        results[instance_id-1] = "❌ Not running"
    return results

def update_all_statuses():
    return [check_subprocess_status(i, config.base_port + i - 1) for i in range(1, config.max_instances + 1)]

def count_running_instances():
    return sum(1 for p in processes.values() if p.poll() is None)

# Gradio interface TTS Test (not anything to do with client requests)
def generate_tts(port, engine_num, text, voice):
    try:
        response = requests.post(f"http://127.0.0.1:{port}/api/tts-generate", data={
            "text_input": text,
            "text_filtering": "standard",
            "character_voice_gen": voice,
            "narrator_enabled": "false",
            "language": "en",
            "output_file_name": "ttsmanager",
            "output_file_timestamp": "true",
            "autoplay": "false",
            "autoplay_volume": "0.8"
        }, timeout=30)  # Increased timeout for TTS generation
        data = response.json()
        if data["status"] == "generate-success":
            return f"http://127.0.0.1:{port}{data['output_file_url']}", f"TTS generated successfully on Engine {engine_num} and Port {port}"
    except requests.Timeout:
        return None, f"Timeout while generating TTS on port {port}"
    except requests.ConnectionError:
        return None, f"Connection error while generating TTS on port {port}"
    except Exception as e:
        return None, f"Unexpected error while generating TTS on port {port}: {str(e)}"
    return None, "Failed to generate TTS"

def test_tts(engine, voice, text):
    if not engine or not voice:
        return None, "Please select both an engine and a voice" 
    try:
        engine_num = int(engine.split()[1])
        port = config.base_port + engine_num - 1
        formatted_text = text.format(engine=engine_num, port=port, voice=voice)
        
        audio_url, debug_msg = generate_tts(port, engine_num, formatted_text, voice)
        if audio_url:
            return audio_url, debug_msg
        else:
            return None, debug_msg
    except Exception as e:
        return None, f"Error in test_tts: {str(e)}"

#############################
# Gradio Code and functions #
#############################
def create_gradio_interface():
    global monitor
    monitor = MonitoringControl()
    with gr.Blocks(title="AllTalk Multi Engine Manager", theme=gr.themes.Base()) as interface:
        with gr.Row():
            gr.Markdown("# AllTalk Multi Engine Manager")
            gr.Markdown("")
            gr.Markdown("")
            dark_mode_btn = gr.Button("Light/Dark Mode", variant="primary", size="sm")
            dark_mode_btn.click(None, None, None,
            js="""() => {
                if (document.querySelectorAll('.dark').length) {
                    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                    // localStorage.setItem('darkMode', 'disabled');
                } else {
                    document.querySelector('body').classList.add('dark');
                    // localStorage.setItem('darkMode', 'enabled');
                }
            }""", show_api=False)
        with gr.Tab("Engine Management"):
            with gr.Row():
                num_instances_input = gr.Slider(minimum=1, maximum=config.max_instances, step=1, value=1, label="Number of Instances to Start")
                start_MEMple_button = gr.Button("Start Instances")
                stop_all_button = gr.Button("Stop All Instances")
                refresh_button = gr.Button("Refresh Status")
            error_box = gr.Textbox(label="Error Messages", visible=False)
            
            instance_statuses = []
            for i in range(0, config.max_instances, 8):  # Create rows with 8 instances each
                with gr.Row():
                    for j in range(8):
                        if i + j < config.max_instances:
                            instance_id = i + j + 1
                            
                            # Use a column for each engine instance to control layout
                            with gr.Column(scale=1, min_width=160):
                                gr.Markdown(f"### Engine Instance {instance_id}")
                                status = gr.Markdown("❌ Not running")
                                
                                with gr.Row():
                                    start_btn = gr.Button("▶️ Start", size="sm", scale=1, min_width=10)
                                with gr.Row():                                    
                                    stop_btn = gr.Button("⏹️ Stop", size="sm", scale=1, min_width=10)
                                    restart_btn = gr.Button("♻️ Re-St", size="sm", scale=1, min_width=10)

                                # Define click behavior for each button
                                start_btn.click(lambda id=instance_id: start_and_update(id, config.base_port + id - 1), outputs=status)
                                stop_btn.click(lambda id=instance_id: stop_and_update(id), outputs=status)
                                restart_btn.click(lambda id=instance_id: restart_and_update(id, config.base_port + id - 1), outputs=status)

                            instance_statuses.append(status)
            
            def refresh_and_update_slider():
                running_count = count_running_instances()
                statuses = update_all_statuses()
                return [gr.update(value=running_count)] + statuses

            refresh_button.click(
                refresh_and_update_slider,
                outputs=[num_instances_input] + instance_statuses
            )
                
            def validate_and_start(num_instances):
                if num_instances < 1:
                    return [gr.update(value=1)] + ["❌ Not running"] * config.max_instances + ["Invalid number of instances"]
                
                results = start_MEMple_instances(num_instances)
                running_count = count_running_instances()
                return [gr.update(value=running_count)] + results + [""]

            start_MEMple_button.click(
                validate_and_start,
                inputs=[num_instances_input],
                outputs=[num_instances_input] + instance_statuses + [error_box]
            )

            def stop_all_and_update():
                results = stop_all_instances()
                return [gr.update(value=1)] + results

            stop_all_button.click(stop_all_and_update, outputs=[num_instances_input] + instance_statuses) 
            
        with gr.Tab("Engines TTS Test"):
            with gr.Row():
                engine_selector = gr.Dropdown(choices=["Please Start an Engine & Refresh List"], value="Please Start an Engine & Refresh List", label="Select TTS Engine", scale=1)
                voice_selector = gr.Dropdown(choices=["Please Start an Engine & Refresh List"], value="Please Start an Engine & Refresh List", label="Select Voice", scale=1)
                test_text = gr.Textbox(label="Enter text to synthesize", value="This is a test of TTS engine number {engine} on port {port}, using the {voice} voice.", scale=2)
            
            with gr.Row():
                refresh_engine_button = gr.Button("Refresh Engine List")
                test_button = gr.Button("Generate TTS")
                
            with gr.Row():
                audio_output = gr.Audio(label="Generated Audio")
                debug_output = gr.Textbox(label="Debug Output", lines=5)
            
            def update_engine_selector():
                running_engines = []
                debug_info = []
                for i, info in tts_instances.items():
                    if is_instance_active(i):
                        running_engines.append(f"Engine {i} (Port {info['port']})")
                        status = 'Ready'
                    else:
                        status = 'Not Ready'
                    debug_info.append(f"Engine {i} (Port {info['port']}): {status}")
                
                debug_str = "\n".join(debug_info)
                return gr.update(choices=running_engines, value=running_engines[0] if running_engines else None), debug_str

            async def update_voice_selector(engine):
                if engine:
                    engine_num = int(engine.split()[1])
                    port = config.base_port + engine_num - 1
                    voices = await get_available_voices(port)
                    debug_str = f"Fetched voices for Engine {engine_num} (Port {port}): {voices}"
                    return gr.update(choices=voices, value=voices[0] if voices else None), debug_str
                return gr.update(choices=[], value=None), "No engine selected"

            refresh_engine_button.click(
                update_engine_selector,
                outputs=[engine_selector, debug_output]
            )

            engine_selector.change(
                update_voice_selector,
                inputs=[engine_selector],
                outputs=[voice_selector, debug_output]
            )

        # Generate TTS when test button is clicked
        test_button.click(test_tts, inputs=[engine_selector, voice_selector, test_text], outputs=[audio_output, debug_output])
          
        with gr.Tab("MEM Settings"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🟦 Engine Instance Management")
                    with gr.Row():
                        base_port_input = gr.Number(
                            value=config.base_port,
                            label=f"Starting Port Number (Current: {config.base_port})",
                            step=1
                        )
                        max_instances_input = gr.Number(
                            value=config.max_instances,
                            label=f"Maximum Engine Instances (Current: {config.max_instances})",
                            step=1
                        )
                        auto_start_engines = gr.Number(
                            value=config.auto_start_engines,
                            label="Auto-start Engine Instances (0 to disable)",
                            step=1
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🟨 Gradio Port & API Port")                                      
                    with gr.Row():
                        gradio_port_input = gr.Number(
                            value=config.gradio_interface_port,
                            label=f"Gradio Interface Port (Current: {config.gradio_interface_port})",
                            step=1
                        )
                        api_server_port_input = gr.Number(
                            value=config.api_server_port,
                            label=f"API Server Port (Current: {config.api_server_port})",
                            step=1
                        )                                     
                                        
            with gr.Row():               
                with gr.Column(scale=2):
                    gr.Markdown("### 🟥 Engine Start-up Time Contol")
                    with gr.Row():
                        max_retries_input = gr.Number(
                            value=config.max_retries,
                            label=f"Max Retries (Current: {config.max_retries})",
                            step=1
                        )
                        initial_wait_input = gr.Number(
                            value=config.initial_wait,
                            label=f"Initial Wait (Current: {config.initial_wait})",
                            step=0.1
                        )
                        backoff_factor_input = gr.Number(
                            value=config.backoff_factor,
                            label=f"Backoff Factor (Current: {config.backoff_factor})",
                            step=0.1
                        )
                with gr.Column(scale=1):
                    gr.Markdown("### 🟪 Debug Settings")
                    with gr.Row():                        
                        debug_mode_input = gr.Dropdown(
                            choices=["Disabled", "Enabled"],
                            value="Enabled" if config.debug_mode else "Disabled",
                            label="Debug Mode"
                        ) 

            with gr.Row():               
                with gr.Column(scale=1):
                    gr.Markdown("### 🟩 Core Queue Management")                  
                    with gr.Row(): 
                        max_queue_time_input = gr.Number(
                            value=config.max_queue_time,
                            label=f"Max Queue Time (seconds) (Current: {config.max_queue_time})",
                            step=1
                        )
                        queue_check_interval_input = gr.Number(
                            value=config.queue_check_interval,
                            label=f"Queue Check Interval (seconds) (Current: {config.queue_check_interval})",
                            step=0.1
                        )
                with gr.Column(scale=3):
                    gr.Markdown("### 🟩 TTS Request Dynamic Timeout Factors")                  
                    with gr.Row():
                        tts_request_timeout_input = gr.Number(
                            value=config.tts_request_timeout,
                            label=f"TTS Request Timeout (seconds) (Current: {config.tts_request_timeout})",
                            step=1
                        )                                                         
                        text_length_factor_input = gr.Number(
                            value=config.text_length_factor,
                            label=f"Text Length Factor Per 100 chrs (Current: {config.text_length_factor})",
                            step=0.1
                        )
                        concurrent_request_factor_input = gr.Number(
                            value=config.concurrent_request_factor,
                            label=f"Concurrent Request Factor (Current: {config.concurrent_request_factor})",
                            step=0.1
                        )
                        diminishing_factor_input = gr.Number(
                            value=config.diminishing_factor,
                            label=f"Diminishing Factor Calc (Current: {config.diminishing_factor})",
                            step=0.1
                        )
                        queue_position_factor_input = gr.Number(
                            value=config.queue_position_factor,
                            label=f"Queue Position Factor (Current: {config.queue_position_factor})",
                            step=0.1
                        )                                                           
                                          
            with gr.Row():                
                settings_status = gr.Textbox(label="Settings Status", scale=3)
                set_settings_button = gr.Button("Save Settings", scale=1)

            def update_settings(new_port, new_max_instances, new_auto_start, new_gradio_port, 
                                new_api_server_port, new_max_retries, new_initial_wait, 
                                new_backoff_factor, new_debug_mode, new_max_queue_time, 
                                new_queue_check_interval, new_tts_request_timeout,
                                new_text_length_factor, new_concurrent_request_factor,
                                new_diminishing_factor, new_queue_position_factor):
                if new_port < 1024:
                    new_port = 7001
                
                new_auto_start = min(int(new_auto_start), int(new_max_instances))
                
                config.base_port = int(new_port)
                config.max_instances = int(new_max_instances)
                config.auto_start_engines = new_auto_start
                config.gradio_interface_port = int(new_gradio_port)
                config.api_server_port = int(new_api_server_port)
                config.max_retries = int(new_max_retries)
                config.initial_wait = float(new_initial_wait)
                config.backoff_factor = float(new_backoff_factor)
                config.debug_mode = (new_debug_mode == "Enabled")
                config.max_queue_time = int(new_max_queue_time)
                config.queue_check_interval = float(new_queue_check_interval)
                config.tts_request_timeout = int(new_tts_request_timeout)
                config.text_length_factor = float(new_text_length_factor)
                config.concurrent_request_factor = float(new_concurrent_request_factor)
                config.diminishing_factor = float(new_diminishing_factor)
                config.queue_position_factor = float(new_queue_position_factor)
                
                config.save()
                
                status_message = (
                    f"Settings updated: Start port: {config.base_port}, Max instances: {config.max_instances}, "
                    f"Auto-start engines: {config.auto_start_engines}, Gradio port: {config.gradio_interface_port}, "
                    f"API Server Port: {config.api_server_port}, "
                    f"Max retries: {config.max_retries}, Initial wait: {config.initial_wait}, "
                    f"Backoff factor: {config.backoff_factor}, Debug mode: {config.debug_mode}, "
                    f"Max queue time: {config.max_queue_time}, "
                    f"Queue check interval: {config.queue_check_interval}, "
                    f"TTS request timeout: {config.tts_request_timeout}, "
                    f"Text length factor: {config.text_length_factor}, "
                    f"Concurrent request factor: {config.concurrent_request_factor}, "
                    f"Diminishing factor: {config.diminishing_factor}, "
                    f"Queue position factor: {config.queue_position_factor}"
                )
                
                return (
                    status_message,
                    gr.update(value=config.base_port, label=f"Starting Port Number (Current: {config.base_port})"),
                    gr.update(value=config.max_instances, label=f"Maximum Instances (Current: {config.max_instances})"),
                    gr.update(value=config.auto_start_engines),
                    gr.update(value=config.gradio_interface_port, label=f"Gradio Interface Port (Current: {config.gradio_interface_port})"),
                    gr.update(value=config.api_server_port, label=f"API Server Port (Current: {config.api_server_port})"),
                    gr.update(value=config.max_retries),
                    gr.update(value=config.initial_wait),
                    gr.update(value=config.backoff_factor),
                    gr.update(value="Enabled" if config.debug_mode else "Disabled"),
                    gr.update(value=config.max_queue_time, label=f"Max Queue Time (seconds) (Current: {config.max_queue_time})"),
                    gr.update(value=config.queue_check_interval, label=f"Queue Check Interval (seconds) (Current: {config.queue_check_interval})"),
                    gr.update(value=config.tts_request_timeout, label=f"TTS Request Timeout (seconds) (Current: {config.tts_request_timeout})"),
                    gr.update(value=config.text_length_factor, label=f"Text Length Factor (Current: {config.text_length_factor})"),
                    gr.update(value=config.concurrent_request_factor, label=f"Concurrent Request Factor (Current: {config.concurrent_request_factor})"),
                    gr.update(value=config.diminishing_factor, label=f"Diminishing Factor (Current: {config.diminishing_factor})"),
                    gr.update(value=config.queue_position_factor, label=f"Queue Position Factor (Current: {config.queue_position_factor})")
                )

            set_settings_button.click(
                update_settings,
                inputs=[base_port_input, max_instances_input, auto_start_engines, gradio_port_input,
                        api_server_port_input, max_retries_input, initial_wait_input, backoff_factor_input,
                        debug_mode_input, max_queue_time_input, queue_check_interval_input, 
                        tts_request_timeout_input, text_length_factor_input, concurrent_request_factor_input,
                        diminishing_factor_input, queue_position_factor_input],
                outputs=[settings_status, base_port_input, max_instances_input, auto_start_engines,
                        gradio_port_input, api_server_port_input, max_retries_input, initial_wait_input,
                        backoff_factor_input, debug_mode_input, max_queue_time_input, queue_check_interval_input,
                        tts_request_timeout_input, text_length_factor_input, concurrent_request_factor_input,
                        diminishing_factor_input, queue_position_factor_input]
            )
              
            with gr.Row():
                gr.Markdown("### 🆘 Settings Help")
            with gr.Tab("Engine Instance Management"):                
                with gr.Row():                                              
                    with gr.Column():
                        gr.Markdown("""
                        These settings control the initialization, quantity, and port assignment of TTS engine instances within MEM. They determine how many engines can run simultaneously and how network ports are allocated to them.

                        ### 🟦 Starting Port Number
                        **Default: 7001**<br>
                        The base port number for TTS engine instances. Each instance uses the next available port. Ensure these ports are free on your system.

                        ### 🟦 Maximum Instances
                        **Default: 8**<br>
                        The maximum number of simultaneous TTS engine instances. Adjust based on your system's capabilities to prevent overload.

                        ### 🟦 Auto-start Engines
                        **Default: 0 (disabled)**<br>
                        Number of TTS engine instances to start automatically on MEM launch. Cannot exceed Maximum Instances.   
                        """)
            with gr.Tab("Gradio Port & API Port"):                
                with gr.Row():                                              
                    with gr.Column():
                        gr.Markdown("""
                        These settings define the network ports for user interface access and API communication. They are crucial for ensuring that users can interact with MEM and that other applications can send requests to MEM.

                        ### 🟨 Gradio Interface Port
                        **Default: 7500**<br>
                        Port for the Gradio web interface. MEM binds to 0.0.0.0 on this port.

                        ### 🟨 API Server Port
                        **Default: 7851**<br>
                        Port for incoming TTS requests to the MEM API. Ensure it's free and not firewalled.
                        """)
            with gr.Tab("Engine Start-up Time Contol"):                
                with gr.Row():                                              
                    with gr.Column():
                        gr.Markdown("""                     
                        These settings manage the initialization process and error handling for individual TTS engine instances. They control how MEM attempts to start each engine, how long it waits between attempts, and how it handles potential failures during the start-up process. These settings are particularly relevant when starting multiple TTS engines simultaneously or during automatic start-up.

                        ### 🟥 Max Retries
                        **Default: 8**<br>
                        Maximum number of attempts MEM will make to connect to a TTS engine before marking it as failed. This helps handle slow-starting engines or temporary network issues.

                        ### 🟥 Initial Wait
                        **Default: 3 (seconds)**<br>
                        Initial delay between retry attempts when connecting to a TTS engine. This gives the engine time to initialize before MEM attempts to connect again.

                        ### 🟥 Backoff Factor
                        **Default: 1.2**<br>
                        Multiplier for wait time between retries, implementing an exponential backoff strategy. This reduces system load during persistent issues by increasing the wait time between each retry attempt.
                        """)
            with gr.Tab("Debug Settings"):                
                with gr.Row():                                              
                    with gr.Column():
                        gr.Markdown("""
                        ### 🟪 Debug Settings                                                        
                        **Default: Disabled**<br>
                        When enabled, MEM outputs additional diagnostic information. Useful for troubleshooting but increases console output.

                        Note: All settings are saved in `mem_config.json` in the MEM script directory. Default settings are used if the file doesn't exist. Most settings take effect immediately, but some (like port changes) may require a MEM restart.
                        """)                                                                   
            with gr.Tab("Core Queue Management"):                        
                with gr.Column():
                    gr.Markdown("""
                    This section defines the fundamental behavior of the request queue system. These settings control how long requests can wait, how often MEM checks for available engines, and the base timeout for TTS requests. They work together to manage the flow of requests through the system.

                    ### 🟩 Max Queue Time
                    **Default: 60 seconds**<br>
                    - This is the maximum time a request can spend waiting in the queue before being sent to a TTS engine.<br>
                    - Once this time is exceeded, the request is removed from the queue and considered failed if it hasn't started processing by a TTS engine.<br>
                    - Dynamic Timeout Factors come into play after a request has left the queue and is being processed by a TTS engine and they determine how long the system will wait for the TTS generation to complete.<br>
                    - Total allowed processing time can be from when a request enters the system to when it completes (or fails) and is the sum of:<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a) Time spent in the queue (limited by Max Queue Time)<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b) Time spent in TTS generation (governed by Dynamic Timeout Factors)
                    - A request's "queue time" stops when it's sent to a TTS engine.
                    - The dynamic timeout for TTS generation is separate from and additional to the queue time, hence the total time a request spends in the system can indeed exceed the Max Queue Time.

                    ### 🟩 Queue Check Interval
                    **Default: 0.1 seconds**<br>
                    Frequency of checking for available TTS engines. Lower values increase responsiveness but may increase CPU usage.

                    These settings work together to balance system responsiveness and resource usage. Adjust Max Queue Time for overall request lifespan, Queue Check Interval for system reactivity.
                    """)               
            with gr.Tab("TTS Request Dynamic Timeout Factors"):                        
                with gr.Column():
                    gr.Markdown("""
                    These advanced settings allow adaptive adjustment of request timeouts based on various factors such as text length, system load, and queue position. They all work to modify the base TTS Request Timeout, creating a flexible timeout system that can adapt to changing conditions and ensure fair processing of requests, specifically designed to manage the waiting time for TTS generation after a request has been dispatched from the queue to an available TTS engine.

                    When They Apply:<br>
                    - These factors come into play after a request has been taken from the queue and sent to a TTS engine for processing. They dynamically adjust the timeout period for each individual TTS generation request based on various factors (Text length, Number of concurrent requests, Position in the queue, Time already spent processing).

                    How They Work:<br>
                    - The system calculates a custom timeout for each request using these factors.<br>
                    - This timeout determines how long the system will wait for the TTS engine to complete the generation before considering it a failed request.<br>

                    Benefit:<br>
                    - Allows longer wait times for more complex requests (e.g., longer text)<br>
                    - Prevents indefinite waiting by setting a maximum timeout<br>
                    - Adapts to system load and request complexity<br>

                    Key Point:<br>
                    - These factors do not affect how long a request waits in the queue. They only apply to the actual TTS generation process after the request has been sent to an engine.

                    ### 🟩 TTS Request Timeout
                    **Default: 30 seconds**<br>
                    Base maximum processing time for a single TTS request. This value serves as the foundation for the Dynamic Timeout Factors. The actual timeout for each request is calculated by applying the Dynamic Timeout Factors to this base value. For a detailed explanation of how this base timeout is adjusted, please refer to the "Dynamic Timeout Factors" section in the settings help.

                    ### 🟩 Text Length Factor
                    **Default: 0.2**<br>
                    Increases the base TTS Request Timeout by this percentage for every 100 characters of text. For example, with a base timeout of 30 seconds and a 200-character text, the adjusted timeout would be 30 * (1 + (0.2 * 2)) = 42 seconds.

                    ### 🟩 Concurrent Request Factor
                    **Default: 0.5**<br>
                    Increases the base TTS Request Timeout by this percentage for each concurrent request. For instance, if there are 2 concurrent requests, the timeout for each would be increased by 50% (0.5 * 2 = 1, so 30 * (1 + 1) = 60 seconds).

                    ### 🟩 Diminishing Factor
                    **Default: 0.5**<br>
                    Reduces the additional time given to long-running requests. This factor is applied to the extra time calculated from other factors. For example, if other factors have increased the timeout by 20 seconds, this might be reduced to 10 seconds (20 * 0.5) for a long-running request.

                    ### 🟩 Queue Position Factor
                    **Default: 1.0**<br>
                    Adds this percentage of the base TTS Request Timeout for each position in the queue. For a request that's second in the queue, this would add 100% of the base timeout (e.g., 30 + 30 = 60 seconds).

                    These factors work together to create a flexible timeout system:
                    1. The base TTS Request Timeout is first adjusted by the Text Length Factor and Concurrent Request Factor.
                    2. The Queue Position Factor then adds additional time based on the request's position.
                    3. For requests that have been processing for a while, the Diminishing Factor reduces the extra time added by other factors.
                    4. The final calculated timeout is capped at the Max Queue Time to prevent excessive wait times.

                    Example calculation:
                    - Base TTS Request Timeout: 30 seconds
                    - Text length: 300 characters
                    - Concurrent requests: 2
                    - Queue position: 3
                    - Request has been processing for a while

                    Calculation:
                    1. Text Length adjustment: 30 * (1 + (0.2 * 3)) = 48 seconds
                    2. Concurrent Request adjustment: 48 * (1 + (0.5 * 2)) = 96 seconds
                    3. Queue Position adjustment: 96 + (30 * 1.0 * 3) = 186 seconds
                    4. Diminishing Factor (assuming it reduces extra time by half): 30 + ((186 - 30) * 0.5) = 108 seconds

                    Final timeout would be 108 seconds, unless this exceeds the Max Queue Time, in which case Max Queue Time would be used instead.

                    This dynamic system allows for flexible handling of various request scenarios while still respecting overall system limits. 
                    """)                         
    
        with gr.Tab("MEM Queue Monitor"):
            gr.Markdown("## Queue and Engine Status")
            
            with gr.Row():
                queue_length = gr.Textbox(label="Total Queue Length", value="0")
                running_engines = gr.Textbox(label="Running Engines", value="0")
                start_button = gr.Button("Start Monitoring")
                stop_button = gr.Button("Stop Monitoring", interactive=False)
                
            with gr.Row():
                engine_status = gr.Dataframe(
                    headers=["Engine", "Port", "Status", "Processing Time", "Char Count", "Text"],
                    label="Engine Status",
                    wrap=True,
                    column_widths=["100px", "70px", "100px", "150px", "100px", "300px"],
                    height=500
                )                
            
            with gr.Row():
                queue_status = gr.Dataframe(
                    headers=["Position", "Wait Time", "Timeout", "Text"],
                    label="Queue Status (Top 10 items)",
                    wrap=True,
                    column_widths=["70px", "100px", "100px", "400px"],
                    height=500
                )
            
            is_monitoring = gr.State(value=False)
            async def update_monitor_data(is_monitoring):
                global monitor
                if not is_monitoring or (monitor and monitor.stop_event.is_set()):
                    return [None] * 4  # Return 4 None values instead of 3

                try:
                    queue_size = len(queue_items)
                    running_engine_count = sum(1 for p in processes.values() if p.poll() is None)
                    
                    queue_data = []
                    for i, item in enumerate(list(queue_items)[:10]):
                        wait_time = datetime.now() - item.start_time
                        timeout = timedelta(seconds=config.max_queue_time) - wait_time
                        queue_data.append([
                            i+1,
                            str(wait_time).split(".")[0],
                            str(timeout).split(".")[0] if timeout > timedelta() else "Timed Out",
                            item.text[:80] + "..." if len(item.text) > 80 else item.text
                        ])
                    
                    engine_data = []
                    for instance, info in tts_instances.items():
                        engine_status = engine_statuses.get(instance, EngineStatus())
                        if instance in processes and processes[instance].poll() is None:
                            if info["locked"] and engine_status.current_request:
                                processing_time = datetime.now() - engine_status.start_time
                                status = "Busy"
                                proc_time = str(processing_time).split('.')[0]
                                char_count = engine_status.char_count
                                text = engine_status.current_request[:50] + "..." if len(engine_status.current_request) > 50 else engine_status.current_request
                            else:
                                status = "Idle"
                                proc_time = ""
                                char_count = ""
                                text = ""
                        else:
                            status = "Not Started"
                            proc_time = ""
                            char_count = ""
                            text = ""
                        
                        engine_data.append([
                            f"Engine {instance}",
                            info['port'],
                            status,
                            proc_time,
                            char_count,
                            text
                        ])
                    
                    await asyncio.sleep(0)  # Yield control to allow cancellation
                    return str(queue_size), str(running_engine_count), queue_data, engine_data
                except Exception as e:
                    print(f"Error in update_monitor_data: {e}")
                    return [None] * 4  # Return 4 None values instead of 3
            
            def start_monitoring():
                if monitor:
                    monitor.stop_event.clear()
                return True, gr.update(interactive=False), gr.update(interactive=True)

            def stop_monitoring():
                if monitor:
                    monitor.stop_event.set()
                return False, gr.update(interactive=True), gr.update(interactive=False)

            start_button.click(
                start_monitoring,
                outputs=[is_monitoring, start_button, stop_button]
            ).then(
                update_monitor_data,
                inputs=[is_monitoring],
                outputs=[queue_length, running_engines, queue_status, engine_status]
            )

            stop_button.click(
                stop_monitoring,
                outputs=[is_monitoring, start_button, stop_button]
            ).then(
                lambda: ("0", "0", [], []),
                outputs=[queue_length, running_engines, queue_status, engine_status]
            )

            # Auto-refresh every 1 seconds when monitoring is active
            interface.load(
                update_monitor_data,
                inputs=[is_monitoring],
                outputs=[queue_length, running_engines, queue_status, engine_status],
                every=1,
                show_progress=False
            )
            
            interface.monitor = monitor  # Attach the MonitoringApp instance to the interface
            routes.App.stop_event = monitor.stop_event     

        with gr.Tab("MEM Load Test"):
            gr.Markdown("## MEM Load Test")
            
            with gr.Row():
                num_requests = gr.Slider(minimum=1, maximum=20, value=3, step=1, label="Number of Requests", scale=2)
                text_length = gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Approximate Text Length (chars)", scale=2)
                voice_dropdown = gr.Dropdown(choices=["Please Start an Engine & Refresh List"], value="Please Start an Engine & Refresh List", label="Voice", scale=2)
                refresh_voices_button = gr.Button("Refresh Voices", scale=1)   
                start_test_button = gr.Button("Start Load Test", scale=1)
            
            with gr.Row():
                summary_text = gr.Textbox(label="Summary", interactive=False, lines=4)
                note_text = gr.Textbox(
                    value=(
                        "All requests are sent simultaneously. The 'completed in' time is calculated from when the requests were sent, "
                        "not the actual TTS processing time per individual request within the TTS engine. This test measures overall "
                        "system responsiveness, including queuing and load balancing, not just TTS generation speed."
                    ),
                    label="Important Note",
                    lines=4,
                    interactive=False,
                )
            
            results_table = gr.Dataframe(
                headers=["Request", "Time (s)", "Status", "Audio URL"],
                label="Load Test Results",
                row_count=(1, "dynamic"),
                col_count=(4, "fixed"),
                interactive=False,
                datatype=["str", "number", "str", "str"]
            )

            async def refresh_voices():
                voices = await fetch_voices()
                return gr.update(choices=voices, value=voices[0] if voices else None)

            async def start_load_test(num_requests, text_length, voice):
                if not voice:
                    yield [], "Please select a voice before starting the load test.", gr.update()
                    return

                # Calculate the actual text length
                base_text = "Testing MEM load balancing system. Request X. "
                repeats = max(1, text_length // len(base_text))
                actual_text = base_text * repeats
                actual_text_length = len(actual_text.replace("X", "1"))

                progress_message = f"Load Test running: {num_requests} TTS requests, each with approximately {actual_text_length} characters"
                yield [], progress_message, gr.update()

                results, summary = await run_load_test_gradio(num_requests, actual_text, voice)
                yield results, summary, gr.update()

            refresh_voices_button.click(
                refresh_voices,
                outputs=[voice_dropdown]
            )

            start_test_button.click(
                start_load_test,
                inputs=[num_requests, text_length, voice_dropdown],
                outputs=[results_table, summary_text, voice_dropdown]
            )

        with gr.Tab("MEM FAQ/Help"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                        ## What is MEM?
                        AllTalk MEM (Multi Engine Manager) is a research tool designed to manage and test multiple instances of different Text-to-Speech (TTS) engines being loaded simultaneously, with a view to a centralised engine being able to handle multiple requests simultaneously.

                        ## Current Status & Support
                        ⚠️ AllTalk MEM is currently in a demonstration, testing, and experimental phase. ⚠️<br>
                        ⚠️ There will be bugs & issues. MEM is not intended for Production Environments. ⚠️ <br>
                        ⚠️ As such there is **NO support being offered on MEM**. ⚠️

                        ## System Capacity
                        **Q: How many engines can my system (GPU/CPU) handle at once?**<br>
                        A: The exact number is unknown and will vary based on your specific hardware configuration and the TTS engines being loaded. Experimentation is required to determine the optimal number for your system. You would have to monitor your system resources and see what works. Dont forget you need to leave RAM/VRAM overhead for actual generation/inference.

                        ## Queue System
                        **Q: Is there an in-built queue system to handle different requests to different loaded engines?**<br>
                        A: MEM incorporates a built-in queue system to manage multiple TTS requests across loaded engine instances:

                        1. All TTS requests are received through the API port (default: 7851).
                        2. The queue system distributes incoming requests among available TTS engine instances.
                        3. If all engines are busy, new requests are held in a queue until an engine becomes available.
                        4. The system continuously checks for available engines to process waiting requests.
                        5. If a request cannot be processed within the allocated time, it will be marked as failed.

                        This queue system aims to balance the load across all running engines efficiently. Advanced features for queue management and dynamic timeout calculations are available and can be configured in the MEM settings.

                        Note: As this is a research and testing implementation, its performance in high-load or production environments is not guaranteed. For load testing, you can use the MEM Load Test to simulate multiple simultaneous requests.
                        
                        ## API Requests to each loaded engine
                        **Q: Can I use the AllTalk API to each engine instance?**<br>
                        A: Yes, each engine loaded will respond fully as if it was a standalone AllTalk instance. So you can use the standard API requests to the port number of each TTS engine loaded in.
                        
                        > **Ready Endpoint -** http://{ipaddress}:{port}/api/ready<br>
                        > **Voice Endpoint -** http://{ipaddress}:{port}/api/voices<br>
                        > **RVC Voices Endpoint -** http://{ipaddress}:{port}/api/rvcvoices<br>
                        > **Current Settings Endpoint -** http://{ipaddress}:{port}/api/currentsettings<br>
                        > **TTS Generation Endpoint -** http://{ipaddress}:{port}/api/tts-generate<br>
                        > **OpenAI Compatible Endpoint -** http://{ipaddress}:{port}/v1/audio/speech<br>
                        
                        ## What happens if it tries to load an engine thats on a port already in use
                        **Q: When a engine instance starts up, what happens if a port is already used by something else?**<br>
                        A: A test is performed of the port before starting each instance, so the engine trying to load on that port number wouldnt load in.                          

                        """) 
                with gr.Column():                    
                    gr.Markdown(""" 
                        ## Handling Multiple Requests
                        **Q: How will different TTS engines from different manufacturers OR how will CUDA/Python handle multiple requests coming in at the same time for TTS generation?**<br>
                        A: The behavior in such scenarios is currently unknown and is part of the research aspect of this tool. Its highly possible some TTS engines may respond in a timely fashion and its also possible they may not. I cannot say how things like CUDA/Python will handle multiple requests to the same hadware. 
                        
                        CUDA has additional code and tools to deal with multiplexing requests and none of these have been implimented. So for example, if you are using a TTS engine that uses a GPU, and 2x different loaded TTS engines make a request of that hardware, you may get a 50/50 spit use of the hardware and you may not. 3x requests, you may get a 33/33/33 hardware load sharing sceanario nad you may not. I cannot say at the moment and havnt tested.
                        
                        Python - As each instance of each TTS engine will be loaded into a seperate Python instance, all memory management and requests should remain segregated from one another, so there should be no bleed over of tensors in CUDA requests etc, however, I have not tested.                             
                                                
                        ## What settings are used for the TTS engines loaded
                        **Q: What exact settings are being used by the TTS engines being loaded in?**<br>
                        A: Whatever you have set in the main AllTalk Gradio interface when you load AllTalk as a standalone, under the `TTS Engine Settings` and the specific TTS engine, those are the settings loaded/used.
                        
                        ## How can I change which TTS engine is loaded
                        **Q: How can you change the TTS engines being loaded?**<br>
                        A: Whatever you have set in the main AllTalk Gradio interface when you load AllTalk as a standalone, that will be the engine loaded in.                      

                        ## Future Updates
                        **Q: Will you be updating this in the future?**<br>
                        A: Updates may be made in the future, but this depends largely on available time and resources.

                        ## Contributions
                        **Q: Can I write some code for it and send a PR?**<br>
                        A: Yes, contributions are welcome! If you have improvements or features you'd like to add, feel free to submit a Pull Request.

                        ## Future Plans
                        If time permits, potential future developments include:<br>

                        - Potential control over each TTS engines's settings and manipulating the JSON files that control AllTalk, so as to handle which engine loads and its settings from the MEM interface.<br>
                        - Developing a basic API management suite for MEM<br>

                        Please note that these are potential plans and not guaranteed features.
                        """)        
            
    return interface

def start_and_update(instance_id, port):
    if is_port_in_use(port):
        return f"Port {port} is already in use"
    result = start_subprocess(instance_id, port)
    update_tts_instances()
    return check_subprocess_status(instance_id, port)

def stop_and_update(instance_id):
    result = stop_subprocess(instance_id)
    update_tts_instances()
    return "❌ Not running"

def restart_and_update(instance_id, port):
    if is_port_in_use(port) and instance_id not in processes:
        return f"Port {port} is already in use"
    result = restart_subprocess(instance_id, port)
    return check_subprocess_status(instance_id, port)

def start_MEMple_instances(num_instances):
    currently_running = count_running_instances()
    
    if currently_running >= num_instances:
        return update_all_statuses()  # All requested instances are already running
    
    instances_to_start = num_instances - currently_running
    start_port, available_ports = find_available_port(config.base_port + currently_running, instances_to_start)
    
    if len(available_ports) < instances_to_start:
        return ["Not enough available ports"] * config.max_instances

    results = update_all_statuses()
    for i, port in enumerate(available_ports[:instances_to_start], start=currently_running+1):
        result = start_subprocess(i, port)
        results[i-1] = result
        time.sleep(10)  # Increase wait time to 10 seconds

    return results

########################
## WEBSERVER # General #
########################
app = Flask(__name__)

# Queue for holding TTS requests
request_queue = queue.Queue()

# Lock for thread-safe operations
lock = threading.Lock()

# Dictionary to keep track of TTS instance status
tts_instances = {}

def initialize_instances():
    global tts_instances
    with lock:
        tts_instances.clear()
        for i in range(1, config.max_instances + 1):
            port = config.base_port + i - 1
            tts_instances[i] = {"port": port, "locked": False}

def get_available_instance():
    with lock:
        for instance, info in tts_instances.items():
            if not info["locked"] and is_instance_active(instance):
                info["locked"] = True
                return instance
    return None

def is_instance_active(instance):
    port = tts_instances[instance]["port"]
    try:
        response = requests.get(f"http://127.0.0.1:{port}/api/ready", timeout=1)
        return response.text.strip() == "Ready"
    except:
        return False

def release_instance(instance):
    with lock:
        if instance in tts_instances:
            tts_instances[instance]["locked"] = False

def update_tts_instances():
    global tts_instances
    with lock:
        tts_instances.clear()
        for i, process in processes.items():
            if process.poll() is None:  # Check if the process is running
                port = config.base_port + i - 1
                tts_instances[i] = {"port": port, "locked": False}

def calculate_dynamic_timeout(text, concurrent_requests, queue_position, total_queue_length, request_start_time):
    base_timeout = config.tts_request_timeout * (1 + (concurrent_requests * 0.5))
    
    text_length_factor = len(text) / 100
    adjusted_timeout = base_timeout * (1 + (text_length_factor * 0.2))
    
    time_in_process = time.time() - request_start_time
    diminishing_factor = max(0, 1 - (time_in_process / base_timeout))
    timeout_with_diminishing = adjusted_timeout * (1 + (diminishing_factor * config.diminishing_factor))
    
    queue_position_factor = queue_position / total_queue_length
    final_timeout = timeout_with_diminishing + (config.tts_request_timeout * queue_position_factor)
    
    return min(final_timeout, config.max_queue_time)  # Ensure we don't exceed max queue time

########################################
## WEBSERVER # Client Facing Endpoints #
########################################
@flask_app.route('/api/tts-generate', methods=['POST'])
@cross_origin(origins='*', methods=['POST', 'OPTIONS'], allow_headers=['Content-Type'])
def tts_generate():
    print("[AllTalk MEM] Received TTS generate request") if config.debug_mode else None
    start_time = datetime.now()
    request_data = request.form.to_dict()
    
    # Add request to queue
    queue_item = QueueItem(request_data['text_input'], start_time)
    queue_items.append(queue_item)
    
    while (datetime.now() - start_time).total_seconds() < config.max_queue_time:
        instance = get_available_instance()
        if instance:
            try:
                print(f"[AllTalk MEM] Processing request with instance {instance}") if config.debug_mode else None
                
                # Update engine status
                engine_statuses[instance] = EngineStatus()
                engine_statuses[instance].current_request = request_data['text_input']
                engine_statuses[instance].start_time = datetime.now()
                engine_statuses[instance].char_count = len(request_data['text_input'])
                
                dynamic_timeout = calculate_dynamic_timeout(
                    text=request_data['text_input'],
                    concurrent_requests=len([i for i in tts_instances.values() if i['locked']]),
                    queue_position=len(queue_items),
                    total_queue_length=len(queue_items),
                    request_start_time=start_time.timestamp()
                )
                
                result = process_tts_request(instance, request_data, dynamic_timeout)
                print(f"[AllTalk MEM] Request processed, result: {result}") if config.debug_mode else None
                
                # Remove request from queue
                queue_items.popleft()
                
                # Reset engine status
                engine_statuses[instance] = EngineStatus()
                
                return jsonify(result)
            except Exception as e:
                print(f"[AllTalk MEM] Error processing request: {str(e)}")
                release_instance(instance)
                return jsonify({"status": "error", "message": str(e)})
        time.sleep(config.queue_check_interval)
    
    # Remove request from queue if it times out
    queue_items.popleft()
    
    print("[AllTalk MEM] No instance available within queue timeout period")
    return jsonify({"status": "error", "message": "No TTS instance available within the queue maximum wait time"})

def process_tts_request(instance, data, timeout):
    port = tts_instances[instance]["port"]
    try:
        response = requests.post(f"http://127.0.0.1:{port}/api/tts-generate", data=data, timeout=timeout)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        release_instance(instance)

@flask_app.route('/v1/audio/speech', methods=['POST'])
@cross_origin(origins='*', methods=['POST', 'OPTIONS'], allow_headers=['Content-Type'])
def openai_speech():
    print("[AllTalk MEM] Received OpenAI-style TTS generate request") if config.debug_mode else None
    start_time = datetime.now()
    request_data = request.json

    # Add request to queue
    queue_item = QueueItem(request_data['input'], start_time)
    queue_items.append(queue_item)

    while (datetime.now() - start_time).total_seconds() < config.max_queue_time:
        instance = get_available_instance()
        if instance:
            try:
                print(f"[AllTalk MEM] Processing OpenAI-style request with instance {instance}") if config.debug_mode else None

                # Update engine status
                engine_statuses[instance] = EngineStatus()
                engine_statuses[instance].current_request = request_data['input']
                engine_statuses[instance].start_time = datetime.now()
                engine_statuses[instance].char_count = len(request_data['input'])

                dynamic_timeout = calculate_dynamic_timeout(
                    text=request_data['input'],
                    concurrent_requests=len([i for i in tts_instances.values() if i['locked']]),
                    queue_position=len(queue_items),
                    total_queue_length=len(queue_items),
                    request_start_time=start_time.timestamp()
                )

                result = process_openai_tts_request(instance, request_data, dynamic_timeout)
                print(f"[AllTalk MEM] OpenAI-style request processed, result: {result}") if config.debug_mode else None

                # Remove request from queue
                queue_items.popleft()

                # Reset engine status
                engine_statuses[instance] = EngineStatus()

                if isinstance(result, requests.Response) and result.status_code == 200:
                    return send_file(
                        io.BytesIO(result.content),
                        mimetype=f'audio/{request_data.get("response_format", "mp3")}',
                        as_attachment=True,
                        download_name=f"speech.{request_data.get('response_format', 'mp3')}"
                    )
                else:
                    return jsonify({"error": "Failed to generate speech"}), 500

            except Exception as e:
                print(f"[AllTalk MEM] Error processing OpenAI-style request: {str(e)}")
                release_instance(instance)
                return jsonify({"error": str(e)}), 500

        time.sleep(config.queue_check_interval)

    # Remove request from queue if it times out
    queue_items.popleft()

    print("[AllTalk MEM] No instance available within queue timeout period for OpenAI-style request")
    return jsonify({"error": "No TTS instance available within the queue maximum wait time"}), 504

def process_openai_tts_request(instance, data, timeout):
    port = tts_instances[instance]["port"]
    try:
        response = requests.post(f"http://127.0.0.1:{port}/v1/audio/speech", json=data, timeout=timeout)
        return response
    except Exception as e:
        return {"error": str(e)}
    finally:
        release_instance(instance)

@flask_app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(flask_app.config['OUTPUT_FOLDER'], filename)

@flask_app.route('/api/ready', methods=['GET'])
async def relay_ready_status():
    status = await fetch_from_any_engine('/api/ready')
    return status, 200 if status == "Ready" else 503

@flask_app.route('/api/voices', methods=['GET'])
async def relay_voices():
    voices = await fetch_from_any_engine('/api/voices')
    return jsonify(voices)

@flask_app.route('/api/rvcvoices', methods=['GET'])
async def relay_rvc_voices():
    rvc_voices = await fetch_from_any_engine('/api/rvcvoices')
    return jsonify(rvc_voices)

@flask_app.route('/api/currentsettings', methods=['GET'])
async def relay_current_settings():
    settings = await fetch_from_any_engine('/api/currentsettings')
    return jsonify(settings)

async def fetch_from_any_engine(endpoint):
    for instance, info in tts_instances.items():
        if is_instance_active(instance):
            port = info['port']
            url = f"http://127.0.0.1:{port}{endpoint}"
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            if endpoint == '/api/ready':
                                return await response.text()
                            else:
                                return await response.json()
                except aiohttp.ClientError:
                    continue  # Try the next engine if this one fails
    return {"status": "error", "message": "No active TTS engine available"}

##########################################
## WEBSERVER # Gradio interface internal #
##########################################
async def fetch_voices():
    if not tts_instances:
        return ["No Engines Loaded"]

    # Get the port of the first active engine
    for info in tts_instances.values():
        port = info['port']
        url = f"http://127.0.0.1:{port}/api/voices"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        voices = data.get('voices', [])
                        return voices if voices else ["No voices found"]
            except aiohttp.ClientError:
                continue  # Try the next engine if this one fails

    return ["No response from engines"]

async def get_available_voices(port):
    url = f"http://127.0.0.1:{port}/api/voices"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data["status"] == "success":
                        return data["voices"]
                    else:
                        print(f"Error fetching voices: {data.get('message', 'Unknown error')}")
                else:
                    print(f"Error fetching voices: HTTP {response.status}")
    except aiohttp.ClientError as e:
        print(f"Network error while fetching voices: {str(e)}")
    except Exception as e:
        print(f"Unexpected error while fetching voices: {str(e)}")
    
    return []  # Return an empty list if there's any error

async def update_voice_selector(engine):
    if engine:
        engine_num = int(engine.split()[1])
        port = config.base_port + engine_num - 1
        voices = await get_available_voices(port)
        debug_str = f"Fetched voices for Engine {engine_num} (Port {port}): {voices}"
        return gr.update(choices=voices, value=voices[0] if voices else None), debug_str
    return gr.update(choices=[], value=None), "No engine selected"

#############################
## WEBSERVER # Load Testing #
#############################
async def run_load_test_gradio(num_requests, base_text, voice):
    api_url = f"http://127.0.0.1:{config.api_server_port}/api/tts-generate"
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            text = base_text.replace("X", str(i+1))
            task = send_tts_request(session, api_url, text, voice, i+1)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    results = []
    for i, response in enumerate(responses):
        if response and response.get("status") == "generate-success":
            audio_url = f"http://127.0.0.1:{config.api_server_port}{response['output_file_url']}"
            # Round time to 1 decimal place
            rounded_time = round(response['time'], 1)
            results.append([f"Request {i+1}", rounded_time, "Success", audio_url])
        else:
            results.append([f"Request {i+1}", None, "Failed", ""])

    successful_requests = sum(1 for r in results if r[2] == "Success")
    failed_requests = num_requests - successful_requests
    summary = f"Load test completed in {total_time:.1f} seconds.\n"
    summary += f"Successful requests: {successful_requests}\n"
    summary += f"Failed requests: {failed_requests}\n"
    summary += f"Actual characters per request: {len(base_text)}"

    return results, summary

async def send_tts_request(session, url, text, voice, request_id):
    start_time = time.time()
    try:
        async with session.post(url, data={
            "text_input": text,
            "text_filtering": "standard",
            "character_voice_gen": voice,
            "narrator_enabled": "false",
            "narrator_voice_gen": "en_US-ljspeech-high.onnx",
            "text_not_inside": "character",
            "language": "en",
            "output_file_name": f"loadtest_{request_id}",
            "output_file_timestamp": "true",
            "autoplay": "false",
            "autoplay_volume": "0.8"
        }) as response:
            result = await response.json()
            end_time = time.time()
            result['time'] = end_time - start_time
            return result
    except Exception as e:
        print(f"Request {request_id} failed: {str(e)}")
        return None

##################################
## WEBSERVER # Thread Management #
##################################
class SilentWSGIRequestHandler(WSGIRequestHandler):
    def log_request(self, *args, **kwargs):
        pass

class ServerThread(threading.Thread):
    def __init__(self, app):
        threading.Thread.__init__(self)
        self.server = make_server('0.0.0.0', config.api_server_port, app, threaded=True, request_handler=SilentWSGIRequestHandler)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print(f"[AllTalk MEM] Starting API server on port {config.api_server_port}") if config.debug_mode else None
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

def start_api_server():
    global server_thread
    server_thread = ServerThread(flask_app)  # Use flask_app here
    server_thread.start()
    print(f"[AllTalk MEM] API server thread started on port {config.api_server_port}") if config.debug_mode else None

def stop_api_server():
    global server_thread
    if server_thread:
        print("[AllTalk MEM] Shutting down API server")
        server_thread.shutdown()
        server_thread.join()
        print("[AllTalk MEM] API server shut down successfully")

##########################
### Shutdown & Cleanup ###
##########################
def graceful_shutdown():
    print("[AllTalk MEM] Initiating graceful shutdown...")
    # Stop all engine instances
    print("[AllTalk MEM] Shutting down all engines...")
    stop_all_instances()
    print("[AllTalk MEM] All engines stopped.")
    # Stop API server
    stop_api_server()
    # Set flags to stop ongoing processes
    stop_monitor.set()
    should_exit.set()
    # Close Gradio interface
    if 'interface' in globals():
        interface.close()
    gr.close_all()
    print("[AllTalk MEM] Graceful shutdown complete.")

######################
### INITIALISATION ###
######################
print(f"[AllTalk MEM] Please use \033[91mCtrl+C\033[0m when exiting otherwise Python")
print(f"[AllTalk MEM] subprocess's will continue running in the background.")
print(f"[AllTalk MEM] ")
print(f"[AllTalk MEM] Ensure you have configured AllTalk TTS engines in the")
print(f"[AllTalk MEM] AllTalk interface before using MEM. MEM does not require")
print(f"[AllTalk MEM] AllTalk to be running and should be used seperately.")
print(f"[AllTalk MEM] ")
print(f"[AllTalk MEM] MEM Server Ready")

def auto_start_engines():
    num_engines = config.auto_start_engines
    if num_engines > 0:
        print(f"[AllTalk MEM] Auto-starting {num_engines} engines...")
        results = start_MEMple_instances(num_engines)
        update_tts_instances()
        print(f"[AllTalk MEM] Auto-start complete.")
        print(f"[AllTalk MEM] {results}")   

# Main execution
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    if WITH_UI:
        print("[AllTalk MEM] Starting gradio...")
        interface = create_gradio_interface()
        # Start the Gradio interface
        interface.launch(quiet=True, server_port=config.gradio_interface_port, prevent_thread_lock=True)
    else:
        print("[AllTalk MEM] Skipping gradio start...")
    # Start the API server
    start_api_server()
    initialize_instances()
    auto_start_engines()
    try:
        # Main loop
        while not should_exit.is_set():
            time.sleep(0.1)  # Short sleep to prevent high CPU usage
    except KeyboardInterrupt:
        print("[AllTalk MEM] Keyboard interrupt received. Shutting down...")
        should_exit.set()
    except Exception as e:
        print(f"[AllTalk MEM] An error occurred: {e}")
    finally:
        graceful_shutdown()
        sys.exit(0)