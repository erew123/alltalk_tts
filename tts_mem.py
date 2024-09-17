import subprocess
import threading
import signal
import sys
import os
import gradio as gr
import time
import requests
from requests.exceptions import RequestException
import socket
import json

# Add these at the beginning of your file, after other imports
CONFIG_FILE = "mem_config.json"

# Dictionary to store all processes
processes = {}
script_path = "tts_server.py"

# Global flag to indicate if the program should exit
should_exit = threading.Event()

def signal_handler(signum, frame):
    print("[AllTalk MEM] Interrupt received, shutting down...")
    should_exit.set()

# Default configuration
default_config = {
    "base_port": 7001,
    "auto_start_engines": 0,
    "max_instances": 8,
    "gradio_interface_port": 7500,
    "max_retries": 8,
    "initial_wait": 3,
    "backoff_factor": 1.2,
    "debug_mode": False
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            loaded_config = json.load(f)
            # Update loaded_config with any missing keys from default_config
            for key, value in default_config.items():
                if key not in loaded_config:
                    loaded_config[key] = value
            # Ensure debug_mode is a boolean
            loaded_config['debug_mode'] = bool(loaded_config['debug_mode'])
            return loaded_config
    return default_config.copy()

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

# Load configuration at the start of the script
config = load_config()
current_base_port = config['base_port']
max_instances = config['max_instances']

############################################
# START-UP # Display initial splash screen #
############################################
print(f"[AllTalk MEM]\033[94m     _    _ _ \033[1;35m_____     _ _     \033[0m  _____ _____ ____  ")
print(f"[AllTalk MEM]\033[94m    / \  | | |\033[1;35m_   _|_ _| | | __ \033[0m |_   _|_   _/ ___| ")
print(f"[AllTalk MEM]\033[94m   / _ \ | | |\033[1;35m | |/ _` | | |/ / \033[0m   | |   | | \___ \ ")
print(f"[AllTalk MEM]\033[94m  / ___ \| | |\033[1;35m | | (_| | |   <  \033[0m   | |   | |  ___) |")
print(f"[AllTalk MEM]\033[94m /_/   \_\_|_|\033[1;35m |_|\__,_|_|_|\_\ \033[0m   |_|   |_| |____/ ")
print(f"[AllTalk MEM]")
print(f"[AllTalk MEM] \033[94m            AllTalk Multi Engine Manager\033[00m")
print(f"[AllTalk MEM]")
print(f"[AllTalk MEM] \033[93m     MEM is not intended for production use and\033[00m")
print(f"[AllTalk MEM] \033[93m      there is NO support being offered on MEM\033[00m")
print(f"[AllTalk MEM]")
print(f"[AllTalk MEM] \033[94mGradio Light:\033[00m \033[92mhttp://127.0.0.1:{config['gradio_interface_port']}\033[00m")
print(f"[AllTalk MEM] \033[94mGradio Dark :\033[00m \033[92mhttp://127.0.0.1:{config['gradio_interface_port']}?__theme=dark\033[00m")
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
    wait_time = config['initial_wait']
    while retries < config['max_retries']:
        try:
            result = func()
            # print(f"[AllTalk MEM] Operation successful on attempt {retries + 1}")            
            return result
        except RequestException as e:
            retries += 1
            # print(f"[AllTalk MEM] Attempt {retries}")            
            if retries == config['max_retries']:
                # print(f"[AllTalk MEM] All {max_retries} attempts failed waiting for this instance of the Engine to load.")                
                return False
            # print(f"[AllTalk MEM] Retrying in {wait_time} seconds...")            
            time.sleep(wait_time)
            wait_time = round(wait_time * config['backoff_factor'], 1)
    return False

def is_server_ready(port, timeout=5):
    def check_ready():
        response = requests.get(f"http://127.0.0.1:{port}/api/ready", timeout=timeout)
        if response.text.strip() == "Ready":
            return True
        raise RequestException("Server not ready")

    return retry_with_backoff(check_ready)

def update_engine_selector():
    running_engines = []
    debug_info = []
    for i, p in processes.items():
        if p.poll() is None:
            port = current_base_port + i - 1
            ready = is_server_ready(port)
            status = 'Ready' if ready else 'Not Ready'
            debug_info.append(f"Engine {i} (Port {port}): {status}")
            if ready:
                running_engines.append(f"Engine {i} (Port {port})")
    
    debug_str = "\n".join(debug_info)
    print(f"Debug info: {debug_str}")  # Add this line for console debugging
    return gr.update(choices=running_engines, value=running_engines[0] if running_engines else None), debug_str

def stop_all_instances():
    instance_ids = list(processes.keys())
    results = ["❌ Not running"] * max_instances
    print(f"[AllTalk MEM] Terminating All TTS Engines")
    for instance_id in instance_ids:
        stop_subprocess(instance_id)
        results[instance_id-1] = "❌ Not running"
    return results

def update_all_statuses(base_port):
    return [check_subprocess_status(i, base_port + i - 1) for i in range(1, max_instances + 1)]

def count_running_instances():
    return sum(1 for p in processes.values() if p.poll() is None)

def get_available_voices(port, max_retries=3, delay=2):
    for _ in range(max_retries):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/api/voices", timeout=5)
            data = response.json()
            if data["status"] == "success":
                return data["voices"]
        except requests.RequestException as e:
            print(f"Error fetching voices from port {port}: {str(e)}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from port {port}")
        time.sleep(delay)
    return []

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
        port = current_base_port + engine_num - 1
        formatted_text = text.format(engine=engine_num, port=port, voice=voice)
        
        audio_url, debug_msg = generate_tts(port, engine_num, formatted_text, voice)
        if audio_url:
            return audio_url, debug_msg
        else:
            return None, debug_msg
    except Exception as e:
        return None, f"Error in test_tts: {str(e)}"

def create_gradio_interface():
    global current_base_port 
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
                num_instances_input = gr.Slider(minimum=1, maximum=max_instances, step=1, value=1, label="Number of Instances to Start")
                start_MEMple_button = gr.Button("Start Instances")
                stop_all_button = gr.Button("Stop All Instances")
                refresh_button = gr.Button("Refresh Status")
            error_box = gr.Textbox(label="Error Messages", visible=False)
            
            instance_statuses = []
            for i in range(0, max_instances, 8):  # Create rows with 8 instances each
                with gr.Row():
                    for j in range(8):
                        if i + j < max_instances:
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
                                start_btn.click(lambda id=instance_id: start_and_update(id, current_base_port + id - 1), outputs=status)
                                stop_btn.click(lambda id=instance_id: stop_and_update(id), outputs=status)
                                restart_btn.click(lambda id=instance_id: restart_and_update(id, current_base_port + id - 1), outputs=status)

                            instance_statuses.append(status)
            
            def refresh_and_update_slider():
                running_count = count_running_instances()
                statuses = update_all_statuses(current_base_port)
                return [gr.update(value=running_count)] + statuses

            refresh_button.click(
                refresh_and_update_slider,
                outputs=[num_instances_input] + instance_statuses
            )
                
            def validate_and_start(num_instances):
                if num_instances < 1:
                    return [gr.update(value=1)] + ["❌ Not running"] * max_instances + ["Invalid number of instances"]
                
                results = start_MEMple_instances(num_instances, current_base_port)
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
        
            def start_MEMple_instances(num_instances, base_port):
                currently_running = count_running_instances()
                
                if currently_running >= num_instances:
                    return update_all_statuses(base_port)  # All requested instances are already running
                
                instances_to_start = num_instances - currently_running
                start_port, available_ports = find_available_port(base_port + currently_running, instances_to_start)
                
                if len(available_ports) < instances_to_start:
                    return ["Not enough available ports"] * max_instances

                results = update_all_statuses(base_port)
                for i, port in enumerate(available_ports[:instances_to_start], start=currently_running+1):
                    result = start_subprocess(i, port)
                    results[i-1] = result
                    time.sleep(10)  # Increase wait time to 10 seconds
            
                return results     
            
            gr.Markdown("## Test TTS Engines")
            with gr.Row():
                engine_selector = gr.Dropdown(label="Select TTS Engine", choices=[], scale=1)
                voice_selector = gr.Dropdown(label="Select Voice", choices=[], scale=1)
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
                # Create a list of keys to iterate over
                process_keys = list(processes.keys())
                for i in process_keys:
                    p = processes.get(i)
                    if p and p.poll() is None:
                        port = current_base_port + i - 1
                        ready = is_server_ready(port)
                        status = 'Ready' if ready else 'Not Ready'
                        debug_info.append(f"Engine {i} (Port {port}): {status}")
                        if ready:
                            running_engines.append(f"Engine {i} (Port {port})")
                
                debug_str = "\n".join(debug_info)
                #print(f"Debug info: {debug_str}")  # un hash # line for console debugging
                return gr.update(choices=running_engines, value=running_engines[0] if running_engines else None), debug_str
            
            def update_voice_selector(engine):
                if engine:
                    engine_num = int(engine.split()[1])
                    port = current_base_port + engine_num - 1
                    voices = get_available_voices(port)
                    debug_str = f"Fetched voices for Engine {engine_num} (Port {port}): {voices}"
                    return gr.update(choices=voices, value=voices[0] if voices else None), debug_str
                return gr.update(choices=[], value=None), "No engine selected"
            
            # Update engine selector when instances are started/stopped
            start_MEMple_button.click(update_engine_selector, outputs=[engine_selector, debug_output])
            stop_all_button.click(update_engine_selector, outputs=[engine_selector, debug_output])
            
            # Add a manual refresh button for the engine selector
            refresh_engine_button.click(update_engine_selector, outputs=[engine_selector, debug_output])

            # Update voice selector when engine is selected
            engine_selector.change(update_voice_selector, inputs=[engine_selector], outputs=[voice_selector, debug_output])
            
            # Generate TTS when test button is clicked
            test_button.click(test_tts, inputs=[engine_selector, voice_selector, test_text], outputs=[audio_output, debug_output])
          
        with gr.Tab("MEM Settings"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Engine Instance Management")
                    with gr.Row():
                        base_port_input = gr.Number(
                            value=config['base_port'],
                            label=f"Starting Port Number (Current: {config['base_port']})",
                            step=1
                        )
                        max_instances_input = gr.Number(
                            value=config['max_instances'],
                            label=f"Maximum Engine Instances (Current: {config['max_instances']})",
                            step=1
                        )
                        auto_start_engines = gr.Number(
                            value=config['auto_start_engines'],
                            label="Auto-start Engine Instances (0 to disable)",
                            step=1
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Gradio Port & Debugging")                                      
                    with gr.Row():
                        gradio_port_input = gr.Number(
                            value=config['gradio_interface_port'],
                            label=f"Gradio Interface Port (Current: {config['gradio_interface_port']})",
                            step=1
                        )                
                        debug_mode_input = gr.Dropdown(
                            choices=["Disabled", "Enabled"],
                            value="Enabled" if config['debug_mode'] else "Disabled",
                            label="Debug Mode"
                        )
                                        
            with gr.Row():               
                with gr.Column(scale=2):
                    gr.Markdown("### Engine Start-up Time Contol")
                    with gr.Row():
                        max_retries_input = gr.Number(
                            value=config['max_retries'],
                            label=f"Max Retries (Current: {config['max_retries']})",
                            step=1
                        )
                        initial_wait_input = gr.Number(
                            value=config['initial_wait'],
                            label=f"Initial Wait (Current: {config['initial_wait']})",
                            step=0.1
                        )
                        backoff_factor_input = gr.Number(
                            value=config['backoff_factor'],
                            label=f"Backoff Factor (Current: {config['backoff_factor']})",
                            step=0.1
                        )
                with gr.Column(scale=1):
                    gr.Markdown("### Save Settings")
                    with gr.Row():                        
                        set_settings_button = gr.Button("Save Settings")
            with gr.Row():                
                settings_status = gr.Textbox(label="Settings Status")


            def update_settings(new_port, new_max_instances, new_auto_start, new_gradio_port, 
                                new_max_retries, new_initial_wait, new_backoff_factor, new_debug_mode):
                global current_base_port, max_instances, config, server_port
                
                if new_port < 1024:
                    new_port = 7001
                
                new_auto_start = min(int(new_auto_start), int(new_max_instances))
                
                current_base_port = int(new_port)
                max_instances = int(new_max_instances)
                server_port = int(new_gradio_port)
                
                config['base_port'] = current_base_port
                config['max_instances'] = max_instances
                config['auto_start_engines'] = new_auto_start
                config['gradio_interface_port'] = server_port
                config['max_retries'] = int(new_max_retries)
                config['initial_wait'] = float(new_initial_wait)
                config['backoff_factor'] = float(new_backoff_factor)
                config['debug_mode'] = (new_debug_mode == "Enabled")
                
                save_config(config)
                
                return (
                    f"Settings updated: Start port: {current_base_port}, Max instances: {max_instances}, "
                    f"Auto-start engines: {config['auto_start_engines']}, Gradio port: {server_port}, "
                    f"Max retries: {config['max_retries']}, Initial wait: {config['initial_wait']}, "
                    f"Backoff factor: {config['backoff_factor']}, Debug mode: {config['debug_mode']}",
                    gr.update(value=current_base_port, label=f"Starting Port Number (Current: {current_base_port})"),
                    gr.update(value=max_instances, label=f"Maximum Instances (Current: {max_instances})"),
                    gr.update(value=config['auto_start_engines']),
                    gr.update(value=server_port, label=f"Gradio Interface Port (Current: {server_port})"),
                    gr.update(value=config['max_retries']),
                    gr.update(value=config['initial_wait']),
                    gr.update(value=config['backoff_factor']),
                    gr.update(value="Enabled" if config['debug_mode'] else "Disabled")
                )

            set_settings_button.click(
                update_settings,
                inputs=[base_port_input, max_instances_input, auto_start_engines, gradio_port_input,
                        max_retries_input, initial_wait_input, backoff_factor_input, debug_mode_input],
                outputs=[settings_status, base_port_input, max_instances_input, auto_start_engines,
                         gradio_port_input, max_retries_input, initial_wait_input, backoff_factor_input,
                         debug_mode_input]
            )
              
            with gr.Row():
                gr.Markdown("""
                All settings are saved in the `mem_config.json` file in the same directory as the MEM script. If `mem_config.json` doesn't exist, default settings will be used. Settings take effect immediately after saving. Some settings (like Gradio Interface Port) may require a restart of MEM to take effect.
                """)
            with gr.Row():                                              
                with gr.Column():
                    gr.Markdown("""
                    ### 🟩 Starting Port Number
                    **Default: 7001**<br>
                    This is the base port number from which MEM starts assigning ports to TTS engine instances. Each subsequent instance will use the next available port number. Ensure this port and the following ones are free on your system.

                    ### 🟩 Maximum Instances
                    **Default: 8**<br>
                    The maximum number of TTS engine instances that can be run simultaneously. This limit helps prevent system overload. Adjust based on your system's capabilities and requirements.

                    ### 🟩 Auto-start Engines
                    **Default: 0 (disabled)**<br>
                    The number of TTS engine instances to automatically start when MEM launches. Set to 0 to disable auto-start. This number cannot exceed the Maximum Instances setting.

                    ### 🟩 Gradio Interface Port
                    **Default: 7500**<br>
                    The port on which the Gradio web interface for MEM will be accessible. Make sure this port is free and not blocked by your firewall. Gradio will bind to all IP address on your machine on this port, aka 0.0.0.0.
                    """)
                    
                with gr.Column():
                    gr.Markdown("""                
                    ### 🟩 Max Retries
                    **Default: 8**<br>
                    The maximum number of attempts MEM will make to connect to a TTS engine instance before considering it failed. This setting helps handle slow-starting engines.

                    ### 🟩 Initial Wait
                    **Default: 3 (seconds)**<br>
                    The initial waiting time between retry attempts (Max Retries) when connecting to a TTS engine instance. This delay helps prevent overwhelming the system with rapid reconnection attempts.

                    ### 🟩Backoff Factor
                    **Default: 1.2**<br>
                    A multiplier applied to the waiting time between retry attempts. With each failed attempt, the wait time (Initial Wait) is multiplied by this factor, implementing an exponential backoff strategy. This helps to reduce system load during persistent issues.

                    ### 🟩 Debug Mode
                    **Default: Disabled**<br>
                    When enabled, MEM will output additional diagnostic information. This is useful for troubleshooting issues but may increase console output significantly. Use this when you need detailed information about MEM's operations .
                    """)               
        
        with gr.Tab("MEM FAQ/Help"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                        ## What is MEM?
                        AllTalk MEM (Multi Engine Manager) is a research tool designed to manage and test multiple instances of different Text-to-Speech (TTS) engines being loaded simultaneously, with a view to a centralised engine being able to handle multiple requests simultaneously.

                        ## Current Status & Support
                        AllTalk MEM is currently in a demonstration, testing, and experimental phase. 
                        
                        ⚠️ MEM is not intended for production use at this time and there is **NO support being offered on MEM**. ⚠️ 

                        ## System Capacity
                        **Q: How many engines can my system (GPU/CPU) handle at once?**<br>
                        A: The exact number is unknown and will vary based on your specific hardware configuration and the TTS engines being loaded. Experimentation is required to determine the optimal number for your system. You would have to monitor your system resources and see what works. Dont forget you need to leave RAM/VRAM overhead for actual generation/inference.

                        ## Queue System
                        **Q: Is there an in-built queue system to handle different requests to different loaded engines?**<br>
                        A: Not currently. AllTalk MEM is presently a demonstration and research tool without an integrated queue system. You are welcome to build your own and test it out.

                        ## Handling Multiple Requests
                        **Q: How will different TTS engines from different manufacturers OR how will CUDA/Python handle multiple requests coming in at the same time for TTS generation?**<br>
                        A: The behavior in such scenarios is currently unknown and is part of the research aspect of this tool. Its highly possible some TTS engines may respond in a timely fashion and its also possible they may not. I cannot say how things like CUDA/Python will handle multiple requests to the same hadware. 
                        
                        CUDA has additional code and tools to deal with multiplexing requests and none of these have been implimented. So for example, if you are using a TTS engine that uses a GPU, and 2x different loaded TTS engines make a request of that hardware, you may get a 50/50 spit use of the hardware and you may not. 3x requests, you may get a 33/33/33 hardware load sharing sceanario nad you may not. I cannot say at the moment and havnt tested.
                        
                        Python - As each instance of each TTS engine will be loaded into a seperate Python instance, all memory management and requests should remain segregated from one another, so there should be no bleed over of tensors in CUDA requests etc, however, I have not tested.
                        
                        ## API Requests to each loaded engine
                        **Q: Can I use the AllTalk API to each engine instance?**<br>
                        A: Yes, each engine loaded will respond fully as if it was a standalone AllTalk instance. So you can use the standard API requests to the port number of each TTS engine loaded in.
  
                        """) 
                with gr.Column():                    
                    gr.Markdown("""                
                        ## What settings are used for the TTS engines loaded
                        **Q: What exact settings are being used by the TTS engines being loaded in?**<br>
                        A: Whatever you have set in the main AllTalk Gradio interface when you load AllTalk as a standalone, under the `TTS Engine Settings` and the specific TTS engine, those are the settings loaded/used.
                        
                        ## How can I change which TTS engine is loaded
                        **Q: How can you change the TTS engines being loaded?**<br>
                        A: Whatever you have set in the main AllTalk Gradio interface when you load AllTalk as a standalone, that will be the engine loaded in.
                        
                        ## What happens if it tries to load on a port in use
                        **Q: When a engine instance starts up, what happens if a port is already used by something else?**<br>
                        A: A test is performed of the port before starting each instance, so the engine trying to load on that port number wouldnt load in.                        

                        ## Future Updates
                        **Q: Will you be updating this in the future?**<br>
                        A: Updates may be made in the future, but this depends largely on available time and resources.

                        ## Contributions
                        **Q: Can I write some code for it and send a PR?**<br>
                        A: Yes, contributions are welcome! If you have improvements or features you'd like to add, feel free to submit a Pull Request.

                        ## Future Plans
                        If time permits, potential future developments include:<br>

                        - Potential control over each TTS engines's settings and manipulating the JSON files that control AllTalk, so as to handle which engine loads and its settings from the MEM interface.<br>
                        - Storing settings in a configuration JSON file (e.g., start port number, number of engines to start, automatic start of engines on script start)<br>
                        - Implementing an incoming queue management system to handle multiple requests and distribute them among loaded engines<br>
                        - Developing a basic API management suite for MEM<br>
                        - Refining and optimizing the existing logi for model loading etc<br>

                        Please note that these are potential plans and not guaranteed features.
                        
                        ## ⚠️ Support for MEM ⚠️
                        
                        ⚠️ MEM is not intended for production use at this time and there is **NO support being offered on MEM**. ⚠️ 
                        
                        ⚠️ Yes, I know there will be bugs. Yes I know there will be issues. This is not designed for Production Environments. ⚠️
                        """)
    return interface

def start_and_update(instance_id, port):
    if is_port_in_use(port):
        return f"Port {port} is already in use"
    result = start_subprocess(instance_id, port)
    return check_subprocess_status(instance_id, port)

def stop_and_update(instance_id):
    result = stop_subprocess(instance_id)
    return "❌ Not running"

def restart_and_update(instance_id, port):
    if is_port_in_use(port) and instance_id not in processes:
        return f"Port {port} is already in use"
    result = restart_subprocess(instance_id, port)
    return check_subprocess_status(instance_id, port)

def start_MEMple_instances(num_instances, base_port):
    currently_running = count_running_instances()
    
    if currently_running >= num_instances:
        return update_all_statuses(base_port)  # All requested instances are already running
    
    instances_to_start = num_instances - currently_running
    start_port, available_ports = find_available_port(base_port + currently_running, instances_to_start)
    
    if len(available_ports) < instances_to_start:
        return ["Not enough available ports"] * max_instances

    results = update_all_statuses(base_port)
    for i, port in enumerate(available_ports[:instances_to_start], start=currently_running+1):
        result = start_subprocess(i, port)
        results[i-1] = result
        time.sleep(10)  # Increase wait time to 10 seconds

    return results

print(f"[AllTalk MEM] Please use \033[91mCtrl+C\033[0m when exiting otherwise Python")
print(f"[AllTalk MEM] subprocess's will continue running in the background.")
print(f"[AllTalk MEM] ")
print(f"[AllTalk MEM] MEM Server Ready")

def launch_interface(interface):
    interface.launch(quiet=True, server_port=config['gradio_interface_port'], prevent_thread_lock=True)

def auto_start_engines():
    num_engines = config['auto_start_engines']
    if num_engines > 0:
        print(f"[AllTalk MEM] Auto-starting {num_engines} engines...")
        results = start_MEMple_instances(num_engines, current_base_port)
        print(f"[AllTalk MEM] Auto-start complete. Results: {results}")

def shutdown():
    print("[AllTalk MEM] Shutting down all engines...")
    stop_all_instances()
    print("[AllTalk MEM] All engines stopped.")

if __name__ == "__main__":
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    interface = create_gradio_interface()
    
    # Start the Gradio interface in a separate thread
    interface_thread = threading.Thread(target=launch_interface, args=(interface,))
    interface_thread.start()
    
    # Auto-start engines
    auto_start_engines()
    
    try:
        # Main loop
        while not should_exit.is_set():
            should_exit.wait(1)  # Wait for 1 second or until should_exit is set
    except Exception as e:
        print(f"[AllTalk MEM] An error occurred: {e}")
    finally:
        print("[AllTalk MEM] Initiating shutdown...")
        shutdown()
        # Give the interface thread a chance to close gracefully
        interface_thread.join(timeout=5)
        print("[AllTalk MEM] Shutdown complete.")
        sys.exit(0)