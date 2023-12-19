try:
    import platform
    import subprocess
    import logging
    from importlib_metadata import distribution, distributions
    import torch
    import os  # Import the os module
except ImportError as e:
    print(f"\033[91mError importing module: {e}\033[0m\n")
    print("\033[94mPlease ensure you started the Text-generation-webUI Python environment with either\033[0m")
    print("\033[92mcmd_linux.sh\033[0m, \033[92mcmd_windows.bat\033[0m, \033[92mcmd_macos.sh\033[0m, or \033[92mcmd_wsl.bat\033[0m")
    print("\033[94mand then try running the diagnostics again.\033[0m")
    exit(1)

try:
    import psutil
except ImportError:
    print("psutil not found. Installing...")
    subprocess.run(['pip', 'install', 'psutil'])
    import psutil

# Additional import for torch
import torch

# Function to get GPU information using subprocess
def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        return result.stdout
    except FileNotFoundError:
        return "NVIDIA GPU information not available"

# Function to check if a port is in use
def is_port_in_use(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False

# Function to log and print system information
def log_system_info():
    # System information
    os_version = platform.system() + " " + platform.version()
    
    # Get CUDA_HOME environment variable
    cuda_home = os.environ.get('CUDA_HOME', 'N/A')

    gpu_info = get_gpu_info()

    # Python environment information
    python_version = platform.python_version()

    # Torch version
    torch_version = torch.__version__

    # System RAM using psutil
    try:
        virtual_memory = psutil.virtual_memory()
        total_ram = f"{virtual_memory.total / (1024 ** 3):.2f} GB"
        available_ram = f"{virtual_memory.available / (1024 ** 3):.2f} GB"
        system_ram = f"{available_ram} available out of {total_ram} total"
    except NameError:
        print("psutil is not installed. Unable to check system RAM.")
        system_ram = "N/A"

    # Port check (if psutil is available)
    port_status = "N/A"
    if 'psutil' in globals():
        port_to_check = 7851
        if is_port_in_use(port_to_check):
            port_status = f"Port {port_to_check} is in use."
        else:
            port_status = f"Port {port_to_check} is available."

    # Package versions using importlib_metadata
    package_versions = {d.metadata['Name']: d.version for d in distributions()}

    # Log and print information
    logging.info(f"OS Version: {os_version}")
    logging.info(f"Note: Windows 11 does call itself 10")
    logging.info(f"Python Version: {python_version}")
    logging.info(f"Torch Version: {torch_version}")
    logging.info(f"System RAM: {system_ram}")
    logging.info(f"Total System RAM: {total_ram}")
    logging.info(f"Available System RAM: {available_ram}")
    logging.info(f"CUDA_HOME: {cuda_home}")
    logging.info(f"\nPort Status: {port_status}")
    logging.info(f"GPU Information:\n{gpu_info}")
    logging.info("Package Versions:")
    for package, version in package_versions.items():
        logging.info(f"{package}>= {version}")

    # Print to screen
    print(f"\033[94mOS Version:\033[0m \033[92m{os_version}\033[0m")
    print(f"\033[94mNote:\033[0m \033[92mWindows 11 does call itself 10\033[0m")
    print(f"\033[94mPython Version:\033[0m \033[92m{python_version}\033[0m")
    print(f"\033[94mTorch Version:\033[0m \033[92m{torch_version}\033[0m")
    print(f"\033[94mSystem RAM:\033[0m \033[92m{system_ram}\033[0m")
    print(f"\033[94mTotal System RAM:\033[0m \033[92m{total_ram}\033[0m")
    print(f"\033[94mAvailable System RAM:\033[0m \033[92m{available_ram}\033[0m")
    print(f"\033[94mCUDA_HOME:\033[0m \033[92m{cuda_home}\033[0m")
    print(f"\033[94mPort Status:\033[0m \033[92m{port_status}\033[0m")
    print("")
    print(f"GPU Information:{gpu_info}")
    print(f"\033[94mDiagnostic log created:\033[0m \033[92mdiagnostics.log\033[0m")

if __name__ == "__main__":
    log_system_info()
