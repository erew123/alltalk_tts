import platform
import subprocess
import logging
import torch
import os
import re
import sys
import glob
import site
import textwrap
import threading
import json
import traceback
import signal
from pathlib import Path
from datetime import datetime
from importlib.metadata import version as get_version, PackageNotFoundError
from packaging import version
if platform.system() == "Windows":
    import winreg

# Check and install psutil if necessary
try:
    import psutil
except ImportError:
    print("psutil not found. Installing...")
    subprocess.run(['pip', 'install', 'psutil'])
    import psutil

# Check and install importlib_metadata if necessary
try:
    from importlib_metadata import distributions
except ImportError:
    print("importlib_metadata not found. Installing...")
    subprocess.run(['pip', 'install', 'importlib_metadata'])
    from importlib_metadata import distributions

# Check and install packaging if necessary
try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import parse as parse_version
except ImportError:
    print("packaging not found. Installing...")
    subprocess.run(['pip', 'install', 'packaging'])
    from packaging.specifiers import SpecifierSet
    from packaging.version import parse as parse_version

try:
    from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
                                 QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QFrame)
    from PyQt6.QtGui import QColor, QDropEvent, QDragEnterEvent, QFont
    from PyQt6.QtCore import Qt, QTimer, QCoreApplication
except ImportError:
    print("PyQt6 not found. Installing...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'PyQt6'])
    from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
                                 QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QFrame)
    from PyQt6.QtGui import QColor, QDropEvent, QDragEnterEvent, QFont
    from PyQt6.QtCore import Qt, QTimer, QCoreApplication


def input_with_timeout(prompt, timeout):
    def input_thread(result):
        try:
            result.append(input(prompt))
        except EOFError:  # Handles the situation where input is interrupted
            result.append(None)

    result = []
    thread = threading.Thread(target=input_thread, args=(result,))
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("\n\n    No input received. Using default 'requirements_standalone.txt'.")
        return None
    else:
        return result[0]

def get_requirements_file():
    this_dir = Path(__file__).parent
    requirements_dir = this_dir / 'system' / 'requirements'
    requirements_files = list(requirements_dir.glob('requirements*.txt'))

    if not requirements_files:
        print("\033[91mNo requirements files found.\033[0m")
        return None

    print("\033[94m\n  Select a requirements file to check against.")
    print("\033[94m  Or press Enter for default 'requirements_standalone.txt'):\033[0m\n")
    for i, file_path in enumerate(requirements_files, start=1):
        print(f"  {i}. {file_path.name}")
    print("\033[94m\n  After 20 seconds, requirements_standalone.txt will be auto selected.\033[0m\n")
    choice = input_with_timeout("\n  Enter the number of your choice: ", 20)
    
    try:
        if choice is None:
            raise ValueError("No input; using default.")
        choice = int(choice) - 1
        return str(requirements_files[choice])
    except (ValueError, IndexError):
        return str(requirements_dir / "requirements_standalone.txt")

def check_visual_cpp_build_tools():
    possible_locations = [
        "C:\\Program Files (x86)\\Microsoft Visual Studio",
        "C:\\Program Files\\Microsoft Visual Studio",
    ]
    
    found_tools = []
    
    for base_path in possible_locations:
        if os.path.exists(base_path):
            for year in ["2022", "2019", "2017", "2015"]:
                vs_path = os.path.join(base_path, year)
                if os.path.exists(vs_path):
                    # Check for Build Tools
                    bt_path = os.path.join(vs_path, "BuildTools")
                    if os.path.exists(bt_path):
                        vcvars_path = os.path.join(bt_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
                        if os.path.exists(vcvars_path):
                            found_tools.append(f"Visual Studio {year} Build Tools (Path: {bt_path})")
                    
                    # Check for full Visual Studio installation
                    common_path = os.path.join(vs_path, "Common7", "IDE", "devenv.exe")
                    if os.path.exists(common_path):
                        found_tools.append(f"Visual Studio {year} (Full) (Path: {vs_path})")
    
    return found_tools

def check_windows_sdk():
    sdk_paths = [
        r"C:\Program Files (x86)\Windows Kits\10",
        r"C:\Program Files\Windows Kits\10",
    ]
    
    found_sdks = []
    for path in sdk_paths:
        if os.path.exists(path):
            include_path = os.path.join(path, "Include")
            if os.path.exists(include_path):
                sdk_versions = [d for d in os.listdir(include_path) if os.path.isdir(os.path.join(include_path, d))]
                for sdk_version in sdk_versions:
                    found_sdks.append((sdk_version, path))
    
    return found_sdks

def check_windows_version():
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
        build = winreg.QueryValueEx(key, "CurrentBuildNumber")[0]
        if int(build) >= 22000:
            return "Windows 11"
        else:
            return "Windows 10"
    except WindowsError:
        return "Unknown Windows version"

def check_setuptools_version():
    try:
        setuptools_version = get_version("setuptools")
        return setuptools_version
    except PackageNotFoundError:
        return None

def check_espeak_ng():
    try:
        result = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True)
        return result.stdout.strip()
    except FileNotFoundError:
        return None

def setup_logging():
    log_file = 'diagnostics.log'
    
    # Close existing handlers
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    # Delete the existing log file if it exists
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            print(f"  Existing {log_file} has been deleted.")
        except Exception as e:
            print(f"  Error deleting existing {log_file}: {e}")
            return  # Exit the function if we can't delete the file
    else:
        print(f"  Creating new {log_file} file.\n\n")

    # Set up logging
    try:
        logging.basicConfig(filename=log_file,
                            filemode='w',
                            level=logging.INFO,
                            format='%(message)s')

        logging.info(f"Diagnostic log created at {datetime.now()}")
        logging.info("="*50 + "\n")
        print(f"  New {log_file} has been set up successfully.\n")
    except Exception as e:
        print(f"  Error setting up logging: {e}")
    print(f"  Gathering System info....")        

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        return result.stdout
    except FileNotFoundError:
        return "NVIDIA GPU information not available"

def get_cpu_info():
    cpu_info = {
        'physical_cores': psutil.cpu_count(logical=False),
        'total_cores': psutil.cpu_count(logical=True),
        'max_frequency': psutil.cpu_freq().max
    }
    return cpu_info

def get_disk_info():
    disk_info = []
    partitions = psutil.disk_partitions(all=True)
    for p in partitions:
        try:
            usage = psutil.disk_usage(p.mountpoint)
            disk_info.append(
                f"Drive: {p.device} | Total: {usage.total / (1024 ** 3):.2f} GB | Used: {usage.used / (1024 ** 3):.2f} GB | Free: {usage.free / (1024 ** 3):.2f} GB | Type: {p.fstype}"
            )
        except Exception as e:
            # Handle various types of errors
            error_type = type(e).__name__
            error_message = str(e)
            
            if isinstance(e, PermissionError):
                status = "Inaccessible or locked (Permission denied)"
            elif isinstance(e, OSError):
                if e.winerror == 1117:  # WinError 1117: I/O device error
                    status = "I/O device error"
                else:
                    status = f"OS Error: {error_message}"
            else:
                status = f"Error: {error_type} - {error_message}"
            
            disk_info.append(f"Drive: {p.device} | Status: {status} | Type: {p.fstype}")
    
    return disk_info

def is_port_in_use(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False

def satisfies_wildcard(installed_version, required_version):
    if '*' in required_version:
        required_parts = required_version.split('.')
        installed_parts = installed_version.split('.')
        for req, inst in zip(required_parts, installed_parts):
            if req != '*' and req != inst:
                return False
        return True
    return False

def test_cuda():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            torch.tensor([1.0, 2.0]).cuda()
            return "Success - CUDA is available and working."
        except Exception as e:
            return f"Fail - CUDA is available but not working. Error: {e}"
    else:
        return "Fail - CUDA is not available."
   
def get_cuda_details():
    try:
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_id)
            device_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
            cuda_version = torch.version.cuda
            return device_name, device_memory, cuda_version
        else:
            return None, None, None
    except Exception as e:
        logging.warning(f"Error getting CUDA details: {e}")
        return None, None, None

def find_files_in_path_with_wildcard(pattern):
    site_packages_path = site.getsitepackages()
    found_paths = []
    sub_directory = "nvidia/cublas"
    if platform.system() == "Linux":
        sub_directory = os.path.join(sub_directory, "lib")
    else:
        sub_directory = os.path.join(sub_directory, "bin")
    for directory in site_packages_path:
        search_directory = os.path.join(directory, sub_directory)
        for file_path in glob.glob(os.path.join(search_directory, pattern)):
            if os.path.isfile(file_path):
                found_paths.append(file_path)
    return found_paths

def get_conda_info():
    conda_exe = os.environ.get('CONDA_EXE')
    if not conda_exe:
        return "CONDA_EXE environment variable not found. Conda might not be installed or properly set up."

    conda_info = {}
    try:
        # Get general conda info
        result = subprocess.run([conda_exe, 'info', '--json'], capture_output=True, text=True)
        if result.returncode == 0:
            conda_info = json.loads(result.stdout)
        else:
            return f"Error running conda info: {result.stderr}"

        # Get list of packages in current environment
        result = subprocess.run([conda_exe, 'list', '--json'], capture_output=True, text=True)
        if result.returncode == 0:
            packages = json.loads(result.stdout)
        else:
            return f"Error getting package list: {result.stderr}"

    except Exception as e:
        return f"Error executing conda: {str(e)}"

    # Extract relevant information
    conda_version = conda_info.get('conda_version', 'Unknown')
    current_env = conda_info.get('active_prefix', 'No active environment')
    env_list = conda_info.get('envs', [])

    # Process package information
    package_info = {
        'conda-forge': {},
        'pkgs/main': {},
        'pkgs/msys2': {},
        'other': {}
    }
    for package in packages:
        name = package.get('name', 'Unknown')
        version = package.get('version', 'Unknown')
        channel = package.get('channel', 'Unknown')
        
        # Skip pypi (pip-installed) packages
        if channel == 'pypi':
            continue
        
        # Categorize packages
        if 'conda-forge' in channel:
            package_info['conda-forge'][name] = {'version': version, 'channel': channel}
        elif channel == 'pkgs/main':
            package_info['pkgs/main'][name] = {'version': version, 'channel': channel}
        elif channel == 'pkgs/msys2':
            package_info['pkgs/msys2'][name] = {'version': version, 'channel': channel}
        elif 'nvidia' not in channel.lower():  # Exclude NVIDIA packages
            package_info['other'][name] = {'version': version, 'channel': channel}

    return {
        'version': conda_version,
        'current_env': current_env,
        'env_list': env_list,
        'conda_exe': conda_exe,
        'packages': package_info
    }

def log_system_info():
    os_version = platform.system() + " " + platform.version()
    cuda_home = os.environ.get('CUDA_HOME', 'N/A')
    gpu_info = get_gpu_info()
    python_version = platform.python_version()
    conda_info = get_conda_info()
    torch_version = torch.__version__
    cuda_test_result = test_cuda()
    cuda_device_name, cuda_device_memory, cuda_version = get_cuda_details()
    cpu_info = get_cpu_info()
    disk_info = get_disk_info()

    try:
        virtual_memory = psutil.virtual_memory()
        total_ram = f"{virtual_memory.total / (1024 ** 3):.2f} GB"
        available_ram = f"{virtual_memory.available / (1024 ** 3):.2f} GB"
        system_ram = f"{available_ram} available out of {total_ram} total"
    except NameError:
        print("psutil is not installed. Unable to check system RAM.")
        system_ram = "N/A"

    port_status = "N/A"
    if 'psutil' in globals():
        port_to_check = 7851
        if is_port_in_use(port_to_check):
            port_status = f"Port {port_to_check} is in use."
        else:
            port_status = f"Port {port_to_check} is available."

    package_versions = {d.metadata['Name']: d.version for d in distributions()}
    python_executable = sys.executable
    python_version_info = sys.version_info
    python_virtual_env = os.environ.get('VIRTUAL_ENV', 'N/A')
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'N/A')
    search_path = sys.path
    path_env = os.environ.get('PATH', 'N/A')
    file_name = 'cublas64_11.*' if platform.system() == "Windows" else 'libcublas.so.11*'
    found_paths = find_files_in_path_with_wildcard(file_name)
    
    
    requirements_file = get_requirements_file()
    
    if requirements_file:
        required_packages = {}
        installed_packages = {}
        try:
            with open(requirements_file, 'r') as req_file:
                requirements = [line.strip() for line in req_file]
                for req in requirements:
                    if ';' in req:
                        continue  # Skip OS-specific requirements
                    match = re.match(r'([^\s>=]+)\s*([>=<]+)\s*([^,]+)', req)
                    if match:
                        package_name, operator, version_spec = match.groups()
                        installed_version = package_versions.get(package_name, 'Not installed')
                        if installed_version != 'Not installed':
                            required_packages[package_name] = (operator, version_spec)
                            installed_packages[package_name] = installed_version
        except FileNotFoundError:
            print(f"\n{requirements_file} not found. Skipping version checks.")
            logging.info(f"NOTE {requirements_file} not found. Skipping version checks.")

    logging.info(f"OPERATING SYSTEM:")
    logging.info(f" OS Version: {os_version}")
    logging.info(f" Note: Windows 11 will list as build is 10.x.22xxx")
    logging.info(f"\nHARDWARE ENVIRONMENT:")
    logging.info(f" CPU: Physical Cores: {cpu_info['physical_cores']}, Total Cores: {cpu_info['total_cores']}, Max Frequency: {cpu_info['max_frequency']} MHz")
    logging.info(f" System RAM: {system_ram}")
    logging.info(f"\nGPU INFORMATION:")
    logging.info(f" {gpu_info}")
    logging.info(f"CUDA:")
    logging.info(f" CUDA Working: {cuda_test_result}")
    logging.info(f" CUDA_HOME   : {cuda_home}")    
    logging.info(f" CUDA Device : {cuda_device_name if cuda_device_name else 'N/A'}")
    logging.info(f" CUDA Memory : {cuda_device_memory:.2f} GB" if cuda_device_memory else "N/A")
    logging.info(f" CUDA Version: {cuda_version if cuda_version else 'N/A'}")
    logging.info("\nDISK INFORMATION:")
    for disk in disk_info:
        logging.info(disk)
    logging.info("\nNETWORK PORT:")
    logging.info(f" Port Status : {port_status}")
    if found_paths:
        logging.info(f" Cublas64_11 Path: {', '.join(found_paths)}")
    else:
        logging.info(f" Cublas64_11 Path: Not found in any search path directories.")
        
    if platform.system() == "Windows":
        windows_version = check_windows_version()
        build_tools = check_visual_cpp_build_tools()
        sdks = check_windows_sdk()
        setuptools_version = check_setuptools_version()
        espeak_ng_version = check_espeak_ng()
        
        logging.info("\nWindows C++ Build tools & Windows SDK:")
        logging.info(f" Windows Version: {windows_version}")
        
        if build_tools:
            logging.info(" Visual C++ Build Tools and/or Visual Studio found:")
            for tool in build_tools:
                logging.info(f" {tool}")
        else:
            logging.info(" No Visual C++ Build Tools or Visual Studio found.")
        
        if sdks:
            logging.info("\nWindows SDK(s) found:")
            for sdk_version, path in sdks:
                logging.info(f" SDK Version: {sdk_version}")
                logging.info(f" Path: {path}")
        else:
            logging.info(" No Windows SDKs found.")
        
        if setuptools_version:
            logging.info(f" Python setuptools version: {setuptools_version}")
        else:
            logging.info(" Python setuptools not found.")
            
            
        logging.info("\nWindows Espeak-ng:")            
        if espeak_ng_version:
            logging.info(f" Espeak-ng version: {espeak_ng_version}")
        else:
            logging.info(" Espeak-ng not found.")      
           
    logging.info("\nPYTHON & PYTORCH:")
    logging.info(f" Torch Version: {torch_version}")
    logging.info(f" Python Version: {python_version}")
    logging.info(f" Python Version Info: {python_version_info}")
    logging.info(f" Python Executable: {python_executable}")
    logging.info(f" Python Virtual Environment: {python_virtual_env} (Should be N/A when in Text-generation-webui Conda Python environment)")
    logging.info(f" Conda Environment: {conda_env}")
    logging.info("\nPython Search Path:")
    for path in search_path:
        logging.info(f"  {path}")
    logging.info("\nOS SEARCHPATH ENVIRONMENT:")
    for path in path_env.split(';'):
        logging.info(f"  {path}")
    logging.info("\nCONDA INFORMATION:")
    if isinstance(conda_info, str):
        logging.info(f" {conda_info}")
    else:
        logging.info(f" Conda Executable: {conda_info['conda_exe']}")
        logging.info(f" Conda Version: {conda_info['version']}")
        logging.info(f" Current Environment: {conda_info['current_env']}")
        logging.info(" Conda Environments:")
        for env in conda_info['env_list']:
            logging.info(f"  {env}")
        
        logging.info("\nCONDA PACKAGES IN CURRENT ENVIRONMENT:")
        for category in ['conda-forge', 'pkgs/main', 'pkgs/msys2', 'other']:
            if conda_info['packages'][category]:
                logging.info(f"\n {category.upper()} PACKAGES:")
                for package, details in conda_info['packages'][category].items():
                    logging.info(f"  {package:<30} Version: {details['version']:<15} Channel: {details['channel']}")    
       
    if required_packages:
        logging.info("\nPACKAGE VERSIONS vs REQUIREMENTS FILE:")
        max_package_length = max(len(package) for package in required_packages.keys())
        for package_name, (operator, required_version) in required_packages.items():
            installed_version = installed_packages.get(package_name, 'Not installed')
            logging.info(f" {package_name.ljust(max_package_length)}  Required: {operator} {required_version.ljust(12)}  Installed: {installed_version}")
    
    logging.info("\nPYTHON PACKAGES:")
    package_versions = {d.metadata['Name']: d.version for d in distributions()}
    max_package_length = max(len(package) for package in package_versions.keys())
    for package, version in package_versions.items():
        logging.info(f" {package:<{max_package_length}} = {version}")

    deepspeed_requirements = (
        f"\nDeepSpeed Installation Requirements:\n"
        f"  OS     : {platform.system()}\n"
        f"  Python : {python_version}\n"
        f"  PyTorch: {torch_version}\n"
        f"  CUDA   : {'N/A' if not cuda_version else cuda_version}\n"
        f"\nTo install DeepSpeed, run the following command based on your environment:"
        f"\n  pip install deepspeed==<version> --no-cache-dir"
        f"\nEnsure to choose the correct DeepSpeed version matching your environment."
    )

    logging.info(deepspeed_requirements)

    print(f"\n\033[94mOS Version:\033[0m \033[92m{os_version}\033[0m")
    print(f"\033[94mOS Ver note:\033[0m \033[92m(Windows 11 will say build is 10.x.22xxx)\033[0m")
    print(f"\033[94mSystem RAM:\033[0m \033[92m{system_ram}\033[0m")
    for disk in disk_info:
        print(f"\033[94mDisk:\033[0m \033[92m{disk}\033[0m")
    print(f"\033[94m\nGPU Information:\033[0m {gpu_info}")
    print(f"\033[94mPort Status :\033[0m \033[92m{port_status}\033[0m")
    print(f"\033[0m\n  If this network port is unavailable because something else is using it, your")
    print(f"\033[0m  firewall or antivirus is blocking it, AllTalk will fail to start.")     
    print(f"\n\033[94mCUDA Device :\033[0m \033[92m{cuda_device_name if cuda_device_name else 'N/A'}\033[0m")
    print(f"\033[94mCUDA Memory :\033[0m \033[92m{cuda_device_memory:.2f} GB\033[0m" if cuda_device_memory else "\033[94mCUDA Memory :\033[0m \033[92mN/A\033[0m")
    print(f"\033[94mCUDA Version:\033[0m \033[92m{cuda_version if cuda_version else 'N/A'}\033[0m")
    if "Fail" in cuda_test_result:
        print(f"\033[91mCUDA Working:\033[0m \033[91m{cuda_test_result}\033[0m")
    else:
        print(f"\033[94mCUDA Working:\033[0m \033[92m{cuda_test_result}\033[0m")
    print(f"\033[94mCUDA_HOME   :\033[0m \033[92m{cuda_home}\033[0m")
    if found_paths:
        print(f"\033[94mCublas64_11 :\033[0m \033[92m{', '.join(found_paths)}\033[0m")
    else:
        print(f"\033[94mCublas64_11 :\033[0m \033[91mNot found in any search path directories.\033[0m")
    print(f"\033[0m\n  If you do not have a CUDA version and CUDA is failing, you will not have your")
    print(f"\033[0m  TTS engines being accelerated with CUDA. CUDA is only available on Nvidia GPU")
    print(f"\033[0m  and is setup by installing PyTorch with a correct CUDA version in your Python")
    print(f"\033[0m  virtual environment.")
    print(f"\033[94m\nPyTorch Version  :\033[0m \033[92m{torch_version}\033[0m")
    print(f"\033[94mPython Version   :\033[0m \033[92m{platform.python_version()}\033[0m")
    print(f"\033[94mPython Executable:\033[0m \033[92m{python_executable}\033[0m")
    print(f"\033[0m\n  AllTalk has been validated to run on Python 3.11.x versions and also PyTorch")
    print(f"\033[0m  2.0.x to 2.2.x. Earlier or later versions of PyTorch and Python may not work.")
    print(f"\033[94m\nConda Environment:\033[0m \033[92m{conda_env}\033[0m")
    print(f"\n\033[94mPython Search Path:\033[0m")
    for path in search_path:
        print(f"  {path}")
    print(f"\033[0m\n  If you are correctly in the AllTalk Python virtual environment, you will")
    print(f"\033[0m  expect to see 'alltalk_environment' as part of the path of the above folders.")
    print(f"\033[0m  If you are running AllTalk as part of Text-generation-webui, you should see ")
    print(f"\033[0m  'text-generation-webui' listed in the path of the above folders. If you dont")
    print(f"\033[0m  see them mentioned, you have probably not started the correct Python virtual")
    print(f"\033[0m  environment.")

    if platform.system() == "Windows":
        print(f"\n\033[94mWindows C++ Build tools & Windows SDK:\033[0m")
        print(f"\033[94m Windows Version:\033[0m \033[92m{windows_version}\033[0m")
        if sdks:
            print("\033[94m Windows SDK:\033[0m \033[92mFound\033[0m")
        else:
            print("\033[94mWindows SDK:\033[0m \033[91mNot Found\033[0m")        
        if build_tools:
            print("\033[94m Visual C++ Build Tools:\033[0m \033[92mFound\033[0m")
        else:
            print("\033[94m Visual C++ Build Tools:\033[0m \033[91mNot Found\033[0m")
        print(f"\033[94m Python setuptools version:\033[0m \033[92m{setuptools_version if setuptools_version else 'Not Found'}\033[0m")
        print(f"\n\033[94mWindows Espeak-ng:\033[0m")
        print(f"\033[94m Espeak-ng:\033[0m \033[92m{'Installed' if espeak_ng_version else 'Not Found'}\033[0m")    
    
    print(f"\n\033[94mConda Information:\033[0m")
    if isinstance(conda_info, str):
        print(f"\033[92m{conda_info}\033[0m")
    else:
        print(f"\033[94m Conda Executable:\033[0m \033[92m{conda_info['conda_exe']}\033[0m")
        print(f"\033[94m Conda Version:\033[0m \033[92m{conda_info['version']}\033[0m")
        print(f"\033[94m Current Environment:\033[0m \033[92m{conda_info['current_env']}\033[0m")
        
        # Check for faiss-cpu in conda-forge and pytorch, and ffmpeg in conda-forge
        print(f"\n\033[94mKey Conda Packages:\033[0m")
        all_conda_packages = {
            **conda_info['packages']['conda-forge'],
            **conda_info['packages']['other']
        }
        
        # Check for faiss-cpu
        if 'faiss-cpu' in all_conda_packages:
            details = all_conda_packages['faiss-cpu']
            channel = "conda-forge" if "conda-forge" in details['channel'] else "pytorch" if "pytorch" in details['channel'] else details['channel']
            print(f"  \033[92mfaiss-cpu\033[0m Version: \033[93m{details['version']:<15}\033[0m Channel: \033[94m{channel}\033[0m")
        else:
            print(f"  \033[91mfaiss-cpu\033[0m Not installed via conda")
        
        # Check for ffmpeg
        if 'ffmpeg' in conda_info['packages']['conda-forge']:
            details = conda_info['packages']['conda-forge']['ffmpeg']
            print(f"  \033[92mffmpeg   \033[0m Version: \033[93m{details['version']:<15}\033[0m Channel: \033[94mconda-forge\033[0m")
        else:
            print(f"  \033[91mffmpeg   \033[0m Not installed via conda-forge")
            
            
            
    if required_packages:
        print("\033[94m\nRequirements file package comparison:\033[0m")
        max_package_length = max(len(package) for package in required_packages.keys())

        for package_name, (operator, required_version) in required_packages.items():
            installed_version = installed_packages.get(package_name, 'Not installed')
            required_version_no_build = required_version.split("+")[0]

            if '*' in required_version:
                condition_met = satisfies_wildcard(installed_version, required_version)
            else:
                required_specifier = SpecifierSet(f"{operator}{required_version_no_build}")
                installed_version = parse_version(installed_version)
                condition_met = installed_version in required_specifier

            color_required = "\033[92m" if condition_met else "\033[91m"
            color_installed = "\033[92m" if condition_met else "\033[91m"

            print(f"  {package_name.ljust(max_package_length)}  Required: {color_required}{operator} {required_version.ljust(12)}\033[0m  Installed: {color_installed}{installed_version}\033[0m")

        print("\033[94m\nRequirements file specifier meanings:\033[0m")
        explanation = textwrap.dedent("""
           == Exact version             != Any version except         < Less than
           <= Less than or equal to     >  Greater than               >= Greater/equal to
           ~ Compatible release         ;  Environment marker         AND Logical AND
           OR Logical OR
        """)
        print(explanation.strip())

    print(f"\033[94m\nDeepSpeed Installation Requirements:\033[0m")
    print("\n  DeepSpeed is HIGHLY specific. It MUST be compiled for the correct major")
    print("  revision of Python (e.g. 3.11.x), PyTorch (e.g. 2.2.x), and the CUDA version")
    print("  that your PyTorch is using (e.g. 11.8 or 12.1). If any of these three")
    print("  components change version number (e.g. you update to PyTorch 2.3.x or Python")
    print("  3.12.x), you will need to uninstall DeepSpeed and install a build that matches")
    print("  your new Python environment.")
    print("\n           Here are the details of your current Python environment:")
    print(f"           OS: \033[92m{platform.system()}\033[0m Python: \033[92m{python_version}\033[0m PyTorch: \033[92m{torch_version}\033[0m CUDA: \033[92m{'N/A' if not cuda_version else cuda_version}\033[0m")
    print("\n  Therefore, you would need a DeepSpeed build for the above configuration. If")
    print("  you change ANY of the versions, remember to reinstall DeepSpeed to ensure")
    print("  compatibility with your updated environment. If the copy of DeepSpeed you have")
    print("  is NOT built to match the above, it will error/fail/crash etc.\n\n")
    print(f"\033[94mA diagnostic log file called \033[92mdiagnostics.log\033[94m has been created in the AllTalk")
    print(f"\033[94mfolder. Up above, on SCREEN you can read a few sections to help you diagnose\033[0m")
    print(f"\033[94many possible problems with your configuration or setup. So please take the\033[0m")
    print(f"\033[94mtime to check them. \033[0m")
    print(f"\033[94m\nIf you are going to ask for support, please upload the \033[92mdiagnostics.log\033[94m file.\033[0m")



class DragDropTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            with open(file_path, 'r') as file:
                self.setText(file.read())

class PackageComparisonTool(QWidget):
    def __init__(self):
        super().__init__()
        try:
            self.initUI()
        except Exception as e:
            print(f"Error in PackageComparisonTool initialization: {str(e)}")
            traceback.print_exc()
            
    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        main_layout = QVBoxLayout()  # Main layout is vertical

        # Set default font and styles for the entire widget
        self.setStyleSheet("""
            QWidget {
                font-size: 10pt;
            }
            QLabel {
                font-weight: bold;
            }
            QPushButton {
                font-size: 10pt;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton[accessibleName="browse"] {
                font-weight: normal;
            }
        """)

        # Instructions
        instructions = QLabel(
            "<b>INSTRUCTIONS:</b><br><br>"
            "1. Base Diagnostics File: Load the base diagnostics log file (e.g. from a working setup). Try `/system/config/basediagnostics.log`<br>"
            "2. Comparison Diagnostics File: Load your current diagnostics log file<br>"
            "3. Click the 'Compare' button to see differences<br>"
            "4. Review results in the table and organized results<br>"
            "5. Use 'Copy Commands' to copy pip commands to manually use at the Python command prompt.<br>"
            "6. Optionally, use 'Run Pip Commands' to apply changes (requires AllTalk's Python environment is activated). You can watch the installation at the terminal/command prompt.<br>"
            "7. Windows users, if your Windows C++ tools, Windows SDK or Espeak-ng are not installed, please re-read the installation requirements on Github.<br>"
        )
        instructions.setWordWrap(True)
        main_layout.addWidget(instructions)

        # Windows-specific information
        self.windows_info_widget = QWidget()
        self.windows_info_layout = QHBoxLayout(self.windows_info_widget)
        self.cpp_build_tools_label = QLabel("C++ Build Tools: <span style='color: gray;'>Checking...</span>")
        self.sdk_label = QLabel("Windows SDK: <span style='color: gray;'>Checking...</span>")
        self.espeak_ng_label = QLabel("Espeak-ng: <span style='color: gray;'>Checking...</span>")
        
        self.windows_info_layout.addWidget(self.cpp_build_tools_label)
        self.windows_info_layout.addWidget(self.sdk_label)
        self.windows_info_layout.addWidget(self.espeak_ng_label)
        
        main_layout.addWidget(self.windows_info_widget)

        # File inputs
        file_layout = QHBoxLayout()
        
        # Base file input
        base_layout = QVBoxLayout()
        base_label = QLabel("Base Diagnostics File (Drag&Drop or Browse):")
        self.base_text = DragDropTextEdit()
        base_browse = QPushButton("Browse for a Base Diagnostic log file")
        base_browse.setAccessibleName("browse")
        base_browse.clicked.connect(lambda: self.browse_file(self.base_text))
        base_layout.addWidget(base_label)
        base_layout.addWidget(self.base_text)
        base_layout.addWidget(base_browse)

        # Compare file input
        compare_layout = QVBoxLayout()
        compare_label = QLabel("Comparison Diagnostics File (Drag&Drop or Browse):")
        self.compare_text = DragDropTextEdit()
        compare_browse = QPushButton("Browse for your Comparison Diagnostic log file")
        compare_browse.setAccessibleName("browse")
        compare_browse.clicked.connect(lambda: self.browse_file(self.compare_text))
        compare_layout.addWidget(compare_label)
        compare_layout.addWidget(self.compare_text)
        compare_layout.addWidget(compare_browse)

        file_layout.addLayout(base_layout)
        file_layout.addLayout(compare_layout)
        
        main_layout.addLayout(file_layout)

        # Compare button
        compare_button = QPushButton("Compare the Base and Comparison log files")
        compare_button.clicked.connect(self.compare_packages)
        main_layout.addWidget(compare_button)
        main_layout.addSpacing(20)

        # Results table
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["Package", "Base Version", "Comparison Version"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        main_layout.addWidget(self.result_table)
        main_layout.addSpacing(20)

        lower_section_layout = QHBoxLayout()

        # Pip commands box
        pip_layout = QVBoxLayout()
        pip_label = QLabel("Pip commands to align versions:")
        pip_layout.addWidget(pip_label)
        
        self.pip_commands = QTextEdit()
        self.pip_commands.setReadOnly(True)
        pip_layout.addWidget(self.pip_commands)
        
        pip_buttons_layout = QHBoxLayout()
        copy_button = QPushButton("Copy Commands")
        copy_button.clicked.connect(self.copy_pip_commands)
        run_pip_button = QPushButton("Run Pip Commands")
        run_pip_button.clicked.connect(self.run_pip_commands)
        pip_buttons_layout.addWidget(copy_button)
        pip_buttons_layout.addWidget(run_pip_button)
        
        pip_layout.addLayout(pip_buttons_layout)
        lower_section_layout.addLayout(pip_layout)

        # Add some spacing between the two sections
        lower_section_layout.addSpacing(20)

        # Organized results box
        org_layout = QVBoxLayout()
        org_label = QLabel("Organized Results:")
        org_layout.addWidget(org_label)
        
        self.org_results = QTextEdit()
        self.org_results.setReadOnly(True)
        org_layout.addWidget(self.org_results)
        
        copy_org_button = QPushButton("Copy Organized Results")
        copy_org_button.clicked.connect(self.copy_organized_results)
        org_layout.addWidget(copy_org_button)
        
        lower_section_layout.addLayout(org_layout)

        # Add the lower section layout to the main layout
        main_layout.addLayout(lower_section_layout)

        self.setLayout(main_layout)
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle('Advanced Package Comparison Tool')

        # Run the checks immediately
        self.update_windows_info_display()

    def browse_file(self, text_edit):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;Log Files (*.log);;All Files (*)")
        if file_name:
            try:
                # Check if file exists and is not empty
                if not os.path.exists(file_name):
                    raise FileNotFoundError(f"File does not exist: {file_name}")
                
                file_size = os.path.getsize(file_name)
                print(f"File size: {file_size} bytes")
                
                if file_size == 0:
                    print(f"Warning: File is empty: {file_name}")
                    return

                # Try different encoding options
                encodings = ['utf-8', 'utf-8-sig', 'ascii', 'latin-1']
                content = ""
                
                for encoding in encodings:
                    try:
                        with open(file_name, 'r', encoding=encoding) as file:
                            content = file.read()
                        print(f"File opened successfully with {encoding} encoding: {file_name}")
                        break
                    except UnicodeDecodeError:
                        print(f"Failed to open with {encoding} encoding, trying next...")
                
                if not content:
                    raise ValueError("Unable to read file content with any encoding")

                text_edit.setText(content)
                print(f"Content length: {len(content)} characters")
                
                # Print first few lines for debugging
                print("First few lines of content:")
                print("\n".join(content.split('\n')[:5]))

            except Exception as e:
                print(f"Error opening file {file_name}: {str(e)}")
                traceback.print_exc()

    def extract_packages(self, text):
        packages = {}
        try:
            start_index = text.find("PYTHON PACKAGES:")
            if start_index != -1:
                package_section = text[start_index:]
                lines = package_section.split('\n')[1:]  # Skip the "PYTHON PACKAGES:" line
                for line in lines:
                    if line.strip() == "":
                        break  # Stop at the first empty line after packages
                    parts = line.split('=')
                    if len(parts) == 2:
                        package_name = parts[0].strip()
                        version = parts[1].strip()
                        packages[package_name.lower()] = version

            # print(f"Extracted {len(packages)} packages")
        except Exception as e:
            print(f"Error extracting packages: {str(e)}")
            traceback.print_exc()

        return packages

    def compare_packages(self):
        base_text = self.base_text.toPlainText()
        compare_text = self.compare_text.toPlainText()
        base_packages = self.extract_packages(base_text)
        compare_packages = self.extract_packages(compare_text)

        self.result_table.setRowCount(0)
        pip_commands = []
        missing = []
        different = []
        matching = []
        additional = []

        all_packages = sorted(set(base_packages.keys()) | set(compare_packages.keys()), key=str.lower)

        for package_lower in all_packages:
            row_position = self.result_table.rowCount()
            self.result_table.insertRow(row_position)

            # Find the original capitalization
            package = next((p for p in base_packages.keys() if p.lower() == package_lower), 
                           next((p for p in compare_packages.keys() if p.lower() == package_lower), package_lower))

            base_version = base_packages.get(package_lower, "N/A")
            compare_version = compare_packages.get(package_lower, "N/A")

            self.result_table.setItem(row_position, 0, QTableWidgetItem(package))
            self.result_table.setItem(row_position, 1, QTableWidgetItem(base_version))
            self.result_table.setItem(row_position, 2, QTableWidgetItem(compare_version))

            if base_version == compare_version:
                for col in range(3):
                    self.result_table.item(row_position, col).setBackground(QColor("lightgreen"))
                matching.append(f"{package}: {base_version}")
            elif base_version == "N/A":
                for col in range(3):
                    self.result_table.item(row_position, col).setBackground(QColor("lightblue"))
                additional.append(f"{package}: {compare_version}")
            elif compare_version == "N/A":
                for col in range(3):
                    self.result_table.item(row_position, col).setBackground(QColor("pink"))
                pip_commands.append(f"pip install {package}=={base_version}")
                missing.append(f"{package}: {base_version}")
            else:
                for col in range(3):
                    self.result_table.item(row_position, col).setBackground(QColor("yellow"))
                pip_commands.append(f"pip install {package}=={base_version}")
                different.append(f"{package}: {base_version} (Base) vs {compare_version} (Compare)")

        self.pip_commands.setText("\n".join(pip_commands))

        organized_results = "Missing in Comparison (present in Base):\n" + "\n".join(sorted(missing)) + "\n\n"  
        organized_results += "Different Versions:\n" + "\n".join(sorted(different)) + "\n\n"
        organized_results += "Matching Versions:\n" + "\n".join(sorted(matching)) + "\n\n"
        organized_results += "Additional in Comparison (not in Base):\n" + "\n".join(sorted(additional))                         
        self.org_results.setText(organized_results)       

    def update_windows_info_display(self):
        if platform.system() == "Windows":
            build_tools = check_visual_cpp_build_tools()
            sdks = check_windows_sdk()
            espeak_ng_version = check_espeak_ng()

            self.update_label(self.cpp_build_tools_label, "C++ Build Tools", 'Found' if build_tools else 'Not Found')
            self.update_label(self.sdk_label, "Windows SDK", 'Found' if sdks else 'Not Found')
            self.update_label(self.espeak_ng_label, "Espeak-ng", 'Installed' if espeak_ng_version else 'Not Found')
            self.windows_info_widget.show()
        else:
            self.windows_info_widget.hide()

    def update_label(self, label, title, status):
        color = "green" if status in ['Found', 'Installed'] else "red"
        label.setText(f"{title}: <span style='color: {color};'>{status}</span>")

    def set_label_color(self, label, status):
        if status in ['Found', 'Installed']:
            label.setStyleSheet("color: green;")
        else:
            label.setStyleSheet("color: red;")  

    def copy_pip_commands(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.pip_commands.toPlainText())

    def copy_organized_results(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.org_results.toPlainText())

    def run_pip_commands(self):
        commands = self.pip_commands.toPlainText().split('\n')
        for command in commands:
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Successfully executed: {command}")
            except subprocess.CalledProcessError as e:
                print(f"Error executing {command}: {e}")        

def cleanup_logging():
    logging.shutdown()

# Global flag to indicate if the application should exit
should_exit = False

def signal_handler(signum, frame):
    global should_exit
    print("\nCtrl+C pressed. Closing the application.")
    should_exit = True

# Set up the signal handler at the global level
signal.signal(signal.SIGINT, signal_handler)

def clear_screen():
    # Detect the operating system and clear the terminal screen
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def is_running_in_gui():
    return os.environ.get('DISPLAY') is not None or os.environ.get('WAYLAND_DISPLAY') is not None

def show_linux_instructions():
    print("\n\033[93m  Linux System Detected\033[0m")
    print("  If the GUI wont start, you may need to install additional dependencies.")
    print("  Please run one of the following commands based on your distribution:\n")
    print("\033[96m  For Ubuntu, Debian, and derivatives:\033[0m")
    print("  sudo apt-get install libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0\n")
    print("\033[96m  For Fedora, CentOS, RHEL, and other RPM-based systems:\033[0m")
    print("  sudo dnf install libxcb libxkbcommon-x11 libxcb-xinerama libxcb-cursor")
    print("                             - or -")
    print("  sudo yum install libxcb libxkbcommon-x11 libxcb-xinerama libxcb-cursor\n")
    print("\033[93m  After installing these packages, please restart the diagnostic tool.\033[0m\n")

def show_menu():
    print("\n\n\n\033[94m  AllTalk Diagnostics Tool Menu\033[0m")
    print("\n  Ensure you have started AllTalk'a Python Environment with the start_environment file")
    print("  before running this tool. TGWUI users will use cmd_{your-os} from the TGWUI folder.")
    print("\n  1. \033[93mGenerate a \033[92mdiagnostics.log\033[93m file\033[0m")
    print("     This option creates a new detailed log file of your system's configuration,")
    print("     including OS, hardware, Python environment, and installed packages. Use this")
    print("     to troubleshoot issues or when seeking support. Old diagnostic.log files will")
    print("     be overwritten.")
    print("\n  2. \033[93mCompare two \033[92mdiagnostics.log\033[93m logs\033[0m")
    print("     This opens a graphical tool to compare two diagnostics.log files. Useful")
    print("     for identifying differences between working and non-working configurations")
    print("     or for comparing system setups.")
    print("\n  9. \033[91mExit\033[0m")
    print("     Close the diagnostics tool.")
    choice = input("\n  Enter your choice (1, 2 or exit with 9): ")
    return choice

def main():
    print(f"\n\033[94m      _    _ _ \033[1;35m_____     _ _     \033[0m  _____ _____ ____  ")
    print(f"\033[94m     / \  | | |\033[1;35m_   _|_ _| | | __ \033[0m |_   _|_   _/ ___| ")
    print(f"\033[94m    / _ \ | | |\033[1;35m | |/ _` | | |/ / \033[0m   | |   | | \___ \ ")
    print(f"\033[94m   / ___ \| | |\033[1;35m | | (_| | |   <  \033[0m   | |   | |  ___) |")
    print(f"\033[94m  /_/   \_\_|_|\033[1;35m |_|\__,_|_|_|\_\ \033[0m   |_|   |_| |____/ ")    

    # Check for GUI environment on non-Windows systems
    if platform.system() != 'Windows' and not is_running_in_gui():
        print("No graphical environment detected. Using offscreen rendering.")
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print(f"Failed to initialize QApplication: {e}")
        print("Attempting to use offscreen platform...")
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QApplication(sys.argv)

    # Use a timer to check for the exit flag
    timer = QTimer()
    timer.timeout.connect(lambda: check_exit(app))
    timer.start(100)  # Check every 100ms

    while not should_exit:
        choice = show_menu()
        if choice == '1':
            setup_logging() 
            log_system_info()
            print("\n     Diagnostics completed. Check the \033[92mdiagnostics.log\033[0m file for details.\n")
            input("  Press Enter to return to the main menu...")
            clear_screen()
        elif choice == '2':
            if platform.system() != "Windows":
                show_linux_instructions()
            try:
                print("\n  Attempting to open Diagnostics GUI Window...")
                ex = PackageComparisonTool()
                ex.show()
                print("  If you don't see a window, it might be hidden or there is a display issue.")
                print("  Press Ctrl+C to return to the menu if no window appears.")
                app.exec()
                print("  GUI window closed.")
            except Exception as e:
                print(f"Error initializing or running PackageComparisonTool: {str(e)}")
                traceback.print_exc()
            clear_screen()
        elif choice == '9':
            cleanup_logging()
            print("\n  Exiting...")
            break
        else:
            print("  Invalid choice. Please try again.")
            clear_screen()

    cleanup_logging()
    app.quit()

def check_exit(app):
    if should_exit:
        cleanup_logging()
        app.quit()

if __name__ == "__main__":
    main()