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
from pathlib import Path

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
    print(f"\033[94m      _    _ _ \033[1;35m_____     _ _     \033[0m  _____ _____ ____  ")
    print(f"\033[94m     / \  | | |\033[1;35m_   _|_ _| | | __ \033[0m |_   _|_   _/ ___| ")
    print(f"\033[94m    / _ \ | | |\033[1;35m | |/ _` | | |/ / \033[0m   | |   | | \___ \ ")
    print(f"\033[94m   / ___ \| | |\033[1;35m | | (_| | |   <  \033[0m   | |   | |  ___) |")
    print(f"\033[94m  /_/   \_\_|_|\033[1;35m |_|\__,_|_|_|\_\ \033[0m   |_|   |_| |____/ ")
    print(f"")

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

logging.basicConfig(filename='diagnostics.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')

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
    partitions = psutil.disk_partitions()
    for p in partitions:
        usage = psutil.disk_usage(p.mountpoint)
        disk_info.append(f" Drive: {p.device} | Total: {usage.total / (1024 ** 3):.2f} GB | Used: {usage.used / (1024 ** 3):.2f} GB | Free: {usage.free / (1024 ** 3):.2f} GB | Type: {p.fstype}")
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

def log_system_info():
    os_version = platform.system() + " " + platform.version()
    cuda_home = os.environ.get('CUDA_HOME', 'N/A')
    gpu_info = get_gpu_info()
    python_version = platform.python_version()
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

    if required_packages:
        logging.info("\nPACKAGE VERSIONS vs REQUIREMENTS FILE:")
        max_package_length = max(len(package) for package in required_packages.keys())
        for package_name, (operator, required_version) in required_packages.items():
            installed_version = installed_packages.get(package_name, 'Not installed')
            logging.info(f" {package_name.ljust(max_package_length)}  Required: {operator} {required_version.ljust(12)}  Installed: {installed_version}")
    
    logging.info("\nPYTHON PACKAGES:")
    for package, version in package_versions.items():
        logging.info(f" {package}>= {version}")

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

if __name__ == "__main__":
    log_system_info()
    sys.exit(1)
