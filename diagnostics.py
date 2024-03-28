try:
    import platform
    import subprocess
    import logging
    from importlib_metadata import distributions
    import torch
    import os  # Import the os module
    import re
    import sys
    import glob
    import site
    import textwrap
    import torch
    import packaging.version
    import packaging.specifiers
    from packaging.specifiers import SpecifierSet
    from packaging.specifiers import SpecifierSet
    from packaging.version import parse as parse_version
    from pathlib import Path
except ImportError as e:
    print(f"\033[91mError importing module: {e}\033[0m\n")
    print("\033[94mPlease ensure you started the Text-generation-webUI Python environment with either\033[0m")
    print("\033[92mcmd_linux.sh\033[0m, \033[92mcmd_windows.bat\033[0m, \033[92mcmd_macos.sh\033[0m, or \033[92mcmd_wsl.bat\033[0m")
    print("\033[94mfrom the text-generation-webui directory, and then try running the diagnostics again.\033[0m")
    exit(1)

this_dir = Path(__file__).parent

try:
    import psutil
except ImportError:
    print("psutil not found. Installing...")
    subprocess.run(['pip', 'install', 'psutil'])
    import psutil

def get_requirements_file():
    this_dir = Path(__file__).parent  # Assuming 'this_dir' is defined as the script's directory
    requirements_dir = this_dir / 'system' / 'requirements'
    requirements_files = list(requirements_dir.glob('requirements*.txt'))  # Using pathlib for globbing
    
    if not requirements_files:
        print("\033[91mNo requirements files found.\033[0m")
        return None

    print("\033[94m\nSelect a requirements file to check against (or press Enter for default 'requirements.txt'):\033[0m\n")
    for i, file_path in enumerate(requirements_files, start=1):
        print(f"    {i}. {file_path.name}")  # Only print the filename

    choice = input("\nEnter the number of your choice: ")
    try:
        choice = int(choice) - 1
        return str(requirements_files[choice])  # Return the full path as a string
    except (ValueError, IndexError):
        return str(requirements_dir / "requirements.txt")  # Return the default path as a string

# Set up logging with filemode='w'
logging.basicConfig(filename='diagnostics.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')  # Custom format

# Function to get GPU information using subprocess
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

# Function to check if a port is in use
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
            # Attempt to create a tensor on GPU
            torch.tensor([1.0, 2.0]).cuda()
            return "Success - CUDA is available and working."
        except Exception as e:
            return f"Fail - CUDA is available but not working. Error: {e}"
    else:
        return "Fail - CUDA is not available."

def find_files_in_path_with_wildcard(pattern):
    # Get the site-packages directory of the current Python environment
    site_packages_path = site.getsitepackages()
    found_paths = []
    # Adjust the sub-directory based on the operating system
    sub_directory = "nvidia/cublas"
    if platform.system() == "Linux":
        sub_directory = os.path.join(sub_directory, "lib")
    else:
        sub_directory = os.path.join(sub_directory, "bin")
    # Iterate over each site-packages directory (there can be more than one)
    for directory in site_packages_path:
        # Construct the search directory path
        search_directory = os.path.join(directory, sub_directory)
        # Use glob to find all files matching the pattern in this directory
        for file_path in glob.glob(os.path.join(search_directory, pattern)):
            if os.path.isfile(file_path):  # Ensure it's a file
                found_paths.append(file_path)
    return found_paths

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
    cuda_test_result = test_cuda()
    cpu_info = get_cpu_info()
    disk_info = get_disk_info()

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

    # Python environment information
    python_executable = sys.executable
    python_version_info = sys.version_info
    python_virtual_env = os.environ.get('VIRTUAL_ENV', 'N/A')

    # Conda environment information
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'N/A')

    # Get Path environment information
    search_path = sys.path
    path_env = os.environ.get('PATH', 'N/A')

    # Check for cublas
    file_name = 'cublas64_11.*' if platform.system() == "Windows" else 'libcublas.so.11*'
    found_paths = find_files_in_path_with_wildcard(file_name)

    # Compare with requirements file
    requirements_file = get_requirements_file()
    if requirements_file:
        required_packages = {}
        installed_packages = {}

        try:
            with open(requirements_file, 'r') as req_file:
                requirements = [line.strip() for line in req_file]
                for req in requirements:
                    # Use regular expression to parse package name and version
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

    # Log and print information
    logging.info(f"OPERATING SYSTEM:")
    logging.info(f" OS Version: {os_version}")
    logging.info(f" Note: Windows 11 will list as build is 10.x.22xxx")
    logging.info(f"\nHARDWARE ENVIRONMENT:")
    logging.info(f" CPU: Physical Cores: {cpu_info['physical_cores']}, Total Cores: {cpu_info['total_cores']}, Max Frequency: {cpu_info['max_frequency']} MHz")
    logging.info(f" System RAM: {system_ram}")
    logging.info(f"\nGPU INFORMATION:{gpu_info}")
    logging.info("DISK INFORMATION:")
    for disk in disk_info:
        logging.info(disk)
    logging.info("\nNETWORK PORT:")
    logging.info(f" Port Status: {port_status}")
    logging.info("\nCUDA:")
    logging.info(f" CUDA Working: {cuda_test_result}")
    logging.info(f" CUDA_HOME: {cuda_home}")
    if found_paths:
        logging.info(f" Cublas64_11 Path: {', '.join(found_paths)}")
    else:
        logging.info(f" Cublas64_11 Path: Not found in any search path directories.")
    logging.info("\nPYTHON & PYTORCH:")
    logging.info(f" Torch Version: {torch_version}")
    logging.info(f" Python Version: {platform.python_version()}")
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
    if required_packages:  # Check if the dictionary is not empty
        logging.info("\nPACKAGE VERSIONS vs REQUIREMENTS FILE:")
        max_package_length = max(len(package) for package in required_packages.keys())
        for package_name, (operator, required_version) in required_packages.items():
            installed_version = installed_packages.get(package_name, 'Not installed')
            logging.info(f" {package_name.ljust(max_package_length)}  Required: {operator} {required_version.ljust(12)}  Installed: {installed_version}")
    logging.info("\nPYTHON PACKAGES:")
    for package, version in package_versions.items():
        logging.info(f" {package}>= {version}")

    # Print to screen
    print(f"\n\033[94mOS Version:\033[0m \033[92m{os_version}\033[0m")
    print(f"\033[94mOS Ver note:\033[0m \033[92m(Windows 11 will say build is 10.x.22xxx)\033[0m")
    print(f"\033[94mSystem RAM:\033[0m \033[92m{system_ram}\033[0m")
    for disk in disk_info:
        print(f"\033[94mDisk:\033[0m \033[92m{disk}\033[0m")
    print(f"\033[94m\nGPU Information:\033[0m {gpu_info}")
    print(f"\033[94mPort Status:\033[0m \033[92m{port_status}\033[0m\n")
    if "Fail" in cuda_test_result:
        print(f"\033[91mCUDA Working:\033[0m \033[91m{cuda_test_result}\033[0m")
    else:
        print(f"\033[94mCUDA Working:\033[0m \033[92m{cuda_test_result}\033[0m")
    print(f"\033[94mCUDA_HOME:\033[0m \033[92m{cuda_home}\033[0m")
    if found_paths:
        print(f"\033[94mCublas64_11 Path:\033[0m \033[92m{', '.join(found_paths)}\033[0m")
    else:
        print(f"\033[94mCublas64_11 Path:\033[0m \033[91mNot found in any search path directories.\033[0m")    
    print(f"\033[94m\nTorch Version:\033[0m \033[92m{torch_version}\033[0m")
    print(f"\033[94mPython Version:\033[0m \033[92m{platform.python_version()}\033[0m")
    print(f"\033[94mPython Executable:\033[0m \033[92m{python_executable}\033[0m")
    print(f"\033[94mConda Environment:\033[0m \033[92m{conda_env}\033[0m")
    print(f"\n\033[94mPython Search Path:\033[0m")
    for path in search_path:
        print(f"  {path}")
    if required_packages:  # Check if the dictionary is not empty
        print("\033[94m\nRequirements file package comparison:\033[0m")
        max_package_length = max(len(package) for package in required_packages.keys())

        for package_name, (operator, required_version) in required_packages.items():
            installed_version = installed_packages.get(package_name, 'Not installed')

            # Exclude build information (e.g., +cu118) before creating the SpecifierSet
            required_version_no_build = required_version.split("+")[0]

            if '*' in required_version:
                condition_met = satisfies_wildcard(installed_version, required_version)
            else:
                # Compare versions using packaging.version
                required_specifier = SpecifierSet(f"{operator}{required_version_no_build}")
                installed_version = parse_version(installed_version)
                condition_met = installed_version in required_specifier

            color_required = "\033[92m" if condition_met else "\033[91m"
            color_installed = "\033[92m" if condition_met else "\033[91m"

            # Print colored output
            print(f"  {package_name.ljust(max_package_length)}  Required: {color_required}{operator} {required_version.ljust(12)}\033[0m  Installed: {color_installed}{installed_version}\033[0m")

        print("\nOn Nvidia Graphics cards machines, if your \033[92mInstalled\033[0m version of \033[92mTorch\033[0m and \033[92mTorchaudio\033[0m does")    
        print("not have \033[92m+cu118\033[0m (Cuda 11.8) or \033[92m+cu121\033[0m (Cuda 12.1) listed after them, you do not have CUDA")
        print("installed for Torch or Torchaudio in this Python environment. This will cause you problems")
        print("with \033[94mAllTalk\033[0m and \033[94mFinetuning.\033[0m You may have to 'pip install' a new version of torch and")
        print("torchaudio, using '\033[94m--upgrade --force-reinstall\033[0m' with the correct version of PyTorch for\033[0m")
        print("your Python environment.\033[0m")
        print("\033[94m\nRequirements file specifier meanings:\033[0m")
        explanation = textwrap.dedent("""
        == Exact version              != Any version except          < Less than               
        <= Less than or equal to      >  Greater than                >= Greater than or equal to
        ~ Compatible release          ;  Environment marker          AND Logical AND           
        OR Logical OR
        """)
        print(explanation.strip())
    print("")
    print(f"\033[94mDiagnostic log created:\033[0m \033[92mdiagnostics.log. \033[94mA brief summary of results is displayed above on\033[0m")
    print(f"\033[94mscreen. Please see the log file for more detail.\033[0m")
    print(f"\033[94m\nPlease upload the log file with any support ticket.\033[0m")

if __name__ == "__main__":
    log_system_info()
