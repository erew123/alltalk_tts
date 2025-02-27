#!/bin/bash

# ANSI color codes
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
MAGENTA='\033[1;35m'
WHITE='\033[1;37m'
L_RED='\033[1;91m'
L_GREEN='\033[1;92m'
L_YELLOW='\033[1;93m'
L_BLUE='\033[1;94m'
L_CYAN='\033[1;96m'
L_MAGENTA='\033[1;95m'
NC='\033[0m' # No Color

# Navigate to the script's directory
cd "$(dirname "$0")"

# Function to check if curl is installed
check_curl() {
    if ! command -v curl &> /dev/null; then
        echo "curl is not available on this system. Please install curl then re-run the script https://curl.se/"
        echo "or perform a manual installation of a Conda Python environment."
        exit 1
    fi
}

# Check if the current directory path contains a space
containsSpace=false
currentPath=$(pwd)
if echo "$currentPath" | grep -q ' '; then
    containsSpace=true
fi

if [ "$containsSpace" = true ]; then
    echo
    echo -e "    ${L_BLUE}ALLTALK LINUX SETUP UTILITY${NC}"
    echo
    echo
    echo -e "    You are trying to install AllTalk in a folder that has a space in the"
    echo -e "    folder path e.g."
    echo 
    echo -e "       /home/${L_RED}program files${NC}/alltalk_tts"
    echo 
    echo -e "    This causes errors with Conda and Python scripts. Please follow this"
    echo -e "    link for reference:"
    echo 
    echo -e "      ${L_CYAN}https://docs.anaconda.com/free/working-with-conda/reference/faq/#installing-anaconda${NC}"
    echo 
    echo -e "    Please use a folder path that has no spaces in it e.g." 
    echo 
    echo -e "       /home/myfiles/alltalk_tts/"
    echo 
    echo
    read -p "Press Enter to continue..." 
    exit 1
else
    # Continue with the main menu
    echo "Continue with the main menu."
fi

# Main Menu
main_menu() {
    while true; do
        clear
        echo
        echo -e "    ${L_BLUE}ALLTALK LINUX SETUP UTILITY${NC}"
        echo
        echo "    INSTALLATION TYPE"
        echo -e "    1) I am using AllTalk as part of ${L_GREEN}Text-generation-webui${NC}"
        echo -e "    2) I am using AllTalk as a ${L_GREEN}Standalone Application${NC}"
        echo
        echo -e "    9)${L_RED} Exit/Quit${NC}"
        echo
        read -p "    Enter your choice: " user_option

        case $user_option in
            1) webui_menu ;;
            2) standalone_menu ;;
            9) exit 0 ;;
            *) echo "Invalid option"; sleep 2 ;;
        esac
    done
}

# Text-generation-webui Menu
webui_menu() {
    while true; do
        clear
        echo
        echo -e "    ${L_BLUE}TEXT-GENERATION-WEBUI SETUP${NC}"
        echo
        echo -e "    Please ensure you have started your Text-generation-webui Python"
        echo -e "    environment. If you have NOT done this, please run ${L_GREEN}/cmd_linux.sh${NC}"
        echo -e "    in the ${L_GREEN}text-generation-webui${NC} folder and then re-run this script."
        echo
        echo "    BASE REQUIREMENTS"
        echo -e "    1) Apply/Re-Apply the requirements for an ${L_GREEN}Text-generation-webui${NC}"
        echo
        echo "    OPTIONAL"
        echo "    2) Git Pull the latest AllTalk updates from Github"
        echo
        echo "    DEEPSPEED"
        echo "    4) Install DeepSpeed."
        echo "    5) Uninstall DeepSpeed."
        echo
        echo "    OTHER"
        echo "    6) Generate a diagnostics file."
        echo
        echo -e "    9)${L_RED} Exit/Quit${NC}"
        echo
        read -p "    Enter your choice: " webui_option

        case $webui_option in
            1) install_nvidia_textgen ;;
            2) tg_gitpull ;;
            4) install_deepspeed ;;
            5) uninstall_deepspeed ;;
            6) generate_diagnostics_textgen ;;
            9) exit 0 ;;
            *) echo "Invalid option"; sleep 2 ;;
        esac
    done
}

install_nvidia_textgen() {
    local requirements_file="system/requirements/requirements_textgen.txt"
    echo "    Installing Requirements from $requirements_file..."
    if ! pip install -r "$requirements_file"; then
        echo
        echo "    There was an error pulling from Github."
        echo "    Please check the output for details."
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi
    echo
    echo "    Requirements installed successfully."
    
    echo "    Installing additional requirements..."
    if ! pip install -r system/requirements/requirements_textgen2.txt; then
        echo
        echo "    There was an error installing additional requirements."
        echo "    Please check the output for details."
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi

    if ! conda install -y pytorch::faiss-cpu; then
        echo
        echo "    There was an error installing faiss-cpu."
        echo "    Please check the output for details."
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi

    if ! conda install -y conda-forge::ffmpeg; then
        echo
        echo "    There was an error installing ffmpeg."
        echo "    Please check the output for details."
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi

    echo
    echo "    Additional requirements installed successfully."
    echo
    echo -e "    To install ${L_YELLOW}DeepSpeed${NC} on Linux, there are additional"
    echo -e "    steps required. Please see the Github or documentation on DeepSeed."
    echo
    read -p "    Press any key to continue. " -n 1
    echo
}


tg_gitpull() {
    echo
    if ! git pull; then
        echo
        echo "    There was an error installing the requirements."
        echo "    Please check the output for details."
        echo
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi
    echo
    echo "    AllTalk Updated from Github. Please re-apply"
    echo "    the latest requirements file. (Option 1)"
    echo
    read -p "    Press any key to continue. " -n 1
    echo
}

# Function to install DeepSpeed
install_deepspeed() {
    clear
    echo
    echo -e "    ${L_BLUE}DEEPSPEED INSTALLATION REQUIREMENTS${NC}"
    echo
    echo -e "    - Please go see this link as the instructions are there.${NC}"
    echo -e "    The instructions there are making it as simple as posisble.${NC}"
    echo -e "      ${L_GREEN}https://github.com/erew123/alltalk_tts/releases/tag/DeepSpeed-14.2-Linux${NC}"
    echo
    read -p "    Have you completed all the above steps? (y/n): " confirm

    if [ "$confirm" != "y" ]; then
        echo -e "    ${RED}DeepSpeed installation cannot proceed without completing the prerequisites.${NC}"
        return
    fi

    echo -e "\n    ${GREEN}Proceeding with DeepSpeed installation...${NC}"

    if [ $? -ne 0 ]; then
        echo -e "    ${RED}There was an error installing DeepSpeed.${NC}"
        return
    fi

    echo -e "    ${GREEN}DeepSpeed installed successfully.${NC}"
    read -p "    Press any key to continue. " -n 1
    echo
}

uninstall_deepspeed() {
    echo "Uninstalling DeepSpeed..."
    pip uninstall -y deepspeed
    if [ $? -ne 0 ]; then
        echo
        echo "    There was an error uninstalling DeepSpeed."
        echo
        echo "    Press any key to return to the menu."
        read -n 1
        return
    fi
    echo
    echo "    DeepSpeed uninstalled successfully."
    echo
    echo "    Press any key to continue."
    read -n 1
}

generate_diagnostics_textgen() {
    # Run diagnostics
    if ! python diagnostics.py; then
        echo
        echo "    There was an error running diagnostics."
        echo
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi
    echo
    echo "    Diagnostics log file generated successfully."
    echo "    Please see diagnostics.log"
    echo
    read -p "    Press any key to continue. " -n 1
    echo
}

# Standalone Menu
standalone_menu() {
    while true; do
        clear
        echo
        echo -e "    ${L_BLUE}ALLTALK STANDALONE APPLICATION SETUP${NC}"
        echo
        echo "    BASE REQUIREMENTS"
        echo "    1) Install AllTalk as a Standalone Application"
        echo
        echo "    OPTIONAL"
        echo "    2) Git Pull the latest AllTalk updates from Github"
        echo "    3) Re-Apply/Update the requirements file"
        echo "    4) Delete AllTalk's custom Python environment"
        echo "    5) Purge the PIP cache"
        echo
        echo "    OTHER"        
        echo "    8) Generate a diagnostics file"
        echo
        echo -e "    9)${L_RED} Exit/Quit${NC}"
        echo
        read -p "    Enter your choice: " standalone_option

        case $standalone_option in
            1) install_custom_standalone ;;
            2) gitpull_standalone ;;
            3) reapply_standalone ;;
            4) delete_custom_standalone ;;
            5) pippurge_standalone ;;
            8) generate_diagnostics_standalone ;;
            9) exit 0 ;;
            *) echo "Invalid option"; sleep 2 ;;
        esac
    done
}


install_custom_standalone() {
    cd "$(dirname "${BASH_SOURCE[0]}")"

    if [[ "$(pwd)" =~ " " ]]; then
        echo "This script relies on Miniconda which can not be silently installed under a path with spaces."
        exit
    fi

    # Deactivate existing conda envs as needed to avoid conflicts
    { conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null

    OS_ARCH=$(uname -m)
    case "${OS_ARCH}" in
        x86_64*)    OS_ARCH="x86_64" ;;
        arm64* | aarch64*) OS_ARCH="aarch64" ;;
        *)          echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64" && exit ;;
    esac

    # Config
    INSTALL_DIR="$(pwd)/alltalk_environment"
    CONDA_ROOT_PREFIX="${INSTALL_DIR}/conda"
    INSTALL_ENV_DIR="${INSTALL_DIR}/env"
    MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-py311_24.4.0-0-Linux-${OS_ARCH}.sh"
    if [ ! -x "${CONDA_ROOT_PREFIX}/bin/conda" ]; then
        echo "Downloading Miniconda from $MINICONDA_DOWNLOAD_URL to ${INSTALL_DIR}/miniconda_installer.sh"
        mkdir -p "${INSTALL_DIR}"
        curl -L "${MINICONDA_DOWNLOAD_URL}" -o "${INSTALL_DIR}/miniconda_installer.sh"
        chmod +x "${INSTALL_DIR}/miniconda_installer.sh"
        bash "${INSTALL_DIR}/miniconda_installer.sh" -b -p "${CONDA_ROOT_PREFIX}"
        echo "Miniconda installed."
    fi

    if [ ! -d "${INSTALL_ENV_DIR}" ]; then
        "${CONDA_ROOT_PREFIX}/bin/conda" create -y --prefix "${INSTALL_ENV_DIR}" -c conda-forge python=3.11.9
        echo "Conda environment created at ${INSTALL_ENV_DIR}."
    fi

    # Activate the environment and install requirements
    source "${CONDA_ROOT_PREFIX}/etc/profile.d/conda.sh"
    conda activate "${INSTALL_ENV_DIR}"
	
    # Install git
    echo "Installing git"
    conda install -y git
	
    # pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 --upgrade --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
    conda install -y pytorch=2.2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install -y nvidia/label/cuda-12.1.0::cuda-toolkit=12.1
    conda install -y pytorch::faiss-cpu
    conda install -y -c conda-forge "ffmpeg=*=*gpl*"
    conda install -y -c conda-forge "ffmpeg=*=h*_*" --no-deps
    echo
    echo
    echo
    echo "    Fix Nvidia's broken symlinks in the /env/lib folder"
    echo
    # Define the environment path
    env_path="${INSTALL_ENV_DIR}/lib"

    echo "    Installing additional requirements."
    echo
    pip install -r system/requirements/requirements_standalone.txt
    curl -LO https://github.com/erew123/alltalk_tts/releases/download/DeepSpeed-14.0/deepspeed-0.14.2+cu121torch2.2-cp311-cp311-manylinux_2_24_x86_64.whl
    pip install --upgrade gradio==4.44.1
    echo Installing DeepSpeed...
    pip install deepspeed-0.14.2+cu121torch2.2-cp311-cp311-manylinux_2_24_x86_64.whl
    rm deepspeed-0.14.2+cu121torch2.2-cp311-cp311-manylinux_2_24_x86_64.whl
    pip install -r system/requirements/requirements_parler.txt
    conda clean --all --force-pkgs-dirs -y
    # Create start_environment.sh to run AllTalk
    cat << EOF > start_environment.sh
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ "$(pwd)" =~ " " ]]; then echo This script relies on Miniconda which can not be silently installed under a path with spaces. && exit; fi
# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null
# config
CONDA_ROOT_PREFIX="$(pwd)/alltalk_environment/conda"
INSTALL_ENV_DIR="$(pwd)/alltalk_environment/env"
# environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export CUDA_PATH="$INSTALL_ENV_DIR"
export CUDA_HOME="$CUDA_PATH"
# activate env
bash --init-file <(echo "source \"$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh\" && conda activate \"$INSTALL_ENV_DIR\"")
EOF
    cat << EOF > start_alltalk.sh
#!/bin/bash
source "${CONDA_ROOT_PREFIX}/etc/profile.d/conda.sh"
conda activate "${INSTALL_ENV_DIR}"
python script.py
EOF
    cat << EOF > start_finetune.sh
#!/bin/bash
export TRAINER_TELEMETRY=0
source "${CONDA_ROOT_PREFIX}/etc/profile.d/conda.sh"
conda activate "${INSTALL_ENV_DIR}"
python finetune.py
EOF
    cat << EOF > start_diagnostics.sh
#!/bin/bash
source "${CONDA_ROOT_PREFIX}/etc/profile.d/conda.sh"
conda activate "${INSTALL_ENV_DIR}"
python diagnostics.py
EOF
    chmod +x start_alltalk.sh
    chmod +x start_environment.sh
    chmod +x start_finetune.sh
    chmod +x start_diagnostics.sh
    echo
    echo
    echo -e "    Run ${L_YELLOW}./start_alltalk.sh${NC} to start AllTalk."
    echo -e "    Run ${L_YELLOW}./start_diagnostics.sh${NC} to start Diagnostics."
    echo -e "    Run ${L_YELLOW}./start_finetune.sh${NC} to start Finetuning."
    echo -e "    Run ${L_YELLOW}./start_environment.sh${NC} to start the AllTalk Python environment."
    echo
    echo -e "    To install ${L_YELLOW}DeepSpeed${NC} on Linux, there are additional"
    echo -e "    steps required. Please see the Github or documentation on DeepSeed."
    echo
    read -p "    Press any key to continue. " -n 1
}

delete_custom_standalone() {
    local env_dir="$PWD/alltalk_environment"
    # Check if the alltalk_environment directory exists
    if [ ! -d "$env_dir" ]; then
        echo "    \"$env_dir\" directory does not exist. No need to delete."
        read -p "    Press any key to continue. " -n 1
        echo
        return
    fi
    # Check if a Conda environment is active and deactivate it
    if [ -n "$CONDA_PREFIX" ]; then
        echo "    Exiting the Conda environment. You may need to start ./atstart.sh again"
        conda deactivate
    fi
    echo "Deleting \"$env_dir\". Please wait."
    rm -rf "$env_dir"
    if [ -d "$env_dir" ]; then
        echo "    Failed to delete \"$env_dir\" folder."
        echo "    Please make sure it is not in use and try again."
    else
        echo "    Environment \"$env_dir\" deleted successfully."
    fi
    read -p "    Press any key to continue. " -n 1
    echo
}

generate_diagnostics_standalone() {
    local env_dir="$PWD/alltalk_environment"
    local conda_root_prefix="${env_dir}/conda"
    local install_env_dir="${env_dir}/env"
    if [ ! -d "${install_env_dir}" ]; then
        echo
        echo "    The Conda environment at '${install_env_dir}' does not exist."
        echo "    Please install the environment before proceeding."
        echo
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi
    source "${conda_root_prefix}/etc/profile.d/conda.sh"
    conda activate "${install_env_dir}"
    if ! python diagnostics.py; then
        echo
        echo "    There was an error running diagnostics."
        echo
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi
    echo
    echo "    Diagnostics completed successfully."
    read -p "    Press any key to continue. " -n 1
    echo
}

gitpull_standalone() {
    local env_dir="$PWD/alltalk_environment"
    local conda_root_prefix="${env_dir}/conda"
    local install_env_dir="${env_dir}/env"
    if [ ! -d "${install_env_dir}" ]; then
        echo
        echo "    The Conda environment at '${install_env_dir}' does not exist."
        echo "    Please install the environment before proceeding."
        echo
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi
    source "${conda_root_prefix}/etc/profile.d/conda.sh"
    conda activate "${install_env_dir}"
    if ! git pull; then
        echo
        echo "    There was an error pulling from Github."
        echo
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi
    echo
    echo "    AllTalk Updated from Github. Please re-apply."
    echo "    the latest requirements file. (Option 3)"
    echo
    read -p "    Press any key to continue. " -n 1
    echo
}

pippurge_standalone() {
    if ! pip cache purge; then
        echo
        echo "    There was an error."
        echo
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi
    echo
    echo "    The PIP cache has been purged."
    echo
    read -p "    Press any key to continue. " -n 1
    echo
}

reapply_standalone() {
    local env_dir="$PWD/alltalk_environment"
    local conda_root_prefix="${env_dir}/conda"
    local install_env_dir="${env_dir}/env"
    if [ ! -d "${install_env_dir}" ]; then
        echo
        echo "    The Conda environment at '${install_env_dir}' does not exist."
        echo "    Please install the environment before proceeding."
        echo
        read -p "    Press any key to return to the menu. " -n 1
        echo
        return
    fi
    source "${conda_root_prefix}/etc/profile.d/conda.sh"
    conda activate "${install_env_dir}"
    echo
    echo "    Downloading and installing PyTorch. This step can take a long time"
    echo "    depending on your internet connection and hard drive speed. Please"
    echo "    be patient."
    pip install torch>=2.2.1+cu121 torchaudio>=2.2.1+cu121 --upgrade --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121
    echo
    echo "    Installing additional requirements."
    echo
    pip install -r system/requirements/requirements_standalone.txt
    pip install -r system/requirements/requirements_parler.txt
    echo
    echo "    Requirements have been re-applied/updated."
    echo
    read -p "    Press any key to continue. " -n 1
    echo
}

# Check for "-silent" install command-line argument
if [[ $1 == "-silent" ]]; then
	install_custom_standalone
else 
	# Start the main menu
	main_menu
fi