# AllTalk TTS

### For those interested, the AllTalk v2 BETA is out. See [here](https://github.com/erew123/alltalk_tts/discussions/245)

### AllTalk V1 Below
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- **Can be run as a** [standalone application](https://github.com/erew123/alltalk_tts/#-quick-setup-text-generation-webui--standalone-installation) **or part of :**
   - **Text-generation-webui** [link](https://github.com/oobabooga/text-generation-webui)
   - **SillyTavern** [link](https://github.com/SillyTavern/SillyTavern)
   - **KoboldCPP** [link](https://github.com/LostRuins/koboldcpp)
- **Simple setup utlilty** Windows & Linux.
- **API Suite and 3rd Party support via JSON calls:** Can be used with 3rd party applications via JSON calls.
- **Model Finetuning:** Train the model specifically on a voice of your choosing for better reproduction.
- **Local/Custom models:** Use any of the XTTSv2 models (API Local and XTTSv2 Local).
- **Bulk TTS Generator/Editor:** Generate hours of TTS into one big file or have something read back to you [demo](https://www.youtube.com/watch?v=hunvXn0mLzc).
- **DeepSpeed:** A 2-3x performance boost generating TTS. [Screenshot](https://github.com/erew123/screenshots/raw/main/deepspeed.jpg)
- **Low VRAM mode:** Great for people with small GPU memory or if your VRAM is filled by your LLM.
- **Custom Start-up Settings:** Adjust your default start-up settings. [Screenshot](https://github.com/erew123/screenshots/raw/main/settingsanddocs.jpg)
- **Narrarator:** Use different voices for main character and narration. [Example Narration](https://vocaroo.com/18nrv7FR6wuA)
- **Optional wav file maintenance:** Configurable deletion of old output wav files. [Screenshot](https://github.com/erew123/screenshots/raw/main/settingsanddocs.jpg)
- **Documentation:** Fully documented with a built in webpage. [Screenshot](https://github.com/erew123/screenshots/raw/main/settingsanddocs.jpg)
- **Clear Console output:** Clear command line output for any warnings or issues.

### 🟦 Screenshots
|![image](https://github.com/erew123/screenshots/raw/main/textgensettings.jpg) | ![image](https://github.com/erew123/screenshots/raw/main/setuputilitys.jpg) | ![image](https://github.com/erew123/screenshots/raw/main/deepspeed.jpg) |![image](https://github.com/erew123/screenshots/raw/main/textgen.jpg) |
|:---:|:---:|:---:|:---:|
|![image](https://github.com/erew123/screenshots/raw/main/settingsanddocs.jpg) | ![image](https://github.com/erew123/screenshots/raw/main/finetune1.jpg) | ![image](https://github.com/erew123/screenshots/raw/main/finetune2.jpg) |![image](https://github.com/erew123/screenshots/raw/main/sillytavern.jpg)|

---

### Index

- 🟦 [Screenshots](https://github.com/erew123/alltalk_tts#-screenshots)
- 🟩 [Installation](https://github.com/erew123/alltalk_tts/#-quick-setup-text-generation-webui--standalone-installation)
- 🟪 [Updating & problems with updating](https://github.com/erew123/alltalk_tts?#-updating)
- 🔵🟢 [DeepSpeed Installation (Windows & Linux)](https://github.com/erew123/alltalk_tts?#-deepspeed-installation-options)
- 🆘 [Support Requests, Troubleshooting & Feature requests](https://github.com/erew123/alltalk_tts#-support-requests-troubleshooting--feature-requests)
- 🟨 [Help with problems](https://github.com/erew123/alltalk_tts?#-help-with-problems)
- ⚫ [Finetuning a model](https://github.com/erew123/alltalk_tts?#-finetuning-a-model)
- ⬜ [AllTalk TTS Generator](https://github.com/erew123/alltalk_tts?#-alltalk-tts-generator)
- 🟠 [API Suite and JSON-CURL](https://github.com/erew123/alltalk_tts?#-api-suite-and-json-curl)
- 🔴 [Future to-do list & Upcoming updates](https://github.com/erew123/alltalk_tts?#-future-to-do-list)

---
### 🛠️ **About this project & me** 
AllTalk is a labour of love that has been developed, supported and sustained in my personal free time. As a solo enthusiast (not a business or team) my resources are inherently limited. This project has been one of my passions, but I must balance it with other commitments.

To manage AllTalk sustainably, I prioritize support requests based on their overall impact and the number of users affected. I encourage you to utilize the comprehensive documentation and engage with the AllTalk community discussion area. These resources often provide immediate answers and foster a supportive user network.

Should your inquiry extend beyond the documentation, especially if it concerns a bug or feature request, I assure you I’ll offer my best support as my schedule permits. However, please be prepared for varying response times, reflective of the personal dedication I bring to AllTalk. Your understanding and patience in this regard are greatly appreciated.

It's important to note that **I am not** the developer of any TTS models utilized by AllTalk, nor do I claim to be an expert on them, including understanding all their nuances, issues, and quirks. For specific TTS model concerns, I’ve provided links to the original developers in the Help section for direct assistance.

Thank you for your continued support and understanding. 

---

### 💖 Showing Your Support
If AllTalk has been helpful to you, consider showing your support through a donation on my [Ko-fi page](https://ko-fi.com/erew123). Your support is greatly appreciated and helps ensure the continued development and improvement of AllTalk.

---

### 🟩 Quick Setup (Text-generation-webui & Standalone Installation)

Quick setup scripts are available for users on Windows 10/11 and Linux. Instructional videos for both setup processes are linked below.

- Ensure that **Git** is installed on your system as it is required for cloning the repository. If you do not have Git installed, visit [**Git's official website**](https://git-scm.com/downloads) to download and install it.
- Windows users must install C++ development tools for Python to compile Python packages. Detailed information and a link to these tools can be found in the help section [**Windows & Python requirements for compiling packages**](https://github.com/erew123/alltalk_tts#-help-with-problems).

<details>
<summary>QUICK SETUP - Text-Generation-webui</summary>
<br>

For a step-by-step video guide, click [here](https://www.youtube.com/watch?v=icn2XS5rUH8).

To set up AllTalk within Text-generation-webui, follow either method:

1. **Download AllTalk Setup**:
   - **Via Terminal/Console (Recommended)**:
     - `cd \text-generation-webui\extensions\`
     - `git clone https://github.com/erew123/alltalk_tts`
   - **Via Releases Page (Cannot be automatically updated after install as its not linked to Github)**:
     - Download the latest `alltalk_tts.zip` from [Releases](https://github.com/erew123/alltalk_tts/releases) and extract it to `\text-generation-webui\extensions\alltalk_tts\`.

2. **Start Python Environment**:
   - In the text-generation-webui folder, start the environment with the appropriate command:
     - Windows: `cmd_windows.bat`
     - Linux: `./cmd_linux.sh`<br><br>
    
     > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

3. **Run AllTalk Setup Script**:
   - Navigate to the AllTalk directory and execute the setup script:
     - `cd extensions`
     - `cd alltalk_tts`
     - Windows: `atsetup.bat`
     - Linux: `./atsetup.sh`

4. **Install Requirements**:
   - Follow the on-screen instructions to install the necessary requirements. It's recommended to test AllTalk's functionality before installing DeepSpeed.

> **Note**: Always activate the Text-generation-webui Python environment before making any adjustments or using Fine-tuning. Additional instructions for Fine-tuning and DeepSpeed can be found within the setup utility and on this documentation page.

</details>

<details>
<summary>QUICK SETUP - Standalone Installation</summary>
<br>

For a step-by-step video guide, click [here](https://www.youtube.com/watch?v=AQYCccDRbaY).

To perform a Standalone installation of AllTalk:

1. **Get AllTalk Setup**:
   - **Via Terminal/Console (Recommended)**:
     - Navigate to your preferred directory: `cd C:\myfiles\`
     - Clone the AllTalk repository: `git clone https://github.com/erew123/alltalk_tts`
   - **Via Releases Page (Cannot be automatically updated after install as its not linked to Github)**:
     - Download `alltalk_tts.zip` from [Releases](https://github.com/erew123/alltalk_tts/releases) and extract it to your chosen directory, for example, `C:\myfiles\alltalk_tts\`.

2. **Start AllTalk Setup**:
   - Open a terminal/command prompt, move to the AllTalk directory, and run the setup script:
     - `cd alltalk_tts`
     - Windows: `atsetup.bat`
     - Linux: `./atsetup.sh`

3. **Follow the Setup Prompts**:
   - Select Standalone Installation and then Option 1 and follow any on-screen instructions to install the required files. DeepSpeed is automatically installed on **Windows** based system, but will only work on Nvidia GPU's. **Linux** based system users will have to follow the DeepSpeed installation instructions.

> If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

> **Important**: Do not use spaces in your folder path (e.g. avoid `/my folder-is-this/alltalk_tts-main`) as this causes issues with Python & Conda.

</details>

Refer to `🟩 Other installation notes` for further details, including information on additional voices, changing IP, character card notes etc.
> If you wish to understand AllTalks start-up screen, please read **Understanding the AllTalk start-up screen** in the Help section.

---

### 🟩 Docker Builds and Google Colab's

While an AllTalk Docker build exists, it's important to note that this version is based on an earlier iteration of AllTalk and was set up by a third party. At some point, my goal is to deepen my understanding of Docker and its compatibility with AllTalk. This exploration may lead to significant updates to AllTalk to ensure a seamless Docker experience. However, as of now, the Docker build should be considered a BETA version and isn't directly supported by me.

As for Google Colab, there is partial compatibility with AllTalk, though with some quirks. I am currently investigating these issues and figuring out the necessary adjustments to enhance the integration. Until I can ensure a smooth experience, I won't be officially releasing any Google Colab implementations of AllTalk.

---

### 🟩 Manual Installation - As part of Text generation web UI (inc. macOSX)
<details>
	<summary>MANUAL INSTALLATION - Text-Generation-webui</summary>

### Manual Installation for Text Generation Web UI

If you're using a Mac or prefer a manual installation for any other reason, please follow the steps below. This guide is compatible with the current release of Text Generation Web UI as of December 2023. Consider updating your installation if it's been a while, [update instructions here](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install).

- For a visual guide on the installation process, watch [this video](https://youtu.be/9BPKuwaav5w).

1. **Navigate to Text Generation Web UI Folder**:
   - Open a terminal window and move to your Text Generation Web UI directory with:
     - `cd text-generation-webui`

2. **Activate Text Generation Web UI Python Environment**:
   - Start the appropriate Python environment for your OS using one of the following commands:
     - For Windows: `cmd_windows.bat`
     - For Linux: `./cmd_linux.sh`
     - For macOS: `cmd_macos.sh`
     - For WSL: `cmd_wsl.bat`
   
   - Loading the Text Generation Web UI's Python environment **is crucial**. If unsure about what a loaded Python environment should look like, refer to this [image](https://github.com/erew123/alltalk_tts/issues/25#issuecomment-1869344442) and [video guide](https://www.youtube.com/watch?v=9BPKuwaav5w).
   > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

3. **Move to Extensions Folder**:
   - `cd extensions`

4. **Clone the AllTalk TTS Repository**:
   - `git clone https://github.com/erew123/alltalk_tts`

5. **Navigate to the AllTalk TTS Folder**:
   - `cd alltalk_tts`

6. **Install Required Dependencies**:
   - Install dependencies for your machine type:
     - For Windows: `pip install -r system\requirements\requirements_textgen.txt`
     - For Linux/Mac: `pip install -r system/requirements/requirements_textgen.txt`

7. **Optional DeepSpeed Installation**:
- If you're using an Nvidia graphics card on Linux or Windows and wish to install **DeepSpeed**, follow the instructions [here](https://github.com/erew123/alltalk_tts?#-deepspeed-installation-options). 
- **Recommendation**: Start Text Generation Web UI and ensure AllTalk functions correctly before installing DeepSpeed.

8. **Start Text Generation Web UI**:
- Return to the main Text Generation Web UI folder using `cd ..` (repeat as necessary).
  - Start the appropriate Python environment for your OS using one of the following commands:
     - For Windows: `start_windows.bat`
     - For Linux: `./start_linux.sh`
     - For macOS: `start_macos.sh`
     - For WSL: `start_wsl.bat`

- Load the AllTalk extension in the Text Generation Web UI **session** tab.
- For any updates to AllTalk or for tasks like Finetuning, always activate the Text Generation Web UI Python environment first.

Refer to `🟩 Other installation notes` for further details, including information on additional voices, changing IP, character card notes etc.

</details>

### 🟩 Manual Installation - As a Standalone Application

<details>
	<summary>MANUAL INSTALLATION - Run AllTalk as a Standalone with Text-generation-webui</summary>

### Running AllTalk as a Standalone Application alongside Text Generation Web UI

If you have AllTalk installed as an extension of Text Generation Web UI but wish to run it as a standalone application, follow these steps:

1. **Activate Text Generation Web UI Python Environment**:
   - Use the appropriate command for your operating system to load the Python environment:
     - Windows: `cmd_windows.bat`
     - Linux: `./cmd_linux.sh`
     - macOS: `cmd_macos.sh`
     - WSL: `cmd_wsl.bat`

2. **Navigate to the AllTalk Directory**:
   - Move to the AllTalk folder with the following commands:
     - `cd extensions`
     - `cd alltalk_tts`

3. **Start AllTalk**:
   - Run AllTalk with the command:
     - `python script.py`

   There are no additional steps required to run AllTalk as a standalone application from this point.
</details>

<details>
	<summary>MANUAL INSTALLATION - Custom Install of AllTalk</summary>

### Custom Installation of AllTalk

Support for custom Python environments is limted. Please read **Custom Python environments Limitations Notice** below this section.

To run AllTalk as a standalone application with a custom Python environment, ensure you install AllTalk's requirements into the environment of your choice. The instructions provided are generalized due to the variety of potential Python environments.

- **Python Compatibility**: The TTS engine requires Python **3.9.x** to **3.11.x**. AllTalk is tested with Python **3.11.x**. [See TTS Engine details](https://pypi.org/project/TTS/).
- **Path Names**: Avoid spaces in path names as this can cause issues.
- **Custom Python Environments**: If encountering issues potentially related to a custom environment, consider testing AllTalk with the quick setup standalone method that builds its own environment.

#### Quick Overview of Python Environments

   > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

#### Building a Custom Python Environment with Miniconda

1. **Initial Setup**:
   - Ensure `python --version` and `pip` commands work in your terminal. Install Python and Pip if necessary.
     - [Python Installation](https://www.python.org/downloads/)
     - [Pip Installation](https://pip.pypa.io/en/stable/installation/)

2. **Install Miniconda**:
   - Download and install Miniconda for your OS from the [Miniconda Website](https://docs.conda.io/projects/miniconda/en/latest/). Use the `Anaconda Prompt` from the Start Menu or Application Launcher to access the base Conda environment.

3. **Clone AllTalk Repository**:
   - Navigate to your desired folder (e.g., `c:\myfiles\`) and clone the AllTalk repository:
     - `git clone https://github.com/erew123/alltalk_tts`
   - Move to the AllTalk folder with the following commands:
     - `cd alltalk_tts`

4. **Create Conda Environment**:
   - Create a Conda environment named `alltalkenv` with Python 3.11.5:
     - `conda create --name alltalkenv python=3.11.5`
   - Activate the new environment:
     - `conda activate alltalkenv`

5. **Install Requirements**:
   - Install dependencies based on your machine type:
     - For Windows: `pip install -r system\requirements\requirements_standalone.txt`
     - For Linux/Mac: `pip install -r system/requirements/requirements_standalone.txt`

6. **Start AllTalk**:
   - Run AllTalk with the following:
     - `python script.py`

**Note**: For updates, DeepSpeed installations, or other modifications, always activate the `alltalkenv` Conda environment first. Custom scripts or batch files can simplify launching AllTalk.

</details>

🟩 **Custom Python environments Limitations Notice**: Given the vast array of Python environments and custom configurations out there, it's challenging for me to guarantee comprehensive support for each unique setup. AllTalk leverages a wide range of scripts and libraries, many of which are developed and maintained outside of my control. As a result, these components might not always behave as expected in every custom Python environment. I'll do my best to assist where I can, but please understand that my ability to help with issues stemming from these external factors may be limited.

---

#### 🟩 Other installation notes
On first startup, AllTalk will download the Coqui XTTSv2 2.0.2 model to its **models** folder (1.8GB space required). Check the command prompt/terminal window if you want to know what its doing. After it says "Model Loaded" the Text generation webUI is usually available on its IP address a few seconds later, for you to connect to in your browser. If you are running a headless system and need to change the IP, please see the Help with problems section down below.

Once the extension is loaded, please find all documentation and settings on the link provided in the interface (as shown in the screenshot below).

**Where to find voices** https://aiartes.com/voiceai or https://commons.wikimedia.org/ or interviews on youtube etc. Instructions on how to cut down and prepare a voice sample are within the built in documentation.

Please read the note below about start-up times and also the note about ensuring your character cards are set up [correctly](https://github.com/erew123/alltalk_tts#-a-note-on-character-cards--greeting-messages)

Some extra voices for AllTalk are downloadable [here](https://drive.google.com/file/d/1bYdZdr3L69kmzUN3vSiqZmLRD7-A3M47/view?usp=drive_link) and [here](https://drive.google.com/file/d/1CPnx1rpkuKvVj5fGr9OiUJHZ_e8DfTzP/view)

#### 🟩 Changing AllTalks IP address & Accessing AllTalk over your Network
<details>
	<summary>Click to expand</summary><br>
	
AllTalk is coded to start on 127.0.0.1, meaning that it will ONLY be accessable to the local computer it is running on. If you want to make AllTalk available to other systems on your network, you will need to change its IP address to match the IP address of your network card/computers current IP address. There are 2x ways to change the IP address:
  1) Start AllTalk and within its web interface and you can edit the IP address on the "AllTalk Startup Settings".
  2) You can edit the `confignew.json`file in a text editor and change `"ip_address": "127.0.0.1",` to the IP address of your choosing.

So, for example, if your computer's network card was on IP address 192.168.0.20, you would change AllTalk's setting to 192.168.1.20 and then **restart** AllTalk. You will need to ensure your machine stays on this IP address each time it is restarted, by setting your machine to have a static IP address.

</details>

#### 🟩 Text-geneneration-webui & Stable-Diffusion Plugin - Load Order & stripped text
<details>
	<summary>Click to expand</summary><br>
	
The Stable Diffusion plugin for Text-generation-webui **strips out** some of the text, which is passed to Stable Diffusion for image/scene generation. Because this text is stripped, its important to consider the load order of the plugins to get the desired result you want. Lets assume the AI has just generated the following message `*He walks into the room with a smile on his face and says* Hello how are you?`. Depending on the load order will change what text reaches AllTalk for generation e.g.

**SD Plugin loaded before AllTalk** - Only `Hi how are you?` is sent to AllTalk, with the `*He walks into the room with a smile on his face and says*` being sent over to SD for image generation. Narration of the scene is not possible.<br><br>
**AllTalk loaded before SD Plugin** - `*He walks into the room with a smile on his face and says* Hello how are you?` is sent to AllTalk with the `*He walks into the room with a smile on his face and says*` being sent over to SD for image generation.<br><br>
The load order can be changed within Text-generation-webui's `settings.yaml` file or `cmd_flags.txt` (depending on how you are managing your extensions).<br><br>
![image](https://github.com/erew123/screenshots/blob/main/atandsdplugin.jpg)
</details>

#### 🟩 A note on Character Cards & Greeting Messages
<details>
	<summary>Click to expand</summary><br>
	
Messages intended for the Narrator should be enclosed in asterisks `*` and those for the character inside quotation marks `"`. However, AI systems often deviate from these rules, resulting in text that is neither in quotes nor asterisks. Sometimes, text may appear with only a single asterisk, and AI models may vary their formatting mid-conversation. For example, they might use asterisks initially and then switch to unmarked text. A properly formatted line should look like this:

`"`Hey! I'm so excited to finally meet you. I've heard so many great things about you and I'm eager to pick your brain about computers.`"` `*`She walked across the room and picked up her cup of coffee`*`

Most narrator/character systems switch voices upon encountering an asterisk or quotation marks, which is somewhat effective. AllTalk has undergone several revisions in its sentence splitting and identification methods. While some irregularities and AI deviations in message formatting are inevitable, any line beginning or ending with an asterisk should now be recognized as Narrator dialogue. Lines enclosed in double quotes are identified as Character dialogue. For any other text, you can choose how AllTalk handles it: whether it should be interpreted as Character or Narrator dialogue (most AI systems tend to lean more towards one format when generating text not enclosed in quotes or asterisks).

With improvements to the splitter/processor, I'm confident it's functioning well. You can monitor what AllTalk identifies as Narrator lines on the command line and adjust its behavior if needed (Text Not Inside - Function).
</details>

#### 🟩 I want to know more about the XTTS AI model used
<details>
	<summary>Click to expand</summary><br>
	
Currently the XTTS model is the main model used by AllTalk for TTS generation. If you want to know more details about the XTTS model, its capabilties or its technical features you can look at resources such as:
- https://docs.coqui.ai/en/latest/models/xtts.html
- https://github.com/coqui-ai/TTS
- https://github.com/coqui-ai/TTS/discussions

</details>

---

### 🟪 Updating

Maintaining the latest version of your setup ensures access to new features and improvements. Below are the steps to update your installation, whether you're using Text-Generation-webui or running as a Standalone Application.

**NOTE** Future updates will be handled by using the `atsetup` utility.<br><br>
**NOTE** If you have an install **prior to 28th March 2024** that you are updating, perform the `git pull` instructions below, then run the `atsetup` utility and select option 1 in either the Standalone ot Text-generation-webui menu (as matches your system). 

<details>
<summary>UPDATING - Text-Generation-webui</summary>
<br>

The update process closely mirrors the installation steps. Follow these to ensure your setup remains current:

1. **Open a Command Prompt/Terminal**:
   - Navigate to your Text-Generation-webui folder with:
     - `cd text-generation-webui`

2. **Start the Python Environment**:
   - Activate the Python environment tailored for your operating system. Use the appropriate command from below based on your OS:
     - Windows: `cmd_windows.bat`
     - Linux: `./cmd_linux.sh`
     - macOS: `cmd_macos.sh`
     - WSL (Windows Subsystem for Linux): `cmd_wsl.bat`<br><br>

   > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

3. **Navigate to the AllTalk TTS Folder**:
   - Move into your extensions and then the alltalk_tts directory:
     - `cd extensions/alltalk_tts`

4. **Update the Repository**:
   - Fetch the latest updates from the repository with:
     - `git pull`

5. **Install Updated Requirements**:
   - Depending on your machine's OS, install the required dependencies using pip:
     - **For Windows Machines**:
       - `pip install -r system\requirements\requirements_textgen.txt`
     - **For Linux/Mac**:
       - `pip install -r system/requirements/requirements_textgen.txt`

5. **DeepSpeed Requirements**:
   - If Text-gen-webui is using a new version of PyTorch, you may need to uninstall and update your DeepSpeed version.
   - Use AllTalks diagnostics or start-up menu to identify your version of PyTorch.
<br><br>
</details>

<details>
<summary>UPDATING - Standalone Application</summary>
<br>

If you installed from a ZIP file, you cannot use a `git pull` to update, as noted in the Quick Setup instructions.

For Standalone Application users, here's how to update your setup:

1. **Open a Command Prompt/Terminal**:
   - Navigate to your AllTalk folder with:
     - `cd alltalk_tts`

2. **Access the Python Environment**:
   - In a command prompt or terminal window, navigate to your `alltalk_tts` directory and start the Python environment:
     - Windows:
       - `start_environment.bat`
     - Linux/macOS:
       - `./start_environment.sh`

> If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

2. **Pull the Latest Updates**:
   - Retrieve the latest changes from the repository with:
     - `git pull`
     
3. **Install Updated Requirements**:
   - Depending on your machine's OS, install the required dependencies using pip:
     - **For Windows Machines**:
       - `pip install -r system\requirements\requirements_standalone.txt`
     - **For Linux/Mac**:
       - `pip install -r system/requirements/requirements_standalone.txt`
<br><br>
</details>

### 🟪 Resolving Update Issues

If you encounter problems during or after an update, following these steps can help resolve the issue by refreshing your installation while preserving your data:

<details>
<summary>RESOLVING - Updates</summary><br>

The process involves renaming your existing `alltalk_tts` directory, setting up a fresh instance, and then migrating your data:

1. **Rename Existing Directory**:
   - First, rename your current `alltalk_tts` folder to keep it safe e.g. `alltalk_tts.old`. This preserves any existing data.

2. **Follow the Quick Setup instructions**:
   - You will now follow the **Quick Setup** instructions, performing the `git clone https://github.com/erew123/alltalk_tts` to pull down a new copy of AllTalk and install the requirements.
     
     > If you're not familiar with Python environments, see **Understanding Python Environments Simplified** in the Help section for more info.

3. **Migrate Your Data**:
   - **Before** starting the AllTalk, transfer the `models`, `voices`, `outputs` folders and also `confignew.json` from `alltalk_tts.old` to the new `alltalk_tts` directory. This action preserves your voice history and prevents the need to re-download the model.

4) **Launch AllTalk**
   - You're now ready to launch AllTalk and check it works correctly.

6. **Final Step**:
    - Once you've verified that everything is working as expected and you're satisfied with the setup, feel free to delete the `alltalk_tts.old` directory to free up space.

</details>

---

### 🔵🟢 DeepSpeed Installation Options
**DeepSpeed requires an Nvidia Graphics card**

#### 🔵 Linux Installation
DeepSpeed requires access to the **Nvidia CUDA Development Toolkit** to compile on a Linux system. It's important to note that this toolkit is distinct and unrealted to your graphics card driver or the CUDA version the Python environment uses. 

<details>
	<summary>Linux DeepSpeed - Text-generation-webui</summary>

### DeepSpeed Installation for Text generation webUI

1. **Nvidia CUDA Development Toolkit Installation**:
   - The toolkit is crucial for DeepSpeed to compile/build for your version of Linux and requires around 3GB's of disk space.
   - Install using your package manager **(Recommended)** e.g. **CUDA Toolkit 11.8** or download directly from [Nvidia CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) (choose 11.8 or 12.1 for Linux).

2. **Open a Terminal Console**:
   - After Nvidia CUDA Development Toolkit installation, access your terminal console.

3. **Install libaio-dev**:
   - Use your Linux distribution's package manager.
     
   - - `sudo apt install libaio-dev` for Debian-based systems
     - `sudo yum install libaio-devel` for RPM-based systems.

4. **Navigate to Text generation webUI Folder**:
   - Change directory to your Text generation webUI folder with `cd text-generation-webui`.
   
5. **Activate Text generation webUI Custom Conda Environment**:
   - Run `./cmd_linux.sh` to start the environment.<br><br>
   
   > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

6. **Set `CUDA_HOME` Environment Variable**:
   - DeepSpeed locates the Nvidia toolkit using the `CUDA_HOME` environment variable.
   - You will only set this temporarily as Text generation webUI sets up its own CUDA_HOME environment each time you use `./cmd_linux.sh` or `./start_linux.sh`

7. **Temporarily Configuring `CUDA_HOME`**:
   - When the Text generation webUI Python environment is active **(step 5)**, set `CUDA_HOME`.
     
   - - `export CUDA_HOME=/usr/local/cuda`
     - `export PATH=${CUDA_HOME}/bin:${PATH}`
     - `export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH`

   - You can confirm the path is set correctly and working by running the command `nvcc --version` should confirm `Cuda compilation tools, release 11.8.`.
   - Incorrect path settings may lead to errors. If you encounter path issues or receive errors like `[Errno 2] No such file or directory` when you run the next step, confirm the path correctness or adjust as necessary.

8. **DeepSpeed Installation**:
   - Install DeepSpeed using `pip install deepspeed`.

9. **Troubleshooting**:
   - Troubleshooting steps for DeepSpeed installation can be located down below.
   - **NOTE**: You **DO NOT** need to set Text-generation-webUI's **--deepspeed** setting for AllTalk to be able to use DeepSpeed. These are two completely separate things and incorrectly setting that on Text-generation-webUI may cause other complications.
</details>
<details>
	<summary>Linux DeepSpeed - Standalone Installation</summary>

### DeepSpeed Installation for Standalone AllTalk

1. **Nvidia CUDA Development Toolkit Installation**:
   - The toolkit is crucial for DeepSpeed to compile/build for your version of Linux and requires around 3GB's of disk space.
   - Install using your package manager **(Recommended)** e.g. **CUDA Toolkit 11.8** or download directly from [Nvidia CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) (choose 11.8 or 12.1 for Linux).

2. **Open a Terminal Console**:
   - After Nvidia CUDA Development Toolkit installation, access your terminal console.
  
3. **Install libaio-dev**:
   - Use your Linux distribution's package manager.
     
   - - `sudo apt install libaio-dev` for Debian-based systems
     - `sudo yum install libaio-devel` for RPM-based systems.

4. **Navigate to AllTalk TTS Folder**:
   - Change directory to your AllTalk TTS folder with `cd alltalk_tts`.

5. **Activate AllTalk Custom Conda Environment**:
   - Run `./start_environment.sh` to start the AllTalk Python environment.
   - This command will start the custom Python environment that was installed with `./atsetup.sh`.<br><br>
   
   > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

6. **Set `CUDA_HOME` Environment Variable**:
   - The DeepSpeed installation routine locates the Nvidia toolkit using the `CUDA_HOME` environment variable. This can be set temporarily for a session or permanently, depending on other requirements you may have for other Python/System environments.
   - For temporary use, proceed to **step 8**. For a permanent solution, see [Conda's manual on setting environment variables](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#set-env-vars).

7. **(Optional) Permanent `CUDA_HOME` Setup**:
   - If you choose to set `CUDA_HOME` permanently, follow the instructions in the provided Conda manual link above.

8. **Configuring `CUDA_HOME`**:
   - When your Python environment is active **(step 5)**, set `CUDA_HOME`.
     
   - - `export CUDA_HOME=/usr/local/cuda`
     - `export PATH=${CUDA_HOME}/bin:${PATH}`
     - `export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH`

   - You can confirm the path is set correctly and working by running the command `nvcc --version` should confirm `Cuda compilation tools, release 11.8.`.
   - Incorrect path settings may lead to errors. If you encounter path issues or receive errors like `[Errno 2] No such file or directory` when you run the next step, confirm the path correctness or adjust as necessary.

9. **DeepSpeed Installation**:
   - Install DeepSpeed using `pip install deepspeed`.

10. **Starting AllTalk TTS WebUI**:
    - Launch the AllTalk TTS interface with `./start_alltalk.sh` and enable DeepSpeed.

### Troubleshooting

- If setting `CUDA_HOME` results in path duplication errors (e.g., `.../bin/bin/nvcc`), you can correct this by unsetting `CUDA_HOME` with `unset CUDA_HOME` and then adding the correct path to your system's PATH variable.
- Always verify paths and compatibility with other CUDA-dependent applications to avoid conflicts.
- If you have multiple versions of the Nvidia CUDA Development Toolkit installed, you will have to specify the version number in step 8 for the CUDA_HOME path.
- If it became necessary to uninstall DeepSpeed, you can do so by start the Python enviroment and then running `pip uninstall deepspeed`<br><br>

</details>

#### 🟢 Windows Installation
You have 2x options for how to setup DeepSpeed on Windows. Pre-compiled wheel files for specific Python, CUDA and Pytorch builds, or manually compiling DeepSpeed.

<details>
	<summary>Windows DeepSpeed - Pre-Compiled Wheels (Quick and Easy)</summary>

### DeepSpeed Installation with Pre-compiled Wheels

1. **Introduction to Pre-compiled Wheels**:
   - The `atsetup.bat` utility simplifies the installation of DeepSpeed by automatically downloading and installing pre-compiled wheel files. These files are tailored for **specific** versions of Python, CUDA, and PyTorch, ensuring compatibility with both the **Standalone Installation** and a standard build of **Text-generation-webui**.

2. **Manual Installation of Pre-compiled Wheels**:
   - If needed, pre-compiled DeepSpeed wheel files that I have built are available on the [Releases Page](https://github.com/erew123/alltalk_tts/releases). You can manually install or uninstall these wheels using the following commands:
     - Installation: `pip install {deep-speed-wheel-file-name-here}`
     - Uninstallation: `pip uninstall deepspeed`
  
3. **Using `atsetup.bat` for Simplified Management**:
   - For those running the Standalone Installation or a standard build of Text-generation-webui, the `atsetup.bat` utility offers the simplest and most efficient way to manage DeepSpeed installations on Windows.

</details>

<details>
	<summary>Windows DeepSpeed - Manual Compilation</summary>

### Manual DeepSpeed Wheel Compilation

1. **Preparation for Manual Compilation**:
   - Manual compilation of DeepSpeed wheels is an advanced process that requires:
     - **1-2 hours** of your time for initial setup and compilation.
     - **6-10GB** of disk space on your computer.
     - A solid technical understanding of Windows environments and Python.

2. **Understanding Wheel Compatibility**:
   - A compiled DeepSpeed wheel is uniquely tied to the specific versions of Python, PyTorch, and CUDA used during its compilation. If any of these versions are changed, you will need to compile a new DeepSpeed wheel to ensure compatibility.

3. **Compiling DeepSpee Resources**:
   - Myself and [@S95Sedan](https://github.com/S95Sedan) have worked to simplify the compilation process. [@S95Sedan](https://github.com/S95Sedan) has notably improved the process for later versions of DeepSpeed, ensuring ease of build on Windows.
   - Because [@S95Sedan](https://github.com/S95Sedan) is now maintaining the instructions for compiling DeepSpeed on Windows, please visit [@S95Sedan](https://github.com/S95Sedan)'s<br>[DeepSpeed GitHub page](https://github.com/S95Sedan/Deepspeed-Windows).

</details>

---

### 🆘 Support Requests, Troubleshooting & Feature requests
I'm thrilled to see the enthusiasm and engagement with AllTalk! Your feedback and questions are invaluable, helping to make this project even better. To ensure everyone gets the help they need efficiently, please consider the following before submitting a support request:

**Consult the Documentation:** A comprehensive guide and FAQ sections (below) are available to help you navigate AllTalk. Many common questions and troubleshooting steps are covered here.

**Search Past Discussions:** Your issue or question might already have been addressed in the discussions area or [closed issues](https://github.com/erew123/alltalk_tts/issues?q=is%3Aissue+is%3Aclosed). Please use the search function to see if there's an existing solution or advice that applies to your situation.

**Bug Reports:** If you've encountered what you believe is a bug, please first check the [Updates & Bug Fixes List](https://github.com/erew123/alltalk_tts/issues/25) to see if it's a known issue or one that's already been resolved. If not, I encourage you to report it by raising a bug report in the [Issues section](https://github.com/erew123/alltalk_tts/issues), providing as much detail as possible to help identify and fix the issue.

**Feature Requests:** The current Feature request list can be [found here](https://github.com/erew123/alltalk_tts/discussions/74). I love hearing your ideas for new features! While I can't promise to implement every suggestion, I do consider all feedback carefully. Please share your thoughts in the [Discussions area](https://github.com/erew123/alltalk_tts/discussions) or via a Feature Request in the [Issues section](https://github.com/erew123/alltalk_tts/issues). 

---

### 🟨 Help with problems

#### &nbsp;&nbsp;&nbsp;&nbsp; 🔄 **Minor updates/bug fixes list** can be found [here](https://github.com/erew123/alltalk_tts/issues/25)

<details>
<summary>🟨 How to make a diagnostics report file</summary><br>

If you are on a Windows machine or a Linux machine, you should be able to use the `atsetup.bat` or `./atsetup.sh` utility to create a diagnositcs file. If you are unable to use the `atsetup` utility, please follow the instructions below.

<details>
	<summary><strong>Manually making a diagnostics report file</strong></summary><br>

1) Open a command prompt window and start the Python environment. Depending on your setup (Text-generation-webui or Standalone AllTalk), the steps to start the Python environment vary:<br>

- **For Text-generation-webui Users**:
  - Navigate to the Text-generation-webui directory:
    - `cd text-generation-webui`
  - Start the Python environment suitable for your OS:
    - Windows: `cmd_windows.bat`
    - Linux: `./cmd_linux.sh`
    - macOS: `cmd_macos.sh`
    - WSL (Windows Subsystem for Linux): `cmd_wsl.bat`
  - Move into the AllTalk directory:
    - `cd extensions/alltalk_tts`

- **For Standalone AllTalk Users**:
  - Navigate to the `alltalk_tts` folder:
    - `cd alltalk_tts`
  - Start the Python environment:
    - Windows: `start_environment.bat`
    - Linux: `./start_environment.sh`<br><br>
    
   > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

2) Run the diagnostics and select the requirements file name you installed AllTalk with:<br>
   - `python diagnostics.py`

3) You will have an on screen output showing your environment setttings, file versions request vs whats installed and details of your graphics card (if Nvidia). This will also create a file called `diagnostics.log` in the `alltalk_tts` folder, that you can upload if you need to create a support ticket on here.<br><br>

![image](https://github.com/erew123/alltalk_tts/assets/35898566/81b9a6e1-c54b-4da0-b85d-3c6fde566d6a)
<br><br></details></details>

### Installation and Setup Issues

<details>
	<summary>🟨 Understanding Python Environments Simplified</summary><br>
	
Think of Python environments like different rooms in your house, each designed for a specific purpose. Just as you wouldn't cook in the bathroom or sleep in the kitchen, different Python applications need their own "spaces" or environments because they have unique requirements. Sometimes, these requirements can clash with those of other applications (imagine trying to cook a meal in a bathroom!). To avoid this, you can create separate Python environments.

#### Why Separate Environments?
Separate environments, like separate rooms, keep everything organized and prevent conflicts. For instance, one Python application might need a specific version of a library or dependency, while another requires a different version. Just as you wouldn't store kitchen utensils in the bathroom, you wouldn't want these conflicting requirements to interfere with each other. Each environment is tailored and customized for its application, ensuring it has everything it needs without disrupting others.

#### How It Works in Practice:

**Standalone AllTalk Installation:** When you install AllTalk standalone, it's akin to adding a new room to your house specifically designed for your AllTalk activities. The setup process, using the atsetup utility, constructs this custom "room" (Python environment `alltalk_environment`) with all the necessary tools and furnishings (libraries and dependencies) that AllTalk needs to function smoothly, without meddling with the rest of your "house" (computer system). The AllTalk environment is started each time you run `start_alltalk` or `start_environment` within the AllTalk folder.

**Text-generation-webui Installation:** Similarly, installing Text-generation-webui is like setting up another specialized room. Upon installation, it automatically creates its own tailored environment, equipped with everything required for text generation, ensuring a seamless and conflict-free operation. The Text-generation-webui environment is started each time you run `start_*your-os-version*` or `cmd_*your-os-version*` within the Text-generation-webui folder.

#### Managing Environments:
Just as you might renovate a room or bring in new furniture, you can also update or modify Python environments as needed. Tools like Conda or venv make it easy to manage these environments, allowing you to create, duplicate, activate, or delete them much like how you might manage different rooms in your house for comfort and functionality.

Once you're in the right environment, by activating it, installing or updating dependencies (the tools and furniture of your Python application) is straightforward. Using pip, a package installer for Python, you can easily add what you need. For example, to install all required dependencies listed in a requirements.txt file, you'd use:

`pip install -r requirements.txt`

This command tells pip to read the list of required packages and versions from the requirements.txt file and install them in the current environment, ensuring your application has everything it needs to operate. It's like having a shopping list for outfitting a room and ensuring you have all the right items delivered and set up.

Remember, just as it's important to use the right tools for tasks in different rooms of your house, it's crucial to manage your Python environments and dependencies properly to ensure your applications run as intended.

#### How do I know if I am in a Python environment?:
When a Python environment starts up, it changes the command prompt to show the Python environment that it currently running within that terminal/console. 

![image](https://github.com/erew123/screenshots/blob/main/pythonenvironment.jpg)
</details>

<details>
	<summary>🟨 Windows & Python requirements for compiling packages <strong>(ERROR: Could not build wheels for TTS)</strong></summary><br>

`ERROR: Microsoft Visual C++ 14.0 or greater is required` or `ERROR: Could not build wheels for TTS.` or `ModuleNotFoundError: No module named 'TTS`

 Python requires that you install C++ development tools on Windows. This is detailed on the [Python site here](https://wiki.python.org/moin/WindowsCompilers). You would need to install `MSVCv142 - VS 2019 C++ x64/x86 build tools` and `Windows 10/11 SDK` from the C++ Build tools section. 
 
 You can get hold of the **Community** edition [here](https://visualstudio.microsoft.com/downloads/) the during installation, selecting `C++ Build tools` and then `MSVCv142 - VS 2019 C++ x64/x86 build tools` and `Windows 10/11 SDK`. 

![image](https://github.com/erew123/screenshots/raw/main/pythonrequirementswindows.jpg)
 
</details>
<details>
	<summary>🟨 Standalone Install - start_{youros}.xx opens and closes instantly and AllTalk doesnt start</summary><br>

This is more than likely caused by having a space ` ` in your folder path e.g. `c:\program files\alltalk_tts`. In this circumstance you would be best moving the folder to a path without a space e.g. `c:\myfiles\alltalk_tts`. You would have to delete the `alltalk_environment` folder and `start_alltalk.bat` or `start_alltalk.sh` and then re-run `atsetup` to re-create the environment and startup files. 
</details>
<details>
	<summary>🟨 I think AllTalks requirements file has installed something another extension doesn't like</summary><br>
	
Ive paid very close attention to **not** impact what Text-generation-webui is requesting on a factory install. This is one of the requirements of submitting an extension to Text-generation-webui. If you want to look at a comparison of a factory fresh text-generation-webui installed packages (with cuda 12.1, though AllTalk's requirements were set on cuda 11.8) you can find that comparison [here](https://github.com/erew123/alltalk_tts/issues/23). This comparison shows that AllTalk is requesting the same package version numbers as Text-generation-webui or even lower version numbers (meaning AllTalk will not update them to a later version). What other extensions do, I cant really account for that.

I will note that the TTS engine downgrades Pandas data validator to 1.5.3 though its unlikely to cause any issues. You can upgrade it back to text-generation-webui default (december 2023) with `pip install pandas==2.1.4` when inside of the python environment. I have noticed no ill effects from it being a lower or higher version, as far as AllTalk goes. This is also the same behaviour as the Coqui_tts extension that comes with Text-generation-webui.

Other people are reporting issues with extensions not starting with errors about Pydantic e.g. ```pydantic.errors.PydanticImportError: BaseSettings` has been moved to the pydantic-settings package. See https://docs.pydantic.dev/2.5/migration/#basesettings-has-moved-to-pydantic-settings for more details.```

Im not sure if the Pydantic version has been recently updated by the Text-generation-webui installer, but this is nothing to do with AllTalk. The other extension you are having an issue with, need to be updated to work with Pydantic 2.5.x. AllTalk was updated in mid december to work with 2.5.x. I am not specifically condoning doing this, as it may have other knock on effects, but within the text-gen Python environment, you can use `pip install pydantic==2.5.0` or `pip install pydantic==1.10.13` to change the version of Pydantic installed.
</details>
<details>
	<summary>🟨 I am having problems getting AllTalk to start after changing settings or making a custom setup/model setup.</summary><br>
	
I would suggest following [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and if you still have issues after that, you can raise an issue [here](https://github.com/erew123/alltalk_tts/issues)
</details>

### Networking and Access Issues

<details>
	<summary>🟨 I cannot access AllTalk from another machine on my Network</summary><br>

You will need to change the IP address within AllTalk's settings from being 127.0.0.1, which only allows access from the local machine its installed on. To do this, please see [Changing AllTalks IP address & Accessing AllTalk over your Network](https://github.com/erew123/alltalk_tts/tree/main?tab=readme-ov-file#-changing-alltalks-ip-address--accessing-alltalk-over-your-network) at the top of this page.

You may also need to allow access through your firewall or Antivirus package to AllTalk.
</details>

<details>
	<summary>🟨 I am running a Headless system and need to change the IP Address manually as I cannot reach the config page</summary><br>
	
To do this you can edit the `confignew.json` file within the `alltalk_tts` folder. You would look for `"ip_address": "127.0.0.1",` and change the `127.0.0.1` to your chosen IP address,then save the file and start AllTalk.<br><br>

When doing this, be careful not to impact the formatting of the JSON file. Worst case, you can re-download a fresh copy of `confignew.json` from this website and that will put you back to a factory setting.
</details>

### Configuration and Usage Issues
<details>
	<summary>🟨 I activated DeepSpeed in the settings page, but I didnt install DeepSpeed yet and now I have issues starting up</summary><br>
	
You can either follow the [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and fresh install your config. Or you can edit the `confignew.json` file within the `alltalk_tts` folder. You would look for '"deepspeed_activate": true,' and change the word true to false `"deepspeed_activate": false,' ,then save the file and try starting again.

If you want to use DeepSpeed, you need an Nvidia Graphics card and to install DeepSpeed on your system. Instructions are [here](https://github.com/erew123/alltalk_tts#-deepspeed-installation-options)
</details>

<details>
	<summary>🟨 I am having problems updating/some other issue where it wont start up/Im sure this is a bug</summary><br>
	
Please see [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating). If that doesnt help you can raise an ticket [here](https://github.com/erew123/alltalk_tts/issues). It would be handy to have any log files from the console where your error is being shown. I can only losely support custom built Python environments and give general pointers. Please create a `diagnostics.log` report file to submit with a support request.

Also, is your text-generation-webui up to date? [instructions here](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install)
</details>

<details>
	<summary>🟨 I see some red "asyncio" messages</summary><br>
	
As far as I am aware, these are to do with the chrome browser the gradio text-generation-webui in some way. I raised an issue about this on the text-generation-webui [here](https://github.com/oobabooga/text-generation-webui/issues/4788) where you can see that AllTalk is not loaded and the messages persist. Either way, this is more a warning than an actual issue, so shouldnt affect any functionality of either AllTalk or text-generation-webui, they are more just an annoyance.
</details>

### Startup, Performance and Compatibility Issues

<details>
	<summary>🟨 Understanding the AllTalk start-up screen</summary><br>

The AllTalk start-up screen provides various bits of information about the detected Python environment and errors.
 
![image](https://github.com/erew123/screenshots/raw/main/alltalkstartup.jpg)

**Config file check**<br>
- Sometimes I need to add/remove something to your existing configuration file settings. Obviously, I don’t want to impact your existing settings, however any new features may need these settings to be created before AllTalk starts up. Ive added extra code that checks `alltalk_tts/system/config/at_configupdate.json` and `alltalk_tts/system/config/at_configdowngrade.json`, either adding or removing items to your configuration as necessary. If a change is made, you will be notified and a backup of the previous configuration file will be created in the `alltalk_tts` folder.

**AllTalk startup Mode**<br>
- informational. This will state if AllTalk has detected it is running as part of Text-generation-webui or as a Standalone Application.
    
**WAV file deletion**<br>
- If you have set deletion of old generated WAV files, this will state the time frame after which they are purged.

**DeepSpeed version**<br>
- What version of DeepSpeed is installed/detected. This will **not** tell you if the version of DeepSpeed is compiled for your Python, PyTorch or CUDA version. Its important to remember that DeepSpeed has to be compiled for the exact version of Python, PyTorch and CUDA that you are using, so please ensure you have the correct DeepSpeed version installed if necessary.

**Model is available**<br>
- AllTalk is checking if your model files exist. This is not a validity check of the actual model files, they can still be corrupted. If files are missing, AllTalk will attempt to download them from Huggingface, however, if Huggingface has an outage/issue or your internet connection has issues, its possible corrupted or incomplete files will be downloaded. Please read `RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory` if you need to confirm your model files are ok.
    
**Current Python Version**<br>
- Informational. Literally tells you the version of Python running in your Python environment.
    
**Current PyTorch Version**<br>
- Informational. Tell tells you the version of **PyTorch** running in your Python environment, however if you have an Nvidia card, you should be running a CUDA based version of Pytorch. This is indicated with a `+cXXX` after the PyTorch version e.g. `2.2.2+cu121` would be PyTorch version 2.2.2 with CUDA 12.1 extensions. If you don’t have the PyTorch CUDA extensions installed, but you do have an Nvidia card, you may need to re-install PyTorch.

**Current CUDA Version**<br>
- Informational. This is linked to the Current PyTorch Version, as detailed above.

**Current TTS Version**<br>
- Informational. The current version of the TTS engine that is running.

**AllTalk Github updated**<br>
- As long as you have an internet connection, this will tell you last time AllTalk was updated on Github. It is checking the [commit list](https://github.com/erew123/alltalk_tts/commits/main/) to see when the last commit was made. As such, this could be simply a documentation update, a bug fix or new features. Its simply there as a guide to let you know the last time something was changed on AllTalk's Github.

**TTS Subprocess**<br>
- When AllTalk reaches this stage, the subprocess that loads in the AI model is starting. This is most likely where an error could occur with loading the TTS model, just after the documentation message.

**AllTalk Settings & Documentation: http ://x.x.x.x**<br>
- The link where you can reach AllTalks built in settings and documentation page. The TTS model will be loading immediately after this is displayed.
<br><br>
</details>

<details>
	<summary>🟨 AllTalk is only loading into CPU, but I have an Nvidia GPU so it should be loading into CUDA</summary><br>
	
This is caused by Pytorch (Torch) not having the CUDA extensions installed (You can check by running the diagnostics). Typically this happens (on Standalone installations) because when the setup routine goes to install Pytorch with CUDA, it looks in the PIP cache and if a previous application has downloaded a version of Pytorch that **doesn't** have CUDA extensions, the PIP installer doesnt recognise this fact and just uses the cached version for installation. To resolve this:

1) On the `atsetup` utility, on the `Standalone menu` select to `Purge the PIP cache`. This will remove cached packages from the PIP cache, meaning it will have to download fresh copies.
2) As we need to force the upgrade to the Python environment, the easiest way to do this will be to use `atsetup` to `Delete AllTalk's custom Python environment`. This means it will have to rebuild the Python environment. **Note**, you may have to run this step twice, as it has to exit the current Python environment, then you have to re-load `atsetup` and select `Delete AllTalk's custom Python environment` again.
3) You can now use `atsetup` to `Install AllTalk as a Standalone Application` which will download fresh copies of everything and re-install the Python environment. 
4) Once this is done you can check if CUDA is now working with the diagnostics or starting AllTalk and checking the model loads into CUDA.

</details>


<details>
	<summary>🟨 RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory</summary><br>
	
This error message is caused by the model being corrupted or damaged in some way. This error can occur if Huggingface, where the model is downloaded from, have an error (when the model is downloaded) or potentailly internet issues occuring while the model is downloaded on first start-up. 

```
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory

ERROR: Application startup failed. Exiting.
[AllTalk Startup] Warning TTS Subprocess has NOT started up yet, Will keep trying for 120 seconds maximum. Please wait.
```

To resolve this, first look in your `alltalk_tts/models/xttsv2_2.0.2` (or whichever) model folder and confirm that the file sizes are correct.

![image](https://github.com/erew123/screenshots/raw/main/modelsfiles.jpg)

You can delete one or more suspect files and a factory fresh copy of that file or files will be downloaded on next start-up of AllTalk.

</details>

<details>
	<summary>🟨 RuntimeError: Found no NVIDIA driver on your system.</summary><br>
	
This error message is caused by DeepSpeed being enabled when you do not have a Nvidia GPU. To resolve this, edit `confignew.json` and change `"deepspeed_activate": true,` to `"deepspeed_activate": false,` then restart AllTalk.

```
  File "C:\alltalk_tts\alltalk_environment\env\Lib\site-packages\torch\cuda\__init__.py", line 302, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx

ERROR:    Application startup failed. Exiting.
[AllTalk Startup] Warning TTS Subprocess has NOT started up yet, Will keep trying for 120 seconds maximum. Please wait.
```

</details>

<details>
	<summary>🟨 raise RuntimeError("PyTorch version mismatch! DeepSpeed ops were compiled and installed.</summary><br>
	
This error message is caused by having DeepSpeed enabled, but you have a version of DeepSpeed installed that was compiled for a different version of Python, PyTorch or CUDA (or any mix of those). You will need to start your Python environment and run `pip uninstall deepspeed` to remove DeepSpeed from your Python environment and then install the correct version of DeepSpeed.

```
raise RuntimeError("PyTorch version mismatch! DeepSpeed ops were compiled and installed 
RuntimeError: PyTorch version mismatch! DeepSpeed ops were compiled and installed with a different version than what is being used at runtime. Please re-install DeepSpeed or switch torch versions. Install torch version=2.1, Runtime torch version=2.2
```

</details>

<details>
	<summary>🟨 Warning TTS Subprocess has NOT started up yet, Will keep trying for 120 seconds maximum. Please wait. It times out after 120 seconds.</summary><br>
	When the subprocess is starting 2x things are occurring:<br><br>

**A)** Its trying to load the voice model into your graphics card VRAM (assuming you have a Nvidia Graphics card, otherwise its your system RAM)<br>
**B)** Its trying to start up the mini-webserver and send the "ready" signal back to the main process.

Before giving other possibilities a go, some people with **old machines** are finding their startup times are **very** slow 2-3 minutes. Ive extended the allowed time within the script from 1 minute to 2 minutes. **If you have an older machine** and wish to try extending this further, you can do so by editing `script.py` and changing `startup_wait_time = 120` (120 seconds, aka 2 minutes) at the top of the script.py file, to a larger value e.g `startup_wait_time = 240` (240 seconds aka 4 minutes).

**Note:** If you need to create a support ticket, please create a `diagnostics.log` report file to submit with a support request. Details on doing this are above.

Other possibilities for this issue are:

1) You are starting AllTalk in both your `CMD FLAG.txt` and `settings.yaml` file. The `CMD FLAG.txt` you would have manually edited and the `settings.yaml` is the one you change and save in the `session` tab of text-generation-webui and you can `Save UI defaults to settings.yaml`. Please only have one of those two starting up AllTalk.

2) You are not starting text-generation-webui with its normal Python environment. Please start it with start_{your OS version} as detailed [here](https://github.com/oobabooga/text-generation-webui#how-to-install) (`start_windows.bat`,`./start_linux.sh`, `start_macos.sh` or `start_wsl.bat`) OR (`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat` and then `python server.py`).
   
3) You have installed the wrong version of DeepSpeed on your system, for the wrong version of Python/Text-generation-webui. You can go to your text-generation-webui folder in a terminal/command prompt and run the correct cmd version for your OS e.g. (`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`) and then you can type `pip uninstall deepspeed` then try loading it again. If that works, please see here for the correct instructions for installing DeepSpeed [here](https://github.com/erew123/alltalk_tts#-deepspeed-installation-options). 

4) You have an old version of text-generation-webui (pre Dec 2023) I have not tested on older versions of text-generation-webui, so cannot confirm viability on older versions. For instructions on updating the text-generation-webui, please look [here](https://github.com/oobabooga/text-generation-webui#how-to-install) (`update_linux.sh`, `update_windows.bat`, `update_macos.sh`, or `update_wsl.bat`).

5) You already have something running on port 7851 on your computer, so the mini-webserver cant start on that port. You can change this port number by editing the `confignew.json` file and changing `"port_number": "7851"` to `"port_number": "7602"` or any port number you wish that isn’t reserved. Only change the number and save the file, do not change the formatting of the document. This will at least discount that you have something else clashing on the same port number.

6) You have antivirus/firewalling that is blocking that port from being accessed. If you had to do something to allow text-generation-webui through your antivirus/firewall, you will have to do that for this too.

7) You have quite old graphics drivers and may need to update them.

8) Something within text-generation-webui is not playing nicely for some reason. You can go to your text-generation-webui folder in a terminal/command prompt and run the correct cmd version for your OS e.g. (`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`) and then you can type `python extensions\alltalk_tts\script.py` and see if AllTalk starts up correctly. If it does then something else is interfering. 

9) Something else is already loaded into your VRAM or there is a crashed python process. Either check your task manager for erroneous Python processes or restart your machine and try again.

10) You are running DeepSpeed on a Linux machine and although you are starting with `./start_linux.sh` AllTalk is failing there on starting. This is because text-generation-webui will overwrite some environment variables when it loads its python environment. To see if this is the problem, from a terminal go into your text-generation-webui folder and `./cmd_linux.sh` then set your environment variable again e.g. `export CUDA_HOME=/usr/local/cuda` (this may vary depending on your OS, but this is the standard one for Linux, and assuming you have installed the CUDA toolkit), then `python server.py` and see if it starts up. If you want to edit the environment permanently you can do so, I have not managed to write full instructions yet, but here is the conda guide [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#set-env-vars).

11) You have built yourself a custom Python environment and something is funky with it. This is very hard to diagnose as its not a standard environment. You may want to updating text-generation-webui and re installing its requirements file (whichever one you use that comes down with text-generation-webui).
</details>

<details>
	<summary>🟨 I have multiple GPU's and I have problems running Finetuning</summary><br>
	
Finetuning pulls in various other scripts and some of those scripts can have issues with multiple Nvidia GPU's being present. Until the people that created those other scripts fix up their code, there is a workaround to temporarily tell your system to only use the 1x of your Nvidia GPU's. To do this:

- **Windows** - You will start the script with `set CUDA_VISIBLE_DEVICES=0 && python finetune.py`<br>
After you have completed training, you can reset back with `set CUDA_VISIBLE_DEVICES=`<br>
   
- **Linux** - You will start the script with `CUDA_VISIBLE_DEVICES=0 python finetune.py`<br>
After you have completed training, you can reset back with `unset CUDA_VISIBLE_DEVICES`<br>

Rebooting your system will also unset this. The setting is only applied temporarily.

Depending on which of your Nvidia GPU's is the more powerful one, you can change the `0` to `1` or whichever of your GPU's is the most powerful.

</details>

<details>
	<summary>🟨 Firefox - Streaming Audio doesnt work on Firefox</summary><br>
	
This is a long standing issue with Mozilla & Firefox and one I am unable to resolve as Mozilla have not resolved the issue with Firefox. The solution is to use another web browser if you want to use Streaming audio. For details of my prior invesitigation please look at this [ticket](https://github.com/erew123/alltalk_tts/issues/143)
</details>

<details>
	<summary>🟨 Hindi Support - Not working or issues</summary><br>
	
Hindi support does not officially exist according to Coqui. Ive added a limited Hindi support at this time, however, It only works with API TTS method and Im sure there will be issues. [ticket](https://github.com/erew123/alltalk_tts/issues/178)
</details>

### Application Specific Issues

<details>
	<summary>🟨 SillyTavern - I changed my IP address and now SillyTavern wont connect with AllTalk</summary><br>
SillyTavern checks the IP address when loading extensions, saving the IP to its configuration only if the check succeeds. For whatever reason, SillyTavern's checks dont always allow changing its IP address a second time.<br><br>

To manually change the IP address:

1) Navigate to the SillyTavern Public folder located at `/sillytavern/public/`.
2) Open the `settings.json` file.
3) Look for the AllTalk section and find the `provider_endpoint` entry.
3) Replace `localhost` with your desired IP address, for example, `192.168.1.64`.

![image](https://github.com/SillyTavern/SillyTavern/assets/35898566/144e4ac4-87dc-4a2b-8a73-39314abed1ca)
</details>

### TTS Generation Issues & Questions

<details>
	<summary>🟨 XTTS - Does the XTTS AI Model Support Emotion Control or Singing?</summary><br>
	
No, the XTTS AI model does not currently support direct control over emotions or singing capabilities. While XTTS infuses generated speech with a degree of emotional intonation based on the context of the text, users cannot explicitly control this aspect. It's worth noting that regenerating the same line of TTS may yield slightly different emotional inflections, but there is no way to directly control it with XTTS.
</details>
<details>
	<summary>🟨 XTTS - Skips, repeats or pronunciation Issues</summary><br>
	
Firstly, it's important to clarify that the development and maintenance of the XTTS AI models and core scripts are handled by [Coqui](https://docs.coqui.ai/en/latest/index.html), with additional scripts and libraries from entities like [huggingface](https://huggingface.co/docs/transformers/en/index) among many other Python scripts and libraries used by AllTalk. 

AllTalk is designed to be a straightforward interface that simplifies setup and interaction with AI TTS models like XTTS. Currently, AllTalk supports the XTTS model, with plans to include more models in the future. Please understand that the deep inner workings of XTTS, including reasons why it may skip, repeat, or mispronounce, along with 3rd party scripts and libraries utilized, are ultimately outside my control.

Although I ensure the text processed through AllTalk is accurately relayed to the XTTS model speech generation process, and I have aimed to mitigate as many issues as much as possible; skips, repeats and bad pronounciation can still occur.

Certain aspects I have not been able to investigate due to my own time limitations, are:<br>

- The impact of DeepSpeed on TTS quality. Is this more likely to cause skips or repetition?
- Comparative performance between different XTTS model versions (e.g., 2.0.3 vs. 2.0.2) regarding audio quality and consistency.

**From my experience and anecdotally gained knowledge:**<br>

- Lower quality voice samples tend to produce more anomalies in generated speech.
- Model finetuning with high-quality voice samples significantly reduces such issues, enhancing overall speech quality.
- Unused/Excessive punctuation causes issues e.g. asterisks `*`, hashes `#`, brackets `(` `)` etc. Many of these AllTalk will filter out.

So for example, the `female_01.wav` file that is provided with AllTalk is a studio quality voice sample, which the XTTS model was trained on. Typically you will find it unlikely that anomolies occur with TTS generation when using this voice sample. Hence good quality samples and finetuning, generally improve results with XTTS.

If you wish to try out the XTTS version 2.0.3 model and see if it works better, you can download it from [here](https://huggingface.co/coqui/XTTS-v2/tree/v2.0.3), replacing all the files within your `/alltalk_tts/models/xttsv2_2.0.2` folder. This is on my list to both test version 2.0.3 more, but also build a more flexible TTS models downloader, that will not only accomdating other XTTS models, but also other TTS engines. If you try the XTTS version 2.0.3 model and gleen any insights, please let me know.
</details>

---

### ⚫ Finetuning a model
If you have a voice that the model doesnt quite reproduce correctly, or indeed you just want to improve the reproduced voice, then finetuning is a way to train your "XTTSv2 local" model **(stored in `/alltalk_tts/models/xxxxx/`)** on a specific voice. For this you will need:

- An Nvidia graphics card. (Please see the help section [note](https://github.com/erew123/alltalk_tts/edit/main/README.md#performance-and-compatibility-issues) if you have multiple Nvidia GPU's). Preferably 12GB+ VRAM on Windows. Minimum 16GB VRAM on Linux.
- 18GB of disk space free (most of this is used temporarily)
- At least 2 minutes of good quality speech from your chosen speaker in mp3, wav or flacc format, in one or more files (have tested as far as 20 minutes worth of audio).
- As a side note, many people seem to think that the Whisper v2 model (used on Step 1) is giving better results at generating training datasets, so you may prefer to try that, as opposed to the Whisper 3 model.

#### ⚫ How will this work/How complicated is it?
Everything has been done to make this as simple as possible. At its simplest, you can literally just download a large chunk of audio from an interview, and tell the finetuning to strip through it, find spoken parts and build your dataset. You can literally click 4 buttons, then copy a few files and you are done. At it's more complicated end you will clean up the audio a little beforehand, but its still only 4x buttons and copying a few files.

#### ⚫ The audio you will use
I would suggest that if its in an interview format, you cut out the interviewer speaking in audacity or your chosen audio editing package. You dont have to worry about being perfect with your cuts, the finetuning Step 1 will go and find spoken audio and cut it out for you. Is there is music over the spoken parts, for best quality you would cut out those parts, though its not 100% necessary. As always, try to avoid bad quality audio with noises in it (humming sounds, hiss etc). You can try something like [Audioenhancer](https://audioenhancer.ai/) to try clean up noisier audio. There is no need to down-sample any of the audio, all of that is handled for you. Just give the finetuning some good quality audio to work with. 

#### ⚫ Can I Finetune a model more than once on more than one voice
Yes you can. You would do these as multiple finetuning's, but its absolutely possible and fine to do. Finetuning the XTTS model does not restrict it to only being able to reproduce that 1x voice you trained it on. Finetuning is generally nuding the model in a direction to learn the ability to sound a bit more like a voice its not heard before. 

#### ⚫ A note about anonymous training Telemetry information & disabling it
Portions of Coqui's TTS trainer scripts gather anonymous training information which you can disable. Their statement on this is listed [here](https://github.com/coqui-ai/Trainer?tab=readme-ov-file#anonymized-telemetry). If you start AllTalk Finetuning with `start_finetuning.bat` or `./start_finetuning.sh` telemetry will be disabled. If you manually want to disable it, please expand the below:

<details>
	<summary>Manually disable telemetry</summary><br>
	
Before starting finetuning, run the following in your terminal/command prompt:

- On Windows by typing `set TRAINER_TELEMETRY=0`
- On Linux & Mac by typing `export TRAINER_TELEMETRY=0`

Before you start `finetune.py`. You will now be able to finetune offline and no anonymous training data will be sent.
</details>

#### ⚫ Prerequisites for Fine-tuning with Nvidia CUDA Development Toolkit 11.8

All the requirements for Finetuning will be installed by using the atsetup utility and installing your correct requirements (Standalone or for Text-generation-webui). The legacy manual instructions are stored below, however these shouldnt be required.

<details>
	<summary>Legacy manual instructions for installing Nvidia CUDA Development Toolkit 11.8</summary><br>
- To perform fine-tuning, a specific portion of the **Nvidia CUDA Development Toolkit v11.8** must be installed. This is crucial for step 1 of fine-tuning. The objective is to minimize the installation footprint by installing only the essential components.
- The **Nvidia CUDA Development Toolkit v11.8** operates independently from your graphics card drivers and the CUDA version utilized by your Python environment.
- This installation process aims to keep the download and install size as minimal as possible, however a full install of the tookit requires 3GB's of disk space.
- When running Finetuning it will require upto 20GB's of temporary disk space, so please ensure you have this space available and preferably use a SSD or NVME drive.

1. **Download the Toolkit**:
   - Obtain the **network install** version of the Nvidia CUDA Development Toolkit 11.8 from [Nvidia's Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).

2. **Run the Installer**:
   - Choose **Custom (Advanced)** installation.
   - Deselect all options initially.
   - Select the following components:
     - `CUDA` > `Development` > `Compiler` > `nvcc`
     - `CUDA` > `Development` > `Libraries` > `CUBLAS` (**both** development and runtime)

3. **Configure Environment Search Path**:
   - It's essential that `nvcc` and CUDA 11.8 library files are discoverable in your environment's search path. Adjustments can be reverted post-fine-tuning if desired.

     **For Windows**:
     - Edit the `Path` environment variable to include `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`.
     - Add `CUDA_HOME` and set its path to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8.`

     **For Linux**:
     - The path may vary by Linux distribution. Here's a generic setup:
       - `export CUDA_HOME=/usr/local/cuda`
       - `export PATH=${CUDA_HOME}/bin:${PATH}`
       - `export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH`
       
       - Consider adding these to your `~/.bashrc` for permanence, or apply temporarily for the current session by running the above commands each time you start your Python environment.

     **Note**: If using Text-generation-webui, its best to set these temporarily.

4. **Verify Installation**:
   - Open a **new** terminal/command prompt to refresh the search paths.
   - In a terminal or command prompt, execute `nvcc --version`.
   - Success is indicated by a response of `Cuda compilation tools, release 11.8.` Specifically, ensure it is version 11.8.

5. **Troubleshooting**:
   - If the correct version isn't reported, recheck your environment path settings for accuracy and potential conflicts with other CUDA versions.
</details>

#### Additional Note on Torch and Torchaudio:
- Ensure Torch and Torchaudio are CUDA-enabled (any version), which is separate from the CUDA Toolkit installation. CUDA 11.8 corresponds to `cu118` and CUDA 12.1 to `cu121` in AllTalk diagnostics.
- Failure to install CUDA for Torch and Torchaudio will result in Step 2 of fine-tuning failing. These requirements are distinct from the CUDA Toolkit installation, so avoid conflating the two.<br>


#### ⚫ Starting Fine-tuning

**NOTE:** Ensure AllTalk has been launched at least once after any updates to download necessary files for fine-tuning.

1. **Close Resource-Intensive Applications**:
   - Terminate any applications that are using your GPU/VRAM to ensure enough resources for fine-tuning.

2. **Organize Voice Samples**:
   - Place your audio samples into the following directory:
     `/alltalk_tts/finetune/put-voice-samples-in-here/`

Depending on your setup (Text-generation-webui or Standalone AllTalk), the steps to start the Python environment vary:

- **For Standalone AllTalk Users**:
  - Navigate to the `alltalk_tts` folder:
    - `cd alltalk_tts`
  - Start the Python environment:
    - Windows: `start_finetune.bat`
    - Linux: `./start_finetune.sh`
    
- **For Text-generation-webui Users**:
  - Navigate to the Text-generation-webui directory:
    - `cd text-generation-webui`
  - Start the Python environment suitable for your OS:
    - Windows: `cmd_windows.bat`
    - Linux: `./cmd_linux.sh`
    - macOS: `cmd_macos.sh`
    - WSL (Windows Subsystem for Linux): `cmd_wsl.bat`
  - Move into the AllTalk directory:
    - `cd extensions/alltalk_tts`
  - **Linux** users only need to run this command:
    ```
     export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
    ```
  - Start the fine-tuning process with the command:
     - `python finetune.py`<br><br>

   > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

3. **Pre-Flight Checklist**:
   - Go through the pre-flight checklist to ensure readiness. Address any issues flagged as "Fail".

4. **Post Fine-tuning Actions**:
   - Upon completing fine-tuning, the final tab will guide you on managing your files and relocating your newly trained model to the appropriate directory.

These steps guide you through the initial preparations, starting the Python environment based on your setup, and the fine-tuning process itself. Ensure all prerequisites are met to facilitate a smooth fine-tuning experience.

#### ⚫ How many Epochs etc is the right amount?

In finetuning the suggested/recommended amount of epochs, batch size, evaluation percent etc is already set. However, there is no absolutely correct answer to what the settings should be, it all depends on what you are doing. 

- If you just want to train a normal human voice that is in an existing language, for most people’s needs, the base settings would work fine. You may choose to increase the epochs up to maybe 20, or run a second round of training if needed.
- If you were training an entirely new language, you would need a huge amount of training data and it requires around 1000 epochs (based on things I can find around the internet of people who tried this).
- If you are training a cartoon style voice in an existing language, it may need well upwards of 40 epochs until it can reproduce that voice with some success.

There are no absolute correct settings, as there are too many variables, ranging from the amount of samples you are using (5 minutes worth? 4 hours worth? etc), if they are similar samples to what the AI model already understands, so on and so forth. Coqui whom originally trained the model usually say something along the lines of, once you’ve trained it X amount, if it sounds good then you are done and if it doesn’t, train it more.

#### ⚫ Evaluation Data Percentage
In the process of finetuning, it's crucial to balance the data used for training the model against the data reserved for evaluating its performance. Typically, a portion of the dataset is set aside as an 'evaluation set' to assess the model's capabilities in dealing with unseen data. On Step 1 of finetuning you have the option to adjust this evaluation data percentage, offering more control over your model training process.<br><br>
**Why Adjust the Evaluation Percentage?**<br><br>
Adjusting the evaluation percentage **can** be beneficial in scenarios with limited voice samples. When dealing with a smaller dataset, allocating a slightly larger portion to training could enhance the model's ability to learn from these scarce samples. Conversely, with abundant data, a higher evaluation percentage might be more appropriate to rigorously test the model's performance. There are currently no absolutely optimal split percentages as it varies by dataset.
- **Default Setting:** The default evaluation percentage is set at 15%, which is a balanced choice for most datasets.
- **Adjustable Range:** Users can now adjust this percentage, but it’s generally recommend keeping it between 5% and 30%.
  - **Lower Bound:** A minimum of 5% ensures that there's enough data to evaluate model performance.
  - **Upper Bound:** Its suggested not exceeding 30% for evaluation to avoid limiting the amount of data available for training.

- **Understanding the Impact:** Before adjusting this setting, it's important to understand its impact on model training and evaluation. Incorrect adjustments can lead to suboptimal model performance.
- **Gradual Adjustments:** For those unfamiliar with the process, we recommend reading up on training data and training sets, then making small, incremental changes and observing their effects.
- **Data Quality:** Regardless of the split, the quality of the audio data is paramount. Ensure that your datasets are built from good quality audio with enough data within them.

#### ⚫ Using a Finetuned model in Text-generation-webui

At the end of the finetune process, you will have an option to `Compact and move model to /trainedmodel/` this will compact the raw training file and move it to `/model/trainedmodel/`. When AllTalk starts up within Text-generation-webui, if it finds a model in this location a new loader will appear in the interface for `XTTSv2 FT` and you can use this to load your finetuned model. <br><br>**Be careful** not to train a new model from the base model, then overwrite your current `/model/trainedmodel/` **if** you want a seperately trained model. This is why there is an `OPTION B` to move your just trained model to `/models/lastfinetuned/`.

#### ⚫ Training one model with multiple voices

At the end of the finetune process, you will have an option to `Compact and move model to /trainedmodel/` this will compact the raw training file and move it to `/model/trainedmodel/`. This model will become available when you start up finetuning. You will have a choice to train the Base Model or the `Existing finetuned model` (which is the one in `/model/trainedmodel/`). So you can use this to keep further training this model with additional voices, then copying it back to `/model/trainedmodel/` at the end of training.

#### ⚫ Do I need to keep the raw training data/model?

If you've compacted and moved your model, its highly unlikely you would want to keep that data, however the choice is there to keep it if you wish. It will be between 5-10GB in size, so most people will want to delete it.

#### ⚫ I have deeper questions about training the XTTS model, where can I find more information?

If you have deeper questions about the XTTS model, its capabilites, the training process etc, anything thats not covered within the above text or the interface of `finetune.py`, please use the following links to research Coqui's documentation on the XTTS model. 

- https://docs.coqui.ai/en/latest/models/xtts.html
- https://github.com/coqui-ai/TTS
- https://github.com/coqui-ai/TTS/discussions

---

### ⬜ AllTalk TTS Generator
AllTalk TTS Generator is the solution for converting large volumes of text into speech using the voice of your choice. Whether you're creating audio content or just want to hear text read aloud, the TTS Generator is equipped to handle it all efficiently. Please see here for a quick [demo](https://www.youtube.com/watch?v=hunvXn0mLzc)<br><br>The link to open the TTS generator can be found on the built-in Settings and Documentation page.<br><br>**DeepSpeed** is **highly** recommended to speed up generation. **Low VRAM** would be best turned off and your LLM model unloaded from your GPU VRAM (unload your model). **No Playback** will reduce memory overhead on very large generations (15,000 words or more). Splitting **Export to Wav** into smaller groups will also reduce memory overhead at the point of exporting your wav files (so good for low memory systems). 

#### ⬜ Estimated Throughput
This will vary by system for a multitude of reasons, however, while generating a 58,000 word document to TTS, with DeepSpeed enabled, LowVram disabled, splitting size 2 and on an Nvidia RTX 4070, throughput was around 1,000 words per minute. Meaning, this took 1 hour to generate the TTS. Exporting to combined wavs took about 2-3 minutes total.

#### ⬜ Quick Start
- **Text Input:** Enter the text you wish to convert into speech in the 'Text Input' box.
- **Generate TTS:** Hit this to start the text-to-speech conversion.
- **Pause/Resume:** Used to pause and resume the playback of the initial generation of wavs or the stream.
- **Stop Playback:** This will stop the current audio playing back. It does not stop the text from being generated however. 
Once you have sent text off to be generated, either as a stream or wav file generation, the TTS server will remain busy until this process has competed. As such, think carefully as to how much you want to send to the server. 
If you are generating wav files and populating the queue, you can generate one lot of text to speech, then input your next lot of text and it will continue adding to the list.
#### ⬜ Customization and Preferences
- **Character Voice:** Choose the voice that will read your text.
- **Language:** Select the language of your text.
- **Chunk Sizes:** Decide the size of text chunks for generation. Smaller sizes are recommended for better TTS quality.
#### ⬜ Interface and Accessibility
- **Dark/Light Mode:** Switch between themes for your visual comfort.
- **Word Count and Generation Queue:** Keep track of the word count and the generation progress.
#### ⬜ TTS Generation Modes
- **Wav Chunks:** Perfect for creating audio books, or anything you want to keep long term. Breaks down your text into manageable wav files and queues them up. Generation begins automatically, and playback will start after a few chunks have been prepared ahead. You can set the volume to 0 if you don’t want to hear playback. With Wav chunks, you can edit and/or regenerate portions of the TTS as needed.
- **Streaming:** For immediate playback without the ability to save. Ideal for on-the-fly speech generation and listening. This will not generate wav files and it will play back through your browser. You cannot stop the server generating the TTS once it has been sent.<br><br>
With wav chunks you can either playback “In Browser” which is the web page you are on, or “On Server” which is through the console/terminal where AllTalk is running from, or "No Playback". Only generation “In Browser” can play back smoothly and populate the Generated TTS List. Setting the Volume will affect the volume level played back both “In Browser” and “On Server”.<br><br>
For generating **large amounts of TTS**, it's recommended to select the **No Playback** option. This setting minimizes the memory usage in your web browser by avoiding the loading and playing of audio files directly within the browser, which is particularly beneficial for handling extensive audio generations. The definition of large will vary depending on your system RAM availability (will update when I have more information as to guidelines). Once the audio is generated, you can export your list to JSON (for safety) and use the **Play List** option to play back your audio.
#### ⬜ Playback and List Management
- **Playback Controls:** Utilize 'Play List' to start from the beginning or 'Stop Playback' to halt at any time.
- **Custom Start:** Jump into your list at a specific ID to hear a particular section.
- **Regeneration and Editing:** If a chunk isn't quite right, you can opt to regenerate it or edit the text directly. Click off the text to save changes and hit regenerate for the specific line.
- **Export/Import List:** Save your TTS list as a JSON file or import one. Note: Existing wav files are needed for playback. Exporting is handy if you want to take your files away into another program and have a list of which wav is which, or if you keep your audio files, but want to come back at a later date, edit one or two lines, regenerate the speech and re-combine the wav’s into one new long wav.
#### ⬜ Exporting Your Audio
- **Export to WAV:** Combine all generated TTS from the list, into one single WAV file for easy download and distribution. Its always recommended to export your list to a JSON before exporting, so that you have a backup, should something go wrong. You can simply re-import the list and try exporting again.<br><br>When exporting, there is a file size limit of 1GB and as such you have the option to choose how many files to include in each block of audio exported. 600 is just on the limit of 1GB, depending on the average file size, so 500 or less is a good amount to work with. You can combine the generated files after if you wish, in Audacity or similar.<br><br>Additionally, lower export batches will lower the memory requirements, so if your system is low on memory (maybe 8 or 16GB system), you can use smaller export batches to keep the memory requirement down.
#### ⬜ Exporting Subtitles (SRT file)
- **Export SRT:** This will scan through all wav files in your list and generate a subtitles file that will match your exported wav file.
#### ⬜ Analyzing generated TTS for errors
- **Analyze TTS:** This will scan through all wav files comparing each ID's orignal text with the TTS generated for that ID and then flag up inconsistences. Its important to understand this is a **best effort** process and **not 100% perfect**, for example:<br><br>
   - Your text may have the word `their` and the automated routine that listens to your generated TTS interprets the word as `there`, aka a spelling difference.
   - Your text may have `Examples are:` (note the colon) and the automated routine that listens to your generated TTS interprets the word as "Examples are` (note no colon as you cannot sound out a colon in TTS), aka a punctuation difference.
   - Your text may have `There are 100 items` and the automated routine that listens to your generated TTS interprets the word as `There are one hundred items`, aka numbers vs the number written out in words.
   - There will be other examples such as double quotes. As I say, please remember this is a **best effort** to help you identify issues.<br>

As such, there is a `% Accuracy` setting. This uses a couple of methods to try find things that are similar e.g. taking the `their` and `there` example from above, it would identify that they both sound the same, so even if the text says `their` and the AI listening to the generated TTS interprets the word as `there`, it will realise that both sound the same/are similar so there is no need to flag that as an error. However, there are limits to this and some things may slip through or get picked up when you would prefer them not to be flagged.

The higher the accuracy you choose, the more things it will flag up, however you may get more unwanted detections. The lower the less detections. Based on my few tests, accuracy settings between 96 to 98 seem to generally give the best results. However, I would highly recommend you test out a small 10-20 line text and test out the **Analyze TTS** button to get a feel for how it responds to different settings, as well as things it flags up.

You will be able to see the ID's and Text (orignal and as interpreted) by looking at the terminal/command prompt window.

The Analyze TTS feature uses the Whisper Larger-v2 AI engine, which will download on first use if necessary. This will require about 2.5GB's of disk space and could take a few minutes to download, depending on your internet connection.

You can use this feature on systems that do not have an Nvidia GPU, however, unless you have a very powerful CPU, expect it to be slow.

#### ⬜ Tricks to get the model to say things correctly
Sometimes the AI model won’t say something the way that you want it to. It could be because it’s a new word, an acronym or just something it’s not good at for whatever reason. There are some tricks you can use to improve the chances of it saying something correctly.

**Adding pauses**<br>
You can use semi-colons ";" and colons ":" to create a pause, similar to a period "." which can be helpful with some splitting issues.

**Acronyms**<br>
Not all acronyms are going to be pronounced correctly. Let’s work with the word `ChatGPT`. We know it is pronounced `"Chat G P T"` but when presented to the model, it doesn’t know how to break it down correctly. So, there are a few ways we could get it to break out "Chat" and the G P and T. e.g.

`Chat G P T.`
`Chat G,P,T.`
`Chat G.P.T.`
`Chat G-P-T.`
`Chat gee pee tea`

All bar the last one are using ways within the English language to split out "Chat" into one word being pronounced and then split the G, P and T into individual letters. The final example, which is to use Phonetics will sound perfectly fine, but clearly would look wrong as far as human readable text goes. The phonetics method is very useful in edge cases where pronunciation difficult.

#### ⬜ Notes on Usage
- For seamless TTS generation, it's advised to keep text chunks under 250 characters, which you can control with the Chunk sizes.
- Generated audio can be played back from the list, which also highlights the currently playing chunk.
- The TTS Generator remembers your settings, so you can pick up where you left off even after refreshing the page.

---

### 🟠 API Suite and JSON-CURL
### 🟠Overview
The Text-to-Speech (TTS) Generation API allows you to generate speech from text input using various configuration options. This API supports both character and narrator voices, providing flexibility for creating dynamic and engaging audio content.

#### 🟠 Ready Endpoint<br>
Check if the Text-to-Speech (TTS) service is ready to accept requests.

- URL: `http://127.0.0.1:7851/api/ready`<br> - Method: `GET`<br> 

   `curl -X GET "http://127.0.0.1:7851/api/ready"`

  Response: `Ready`

#### 🟠 Voices List Endpoint<br>
Retrieve a list of available voices for generating speech.

- URL: `http://127.0.0.1:7851/api/voices`<br> - Method: `GET`<br>

   `curl -X GET "http://127.0.0.1:7851/api/voices"`

   JSON return: `{"voices": ["voice1.wav", "voice2.wav", "voice3.wav"]}`

#### 🟠 Current Settings Endpoint<br>
Retrieve a list of available voices for generating speech.

- URL: `http://127.0.0.1:7851/api/currentsettings`<br> - Method: `GET`<br>

   `curl -X GET "http://127.0.0.1:7851/api/currentsettings"`

   JSON return: ```{"models_available":[{"name":"Coqui","model_name":"API TTS"},{"name":"Coqui","model_name":"API Local"},{"name":"Coqui","model_name":"XTTSv2 Local"}],"current_model_loaded":"XTTSv2 Local","deepspeed_available":true,"deepspeed_status":true,"low_vram_status":true,"finetuned_model":false}```

  `name & model_name` = listing the currently available models.<br>
  `current_model_loaded` = what model is currently loaded into VRAM.<br>
  `deepspeed_available` = was DeepSpeed detected on startup and available to be activated.<br>
  `deepspeed_status` = If DeepSpeed was detected, is it currently activated.<br>
  `low_vram_status` = Is Low VRAM currently enabled.<br>
  `finetuned_model` = Was a finetuned model detected. (XTTSv2 FT).<br>

#### 🟠 Preview Voice Endpoint
Generate a preview of a specified voice with hardcoded settings.

- URL: `http://127.0.0.1:7851/api/previewvoice/`<br> - Method: `POST`<br> - Content-Type: `application/x-www-form-urlencoded`<br>

   `curl -X POST "http://127.0.0.1:7851/api/previewvoice/" -F "voice=female_01.wav"`

   Replace `female_01.wav` with the name of the voice sample you want to hear.

   JSON return: `{"status": "generate-success", "output_file_path": "/path/to/outputs/api_preview_voice.wav", "output_file_url": "http://127.0.0.1:7851/audio/api_preview_voice.wav"}`

#### 🟠 Switching Model Endpoint<br>

- URL: `http://127.0.0.1:7851/api/reload`<br> - Method: `POST`<br><br>
   `curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=API%20Local"`<br>
   `curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=API%20TTS"`<br>
   `curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=XTTSv2%20Local"`<br>

   Switch between the 3 models respectively.

   `curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=XTTSv2%20FT"`<br>

   If you have a finetuned model in `/models/trainedmodel/` (will error otherwise)

   JSON return `{"status": "model-success"}`

#### 🟠 Switch DeepSpeed Endpoint<br>

- URL: `http://127.0.0.1:7851/api/deepspeed`<br> - Method: `POST`<br><br>
   `curl -X POST "http://127.0.0.1:7851/api/deepspeed?new_deepspeed_value=True"`

   Replace True with False to disable DeepSpeed mode.

   JSON return `{"status": "deepspeed-success"}`

#### 🟠 Switching Low VRAM Endpoint<br>

- URL: `http://127.0.0.1:7851/api/lowvramsetting`<br> - Method: `POST`<br><br>
   `curl -X POST "http://127.0.0.1:7851/api/lowvramsetting?new_low_vram_value=True"`

   Replace True with False to disable Low VRAM mode.

   JSON return `{"status": "lowvram-success"}`

### 🟠 TTS Generation Endpoint (Standard Generation)
Streaming endpoint details are further down the page.

- URL: `http://127.0.0.1:7851/api/tts-generate`<br> - Method: `POST`<br> - Content-Type: `application/x-www-form-urlencoded`<br>

### 🟠 Example command lines (Standard Generation)
Standard TTS generation supports Narration and will generate a wav file/blob. Standard TTS speech Example (standard text) generating a time-stamped file<br>

`curl -X POST "http://127.0.0.1:7851/api/tts-generate" -d "text_input=All of this is text spoken by the character. This is text not inside quotes, though that doesnt matter in the slightest" -d "text_filtering=standard" -d "character_voice_gen=female_01.wav" -d "narrator_enabled=false" -d "narrator_voice_gen=male_01.wav" -d "text_not_inside=character" -d "language=en" -d "output_file_name=myoutputfile" -d "output_file_timestamp=true" -d "autoplay=true" -d "autoplay_volume=0.8"`<br>

Narrator Example (standard text) generating a time-stamped file

`curl -X POST "http://127.0.0.1:7851/api/tts-generate" -d "text_input=*This is text spoken by the narrator* \"This is text spoken by the character\". This is text not inside quotes." -d "text_filtering=standard" -d "character_voice_gen=female_01.wav" -d "narrator_enabled=true" -d "narrator_voice_gen=male_01.wav" -d "text_not_inside=character" -d "language=en" -d "output_file_name=myoutputfile" -d "output_file_timestamp=true" -d "autoplay=true" -d "autoplay_volume=0.8"`<br>

Note that if your text that needs to be generated contains double quotes you will need to escape them with `\"` (Please see the narrator example).

### 🟠 Request Parameters
🟠 **text_input**: The text you want the TTS engine to produce. Use escaped double quotes for character speech and asterisks for narrator speech if using the narrator function. Example:

`-d "text_input=*This is text spoken by the narrator* \"This is text spoken by the character\". This is text not inside quotes."`

🟠 **text_filtering**: Filter for text. Options:

- **none** No filtering. Whatever is sent will go over to the TTS engine as raw text, which may result in some odd sounds with some special characters.<br>
- **standard** Human-readable text and a basic level of filtering, just to clean up some special characters.<br>
- **html** HTML content. Where you are using HTML entity's like &quot;<br>

`-d "text_filtering=none"`<br>
`-d "text_filtering=standard"`<br>
`-d "text_filtering=html"`<br>

Example:

- **Standard Example**: `*This is text spoken by the narrator* "This is text spoken by the character" This is text not inside quotes.`<br>
- **HTML Example**: `&ast;This is text spoken by the narrator&ast; &quot;This is text spoken by the character&quot; This is text not inside quotes.`<br>
- **None**: `Will just pass whatever characters/text you send at it.`<br>

🟠 **character_voice_gen**: The WAV file name for the character's voice.<br>

`-d "character_voice_gen=female_01.wav"`

🟠 **narrator_enabled**: Enable or disable the narrator function. If true, minimum text filtering is set to standard. Anything between double quotes is considered the character's speech, and anything between asterisks is considered the narrator's speech.

`-d "narrator_enabled=true"`<br>
`-d "narrator_enabled=false"` 

🟠 **narrator_voice_gen**: The WAV file name for the narrator's voice.

`-d "narrator_voice_gen=male_01.wav"`

🟠 **text_not_inside**: Specify the handling of lines not inside double quotes or asterisks, for the narrator feature. Options:

- **character**: Treat as character speech.<br>
- **narrator**: Treat as narrator speech.<br>

`-d "text_not_inside=character"`<br>
`-d "text_not_inside=narrator"`

🟠 **language**: Choose the language for TTS. Options:

`ar` Arabic<br>
`zh-cn` Chinese (Simplified)<br>
`cs` Czech<br>
`nl` Dutch<br>
`en` English<br>
`fr` French<br>
`de` German<br>
`hi` Hindi (Please see this re Hindi support, which is very limited https://github.com/erew123/alltalk_tts/issues/178) <br>
`hu` Hungarian<br>
`it` Italian<br>
`ja` Japanese<br>
`ko` Korean<br>
`pl` Polish<br>
`pt` Portuguese<br>
`ru` Russian<br>
`es` Spanish<br>
`tr` Turkish<br>

`-d "language=en"`<br>

🟠 **output_file_name**: The name of the output file (excluding the .wav extension).

`-d "output_file_name=myoutputfile"`<br>

🟠 **output_file_timestamp**: Add a timestamp to the output file name. If true, each file will have a unique timestamp; otherwise, the same file name will be overwritten each time you generate TTS.

`-d "output_file_timestamp=true"`<br>
`-d "output_file_timestamp=false"`

🟠 **autoplay**: Enable or disable playing the generated TTS to your standard sound output device at time of TTS generation.

`-d "autoplay=true"`<br>
`-d "autoplay=false"`

🟠 **autoplay_volume**: Set the autoplay volume. Should be between 0.1 and 1.0. Needs to be specified in the JSON request even if autoplay is false.

`-d "autoplay_volume=0.8"`

### 🟠 TTS Generation Response
The API returns a JSON object with the following properties:

- **status** Indicates whether the generation was successful (generate-success) or failed (generate-failure).<br>
- **output_file_path** The on-disk location of the generated WAV file.<br>
- **output_file_url** The HTTP location for accessing the generated WAV file for browser playback.<br>
- **output_cache_url** The HTTP location for accessing the generated WAV file as a pushed download.<br>

Example JSON TTS Generation Response:

`{"status":"generate-success","output_file_path":"C:\\text-generation-webui\\extensions\\alltalk_tts\\outputs\\myoutputfile_1704141936.wav","output_file_url":"http://127.0.0.1:7851/audio/myoutputfile_1704141936.wav","output_cache_url":"http://127.0.0.1:7851/audiocache/myoutputfile_1704141936.wav"}`

### 🟠 TTS Generation Endpoint (Streaming Generation)
Streaming TTS generation does NOT support Narration and will generate an audio stream. Streaming TTS speech JavaScript Example:<br>

- URL: `http://localhost:7851/api/tts-generate-streaming`<br> - Method: `POST`<br> - Content-Type: `application/x-www-form-urlencoded`<br><br>

```
// Example parameters
const text = "Here is some text";
const voice = "female_01.wav";
const language = "en";
const outputFile = "stream_output.wav";
// Encode the text for URL
const encodedText = encodeURIComponent(text);
// Create the streaming URL
const streamingUrl = `http://localhost:7851/api/tts-generate-streaming?text=${encodedText}&voice=${voice}&language=${language}&output_file=${outputFile}`;
// Create and play the audio element
const audioElement = new Audio(streamingUrl);
audioElement.play(); // Play the audio stream directly
```
- **Text (text):** This is the actual text you want to convert to speech. It should be a string and must be URL-encoded to ensure that special characters (like spaces and punctuation) are correctly transmitted in the URL. Example: `Hello World` becomes `Hello%20World` when URL-encoded.<br>
- **Voice (voice):** This parameter specifies the voice type to be used for the TTS. The value should match one of the available voice options in AllTalks voices folder. This is a string representing the file, like `female_01.wav`.<br>
- **Language (language):** This setting determines the language in which the text should be spoken. A two-letter language code (like `en` for English, `fr` for French, etc.).<br>
- **Output File (output_file):** This parameter names the output file where the audio will be streamed. It should be a string representing the file name, such as `stream_output.wav`. AllTalk will not save this as a file in its outputs folder.<br>

---

### 🔴 Future to-do list
- I am maintaining a list of things people request [here](https://github.com/erew123/alltalk_tts/discussions/74)
- Possibly add some additional TTS engines (TBD).
- Have a break!
