# AllTalk TTS v2 BETA
This is the BETA of v2. To be clear, that means they may well be bugs, issues, missing/incomplete documentation, a variety of potential problems etc (This is what BETA refers to in the software world for those whom dont know). This means you may need a bit of technical know how to deal with things that could come up and should only continue if you feel comfortable with that.

Github discussions on the BETA are [here in the discussion board](https://github.com/erew123/alltalk_tts/discussions/245)

#### If you are going to run the BETA, please read all the info below....

The BETA has been tested on Windows 11 and Lunux (Ubuntu). The Standalone installations `atsetup` **should** take care of everything for you, all the way through to having DeepSpeed up and running on both Windows and Linux.

Seperate from Standalone installations, installation within the Text-generation-webui Python environment should also work, however, I have had to various bits of code last minute and not had an opportuninty to deeply test this yet.

### Screenshots
Please go see [here](https://github.com/erew123/alltalk_tts/discussions/237)

### New Features include

- Multiple TTS engines. Currently XTTS, Piper, Parler and VITS. New engines are easy to install with some coding experience. In time I will add in all engine Coqui supports as they should be easy.
  - Each engine has its own custom settings
  - You can download the models for each engine within its settings area.
  - FYI there is no limit on how many XTTS models AllTalk will now find/work with.
- AllTalk starts up on 0.0.0.0, meaning it will bind to ALL available IP addresses. There is no setting its IP address any more.
- Updated Coqui TTS engine.
- Gradio web interface (see the screenshots)
- Retrieval based Voice Conversion, AKA, RVC voice Pipeline, re-written to work on Python 3.11.
- TTS Generator will now use any TTS engine.
- Vastly updated API Suite. **V2 is not 100% compatable with v1. (See here for an explanation https://github.com/erew123/alltalk_tts/issues/166)**. 
  - New SillyTavern extension is included in the `/system/` folder.
  - New remote TGWUI extension if you want to run your TTS elsewhere from the AllTalk server.
- OpenAI compatable endpoint/API. Meaning this should work as an alternative endpoint with any software that can send a TTS generation request to OpenAI.
- Fully customisable API Settings. I have **NOT** tested any limits of any TTS engines.
- Updated Finetuning for XTTS models
- Audio Transcoding to about 6x formats (mp3, opus, etc)
- About 50 gradio interface themes if you arent happy with the standard Gradio one.
- Documentaton has been about 80% updated. Its built into the interface.
- Lots and lots of other things too numerous to mention.

There is a welcome screen that covers a few other bits, along with the built in documentation.

### To install the BETA build, read the notes below for the type of system you have, the read the Quick Setup section to perform the install.

---

### Windows Systems
- You need to install **Espeak-ng**. You will find a copy of this in the `...\alltalk_tts_v2\system\espeak-ng\` folder. If you dont install it, you will get a warning and various TTS engines will probably crash.
- If you have NEVER run Python on your system EVER, Windows users must install C++ development tools for Python to compile Python packages. This ia a Python requirement not an AllTalk specifc requirement. Detailed information and a link to these tools can be found in the help section [**Windows & Python requirements for compiling packages**](https://github.com/erew123/alltalk_tts#-help-with-problems).

### Linux Systems
You need to install a few bits (depending on your Linux flavour), otherwise DeepSpeed will fail and some TTS engines not work. At your terminal type the following:
- **Debian-based systems** `sudo apt install libaio-dev espeak-ng ffmpeg gcc g++`
- **RPM-based systems** `sudo yum install libaio-devel espeak-ng ffmpeg gcc g++`

As mentioned, the **atsetup.sh** install for Standalone installations **should** install deepspeed automatically. 

TGWUI - If however you are wanting to install in TGWUI and you can look here for a simpler way to get DeepSpeed installed here https://github.com/erew123/alltalk_tts/releases/tag/DeepSpeed-14.2-Linux, however, after 20+ hours wrestling with the conda python toolkit installations (on Linux), it looks like Nvidia may have a fault in their installation that can break 6x of the symlinks in the conda environment folder. They are reasonably easy to fix **if needed** and the script in `/system/config/fixsymlinks.sh` **may** do it for you, though I havnt tested on every OS with every eventuality etc.

### Mac Systems
I do not have a Mac system personally, so cannot test things out. In theory, all the TTS engines **should** work on Mac, however Ive been unable to write an installation routine for AllTalk. I have a loose idea of what to do. You would need to setup a miniconda for mac with Python 3.11.9 and Pytorch 2.2.1. You would need to install (brew) espeak-ng and ffmpeg (like Linux). I think at that point in time, once you have a working environment, you should be able to use the `requirements_standalone.txt` file with pip, though you may want to remove the nvidia cudnn bits, as mac cant use those. You would have to `conda install pytorch::faiss-cpu` and also `conda install conda-forge::ffmpeg`. The sticking point, if you wanted to use RVC is that you need a version of Fairseq that would work on Mac and Python 3.11.x. The only way I can think of doing that would be to compile a build of this https://github.com/VarunGumma/fairseq (if its possible), though I have no idea how to do that on a mac. So using RVC on Mac may be a no-no, or it may be possible for someone who wants to try figuring out Fairseq 12.2+ on mac.

### ALL Systems - General requirements
When I say general requirements, this is what I have tested it on. If you vary outside of this setup, expect issues. I will not have time to figure out why it doesnt work on Python version x and Pytorch version x, I dont have the resource/time available.<br>
- **Python version** 3.11.9<br>
- **PyTorch version** 2.2.1 with CUDA 12.1<br>
- **Standalone Installation Disk space** Used **during** installation may go as high as **23GB** of space. This is cleared down at the end of installation. Windows users can expect about 12GB of disk space before you install any models. Linux users can expect about 16GB of space, due to the extra need for the CUDA Toolkit.<br>
- **TGWUI Installation Disk Space** Will add about 2-3 GB onto the TGWUI Python environment.<br>

---

### 🟩 Quick Setup (Text-generation-webui & Standalone Installation)

Quick setup scripts are available for users on Windows 10/11 and Linux. Instructional videos for both setup processes are linked below.

- Ensure that **Git** is installed on your system as it is required for cloning the repository. If you do not have Git installed, visit [**Git's official website**](https://git-scm.com/downloads) to download and install it.
- **Important**: Do not use dashes or spaces in your folder path (e.g. avoid `/my folder-is-this/alltalk_tts-main`) as this causes issues with Python.

### **NOTE. IF YOU HAVE AN EXISTING ALLTALK INSTALL** this will want to create a folder called `alltalk_tts` so you will need to re-name your old `alltalk_tts` to something else OR put this in a different folder/location. 
### **NOTE**. I am not saying this is a over-the-top of an old alltalk upgrade ATM. It needs the Python environment rebuilding, hence a fresh install is needed for now. 

<details>
<summary>QUICK SETUP - Text-Generation-webui</summary>
<br>

For a step-by-step video guide, click [here](https://www.youtube.com/watch?v=icn2XS5rUH8).

To set up AllTalk within Text-generation-webui, follow either method:

1. **Download AllTalk Setup**:
   - **Via Terminal/Console (Recommended)**:
     - `cd \text-generation-webui\extensions\`
     - `git clone -b alltalkbeta https://github.com/erew123/alltalk_tts`

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
     - Clone the AllTalk repository: `git clone -b alltalkbeta https://github.com/erew123/alltalk_tts`

2. **Start AllTalk Setup**:
   - Open a terminal/command prompt, move to the AllTalk directory, and run the setup script:
     - `cd alltalk_tts`
     - Windows: `atsetup.bat`
     - Linux: `./atsetup.sh`

3. **Follow the Setup Prompts**:
   - Select Standalone Installation and then Option 1 and follow any on-screen instructions to install the required files. DeepSpeed is automatically installed, but will only work on Nvidia GPU's.

> If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

</details>

---


### 🟪 Updating

As long as you did the `git clone` method to setup initially, you will be able to go into the folder and use `git pull` to download updates.

There was an issue with the `.gitignore` file, so if you cannot git pull on a build that is pre 11 June 2024, you can `git reset --hard origin/alltalkbeta` and then git pull (your config file will be reset).

---

### 🆘 Support Requests, Troubleshooting, BETA Discussions & Feature requests
Current concerns with the v2 BETA are around, is this working and what needs doing to bring it or the documentation up to speed and not trying to introduce lots of additional features at this time. AKA, I want to make sure its stable etc.

If you wish to code something yourself though, thats perfectly to do and youre welcome to discuss that with me if needed.

General discussions on the BETA should be [here in the discussion board](https://github.com/erew123/alltalk_tts/discussions/245)

If you have a specifc technical problem, if you think its a quick/simple thing you can discuss it on the discussions board, however, if its a big issue or going to turn into one, lets keep that on an Issues ticket.

---
