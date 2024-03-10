# AllTalk TTS
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- **Can be run as a** [standalone](https://github.com/erew123/alltalk_tts/#-quick-setup-text-generation-webui--standalone-installation) **or part of** [Text-generation-webui](https://github.com/erew123/alltalk_tts/#-quick-setup-text-generation-webui--standalone-installation) **using the a quick setup utility**
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
- **API Suite and 3rd Party support via JSON calls:** Can be used with 3rd party applications via JSON calls.
- **SillyTavern integration:** Full integration with SillyTavern. [Screenshot](https://github.com/erew123/screenshots/raw/main/sillytavern.jpg)

### Index

- 🟩 [Installation](https://github.com/erew123/alltalk_tts/#-quick-setup-text-generation-webui--standalone-installation)
- 🟪 [Updating & problems with updating](https://github.com/erew123/alltalk_tts?#-updating)
- 🟫 [Screenshots](https://github.com/erew123/alltalk_tts#-screenshots)
- 🟨 [Help with problems](https://github.com/erew123/alltalk_tts?#-help-with-problems)
- ⚫ [Finetuning a model](https://github.com/erew123/alltalk_tts?#-finetuning-a-model)
- 🔵🟢🟡 [DeepSpeed Installation (Windows & Linux)](https://github.com/erew123/alltalk_tts?#-deepspeed-installation-options)
- ⬜ [AllTalk TTS Generator](https://github.com/erew123/alltalk_tts?#-alltalk-tts-generator)
- 🟠 [API Suite and JSON-CURL](https://github.com/erew123/alltalk_tts?#-api-suite-and-json-curl)
- 🔴 [Future to-do list & Upcoming updates](https://github.com/erew123/alltalk_tts?#-future-to-do-list)

### 🔄 Feature requests, Updates & Bug fixes
Please check the below link to find a list of all recent updates and changes.
#### &nbsp;&nbsp;&nbsp;&nbsp;🔄 **Updates list & bug fixes list** can be found [here](https://github.com/erew123/alltalk_tts/issues/25)
#### &nbsp;&nbsp;&nbsp;&nbsp;🔄 **Current Feature request list** can be found [here](https://github.com/erew123/alltalk_tts/discussions/74)

I welcome your input and ideas for new features, suggestions, and improvements. Feel free to share your thoughts and collaborate in the discussions area. If you find this project valuable and would like to show your appreciation, you can make a donation on my [Ko-fi](https://ko-fi.com/erew123) page. Your support goes a long way in ensuring that I can continue to deliver even better features and experiences.

**ERROR** `ImportError: cannot import name 'SampleOutput' from 'transformers.generation.utils'` please see this issue [here](https://github.com/erew123/alltalk_tts/issues/82)

### 🟩 Quick Setup (Text-generation-webui & Standalone Installation)
For Windows 10/11 and Linux machines there is a quick setup script. Please note, Python on Windows requires you install the C++ development [tools](https://wiki.python.org/moin/WindowsCompilers) to compile packages, further details can be found in the help section.

Click to expand the correct section below:
<details>
	<summary>QUICK SETUP - Text-Generation-webui</summary><br>

 If you wish to see this as a video, please go [here](https://www.youtube.com/watch?v=icn2XS5rUH8)
1) To download the AllTalk setup you can either:
   - A) Go to the [Releases page](https://github.com/erew123/alltalk_tts/releases) and download the latest `alltalk_tts.zip` then extract it to the text-generation-webui extensions folder<br>e.g. `\text-generation-webui\extensions\alltalk_tts\`.<br><br>
   - B) Go to a terminal/console, move into the `\text-generation-webui\extensions\` folder<br>and `git clone https://github.com/erew123/alltalk_tts`<br><br>
3) In a terminal/command prompt, in the text-generation-webui folder you will start its Python environment with either `cmd_windows.bat` or `./cmd_linux.sh`
4) Move into the AllTalk folder e.g. `cd extensions` then `cd alltalk_tts`
5) Start the AllTalk setup script `atsetup.bat` or `./atsetup.sh`
6) Follow the on-screen prompts and install the correct requirements files that you need. It's recommended to test AllTalk works before installing DeepSpeed.

   Any time you need to make changes to AllTalk, or use Finetuning etc, always start the Text-generation-webui Python environment first.

   Please read the `🟩 Other installation notes` (also additional voices are available there).

   Finetuning & DeepSpeed have other installation requirements (depending on your OS) so please read any instructions in the setup utility and refer back here to this page for detailed instructions (as needed).<br><br>
</details>
<details>
	<summary>QUICK SETUP - Standalone Installation</summary><br>

 If you wish to see this as a video, please go [here](https://www.youtube.com/watch?v=AQYCccDRbaY)
1) To download the AllTalk setup you can either:
   - A) Go to the [Releases page](https://github.com/erew123/alltalk_tts/releases) and download the latest `alltalk_tts.zip` and extract it to the folder of your choice<br>e.g. `C:\myfiles\alltalk_tts\`.<br><br>
   - B) Go to a terminal/console, move into the folder of your choice e.g `C:\myfiles\` folder<br>and `git clone https://github.com/erew123/alltalk_tts`<br><br>
4) In a terminal/command prompt, move into the AllTalk folder e.g. `cd alltalk_tts`
5) Start the AllTalk setup script `atsetup.bat` or `./atsetup.sh`
6) Follow the on-screen prompts and install the correct requirements files that you need. It's recommended to test AllTalk works before installing DeepSpeed.

   DeepSpeed on Windows machines will be installed as standard. Linux machines have other requirements which are detailed within the setup utility and on this page.

   Please read the `🟩 Other installation notes` (also additional voices are available there).

   You cannot have a dash in your folder path e.g. `c:\myfiles\alltalk_tts-main` so please ensure you remove any `-` from your folder path. This is a conda specific requirement and will cause AllTalk not to start at all.

   Finetuning has other installation requirements so please read any instructions in the setup utility and refer back here to this page for detailed instructions.<br><br>
</details>

### 🟩 Manual Installation - As part of Text generation web UI
On Mac's or if you wish to perform a manual installation. Click to expand the correct section below:
<details>
	<summary>MANUAL INSTALLATION - Text-Generation-webui</summary><br>

This has been tested on the current Dec 2023 release of Text generation webUI. If you have not updated it for a while, you may wish to update Text generation webUI, [instructions here](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install)

- If you want to watch a video of how to do the below [link here](https://youtu.be/9BPKuwaav5w)<br>

1) In a command prompt/terminal window you need to move into your Text generation webUI folder:<br><br>
`cd text-generation-webui`

2) Start the Text generation webUI Python environment for your OS with whichever **one** of the below is correct for your OS:<br><br>
`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`<br><br> Loading Text-generation-webui's Python Environment is $\textcolor{red}{\textsf{VERY IMPORTANT}}$. If you are uncertain what a loaded Python environment looks like, image [here](https://github.com/erew123/alltalk_tts/issues/25#issuecomment-1869344442) and video [here](https://www.youtube.com/watch?v=9BPKuwaav5w)

3) Move into your extensions folder:<br><br>
`cd extensions`

4) Once there git clone this repository:<br><br>
`git clone https://github.com/erew123/alltalk_tts`

5) Move into the **alltalk_tts** folder:<br><br>
`cd alltalk_tts`

6) Install one of the two requirements files. Whichever one of the two is correct for your machine type:<br><br>
**Nvidia graphics card machines** - `pip install -r requirements_nvidia.txt`<br><br>
**Other machines (mac, amd etc)** - `pip install -r requirements_other.txt`

7) **(Optional DeepSpeed)** If you have an Nvidia Graphics card on a system running Linux or Windows and wish to use **DeepSpeed** please follow these instructions [here](https://github.com/erew123/alltalk_tts?#-deepspeed-installation-options). **However**, I would highly reccommend before you install DeepSpeed, you start text-generation-webui up, confirm AllTalk starts correctly and everything is working, as DeepSpeed can add another layer of complications troubleshooting any potential start-up issues. If necessary you can `pip uninstall deepspeed`.

8) You can now start move back to the main Text generation webUI folder `cd ..` (a few times), start Text generation webUI with whichever **one** of the startup scripts is correct for your OS (`start_windows.bat`,`./start_linux.sh`, `start_macos.sh` or `start_wsl.bat`)  and load the AllTalk extension in the Text generation webUI **session** tab.<br><br> Starting Text-generation-webui with its correct start-up script is $\textcolor{red}{\textsf{VERY IMPORTANT}}$.

   Any time you need to make changes to AllTalk, or use Finetuning etc, always start the Text-generation-webui Python environment first.

   Please read the `🟩 Other installation notes` (also additional voices are available there).

   Finetuning & DeepSpeed have other installation requirements (depending on your OS) so please read any instructions in the setup utility and refer back here to this page for detailed instructions (as needed).<br><br>

</details>

### 🟩 Manual Installation - As a Standalone Application
On Mac's or if you wish to perform a manual installation. Click to expand the correct section below:
<details>
	<summary>MANUAL INSTALLATION - I want to run AllTalk as a standalone when installed with Text-generation-webui</summary><br>
	
If you already have AllTalk as a extension of Text-generation-webui, and wish to run it as standalone, load Text-generation-webui's Python environment `cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`, move into the AllTalk folder `cd extensions` > `cd alltalk_tts` and start AllTalk with `python script.py`. There is nothing beyond this you would need to do.
</details>
<details>
	<summary>MANUAL INSTALLATION - I wish to do a custom install of AllTalk</summary><br>
	
AllTalk will run as a standalone app, as long as you install its requirements files into whatever Python environment you are using. You can follow the steps to install the AllTalk's requirements into whatever Python environment you wish. Because I dont know what Python environment you are wanting to use, I can only give you a loose set of installation instructions. 

Please note, at time of writing, the TTS engine requires Python **3.9.x** to **3.11.x** [TTS Engine details here](https://pypi.org/project/TTS/). AllTalk and its requirements are tested on Python **3.11.x**.

#### 🟩 A very quick understanding of Python Environments.
Different Python applications have different requirements, some of those requirement’s conflict with other Python applications requirements. To work around this problem, you can create different Python environments that remain separated from one another. A simple way of looking at Python environments, is just like how your house has different rooms for specific purposes (Kitchen, Bathroom, Bedroom etc). You can create a Python environment that is built/customised specifically for your current applications needs/purposes and will **not** interfere with any other Python applications environments/installations. <br><br>If you are adept at managing Python environments, have an existing Python environment and know that you won’t cause any conflicts by installing AllTalk's requirements within that Python environment, then load up your Python environment and install the requirements. For everyone else, here is a basic guide on installing AllTalk in its own custom Python environment (there will be small variations between OS's, but the principle is the same).

**Note:** A standard VENV can cause module path issues, hence Conda is the correct method to create a Python environment for AllTalk. 

#### 🟩 Building a custom Python Environment with Miniconda
1) Open a terminal/command prompt, and confirm that both `python --version` and `pip` both work. Neither of them should give an error message. If they do you may need to install Python (and maybe Pip).<br><br>[Python Website](https://www.python.org/downloads/)<br>[Pip Website](https://pip.pypa.io/en/stable/installation/)<br><br>Once you have those working you can now continue on.<br><br>

2) Assuming you don't already have Miniconda installed, we will need to download and install Minoconda as this is what we will use to build our custom Python environment.<br><br>[Miniconda Website](https://docs.conda.io/projects/miniconda/en/latest/)<br><br>Download the Miniconda version that is correct for your OS and then install it. Miniconda will create some items in your Start Menu or Application Launcher called something like `Anaconda Prompt`<br><br> You will start the `Anaconda Prompt` and your prompt in your terminal/command prompt will say something like: <br><br>`(base) C:\users\myaccount\>`<br><br> The important bit is that it has the `(base)`, the location after that doesn't matter. `(base)` signifies we are in the base conda Python environment and can now create custom Python environments with Conda.<br><br>
  
3) In your Anaconda Prompt command prompt/terminal, move into the folder where you want to download AllTalk to and then git clone this repository. For simplicity I am going to assume that location is `c:\myfiles\`. So you will:<br><br>`cd myfiles`<br><br>`git clone https://github.com/erew123/alltalk_tts`<br><br>

4) Now we will create our custom Conda Python 3.11.5 environment and give it the name alltalkenv by typing the following at the prompt:<br><br>`conda create --name alltalkenv python=3.11.5`<br><br>You will be prompted if you want to continue and you say yes.<br><br>Once that process has completed, we now need to change from the `(base)` Conda Python environment, to our newly created `(alltalkenv)` environment. So type the following:<br><br>`conda activate alltalkenv`<br><br>Your prompt should now change from `(base) C:\myfiles\>` to `(alltalkenv) C:\myfiles\>`<br><br>

5) **If you are NOT using an Nvidia card and CUDA, skip to step 6**. To force PyTorch to install with CUDA, perform the following:<br><br>`pip cache purge` (Clear the current Pip cache)<br><br>`pip uninstall torch torchaudio` (Uninstall Torch and Torchaudio from the `alltalkenv` environment)<br><br>
You can now either install the CUDA 11.8 or 12.1 version of Torch and Torchaudio with **one** of the following<br><br>**CUDA 11.8** > `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`<br><br>**CUDA 12.1** > `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`<br><br>

6) Move into the alltalk_tts folder and install one of the two requirements files. Whichever one of the two is correct for your machine type:<br><br>`cd alltalk_tts`<br><br>
**Nvidia graphics card machines** - `pip install -r requirements_nvidia.txt`<br><br>
**Other machines (mac, amd etc)** - `pip install -r requirements_other.txt`<br><br>

7) Start AllTalk with `python script.py`<br><br>

🟩 Anytime you wish to use AllTalk, update its requirements, install some other requirements such as DeepSpeed, you will need to start its Conda Python Environment in the terminal/prompt, which you will do with `conda activate alltalkenv` as long as the Conda executable is accessible at the command prompt/terminal you are in.<br><br>It is fully possible to create a batch file or script file that you can use as a launcher from your Start Menu/Application Launcher that will start the environment and AllTalk for you. 

Deepspeed and other such things can be installed. Please read the relevant instructions for those items, however, make the relevant changes to load your correct Python environment when installing any requirements files and starting AllTalk.<br><br>

   Any time you need to make changes to AllTalk, or use Finetuning etc, always start your Python environment first.

   Please read the `🟩 Other installation notes` (also additional voices are available there).

   Finetuning & DeepSpeed have other installation requirements (depending on your OS) so please read any instructions in the setup utility and refer back here to this page for detailed instructions (as needed).<br><br>

</details>


#### 🟩 Other installation notes
On first startup, AllTalk will download the Coqui XTTSv2 2.0.2 model to its **models** folder (1.8GB space required). Check the command prompt/terminal window if you want to know what its doing. After it says "Model Loaded" the Text generation webUI is usually available on its IP address a few seconds later, for you to connect to in your browser. If you are running a headless system and need to change the IP, please see the Help with problems section down below.

Once the extension is loaded, please find all documentation and settings on the link provided in the interface (as shown in the screenshot below).

**Where to find voices** https://aiartes.com/voiceai or https://commons.wikimedia.org/ or interviews on youtube etc. Instructions on how to cut down and prepare a voice sample are within the built in documentation.

Please read the note below about start-up times and also the note about ensuring your character cards are set up [correctly](https://github.com/erew123/alltalk_tts#-a-note-on-character-cards--greeting-messages)

Some extra voices for AllTalk are downloadable [here](https://drive.google.com/file/d/1bYdZdr3L69kmzUN3vSiqZmLRD7-A3M47/view?usp=drive_link) and [here](https://drive.google.com/file/d/1CPnx1rpkuKvVj5fGr9OiUJHZ_e8DfTzP/view)

#### 🟩 Changing the IP address
AllTalk is coded to start on 127.0.0.1, meaning that it will ONLY be accessable to the local computer it is running on. If you want to make AllTalk available to other systems on your network, you will need to change its IP address to match the IP address of your network card. There are 2x ways to change the IP address:

1) Start AllTalk and within its web interface and you can edit the IP address on the "AllTalk Startup Settings".
2) You can edit the `confignew.json`file in a text editor and change `"ip_address": "127.0.0.1",` to the IP address of your choosing.

So, for example, if your computer's network card was on IP address 192.168.0.20, you would change AllTalk's setting to 192.168.1.20 and then **restart** AllTalk. You will need to ensure your machine stays on this IP address each time it is restarted, by setting your machine to have a static IP address.

#### 🟩 A note on Character Cards & Greeting Messages
Messages intended for the Narrator should be enclosed in asterisks `*` and those for the character inside quotation marks `"`. However, AI systems often deviate from these rules, resulting in text that is neither in quotes nor asterisks. Sometimes, text may appear with only a single asterisk, and AI models may vary their formatting mid-conversation. For example, they might use asterisks initially and then switch to unmarked text. A properly formatted line should look like this:

`"`Hey! I'm so excited to finally meet you. I've heard so many great things about you and I'm eager to pick your brain about computers.`"` `*`She walked across the room and picked up her cup of coffee`*`

Most narrator/character systems switch voices upon encountering an asterisk or quotation marks, which is somewhat effective. AllTalk has undergone several revisions in its sentence splitting and identification methods. While some irregularities and AI deviations in message formatting are inevitable, any line beginning or ending with an asterisk should now be recognized as Narrator dialogue. Lines enclosed in double quotes are identified as Character dialogue. For any other text, you can choose how AllTalk handles it: whether it should be interpreted as Character or Narrator dialogue (most AI systems tend to lean more towards one format when generating text not enclosed in quotes or asterisks).

With improvements to the splitter/processor, I'm confident it's functioning well. You can monitor what AllTalk identifies as Narrator lines on the command line and adjust its behavior if needed (Text Not Inside - Function).

### 🟪 Updating
<details>
	<summary>UPDATING - I am using Text-Generation-webui</summary><br>
This is pretty much a repeat of the installation process. 

1) In a command prompt/terminal window you need to move into your Text generation webUI folder:<br><br>
`cd text-generation-webui` and start the Python environment for your OS with whichever **one** of the below is correct for your OS:<br><br>
`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`<br>

2) Move into your extensions and alltalk_tts folder e.g. `cd extensions` then `cd alltalk_tts`

3) At the command prompt/terminal, type `git pull`

4) Install the correct requirements for your machine:<br><br>
**Nvidia graphics card machines** - `pip install -r requirements_nvidia.txt`<br><br>
**Other machines (mac, amd etc)** - `pip install -r requirements_other.txt`<br><br>
</details>
<details>
	<summary>UPDATING - I am running as a Standalone Application</summary><br>
	
1) In a command prompt/terminal window you need to move into your `alltalk_tts` folder and run `start_environment.bat` or `/start_environment.sh` to load the Python environment.

2) At the command prompt/terminal, type `git pull` and wait for it to complete the download.

3) Install the correct requirements for your machine:<br><br>
**Nvidia graphics card machines** - `pip install -r requirements_nvidia.txt`<br><br>
**Other machines (mac, amd etc)** - `pip install -r requirements_other.txt`<br><br>
</details>

#### 🟪 Updating "git pull" error

<details>
	<summary>Click to expand</summary><br>
	
I did leave a mistake in the `/extensions/alltalk_tts/.gitignore` file at one point. If your `git pull` doesnt work, you can either follow the Problems Updating section below, or edit the `.gitignore` file and **replace its entire contents** with the below, save the file, then re-try the `git pull`<br><br>
```
voices/*.*
models/*.*
outputs/*.*
finetune/*.*
config.json
confignew.json
models.json
diagnostics.log
```
</details>

#### 🟪 Updating other problems

<details>
	<summary>Click to expand</summary><br>

If you do experience any problems, the simplest method to resolve this will be:

1) re-name the existing `alltalk_tts` folder to something like `alltalk_tts.old`

2) Start a console/terminal then:<br><br>
`cd text-generation-webui` and start your python environment `cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`

3) Move into the extensions folder, same as if you were doing a fresh installation:<br><br>
`cd extensions` then<br><br>
`git clone https://github.com/erew123/alltalk_tts`

This will download a fresh installation. 

3) Move into the **alltalk_tts** folder:<br><br>
`cd alltalk_tts`

4) Install the correct requirements for your machine:<br><br>
**Nvidia graphics card machines** - `pip install -r requirements_nvidia.txt`<br><br>
**Other machines (mac, amd etc)** - `pip install -r requirements_other.txt`

5) Before starting it up, copy/merge the `models`, `voices` and `outputs` folders over from the `alltalk_tts.old` folder to the newly created `alltalk_tts` folder. This will keep your voices history and also stop it re-downloading the model again.

You can now start text-generation-webui or AllTalk (standalone) and it should start up fine. You will need to re-set any saved configuration changes on the configuration page. 

Assuming its all working fine and you are happy, you can delete the old alltalk_tts.old folder.
</details>

### 🟫 Screenshots
|![image](https://github.com/erew123/screenshots/raw/main/textgensettings.jpg) | ![image](https://github.com/erew123/screenshots/raw/main/setuputilitys.jpg) | ![image](https://github.com/erew123/screenshots/raw/main/deepspeed.jpg) |![image](https://github.com/erew123/screenshots/raw/main/textgen.jpg) |
|:---:|:---:|:---:|:---:|
|![image](https://github.com/erew123/screenshots/raw/main/settingsanddocs.jpg) | ![image](https://github.com/erew123/screenshots/raw/main/finetune1.jpg) | ![image](https://github.com/erew123/screenshots/raw/main/finetune2.jpg) |![image](https://github.com/erew123/screenshots/raw/main/sillytavern.jpg)|

### 🟨 Help with problems

#### &nbsp;&nbsp;&nbsp;&nbsp; 🔄 **Minor updates/bug fixes list** can be found [here](https://github.com/erew123/alltalk_tts/issues/25)

#### 🟨 How to make a diagnostics report file
If you are on a Windows machine or a Linux machine, you should be able to use the `atsetup.bat` or `./atsetup.sh` utility to create a diagnositcs file. If you are unable to use the `atsetup` utility, please follow the instructions below.
<details>
	<summary>Click to expand</summary>
	
1) Open a command prompt window, move into your **text-generation-webui** folder, you can now start the Python environment for text-generation-webui:<br><br>
`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`

2) Move into the **alltalk_tts** folder:<br><br>
`cd extensions` and then `cd alltalk_tts`

3) Run the diagnostics and select the requirements file name you installed AllTalk with:<br><br>
`python diagnostics.py`

4) You will have an on screen output showing your environment setttings, file versions request vs whats installed and details of your graphics card (if Nvidia). This will also create a file called `diagnostics.log` in the `alltalk_tts` folder, that you can upload if you need to create a support ticket on here.<br><br>

![image](https://github.com/erew123/alltalk_tts/assets/35898566/81b9a6e1-c54b-4da0-b85d-3c6fde566d6a)
<br><br></details>


#### 🟨 [AllTalk Startup] Warning TTS Subprocess has NOT started up yet, Will keep trying for 120 seconds maximum. Please wait. It times out after 120 seconds.

<details>
	<summary>Click to expand</summary><br>
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

#### 🟨 Windows & Python requirements for compiling packages

`ERROR: Microsoft Visual C++ 14.0 or greater is required` or `ERROR: Could not build wheels for TTS.` or `ModuleNotFoundError: No module named 'TTS`

<details>
	<summary>Click to expand</summary><br>

 Python requires that you install C++ development tools on Windows. This is detailed on the [Python site here](https://wiki.python.org/moin/WindowsCompilers). You would need to install `MSVCv142 - VS 2019 C++ x64/x86 build tools` and `Windows 10/11 SDK` from the C++ Build tools section. 
 
 You can get hold of the **Community** edition [here](https://visualstudio.microsoft.com/downloads/) the during installation, selecting `C++ Build tools` and then `MSVCv142 - VS 2019 C++ x64/x86 build tools` and `Windows 10/11 SDK`. 

![image](https://github.com/erew123/screenshots/raw/main/pythonrequirementswindows.jpg)
 
</details>

#### 🟨 I think AllTalks requirements file has installed something another extension doesn't like
<details>
	<summary>Click to expand</summary><br>
	
Ive paid very close attention to **not** impact what Text-generation-webui is requesting on a factory install. This is one of the requirements of submitting an extension to Text-generation-webui. If you want to look at a comparison of a factory fresh text-generation-webui installed packages (with cuda 12.1, though AllTalk's requirements were set on cuda 11.8) you can find that comparison [here](https://github.com/erew123/alltalk_tts/issues/23). This comparison shows that AllTalk is requesting the same package version numbers as Text-generation-webui or even lower version numbers (meaning AllTalk will not update them to a later version). What other extensions do, I cant really account for that.

I will note that the TTS engine downgrades Pandas data validator to 1.5.3 though its unlikely to cause any issues. You can upgrade it back to text-generation-webui default (december 2023) with `pip install pandas==2.1.4` when inside of the python environment. I have noticed no ill effects from it being a lower or higher version, as far as AllTalk goes. This is also the same behaviour as the Coqui_tts extension that comes with Text-generation-webui.

Other people are reporting issues with extensions not starting with errors about Pydantic e.g. ```pydantic.errors.PydanticImportError: BaseSettings` has been moved to the pydantic-settings package. See https://docs.pydantic.dev/2.5/migration/#basesettings-has-moved-to-pydantic-settings for more details.```

Im not sure if the Pydantic version has been recently updated by the Text-generation-webui installer, but this is nothing to do with AllTalk. The other extension you are having an issue with, need to be updated to work with Pydantic 2.5.x. AllTalk was updated in mid december to work with 2.5.x. I am not specifically condoning doing this, as it may have other knock on effects, but within the text-gen Python environment, you can use `pip install pydantic==2.5.0` or `pip install pydantic==1.10.13` to change the version of Pydantic installed.
</details>

#### 🟨 I activated DeepSpeed in the settings page, but I didnt install DeepSpeed yet and now I have issues starting up
<details>
	<summary>Click to expand</summary><br>
	
You can either follow the [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and fresh install your config. Or you can edit the `confignew.json` file within the `alltalk_tts` folder. You would look for '"deepspeed_activate": true,' and change the word true to false `"deepspeed_activate": false,' ,then save the file and try starting again.<br><br>

If you want to use DeepSpeed, you need an Nvidia Graphics card and to install DeepSpeed on your system. Instructions are [here](https://github.com/erew123/alltalk_tts#-deepspeed-installation-options)
</details>

#### 🟨 I cannot access AllTalk from another machine
<details>
	<summary>Click to expand</summary><br>
You will need to change the IP address within AllTalk's settings from being 127.0.0.1, which only allows access from the local machine its installed on. To do this, please see [here](https://github.com/erew123/alltalk_tts/edit/main/README.md#-changing-the-ip-address)<br><br>

You may also need to allow access through your firewall or Antivirus package to AllTalk.
</details>

#### 🟨 I am running a Headless system and need to change the IP Address manually as I cannot reach the config page.
<details>
	<summary>Click to expand</summary><br>
	
To do this you can edit the `confignew.json` file within the `alltalk_tts` folder. You would look for `"ip_address": "127.0.0.1",` and change the `127.0.0.1` to your chosen IP address,then save the file and start AllTalk.<br><br>

When doing this, be careful not to impact the formatting of the JSON file. Worst case, you can re-download a fresh copy of `confignew.json` from this website and that will put you back to a factory setting.
</details>

#### 🟨 I am having problems updating/some other issue where it wont start up/Im sure this is a bug
<details>
	<summary>Click to expand</summary><br>
	
Please see [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating). If that doesnt help you can raise an ticket [here](https://github.com/erew123/alltalk_tts/issues). It would be handy to have any log files from the console where your error is being shown. I can only losely support custom built Python environments and give general pointers. Please create a `diagnostics.log` report file to submit with a support request.<br><br>

Also, is your text-generation-webui up to date? [instructions here](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install)
</details>

#### 🟨 Standalone Install - start_{youros}.xx opens and closes instantly and AllTalk doesnt start
<details>
	<summary>Click to expand</summary><br>

This is more than likely caused by having a `-` in your folder path e.g. `c:\myfiles\alltalk_tts-main`. In this circumstance you would be best renaming the folder to remove the `-` from its name e.g. `c:\myfiles\alltalk_tts`, delete the `alltalk_environment` folder and `start_alltalk.bat` or `start_alltalk.sh` and then re-run `atsetup` to re-create the environment and startup files. 

</details>

#### 🟨 I am having problems getting AllTalk to start after changing settings or making a custom setup/model setup.

<details>
	<summary>Click to expand</summary><br>
	
I would suggest following [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and if you still have issues after that, you can raise an issue [here](https://github.com/erew123/alltalk_tts/issues)
</details>

#### 🟨 I see some red "asyncio" messages

<details>
	<summary>Click to expand</summary><br>
	
As far as I am aware, these are to do with the chrome browser the gradio text-generation-webui in some way. I raised an issue about this on the text-generation-webui [here](https://github.com/oobabooga/text-generation-webui/issues/4788) where you can see that AllTalk is not loaded and the messages persist. Either way, this is more a warning than an actual issue, so shouldnt affect any functionality of either AllTalk or text-generation-webui, they are more just an annoyance.
</details>

#### 🟨 I have multiple GPU's and I have problems running Finetuning

<details>
	<summary>Click to expand</summary><br>
	
Finetuning pulls in various other scripts and some of those scripts can have issues with multiple Nvidia GPU's being present. Until the people that created those other scripts fix up their code, there is a workaround to temporarily tell your system to only use the 1x of your Nvidia GPU's. To do this:

- **Windows** - You will start the script with `set CUDA_VISIBLE_DEVICES=0 && python finetune.py`<br>
After you have completed training, you can reset back with `set CUDA_VISIBLE_DEVICES=`<br>
   
- **Linux** - You will start the script with `CUDA_VISIBLE_DEVICES=0 python finetune.py`<br>
After you have completed training, you can reset back with `unset CUDA_VISIBLE_DEVICES`<br>

Rebooting your system will also unset this. The setting is only applied temporarily.

Depending on which of your Nvidia GPU's is the more powerful one, you can change the `0` to `1` or whichever of your GPU's is the most powerful.

</details>

### ⚫ Finetuning a model
If you have a voice that the model doesnt quite reproduce correctly, or indeed you just want to improve the reproduced voice, then finetuning is a way to train your "XTTSv2 local" model **(stored in `/alltalk_tts/models/xxxxx/`)** on a specific voice. For this you will need:

- An Nvidia graphics card. (Please see this [note](https://github.com/erew123/alltalk_tts#-i-have-multiple-gpus-and-i-have-problems-running-finetuning) if you have multiple Nvidia GPU's).
- To install a few portions of the Nvidia CUDA 11.8 Toolkit (this will not impact text-generation-webui's cuda setup.
- 18GB of disk space free (most of this is used temporarily)
- At least 2 minutes of good quality speech from your chosen speaker in mp3, wav or flacc format, in one or more files (have tested as far as 20 minutes worth of audio).
- As a side note, many people seem to think that the Whisper v2 model (used on Step 1) is giving better results at generating training datasets, so you may prefer to try that, as opposed to the Whisper 3 model.

#### ⚫ How will this work/How complicated is it?
Everything has been done to make this as simple as possible. At its simplest, you can literally just download a large chunk of audio from an interview, and tell the finetuning to strip through it, find spoken parts and build your dataset. You can literally click 4 buttons, then copy a few files and you are done. At it's more complicated end you will clean up the audio a little beforehand, but its still only 4x buttons and copying a few files.

#### ⚫ The audio you will use
I would suggest that if its in an interview format, you cut out the interviewer speaking in audacity or your chosen audio editing package. You dont have to worry about being perfect with your cuts, the finetuning Step 1 will go and find spoken audio and cut it out for you. Is there is music over the spoken parts, for best quality you would cut out those parts, though its not 100% necessary. As always, try to avoid bad quality audio with noises in it (humming sounds, hiss etc). You can try something like [Audioenhancer](https://audioenhancer.ai/) to try clean up noisier audio. There is no need to down-sample any of the audio, all of that is handled for you. Just give the finetuning some good quality audio to work with. 

#### ⚫ A note about anonymous training Telemetry information & disabling it
Portions of Coqui's TTS trainer scripts gather anonymous training information which you can disable. Their statement on this is listed [here](https://github.com/coqui-ai/Trainer?tab=readme-ov-file#anonymized-telemetry). Although I have tried to pass `TRAINER_TELEMETRY=0` through AllTalk, it appears you will have to set this manually if you wish to finetune on a non-internet enabled computer or disable the anonymous data being sent. You can do this by:

- On Windows by typing `set TRAINER_TELEMETRY=0`
- On Linux & Mac by typing `export TRAINER_TELEMETRY=0`

Before you start `finetune.py`. You will now be able to finetune offline and no anonymous training data will be sent. 

#### ⚫ Important requirements CUDA 11.8
As mentioned you must have a small portion of the Nvidia CUDA Toolkit **11.8** installed. Not higher or lower versions. Specifically **11.8**. You do not have to uninstall any other versions, change any graphics drivers, reinstall torch or anything like that. This requirement is for Step1 of Finetuning. To keep the download+install as small as possible, you will need to:
- Download the **xxx (network)** install of the Nvidia CUDA Toolkit 11.8 from [here](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- Run the installer. At minimum, you need to [minimally] install the `nvcc` compiler and the `CUBLAS` development and runtime libraries:
  - Select **Custom Advanced** as your installation type.
  - Uncheck all the checkboxes in the list.
  - Now check the following elements:
    - `CUDA` > `Development` > `Compiler` > `nvcc`
    - `CUDA` > `Development` > `Compiler` > `Libraries` > `CUBLAS`
    - `CUDA` > `Runtime` > `Libraries` > `CUBLAS`
  - You can now proceed through the install.
- When that has installed, open a terminal/command prompt and type `nvcc --version`. If it reports back `Cuda compilation tools, release 11.8.` you are good to go. **Specifically, 11.8**. If not continue to the next step.
- For both Windows and Linux, you need to ensure that `nvcc` and the 11.8 cuda library files are in your environments search path. You can undo the changes below after finetuning if you prefer:
  - **Windows**: Edit the Windows `Path` environment variable and add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
  - **Linux**: The path may be different depending on what flavour of Linux you are running, so you may need to seek out specific instructions on the internet. Generic paths **may** be:
    - `export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}` and
    - `export LD_LIBRARY_PATH=/usr/local/cuda-11.8/bin`
    - Add these to your '~/.bashrc' if you want this to be permanent and not something you have to set each time you open a new terminal.
- When you have made the changes, open a **new** terminal/command prompt (in order to load the new search paths) and `nvcc --version`. It should report back `Cuda compilation tools, release 11.8.` at which point, you are good to go.
- If it doesn't report that, check you have correctly set the search environment paths, dont have overlapping other versions of cuda paths etc.

**Note:** Its also important that your Torch and Torchaudio have cuda installed (of any version). If you run the AllTalk diagnostics you can see your Torch and Torchaudio versions listed there. Cuda 11.8 will be listed as `cu118` and Cuda 12.1 as `cu121`. This Torch and Torchaudio is seperate to the above requirement to have the Nvidia CUDA Toolkit installed, so dont confuse the two different requirements. If you dont have Cuda installed on Torch and Torchaudio, Step 2 of Finetuning will fail.

#### ⚫ Starting Finetuning
**NOTE:** Please make sure you have started AllTalk at least once after updating, so that it downloads the additional files needed for finetuning. 

1) Close all other applications that are using your GPU/VRAM and copy your audio samples into:<br><br>
   `/alltalk_tts/finetune/put-voice-samples-in-here/`
3) In a command prompt/terminal window you need to move into your Text generation webUI folder:<br><br>
`cd text-generation-webui`

4) Start the Text generation webUI Python environment for your OS:<br><br>
`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`

5) You can double check your search path environment still works correctly with `nvcc --version`. It should report back 11.8:<br><br>
   `Cuda compilation tools, release 11.8.`

7) Move into your extensions folder:<br><br>
`cd extensions`

8) Move into the **alltalk_tts** folder:<br><br>
`cd alltalk_tts`

9) Install the finetune requirements file: `pip install -r requirements_finetune.txt`

10) Type `python finetune.py` and it should start up.
11) Follow the on-screen instructions when the web interface starts up.
12) When you have finished finetuning, the final tab will tell you what to do with your files and how to move your newly trained model to the correct location on disk.

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

### 🔵🟢🟡 DeepSpeed Installation Options
**NOTE**: You **DO NOT** need to set Text-generation-webUI's **--deepspeed** setting for AllTalk to be able to use DeepSpeed. These are two completely separate things and incorrectly setting that on Text-generation-webUI may cause other complications.

#### 🔵 Linux Installation
<details>
	<summary>Click to expand: Linux DeepSpeed installation</summary>

➡️DeepSpeed requires an Nvidia Graphics card!⬅️

- #### DeepSpeed Installation for Text generation webUI

1) Preferably use your built in package manager to install CUDA tools. Alternatively download and install the Nvidia Cuda Toolkit for Linux [Nvidia Cuda Toolkit 11.8 or 12.1](https://developer.nvidia.com/cuda-toolkit-archive)<br><br>
2) Open a terminal console.<br><br>
3) Install libaio-dev (however your Linux version installs things) e.g. `sudo apt install libaio-dev`<br><br>
4) Move into your Text generation webUI folder e.g. `cd text-generation-webui`<br><br>
5) Start the Text generation webUI Python environment `./cmd_linux.sh`<br><br>
6) Text generation webUI **overwrites** the **CUDA_HOME** environment variable each time you `./cmd_linux.sh` or `./start_linux.sh`, so you will need to either permanently change within the python environment OR set CUDA_HOME it each time you `./cmd_linux.sh`. Details to change it each time are on the next step. Below is a link to Conda's manual and changing environment variables permanently though its possible changing it permanently could affect other extensions, you would have to test.<br> <br>
[Conda manual - Environment variables](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#set-env-vars)<br><br>
7) You can temporarily set the **CUDA_HOME** environment with (Standard paths on Ubuntu, but could vary on other Linux flavours):<br><br>
`export CUDA_HOME=/etc/alternatives/cuda`<br><br>
**every** time you run `./cmd_linux.sh`.<br> <br>
If you try to start DeepSpeed with the CUDA_HOME path set incorrectly, expect an error similar to `[Errno 2] No such file or directory: /home/yourname/text-generation-webui/installer_files/env/bin/nvcc`<br> <br>
You can also add the `export` line into the start scripts just below the `conda active .......` line, to avoid running the command each time. <br> <br>
8) Now install deepspeed with pip install deepspeed<br><br>
9) You can now start Text generation webUI `python server.py` ensuring to activate your extensions.<br><br>
Just to reiterate, starting Text-generation-webUI with `./start_linux.sh` will overwrite the CUDA_HOME variable unless you have permanently changed it, hence always starting it with `./cmd_linux.sh` **then** setting the environment variable manually (step 7) and **then** `python server.py`, which is how you would need to run it each time, unless you permanently set the environment variable for CUDA_HOME within Text-generation-webUI's standard Python environment.
<br><br>
**Removal** - If it became necessary to uninstall DeepSpeed, you can do so with `./cmd_linux.sh` and then `pip uninstall deepspeed`<br><br>

- #### DeepSpeed Installation for standalone alltalk_tts app

1) Preferably use your built in package manager to install CUDA tools. Alternatively download and install the Nvidia Cuda Toolkit for Linux [Nvidia Cuda Toolkit 11.8 or 12.1](https://developer.nvidia.com/cuda-toolkit-archive)<br><br>
2) Open a terminal console.<br><br>
3) Install libaio-dev (however your Linux version installs things) e.g. `sudo apt install libaio-dev` If you are using an RPM based distribution the package will probably be named `libaio-devel`.<br><br> 
4) Navigate to your alltalk_tts folder e.g. `cd alltext_tts`<br><br>
5) Start the alltalk_tts Python environment `conda activate alltalkenv`<br><br>
6) You will need to set the **CUDA_HOME** environment variable. Details to change it each time are on the next step. Below is a link to Conda's manual and changing environment variables permanently though its possible changing it permanently could affect other extensions, you would have to test.<br> <br>
[Conda manual - Environment variables](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#set-env-vars)<br><br>
7) Set the environment path for CUDA_HOME e.g. `export CUDA_HOME=/usr/local/cuda-12.1/bin` or `export CUDA_HOME=/etc/alternative/cuda`.  Verify the path on your Linux distro before setting this variable.<br>
If you try to start DeepSpeed with the CUDA_HOME path set incorrectly, expect an error similar to `[Errno 2] No such file or directory` or `CUDA_HOME does not exist`<br> <br>
If you set the `env` variable and you receive an error of `[Errno 2] No such file or directory: '/usr/local/cuda-12.1/bin/bin/nvcc'` with a double `bin/bin` path, you can remove the env variable with `unset CUDA_HOME` and add the path to your linux PATH env variable with: `export PATH=$PATH:/usr/local/cuda-12.1/bin`. (Verify the path on your Linux distro before setting this variable.)  However if you are using other applications that use CUDA, you will want to verify that configuring the path did not break those applications. <br><br> 
8) Now install deepspeed with `pip install deepspeed`<br><br>
9) You can now start the alltalk_tts webUI with `python script.py`<br><br>
<br><br>
**Removal** - If it became necessary to uninstall DeepSpeed, you can do so by entering your `alltalkenv` enviroment `conda activate alltalkenv` and then running `pip uninstall deepspeed`<br><br>


</details>

#### 🟢🟡 Windows Installation
<details>
	<summary>Click to Expand: Windows DeepSpeed options</summary><br>
	
DeepSpeed v11.2 will work on the current default text-generation-webui Python 3.11 environment! You have 2x options for how to setup DeepSpeed on Windows. A quick way (🟢Option 1) and a long way (🟡Option 2).

Thanks to [@S95Sedan](https://github.com/S95Sedan) - They managed to get DeepSpeed 11.2 working on Windows via making some edits to the original Microsoft DeepSpeed v11.2 installation. The original post is [here](https://github.com/oobabooga/text-generation-webui/issues/4734#issuecomment-1843984142).

#### 🟢 OPTION 1 - Quick and easy!
<details>
	<summary>Click to expand: Pre-Compiled Wheel Deepspeed v11.2 (Python 3.11 and 3.10)</summary>
➡️DeepSpeed requires an Nvidia Graphics card!⬅️<br>

1) Download the correct wheel version for your Python/Cuda from [here](https://github.com/erew123/alltalk_tts/releases/tag/deepspeed) and save the file it inside your **text-generation-webui** folder.

2) Open a command prompt window, move into your **text-generation-webui folder**, you can now start the Python environment for text-generation-webui:<br><br>
`cmd_windows.bat`

3) With the file that you saved in the **text-generation-webui folder** you now type the following, replacing YOUR-VERSION with the name of the file you have:<br><br>
`pip install "deepspeed-0.11.2+YOUR-VERSION-win_amd64.whl"`
   
5) This should install through cleanly and you should now have DeepSpeed v11.2 installed within the Python 3.11/3.10 environment of text-generation-webui.
   
6) When you start up text-generation-webui, and AllTalk starts, you should see **[AllTalk Startup] DeepSpeed Detected**
    
7) Within AllTalk, you will now have a checkbox for **Activate DeepSpeed** though remember you can only change **1x setting every 15 or so seconds**, so dont try to activate DeepSpeed **and** LowVRAM/Change your model simultantiously. Do one of those, wait 15-20 seconds until the change is confirmed in the terminal/command prompt, then you can change the other. When you are happy it works, you can set the default start-up settings in the settings page.
<br><br>
**Removal** - If it became necessary to uninstall DeepSpeed, you can do so with `cmd_windows.bat` and then `pip uninstall deepspeed`<br><br>
</details>

#### 🟡 OPTION 2 - A bit more complicated!
<details>
	<summary>Click to expand: Manual Build DeepSpeed v11.2 (Python 3.11 and 3.10)</summary>
➡️DeepSpeed requires an Nvidia Graphics card!⬅️<br><br>

This will take about 1 hour to complete and about 6GB of disk space.<br>

1. Download the 11.2 release of [DeepSpeed](https://github.com/microsoft/DeepSpeed/releases/tag/v0.11.2) extract it to a folder. 
2. Install Visual C++ build tools, such as [VS2019 C++ x64/x86](https://learn.microsoft.com/en-us/visualstudio/releases/2019/redistribution#vs2019-download) build tools.
3. Download and install the [Nvidia Cuda Toolkit 11.8 or 12.1](https://developer.nvidia.com/cuda-toolkit-archive)
4. **OPTIONAL** If you do not have an python environment already created and you are **not** going to use Text-generation-webui's environment, you can install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html), then at a command prompt, create and activate your environment with:<br><br>
```conda create -n pythonenv python=3.11```<br>
```activate pythonenv```<br>

5. Launch the Command Prompt cmd with Administrator privilege as it requires admin to allow creating symlink folders.
6. If you are using the **Text-generation-webui** python environment, then in the `text-generation-webui` folder you will run `cmd_windows.bat` to start the python evnironment.<br><br>
Otherwise Install PyTorch, 2.1.0 with CUDA 11.8 or 12.1 into your Python 3.1x.x environment e.g:<br><br>
```activate pythonenv``` (activate your python environment)<br>
```conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia```<br>
or<br>
```conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia```<br>

9. Set your CUDA Windows environment variables in the command prompt to ensure that CUDA_HOME and CUDA_PATH are set to your Nvidia Cuda Toolkit path. (The folder above the bin folder that nvcc.exe is installed in). Examples are:<br><br>
```set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1```<br>
```set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1```<br>
or<br>
```set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8```<br>
```set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8```<br>

10. Navigate to wherever you extracted the deepspeed folder in the Command Prompt:<br><br>
```cd c:\DeepSpeed-0.11.2``` (wherever you extracted it to)

11. Modify the following files:<br>
**(These modified files are included in the git-pull of AllTalk, in the DeepSpeed Windows folder and so can just be copied over the top of the exsting folders/files, but if you want to modify them yourself, please follow the below)**<br>

deepspeed-0.11.2/build_win.bat - at the top of the file, add:<br><br>
 ```set DS_BUILD_EVOFORMER_ATTN=0```

deepspeed-0.11.2/csrc/quantization/pt_binding.cpp - lines 244-250 - change to:
```
    std::vector<int64_t> sz_vector(input_vals.sizes().begin(), input_vals.sizes().end());
    sz_vector[sz_vector.size() - 1] = sz_vector.back() / devices_per_node;  // num of GPU per nodes
    at::IntArrayRef sz(sz_vector);
    auto output = torch::empty(sz, output_options);

    const int elems_per_in_tensor = at::numel(input_vals) / devices_per_node;
    const int elems_per_in_group = elems_per_in_tensor / (in_groups / devices_per_node);
    const int elems_per_out_group = elems_per_in_tensor / out_groups;
```

deepspeed-0.11.2/csrc/transformer/inference/csrc/pt_binding.cpp
lines 541-542 - change to:
```
									 {static_cast<unsigned>(hidden_dim * InferenceContext::Instance().GetMaxTokenLength()),
									  static_cast<unsigned>(k * InferenceContext::Instance().GetMaxTokenLength()),
```

lines 550-551 - change to:
```
						 {static_cast<unsigned>(hidden_dim * InferenceContext::Instance().GetMaxTokenLength()),
						  static_cast<unsigned>(k * InferenceContext::Instance().GetMaxTokenLength()),
```
line 1581 - change to:
```
		at::from_blob(intermediate_ptr, {input.size(0), input.size(1), static_cast<int64_t>(mlp_1_out_neurons)}, options);
```

deepspeed-0.11.2/deepspeed/env_report.py
line 10 - add:
```
import psutil
```
line 83 - 100 - change to:
```
def get_shm_size():
    try:
        temp_dir = os.getenv('TEMP') or os.getenv('TMP') or os.path.join(os.path.expanduser('~'), 'tmp')
        shm_stats = psutil.disk_usage(temp_dir)
        shm_size = shm_stats.total
        shm_hbytes = human_readable_size(shm_size)
        warn = []
        if shm_size < 512 * 1024**2:
            warn.append(
                f" {YELLOW} [WARNING] Shared memory size might be too small, consider increasing it. {END}"
            )
            # Add additional warnings specific to your use case if needed.
        return shm_hbytes, warn
    except Exception as e:
        return "UNKNOWN", [f"Error getting shared memory size: {e}"]
```

10. While still in your command line with python environment enabled run:<br>
```build_win.bat``` and wait 10-20 minutes.

11. Now `cd dist` to go into your dist folder and you can now `pip install deepspeed-YOURFILENAME.whl` (or whatever your WHL file is called).
<br><br>
**Removal** - If it became necessary to uninstall DeepSpeed, you can do so with `cmd_windows.bat` and then `pip uninstall deepspeed`<br><br>
</details>
</details>

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
#### ⬜ Customization and Preferences
- **Character Voice:** Choose the voice that will read your text.
- **Language:** Select the language of your text.
- **Chunk Sizes:** Decide the size of text chunks for generation. Smaller sizes are recommended for better TTS quality.
#### ⬜ Interface and Accessibility
- **Dark/Light Mode:** Switch between themes for your visual comfort.
- **Word Count and Generation Queue:** Keep track of the word count and the generation progress.
#### ⬜ Notes on Usage
- For seamless TTS generation, it's advised to keep text chunks under 250 characters, which you can control with the Chunk sizes.
- Generated audio can be played back from the list, which also highlights the currently playing chunk.
- The TTS Generator remembers your settings, so you can pick up where you left off even after refreshing the page.

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
<br>

### 🔴 Future to-do list
- I am maintaining a list of things people request [here](https://github.com/erew123/alltalk_tts/discussions/74)
- Possibly add some additional TTS engines (TBD).
- Have a break!
