# AllTalk TTS
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- **Custom Start-up Settings:** Adjust your default start-up settings. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Narrarator:** Use different voices for main character and narration. [Example Narration](https://vocaroo.com/18fYWVxiQpk1)
- **Low VRAM mode:** Great for people with small GPU memory or if your VRAM is filled by your LLM. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **DeepSpeed:** A 3-4x performance boost generating TTS. [DeepSpeed Windows/Linux Instructions](https://github.com/erew123/alltalk_tts?tab=readme-ov-file#deepspeed-installation-options) [Screenshot](https://github.com/erew123/alltalk_tts/assets/35898566/548619c8-5f1b-47d0-a73d-54d2fee3f3db)
- **Local/Custom models:** Use any of the XTTSv2 models (API Local and XTTSv2 Local).
- **Optional wav file maintenance:** Configurable deletion of old output wav files. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Documentation:** Fully documented with a built in webpage. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Console output** Clear command line output for any warnings or issues.
- **Standalone/3rd Party support via JSON calls** Can be used with 3rd party applications via JSON calls.

## Index

- 🟩 [Installation](https://github.com/erew123/alltalk_tts?#-installation-on-text-generation-web-ui)
- 🟪 [Updating & problems with updating](https://github.com/erew123/alltalk_tts?#-updating)
- 🟫 [Screenshots](https://github.com/erew123/alltalk_tts#-screenshots)
- 🟨 [Help with problems](https://github.com/erew123/alltalk_tts?#-help-with-problems)
- 🔵🟢🟡 [DeepSpeed Installation (Windows & Linux)](https://github.com/erew123/alltalk_tts?#-deepspeed-installation-options)
- 🔴 [Future to-do list & Upcoming updates](https://github.com/erew123/alltalk_tts?#-future-to-do-list)

#### Updates
The latest build (13 Dec 2023) has had the entire text filtering engine and narration engine rebuilt from scratch. It's highly complicated how its actually working, but the end result it a much clearer TTS output and much better control over the narrator option and how to handle text that isnt within quotes or asterisks. It does however mean you need to ensure your character card is set up correctly if using the narrator function. Details are below in the installation notes.

DeepSpeed **v11.2** can be installed within the **default text-generation-webui Python 3.11 environment**. Installs in custom Python environments are possible, but can be more complicated. Instructions [here](https://github.com/erew123/alltalk_tts#deepspeed-112-for-windows--python-311) (or scroll down).

## 🟩 Installation on Text generation web UI
This has been tested on the current Dec 2023 release of Text generation webUI. If you have not updated it for a while, you may wish to update Text generation webUI, [instructions here](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install)

1) In a command prompt/terminal window you need to move into your Text generation webUI folder:<br><br>
`cd text-generation-webui`

2) Start the Text generation webUI Python environment for your OS:<br><br>
`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`

3) Move into your extensions folder:<br><br>
`cd extensions`

4) Once there git clone this repository:<br><br>
`git clone https://github.com/erew123/alltalk_tts`

5) Move into the **alltalk_tts** folder:<br><br>
`cd alltalk_tts`

6) Install the requirements:<br><br>
*Nvidia graphics card machines* - `pip install -r requirements_nvidia.txt`<br><br>
*Other machines (mac, amd etc)* - `pip install -r requirements_other.txt`

7) **(Optional DeepSpeed)** If you have an Nvidia Graphics card on a system running Linux or Windows and wish to use **DeepSpeed** please follow these instructions [here](https://github.com/erew123/alltalk_tts?#-deepspeed-installation-options). **However**, I would highly reccommend before you install DeepSpeed, you start text-generation-webui up, confirm AllTalk starts correctly and everything is working, as DeepSpeed can add another layer of complications troubleshooting any potential start-up issues. If necessary you can `pip uninstall deepspeed`.

8) You can now start move back to the main Text generation webUI folder `cd ..` (a few times), start Text generation webUI (`start_windows.bat`,`./start_linux.sh`, `start_macos.sh` or `start_wsl.bat`)  and load the AllTalk extension in the Text generation webUI **session** tab.
   
9) Please read the note below about start-up times and also the note about ensuring your character cards are set up [correctly](https://github.com/erew123/alltalk_tts#the-one-thing-i-cant-easily-work-around)

10) Some extra voices downloadable [here](https://drive.google.com/file/d/1bYdZdr3L69kmzUN3vSiqZmLRD7-A3M47/view?usp=drive_link)

#### 🟩 Other installation notes
On first startup, AllTalk will download the Coqui XTTSv2 2.0.2 model to its **models** folder (1.8GB space required). Check the command prompt/terminal window if you want to know what its doing. After it says "Model Loaded" the Text generation webUI is usually available on its IP address a few seconds later, for you to connect to in your browser.

Once the extension is loaded, please find all documentation and settings on the link provided in the interface (as shown in the screenshot below).

**Where to find voices** https://aiartes.com/voiceai or https://commons.wikimedia.org/ or interviews on youtube etc. Instructions on how to cut down and prepare a voice sample are within the built in documentation.

#### 🟩 The one thing I cant easily work around
Narrator function specific - With a RP chat with your AI, **on your character card** `parameters menu` > `character tab` > `greeting` make sure that anything in there that is the **narrator is in asterisks** and anything **spoken is in double quotes**, then hit the `save` (💾) button. Greeting paragraphs/sentences are handled differently from how the AI sends text and so its difficut to account for them both.

I could force a delimeter in at this stage, but I know it would/may affect things further down the line in the chat and I need a good think about that before just making a change. This issue **only** affects the greeting card/start of conversation and the "example" card that comes with text-generation-webui suffers this issue (if you want to try it for yourself). So you would put double quotes around like this (from the example card):

`"`*Hey! I'm so excited to finally meet you. I've heard so many great things about you and I'm eager to pick your brain about computers. I'm sure you have a wealth of knowledge that I can learn from.*`"`

## 🟪 Updating
This is pretty much a repeat of the installation process. 

1) In a command prompt/terminal window you need to move into your Text generation webUI folder:<br><br>
`cd text-generation-webui`

2) Move into your extensions and alltalk_tts folder:<br><br>
`cd extensions` then `cd alltalk_tts`

3) At the command prompt/terminal, type:<br><br>
`git pull`

4) Install the requirements:<br><br>
*Nvidia graphics card machines* - `pip install -r requirements_nvidia.txt`<br><br>
*Other machines (mac, amd etc)* - `pip install -r requirements_other.txt`

#### 🟪 **git pull error** 
I did leave a mistake in the `/extensions/alltalk_tts/.gitignore` file at one point. If your `git pull` doesnt work, you can either follow the Problems Updating section below, or edit the `.gitignore` file and **replace its entire contents** with the below, save the file, then re-try the `git pull`<br><br>
```
voices/*.*
models/*.*
outputs/*.*
config.json
confignew.json
models.json
```
#### 🟪 Problems Updating

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

4) Install the requirements:<br><br>
*Nvidia graphics card machines* - `pip install -r requirements_nvidia.txt`<br><br>
*Other machines (mac, amd etc)* - `pip install -r requirements_other.txt`

5) Before starting it up, copy/merge the `models`, `voices` and `outputs` folders over from the `alltalk_tts.old` folder to the newly created `alltalk_tts` folder. This will keep your voices history and also stop it re-downloading the model again.

You can now start text-generation-webui or AllTalk (standalone) and it should start up fine. You will need to re-set any saved configuration changes on the configuration page. 

Assuming its all working fine and you are happy, you can delete the old alltalk_tts.old folder.

## 🟫 Screenshots
|![image](https://github.com/erew123/alltalk_tts/assets/35898566/4ca9b4c7-60ce-4ac6-82e5-fd1989b84644) | ![image](https://github.com/erew123/alltalk_tts/assets/35898566/b0e13dba-c6b1-4ab7-845d-244ac1158330) |
|:---:|:---:|
|![image](https://github.com/erew123/alltalk_tts/assets/35898566/548619c8-5f1b-47d0-a73d-54d2fee3f3db) | ![image](https://github.com/erew123/alltalk_tts/assets/35898566/e35e987c-543a-486b-b4fb-ee6ebe6f59c6) |

## 🟨 Help with problems

#### 🟨 I activated DeepSpeed in the settings page, but I didnt install DeepSpeed yet and now I have issues starting up

You can either follow the [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and fresh install your config. Or you can edit the `confignew.json` file within the `alltalk_tts` folder. You would look for '"deepspeed_activate": true,' and change the word true to false `"deepspeed_activate": false,' ,then save the file and try starting again.

If you want to use DeepSpeed, you need an Nvidia Graphics card and to install DeepSpeed on your system. Instructions are [here](https://github.com/erew123/alltalk_tts#-deepspeed-installation-options)

#### 🟨 I am having problems updating/some other issue where it wont start up/Im sure this is a bug

Please see [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating). If that doesnt help you can raise an ticket [here](https://github.com/erew123/alltalk_tts/issues). It would be handy to have any log files from the console where your error is being shown. I can only losely support custom built Python environments and give general pointers.

Also, is your text-generation-webui up to date? [instructions here](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install)

#### 🟨 I get "[AllTalk Startup] Warning TTS Subprocess has NOT started up yet, Will keep trying for 60 seconds maximum. Please wait." and then it times out after 60 seconds.

When the subprocess is starting 2x things are occurring:

**A)** Its trying to load the voice model into your graphics card VRAM (assuming you have a Nvidia Graphics card, otherwise its your system RAM)<br>
**B)** Its trying to start up the mini-webserver and send the "ready" signal back to the main process.

Possibilities for this issue are (in no particular order):

1) You are starting AllTalk in both your `CMD FLAG.txt` and `settings.yaml` file. The `CMD FLAG.txt` you would have manually edited and the `settings.yaml` is the one you change and save in the `session` tab of text-generation-webui and you can `Save UI defaults to settings.yaml`. Please only have one of those two starting up AllTalk.

2) You already have something running on port 7851 on your computer, so the mini-webserver cant start on that port. You can change this port number by editing the `confignew.json` file and changing `"port_number": "7851"` to `"port_number": "7602"` or any port number you wish that isn’t reserved. Only change the number and save the file, do not change the formatting of the document. This will at least discount that you have something else clashing on the same port number.

3) You have antivirus/firewalling that is blocking that port from being accessed. If you had to do something to allow text-generation-webui through your antivirus/firewall, you will have to do that for this too.

4) You have an old version of text-generation-webui (pre Dec 2023) I have not tested on older versions of text-generation-webui, so cannot confirm viability on older versions. For instructions on updating the text-generation-webui, please look [here](https://github.com/oobabooga/text-generation-webui#how-to-install) (`update_linux.sh`, `update_windows.bat`, `update_macos.sh`, or `update_wsl.bat`).

5) You are not starting text-generation-webui with its normal Python environment. Please start it with start_{your OS version} as detailed [here](https://github.com/oobabooga/text-generation-webui#how-to-install) (`start_windows.bat`,`./start_linux.sh`, `start_macos.sh` or `start_wsl.bat`) OR (`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat` and then `python server.py`). 

6) You have quite old graphics drivers and may need to update them.

7) You have installed the wrong version of DeepSpeed on your system, for the wrong version of Python/Text-generation-webui. You can go to your text-generation-webui folder in a terminal/command prompt and run the correct cmd version for your OS e.g. (`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`) and then you can type `pip uninstall deepspeed` then try loading it again. If that works, please see here for the correct instructions for installing DeepSpeed [here](https://github.com/erew123/alltalk_tts#-deepspeed-installation-options).

8) Something within text-generation-webui is not playing nicely for some reason. You can go to your text-generation-webui folder in a terminal/command prompt and run the correct cmd version for your OS e.g. (`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`) and then you can type `python extensions\alltalk_tts\script.py` and see if AllTalk starts up correctly. If it does then something else is interfering. 

9) You have not installed the requirements file. Please check the installation instructions.

10) Something else is already loaded into your VRAM or there is a crashed python process. Either check your task manager for erroneous Python processes or restart your machine and try again.

11) You are running DeepSpeed on a Linux machine and although you are starting with `./start_linux.sh` AllTalk is failing there on starting. This is because text-generation-webui will overwrite some environment variables when it loads its python environment. To see if this is the problem, from a terminal go into your text-generation-webui folder and `./cmd_linux.sh` then set your environment variable again e.g. `export CUDA_HOME=/usr/local/cuda` (this may vary depending on your OS, but this is the standard one for Linux, and assuming you have installed the CUDA toolkit), then `python server.py` and see if it starts up. If you want to edit the environment permanently you can do so, I have not managed to write full instructions yet, but here is the conda guide [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#set-env-vars).

12) Unlikely but you possibly have a corrupted voice model file. You can delete the `xttsv2_2.0.2` folder inside your `\alltalk_tts\models` folder and it will download an updated model on next startup.

13) You have built yourself a custom Python environment and something is funky with it. This is very hard to diagnose as its not a standard environment. You may want to updating text-generation-webui and re installing its requirements file (whichever one you use that comes down with text-generation-webui).

#### 🟨 I am having problems getting AllTalk to start after changing settings or making a custom setup/model setup.

I would suggest following [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and if you still have issues after that, you can raise an issue [here](https://github.com/erew123/alltalk_tts/issues)

#### 🟨 I see some red "asyncio" messages

As far as I am aware, these are to do with the chrome browser the gradio text-generation-webui in some way. I raised an issue about this on the text-generation-webui [here](https://github.com/oobabooga/text-generation-webui/issues/4788) where you can see that AllTalk is not loaded and the messages persist. Either way, this is more a warning than an actual issue, so shouldnt affect any functionality of either AllTalk or text-generation-webui, they are more just an annoyance.

## 🔵🟢🟡 DeepSpeed Installation Options
#### 🔵 Linux Installation
<details>
	<summary>Click to expand: Linux DeepSpeed installation</summary>

➡️DeepSpeed requires an Nvidia Graphics card!⬅️

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
9) Now install deepspeed with pip install deepspeed<br><br>
10) You can now start Text generation webUI `python server.py` ensuring to activate your extensions.<br><br>
Just to reiterate, starting Text-generation-webUI with `./start_linux.sh` will overwrite the CUDA_HOME variable unless you have permanently changed it, hence always starting it with `./cmd_linux.sh` **then** setting the environment variable manually (step 7) and **then** `python server.py`, which is how you would need to run it each time, unless you permanently set the environment variable for CUDA_HOME within Text-generation-webUI's standard Python environment.
<br><br>
**Removal** - If it became necessary to uninstall DeepSpeed, you can do so with `./cmd_linux.sh` and then `pip uninstall deepspeed`<br><br>
</details>
	
#### 🟢🟡 Windows Installation
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

### 🔴 Future to-do list
- Complete & document the new/full standalone mode API.
- Voice output within the command prompt/terminal (TBD).
- Correct an issue on incorrect output folder path when running as a standalone app.
- Correct a few spelling mistakes in the documnentation.
- Possibly add some additional TTS engines (TBD).
- Have a break!
