# AllTalk TTS
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- **Custom Start-up Settings:** Adjust your default start-up settings. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Narrarator:** Use different voices for main character and narration. [Example Narration](https://vocaroo.com/18fYWVxiQpk1)
- **Low VRAM mode:** Improve generation performance if your VRAM is filled by your LLM. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **DeepSpeed:** A 3-4x performance boost generating TTS. [DeepSpeed Windows/Linux Instructions](https://github.com/erew123/alltalk_tts?tab=readme-ov-file#deepspeed-installation-options) [Screenshot](https://github.com/erew123/alltalk_tts/assets/35898566/548619c8-5f1b-47d0-a73d-54d2fee3f3db)
- **Local/Custom models:** Use any of the XTTSv2 models (API Local and XTTSv2 Local).
- **Optional wav file maintenance:** Configurable deletion of old output wav files. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Documentation:** Fully documented with a built in webpage. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Console output** Clear command line output for any warnings or issues.
- **Standalone/3rd Party support via JSON calls** Can be used with 3rd party applications via JSON calls.

## Index

- 🟩 [Installation](https://github.com/erew123/alltalk_tts?#-installation-on-text-generation-web-ui)
- 🟪 [Updating](https://github.com/erew123/alltalk_tts?#-updating)
- 🟫 [Screenshots](https://github.com/erew123/alltalk_tts#-screenshots)
- 🟨 [Help with problems](https://github.com/erew123/alltalk_tts?#-help-with-problems)
- 🔵🟢🟡 [DeepSpeed Installation (Windows & Linux)](https://github.com/erew123/alltalk_tts?#-deepspeed-installation-options)

#### Updates
The latest build (13 Dec 2023) has had the entire text filtering engine and narration engine rebuilt from scratch. It's highly complicated how its actually working, but the end result it a much clearer TTS output and much better control over the narrator option and how to handle text that isnt within quotes or asterisks. Its a highly recommened update, for the improved quality it gives to the TTS output, if nothing else.

Should you want the older version of the narrator engine+text filtering, I will leave this older copy [here](https://github.com/erew123/alltalk_tts/releases/tag/v1-old-narrator)

DeepSpeed **v11.2** can be installed within the **default text-generation-webui Python 3.11 environment**. Instructions [here](https://github.com/erew123/alltalk_tts#deepspeed-112-for-windows--python-311) (or scroll down). Please note, this is **not** an official Microsoft method (currently) so they will not support you with this style installation. Officially, only DeepSpeed v8.3 is installing on Python 3.9.x.

#### The one thing I cant easily work around
With a RP chat with your AI, **on your character card** `parameters menu` > `character tab` > `greeting` make sure that anything in there that is the **narrator is in asterisks** and anything **spoken is in double quotes**, then hit the `save` (disk) button. Greeting paragraphs/sentences are handled differently from how the AI sends text and so its difficut to account for them both.

I could force a delimeter in at this stage, but I know it would/may affect things further down the line in the chat and I need a good think about that before just making a change. This issue **only** affects the greeting card/start of conversation and the "example" card that comes with text-generation-webui suffers this issue (if you want to try it for yourself). So you would put double quotes around like this (from the example card):

`"`Hey! I'm so excited to finally meet you. I've heard so many great things about you and I'm eager to pick your brain about computers. I'm sure you have a wealth of knowledge that I can learn from.`"`

## 🟩 Installation on Text generation web UI
This has been tested on the current Dec 2023 release of Text generation webUI. If you have not updated it for a while, you may wish to update Text generation webUI, [instructions here](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install)

1) In a command prompt/terminal window you need to move into your Text generation webUI folder:

`cd text-generation-webui`

2) Start the Text generation webUI Python environment for your OS:

`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`

3) Move into your extensions folder:

`cd extensions`

4) Once there git clone this repository:

`git clone https://github.com/erew123/alltalk_tts`

5) Move into the **alltalk_tts** folder:

`cd alltalk_tts`

6) Install the requirements:

*Nvidia graphics card machines* - `pip install -r requirements_nvidia.txt`

*Other machines (mac, amd etc)* - `pip install -r requirements_other.txt`

7) **(Optional DeepSpeed)** If you have an Nvidia Graphics card on a system running Linux or Windows and wish to use **DeepSpeed** please follow these instructions [here](https://github.com/erew123/alltalk_tts?#-deepspeed-installation-options). Though you may wish to check things are generally working with the below steps before installing DeepSpeed.

8) You can now start move back to the main Text generation webUI folder `cd ..` (a few times), start Text generation webUI (`start_windows.bat`,`./start_linux.sh`, `start_macos.sh` or `start_wsl.bat`)  and load the AllTalk extension in the Text generation webUI **session** tab.
   
9) Please read the note below about start-up times and also the note above about ensuring your character cards are set up [correctly](https://github.com/erew123/alltalk_tts#the-one-thing-i-cant-easily-work-around)

10) Some extra voices downloadable [here](https://drive.google.com/file/d/1bYdZdr3L69kmzUN3vSiqZmLRD7-A3M47/view?usp=drive_link)

**Note: It can take a while to start up.** Check the command prompt/terminal window if you want to know what its doing. After it says "Model Loaded" the Text generation webUI is usually available on its IP address a few seconds later, for you to connect to in your browser.

**Documentation:** Click on the link when inside Text generation webUI as shown in the screenshot [here](https://github.com/erew123/alltalk_tts#screenshots)

**Where to find voices** https://aiartes.com/voiceai or https://commons.wikimedia.org/ or interviews on youtube etc. Instructions on how to cut down and prepare a voice sample are within the built in documentation.

### 🟩 Other installation notes
On first startup, AllTalk will download the Coqui XTTSv2 2.0.2 model to its **models** folder (1.8GB space required). You can customse your model or use the TTS latest model within the interface (details in documentation).

Once the extension is loaded, please find all documentation and settings on the link provided in the interface (as shown in the screenshot below).

To start AllTalk every Text generation webUI loads, edit the Text generation webUI `CMD_FLAGS.txt` file in the main `text-generation-webui` folder and add `--extensions alltalk_tts`.

### 🟪 Updating
This is pretty much a repeat of the installation process. 

1) In a command prompt/terminal window you need to move into your Text generation webUI folder:

`cd text-generation-webui`

2) Move into your extensions folder:

`cd extensions`

3) At the command prompt/terminal, type:

`git pull https://github.com/erew123/alltalk_tts`

This should now download any updates/changes.

### 🟪 Problems Updating

If you do experience any problems, the simplest method to resolve this will be:

1) re-name the existing `alltalk_tts` folder to something like `alltalk_tts.old`

2) Start a console/terminal then:

`cd text-generation-webui` and start your python environment `cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`

3) Move into the extensions folder, same as if you were doing a fresh installation:

`cd extensions` then 

`git clone https://github.com/erew123/alltalk_tts`

This will download a fresh installation. 

3) Move into the **alltalk_tts** folder:

`cd alltalk_tts`

4) Install the requirements:

*Nvidia graphics card machines* - `pip install -r requirements_nvidia.txt`

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

You can either follow the [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and fresh install your config. Or you can edit the `config.json` file within the `alltalk_tts` folder. You would look for '"deepspeed_activate": true,' and change the word true to false `"deepspeed_activate": false,' ,then save the file and try starting again.

If you want to use DeepSpeed, you need an Nvidia Graphics card and to install DeepSpeed on your system. Instructions are [here](https://github.com/erew123/alltalk_tts?tab=readme-ov-file#deepspeed-installation-options).

#### 🟨 I am having problems getting AllTalk to start after changing settings or making a custom setup/model setup.

I would suggest following [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and if you still have issues after that, you can raise an issue [here](https://github.com/erew123/alltalk_tts/issues)

#### 🟨 I see some red "asyncio" messages

As far as I am aware, these are to do with the chrome browser the gradio text-generation-webui in some way. I raised an issue about this on the text-generation-webui [here](https://github.com/oobabooga/text-generation-webui/issues/4788) where you can see that AllTalk is not loaded and the messages persist. Either way, this is more a warning than an actual issue, so shouldnt affect any functionality of either AllTalk or text-generation-webui, they are more just an annoyance.

#### 🟨 I am having problems updating/some other issue where it wont start up/Im sure this is a bug

Please see [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating). If that doesnt help you can raise an issue [here](https://github.com/erew123/alltalk_tts/issues). It would be handy to have any log files from the console where your error is being shown. I can only losely support custom built Python environments and give general pointers.

Also, is your text-generation-webui up to date? [instructions here](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install)

## 🔵🟢🟡 DeepSpeed Installation Options
**Note:** ➡️DeepSpeed requires an Nvidia Graphics card!⬅️
### 🔵 For Linux
Covered in the online/buit-in documentation, but a nice easy install.

### 🟢🟡 For Windows & Python 3.11
DeepSpeed v11.1 and v11.2 will work on the current text-generation-webui Python 3.11 environment! You have 2x options for how to setup DeepSpeed on Windows. A quick way (🟢Option 1) and a long way (🟡Option 2).

Thanks to [@S95Sedan](https://github.com/S95Sedan) - They managed to get DeepSpeed 11.2 working on Windows via making some edits to the original Microsoft DeepSpeed v11.2 installation. The original post is [here](https://github.com/oobabooga/text-generation-webui/issues/4734#issuecomment-1843984142).

#### 🟢 OPTION 1 - Quick and easy!
#### Pre-Compiled Wheel Deepspeed v11.1 (for Windows and Python 3.11) ➡️DeepSpeed requires an Nvidia Graphics card!⬅️
[@S95Sedan](https://github.com/S95Sedan) has kindly provided a pre-compiled DeepSpeed v11.1 wheel file, which you can download and use. To use this, you will need to:

**Note:** In my tests, with this method you will **not** need to install the Nvidia CUDA toolkit to make this work, but AllTalk may warn you when starting DeepSpeed that it doesnt see the CUDA Toolkit, however, it works fine for TTS purposes.

1) Download the file [deepspeed-0.11.1+e9503fe-cp311-cp311-win_amd64.whl](https://drive.google.com/file/d/1PFsf6uSPY5Cb4o9VxiZ7DLv-j35L7Y41/view?usp=sharing) by going to the link and clicking the **download** icon at the top right of the screen and save the file it inside your **text-generation-webui** folder.

2) Open a command prompt window, move into your **text-generation-webui folder**, you can now start the Python environment for text-generation-webui:

`cmd_windows.bat`

3) With the file that you saved in the **text-generation-webui folder** you now type the following:

`pip install "deepspeed-0.11.1+e9503fe-cp311-cp311-win_amd64.whl"`
   
4) This should install through cleanly and you should now have DeepSpeed v11.1 installed within the Python 3.11 environment of text-generation-webui.
   
5) When you start up text-generation-webui, and AllTalk starts, you should see **[AllTalk Startup] DeepSpeed Detected**
    
6) Within AllTalk, you will now have a checkbox for **Activate DeepSpeed** though remember you can only change **1x setting every 15 or so seconds**, so dont try to activate DeepSpeed **and** LowVRAM/Change your model simultantiously. Do one of those, wait 15-20 seconds until the change is confirmed in the terminal/command prompt, then you can change the other. When you are happy it works, you can set the default start-up settings in the settings page.

#### 🟡 OPTION 2 - A bit more complicated!
#### Manual Build DeepSpeed v11.2 (for Windows and Python 3.11) ➡️DeepSpeed requires an Nvidia Graphics card!⬅️
DeepSpeed Version 11.2 with CUDA 12.1 - Installation Instructions:

1. Download the 11.2 release of [DeepSpeed](https://github.com/microsoft/DeepSpeed/releases/tag/v0.11.2) extract it to a folder. 
2. Install Visual C++ build tools, such as [VS2019 C++ x64/x86](https://learn.microsoft.com/en-us/visualstudio/releases/2019/redistribution#vs2019-download) build tools.
3. Download and install the [Nvidia Cuda Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
4. Edit your Windows environment variables to ensure that CUDA_HOME and CUDA_PATH are set to your Nvidia Cuda Toolkit path. (The folder above the bin folder that nvcc.exe is installed in). Examples are:<br>
```set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1```<br>
```set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1```<br>

5. OPTIONAL If you do not have an python environment already created, you can install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html), then at a command prompt, create and activate your environment with:<br>
```conda create -n pythonenv python=3.11```<br>
```activate pythonenv```<br>

6. Launch the Command Prompt cmd with Administrator privilege as it requires admin to allow creating symlink folders.
7. Install PyTorch, 2.1.0 with CUDA 12.1 into your Python 3.11 environment e.g:<br>
```activate pythonenv``` (activate your python environment)<br>
```conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia```

8. In your python environment check that your CUDA_HOME and CUDA_PATH are still pointing to the correct location.<br>
```set``` (to list and check the windows environment variables. Refer to step 4 if not)

9. Navigate to your deepspeed folder in the Command Prompt:<br>
```cd c:\deepspeed``` (wherever you extracted it to)

10. Modify the following files:<br>

**(These modified files are included in the git-pull of AllTalk, but if you want to modify them yourself, please follow the below)**

deepspeed-0.11.2/build_win.bat** - at the top of the file, add:<br>
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

11. While still in your command line with python environment enabled run:<br>
```build_win.bat```

12. Now cd dist to go into your dist folder and you can now pip install deepspeed-YOURFILENAME.whl (or whatever your WHL file is called).
