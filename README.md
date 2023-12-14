# AllTalk TTS
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- **Custom Start-up Settings:** Adjust your default start-up settings. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Narrarator:** Use different voices for main character and narration. [Example Narration](https://vocaroo.com/18fYWVxiQpk1)
- **Low VRAM mode:** Improve generation performance if your VRAM is filled by your LLM. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **DeepSpeed:** When DeepSpeed is installed you can get a 3-4x performance boost generating TTS.
- **Local/Custom models:** Use any of the XTTSv2 models (API Local and XTTSv2 Local).
- **Optional wav file maintenance:** Configurable deletion of old output wav files. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Documentation:** Fully documented with a built in webpage. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Console output** Clear command line output for any warnings or issues.
- **Standalone/3rd Party support via JSON calls** Can be used with 3rd party applications via JSON calls.

#### Updates
The latest build (13 Dec 2023) has had the entire text filtering engine and narration engine rebuilt from scratch. It's highly complicated how its actually working, but the end result it a much clearer TTS output and much better control over the narrator option and how to handle text that isnt within quotes or asterisks. Its a highly recommened update, for the improved quality it gives to the TTS output, if nothing else.

Should you want the older version of the narrator engine+text filtering, I will leave this older copy [here](https://github.com/erew123/alltalk_tts/releases/tag/v1-old-narrator)

DeepSpeed **v11.2** can be installed within the **default text-generation-webui Python 3.11 environment**. Instructions [here](https://github.com/erew123/alltalk_tts#deepspeed-112-for-windows--python-311). Please note, this is **not** an official Microsoft method (currently) so they will not support you with this style installation. Officially, only DeepSpeed v8.3 is installing on Python 3.9.x.

#### The one thing I cant easily work around
With a RP chat with your AI, **on your character card** `parameters menu` > `character tab` > `greeting` make sure that anything in there that is the **narrator is in asterisks** and anything **spoken is in double quotes**, then hit the `save` (disk) button. Greeting paragraphs/sentences are handled differently from how the AI sends text and so its difficut to account for them both.

I could force a delimeter in at this stage, but I know it would/may affect things further down the line in the chat and I need a good think about that before just making a change. This issue **only** affects the greeting card/start of conversation and the "example" card that comes with text-generation-webui suffers this issue (if you want to try it for yourself). So you would put double quotes around like this (from the example card):

`"`Hey! I'm so excited to finally meet you. I've heard so many great things about you and I'm eager to pick your brain about computers. I'm sure you have a wealth of knowledge that I can learn from.`"`

## Installation on Text generation web UI
This has been tested on the current Dec 2023 release of Text generation webUI. If you have not updated it for a while, you may wish to update Text generation webUI, instructions [here](https://github.com/oobabooga/text-generation-webui#getting-updates)

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

7) You can now start move back to the main Text generation webUI folder `cd ..` (a few times), start Text generation webUI (`start_windows.bat`,`./start_linux.sh`, `start_macos.sh` or `start_wsl.bat`)  and load the AllTalk extension in the Text generation webUI **session** tab.
   
8) Please read the note below about start-up times and also the note above about ensuring your character cards are set up [correctly](https://github.com/erew123/alltalk_tts#the-one-thing-i-cant-easily-work-around)

**Note: It can take a while to start up.** Check the command prompt/terminal window if you want to know what its doing. After it says "Model Loaded" the Text generation webUI is usually available on its IP address a few seconds later, for you to connect to in your browser.

**Documentation:** Click on the link when inside Text generation webUI as shown in the screenshot [here](https://github.com/erew123/alltalk_tts#screenshots)

**Where to find voices** https://aiartes.com/voiceai or https://commons.wikimedia.org/ or interviews on youtube etc. Instructions on how to cut down and prepare a voice sample are within the built in documentation.

### Other installation notes
On first startup, AllTalk will download the Coqui XTTSv2 2.0.2 model to its **models** folder (1.8GB space required). You can customse your model or use the TTS latest model within the interface (details in documentation).

Once the extension is loaded, please find all documentation and settings on the link provided in the interface (as shown in the screenshot below).

To start AllTalk every Text generation webUI loads, edit the Text generation webUI `CMD_FLAGS.txt` file in the main `text-generation-webui` folder and add `--extensions alltalk_tts`.

## Screenshots
|![image](https://github.com/erew123/alltalk_tts/assets/35898566/a4d983ab-f9e1-42dd-94ee-a85043f74ab2) | ![image](https://github.com/erew123/alltalk_tts/assets/35898566/3497d656-9729-4cb7-8d0d-6367078794ee) |
|:---:|:---:|

## DeepSpeed 11.2 (for Windows & Python 3.11)
DeepSpeed v11.2 will work on the current text-generation-webui Python 3.11 environment! 

Thanks to [@S95Sedan](https://github.com/S95Sedan) - They managed to get DeepSpeed 11.2 working on Windows via making some edits to the original Microsoft DeepSpeed v11.2 installation. The original post is [here](https://github.com/oobabooga/text-generation-webui/issues/4734#issuecomment-1843984142).

#### Pre-Compiled Wheel (for Windows and Python 3.11) - Quick and easy!
[@S95Sedan](https://github.com/S95Sedan) has kindly provided a pre-compiled wheel file, which you can download and use [deepspeed-0.11.1+e9503fe-cp311-cp311-win_amd64.rar.zip](https://github.com/oobabooga/text-generation-webui/files/13593455/deepspeed-0.11.1%2Be9503fe-cp311-cp311-win_amd64.rar.zip). To use this, you will need to:

1) Download the file and put it inside your **text-generation-webui** folder.
   
3) Extract out the zip file [deepspeed-0.11.1+e9503fe-cp311-cp311-win_amd64.rar.zip](https://github.com/oobabooga/text-generation-webui/files/13593455/deepspeed-0.11.1%2Be9503fe-cp311-cp311-win_amd64.rar.zip), which will give you a RAR file *(this is because github wont allow rar files, only zip, so it had to be compressed twice)*.
   
5) Extract out the rar file `deepspeed-0.11.1+e9503fe-cp311-cp311-win_amd64.rar`.
   
7) That should now have extracted a file called `deepspeed-0.11.1+e9503fe-cp311-cp311-win_amd64.whl` **note the .whl extension on it**.
   
9) Still in the **text-generation-webui folder**, you can now start the Python environment for text-generation-webui:

`cmd_windows.bat`

6) Move into the folder where the `whl` file was extracted to and then

`pip install "deepspeed-0.11.1+e9503fe-cp311-cp311-win_amd64.whl"`
   
9) This should install through cleanly and you should now have DeepSpeed 11.2 installed within the Python 3.11 environment of text-generation-webui.
   
10) When you start up text-generation-webui, you should note that AllTalk's startup says **[AllTalk Startup] DeepSpeed Detected**
    
12) Within AllTalk, you will now have a checkbox for **Activate DeepSpeed" though remember you can only change 1x setting every 15 or so seconds, so dont try to activate DeepSpeed **and* LowVRAM/Change your model simultantiously. Do one of those, wait 15-20 seconds until the change is confirmed in the console, then you can change the other. When you are happy it works, you can set the default start-up settings in the settings page.

#### Manual Build (for Windows and Python 3.11) - A bit more complicated!
To perform a manual build of DeepSpeed 11.2, you would follow the instructions for creating DeepSpeed v8.3, but in its place, you would download DeepSpeed v11.2 **Source code (zip)** [here](https://github.com/microsoft/DeepSpeed/releases/tag/v0.11.2) **and** you would use the text-generation-webui's Python environment `cmd_windows.bat`. Extract the downloaded file and you would have to make some file edits to the files in that folder before you can compile DeepSpeed v11.2. As Follows:

**DeepSpeed-0.11.2\build_win.bat** At the top of the file, add:

`set DS_BUILD_EVOFORMER_ATTN=0`

**DeepSpeed-0.11.2\csrc\quantization\pt_binding.cpp - lines 244-250** change to:

```
    std::vector<int64_t> sz_vector(input_vals.sizes().begin(), input_vals.sizes().end());
    sz_vector[sz_vector.size() - 1] = sz_vector.back() / devices_per_node;  // num of GPU per nodes
    at::IntArrayRef sz(sz_vector);
    auto output = torch::empty(sz, output_options);

    const int elems_per_in_tensor = at::numel(input_vals) / devices_per_node;
    const int elems_per_in_group = elems_per_in_tensor / (in_groups / devices_per_node);
    const int elems_per_out_group = elems_per_in_tensor / out_groups;
```

**DeepSpeed-0.11.2\csrc\transformer\inference\csrc\pt_binding.cpp - lines 541-542** change to:

```
									 {static_cast<unsigned>(hidden_dim * InferenceContext::Instance().GetMaxTokenLength()),
									  static_cast<unsigned>(k * InferenceContext::Instance().GetMaxTokenLength()),
```

**DeepSpeed-0.11.2\csrc\transformer\inference\csrc\pt_binding.cpp - lines 550-551** change to:

```
						 {static_cast<unsigned>(hidden_dim * InferenceContext::Instance().GetMaxTokenLength()),
						  static_cast<unsigned>(k * InferenceContext::Instance().GetMaxTokenLength()),
```
**DeepSpeed-0.11.2\csrc\transformer\inference\csrc\pt_binding.cpp - line 1581** change to:

```
		at::from_blob(intermediate_ptr, {input.size(0), input.size(1), static_cast<int64_t>(mlp_1_out_neurons)}, options);
```

**DeepSpeed-0.11.2\deepspeed\env_report.py - line 10** add:

```
import psutil
```

**DeepSpeed-0.11.2\deepspeed\env_report.py - line 83 - 100** change to:

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

You can now compile DeepSpeed and build your whl (wheel) file.

