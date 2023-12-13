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

#### The one thing I cant easily work around
With a RP chat with your AI, on your character card `parameters menu` > `character tab` > `greeting` make sure that anything in there that is the narrator is in asterisks and anything spoken is in double quotes, then hit the `save` (disk) button. There is something about starting a sentence, on the greeting with asterisks, but then text never having another delimiter. I could force one in at this stage, but I know it would affect things further down the line in the chat. This only affects the greeting card. I do have thoughts how to resolve this in future, but I have to account for all the other logic further down the chain and re-test everything, so the simple fix is just make sure your character greeting card has double quotes around text and asterisks around narrator. Besides, thats good practise as it should educate your AI to carry forward with the conversation in that format. You can see the behaviour I detail occuring on the "example" assistant that comes with text-generation-webui.

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
