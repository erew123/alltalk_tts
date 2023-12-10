# AllTalk TTS
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- **Custom Startup Settings:** Adjust your standard startup settings. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Nararator:** Use different voices for main character and narration. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Low VRAM mode:** Improve generation performance if your VRAM is filled by your LLM. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **DeepSpeed:** When DeepSpeed is installed you can get a 3-4x performance boost generating TTS.
- **Local/Custom models:** Use any of the XTTSv2 models (API Local and XTTSv2 Local).
- **Optional wav file maintenance:** Configurable deletion of old output wav files. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Documentation:** Fully documented with a built in webpage. [Screenshot](https://github.com/erew123/alltalk_tts#screenshots)
- **Console output** Clear command line output for any warnings or issues.
- **3rd Party support via JSON calls** Can be used with 3rd party applications via JSON calls.

**Note: Mac compatibility is not yet guaranteed due to CUDA requirement**

## Installation on Text generation web UI
This has been tested on the current Dec 2023 release of Text generation webUI. If you have not updated it for a while, you may wish to update Text generation webUI, instructions [here](https://github.com/oobabooga/text-generation-webui#getting-updates)

1) In a command prompt/terminal window you need to move into your Text generation webUI folder:

`cd text-generation-webui`

2) Start the Text generation webUI Python environment for your OS:

`cmd_windows.bat` or `./cmd_linux.sh` ( Mac users, read note below above re current Mac support)

3) Move into your extensions folder:

`cd extensions`

4) Once there git clone this repository:

`git clone https://github.com/erew123/alltalk_tts`

5) Move into the **alltalk_tts** folder:

`cd alltalk_tts`

6) Install the requirements:

`pip install -r requirements.txt`

7) You can now start move back to the main Text generation webUI folder `cd ..` (a few times), start Text generation webUI (`start_windows.bat` or `./start_linux.sh`)  and load the AllTalk extension in the Text generation webUI **session** tab.

**Note: It can take a while to start up.** Check the command prompt/terminal window if you want to know what its doing. After it says "Model Loaded" the Text generation webUI is usually available on its IP address a few seconds later, for you to connect to in your browser.

**Documentation:** Click on the link when inside Text generation webUI as shown in the screenshot [here](https://github.com/erew123/alltalk_tts#screenshots)

### Other installation notes
On first startup, AllTalk will download the Coqui XTTSv2 2.0.2 model to its **models** folder (1.8GB space required). You can customse your model or use the TTS latest model within the interface (details in documentation).

Once the extension is loaded, please find all documentation and settings on the link provided in the interface (as shown in the screenshot below).

To start AllTalk every Text generation webUI loads, edit the Text generation webUI `CMD_FLAGS.txt` file in the main `text-generation-webui` folder and add `--extensions alltalk_tts`.

## Screenshots
|![image](https://github.com/erew123/alltalk_tts/assets/35898566/1ef942b9-654e-42cc-9144-cf48597ec791) | ![Screenshot 2023-12-09 190108](https://github.com/erew123/alltalk_tts/assets/35898566/ecd75913-5c33-4a99-810c-15b74cc6c91a) |
|:---:|:---:|
