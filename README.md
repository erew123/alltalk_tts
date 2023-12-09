# AllTalk TTS
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- **Custom Startup Settings:** Adjust your standard startup settings. [Screenshot](https://github.com/erew123/alltalk_tts#settings-page)
- **Nararator:** Use different voices for main character and narration. [Screenshot](https://github.com/erew123/alltalk_tts#main-page)
- **Low VRAM mode:** Improve generation performance if your VRAM is filled by your LLM. [Screenshot](https://github.com/erew123/alltalk_tts#main-page)
- **DeepSpeed:** When DeepSpeed is installed you can get a 3-4x performance boost generating TTS.
- **Local/Custom models:** Use any of the XTTSv2 models (API Local and XTTSv2 Local).
- **Wav file maintenance:** Configurable deletion of old output wav files. [Screenshot](https://github.com/erew123/alltalk_tts#settings-page)
- **Documentation:** Fully documented with a built in webpage. [Screenshot](https://github.com/erew123/alltalk_tts#settings-page)
- **Console output** Clear command line output for any warnings or issues.
- **3rd Party support via JSON calls** Can be used with 3rd party applications via JSON calls.

**Note: Mac compatibility is not yet guaranteed due to CUDA requirement**

## Installation on Text generation web UI
This has been tested on the current Dec 2023 release of Text generation webUI. If you have not updated it for a while, you may wish to update Text generation webUI, instructions [here](https://github.com/oobabooga/text-generation-webui#getting-updates)

1) In a command prompt/terminal window you need to move into your Text generation webUI folder:

`cd text-generation-webui`

2) Start the Text generation webUI Python environment for your OS:

`cmd_windows.bat` or `./cmd_linux.sh` (read note below about Mac support)

3) Move into your extensions folder:

`cd extensions`

4) Once there git clone this repository:

`git clone https://github.com/erew123/alltalk_tts`

5) Install the requirements:

`pip install -r requirements.txt`

6) You can now start Text generation webUI and load the AllTalk extension in the Text generation webUI **session** tab.

### Other installation notes
On first startup, AllTalk will download the Coqui XTTSv2 2.0.2 model to its **models** folder (1.8GB space required). You can customse your model or use the TTS latest model within the interface (details in documentation).

Once the extension is loaded, please find all documentation and settings on the link provided in the interface (as shown in the screenshot below).

To start AllTalk every Text generation webUI loads, edit the Text generation webUI `CMD_FLAGS.txt` file in the main `text-generation-webui` folder and add `--extensions alltalk_tts`.

## Screenshots
### Main Page
![image](https://github.com/erew123/alltalk_tts/assets/35898566/77007001-2bec-4cf7-a427-524ceb8bd1a6)

### Settings Page
![Screenshot 2023-12-09 190108](https://github.com/erew123/alltalk_tts/assets/35898566/ecd75913-5c33-4a99-810c-15b74cc6c91a)
