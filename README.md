# AllTalk TTS
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- **Custom Startup Settings:** Adjust your standard startup settings.
- **Nararator:** Use different voices for main character and narration.
- **Low VRAM mode:** Improve generation performance if your VRAM is filled by your LLM.
- **DeepSpeed:** When DeepSpeed is installed you can get a 3-4x performance boost generating TTS.
- **Local/Custom models:** Use any of the XTTSv2 models (API Local and XTTSv2 Local).
- **Wav file maintenance:** Configurable deletion of old output wav files.
- **Documentation:** Fully documented with a built in webpage.
- **Console output** Clear command line output for any warnings or issues.
- **3rd Party support via JSON calls** Can be used with 3rd party applications via JSON calls.

**Note: Mac compatibility is not yet guaranteed due to CUDA requirement**

## Installation on Text generation web UI
In a command prompt/terminal window you need to move into your Text generation webUI folder:

`cd text-generation-webui`

Start the Text generation webUI Python environment:

`cmd_windows.bat` or `./cmd_linux.sh` (read note below about Mac support)

move into your extensions folder:

`cd extensions`

Once there git clone this repository e.g.

```git clone https://github.com/erew123/alltalk_tts```

Install the requirements:

```pip install -r requirements.txt```

You can now start Text generation webUI and load the AllTalk extension in the **session** tab.

AllTalk will download the Coqui XTTSv2 2.0.2 model to its **models** folder (1.8GB space required). You can customse your model or use the TTS latest model within the interface.

Once the extension is loaded, please find all documentation and settings on the link provided in the interface.

You can now start up Text generation web UI and activate AllTalk in your extensions. To start it every time when Text generation webUI loads, edit the `CMD_FLAGS.txt` file in the main `text-generation-webui` folder and add `--extensions alltalk_tts`

## Main Page
![image](https://github.com/oobabooga/text-generation-webui/assets/35898566/aca0a031-5426-4239-abac-cc3149c4d8c4)

## Settings Page
![image](https://github.com/oobabooga/text-generation-webui/assets/35898566/dbb731c9-761f-4a54-9c30-96839d2bb973)
