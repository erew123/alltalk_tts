# AllTalk TTS
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- **Default Settings:** Adjust your standard startup settings.
- **Low VRAM mode:** Improve generation performance if your VRAM is filled by your LLM.
- **DeepSpeed:** When DeepSpeed is installed you can get a 3-4x performance boost generating TTS.
- **Local/Custom models:** Use any of the XTTSv2 models (API Local and XTTSv2 Local).
- **Wav file maintenance:** Configurable deletion of old output wav files.
- **Documentation:** Fully documented with a built in webpage.
- **Console output** Clear command line output for any warnings or issues.
- **3rd Party support via JSON calls** Can be used with 3rd party applications via JSON calls.

**Note: Mac compatibility is not yet guaranteed due to CUDA requirement**

## Installation on Text generation web UI
In a command prompt/terminal window you need to move into your Text generation webUI folder e.g. `cd text-generation-webui`
and then into your extensions folder e.g. `cd extensions`

Once there git clone this repository e.g.

```git clone https://github.com/erew123/alltalk_tts```

You can now start up Text generation web UI and activate AllTalk in your extensions. To start it every time when Text generation webUI loads, edit the `CMD_FLAGS.txt` file in the main `text-generation-webui` folder and add `--extensions alltalk_tts`

## Main Page
![image](https://github.com/oobabooga/text-generation-webui/assets/35898566/aca0a031-5426-4239-abac-cc3149c4d8c4)

## Settings Page
![image](https://github.com/oobabooga/text-generation-webui/assets/35898566/dbb731c9-761f-4a54-9c30-96839d2bb973)
