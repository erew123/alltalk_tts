# AllTalk TTS
AllTalk is an updated version of the Coqui_tts extension for Text Generation web UI. Features include:

- Settings page to adjust your base startup settings.
- Low VRAM mode to improve performance where your LLM takes up your VRAM.
- DeepSpeed voice processing, giving a 3-4x performance boost (requires DeepSpeed installing).
- Local/Custom models (API Local and XTTSv2 Local).
- Configurable deletion of old wav file outputs.
- Built in manual & settings on a web page.
- Clear command line output for any warnings or issues.
- Can be used with 3rd party applications via JSON calls.

**Mac compatibility is not yet guaranteed due to CUDA requirement**

## Installation on Text generation web UI
In a command prompt/terminal window you need to move into your Text generation webUI folder e.g. `cd text-generation-webui`
and then into your extensions folder e.g. `cd extensions`

Once there git clone this repository e.g.

```git clone https://github.com/erew123/alltalk_tts```

You can now start up Text generation web UI and activate AllTalk in your extensions. To start it every time when Text generation webUI loads, edit the `CMD_FLAGS.txt` file in the main `text-generation-webui` folder and add `--extensions alltalk_tts`
