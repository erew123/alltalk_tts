import os
import json
import pathlib as Path
import gradio as gr

def modify_config():
    config_file = os.path.join("confignew.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    config["firstrun_splash"] = False
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    return f"Welcome screen is disabled."

def alltalk_welcome(analytics_enabled=False):
    welcome = """
    ## 🟧 AllTalk v2 BETA
    For detailed information on using AllTalk, please look at the WIKI Github page https://github.com/erew123/alltalk_tts/wiki for the most up-to-date information, however some information is built in on the Documentation and Help tab section up above.

    Please use the **Light/Dark Mode** button at the top right of this page to make it easier to read. I would suggest reading the TLDR's on this page before you disable this splash screen.
    """  
    
    welcome2 = """    
    ### 🟧 TLDR what you need to do
    > **Generating TTS & Swapping TTS Engines** - You can do this on the Generate TTS tab, Swap Engines (Piper, Parler, VITS, XTTS etc) and also choose which model to load for the currently loaded TTS Engine. There is a Generate help tab on the Generate TTS page for more details.<br>
    
    > **Download Voices/Models** - Go to the `TTS Engines Settings` > Choose the engine you want > `Model/Voices Download`. Additional instructions/information about each TTS engine, its specific settings, how to use it, are all stored here.<br>
    
    > **RVC & RVC Voices** - To make RVC work, you have to enable RVC in `Gloabl Settings` > `RVC Settings`. RVC will download about 800MB of models that it needs to work. [RVC WIKI page link](https://github.com/erew123/alltalk_tts/wiki/RVC-(Retrieval%E2%80%90based-Voice-Conversion)<br>

    ### 🟧 Error Message at the Console/Terminal       
    > I am mantaining an up-to-date list of known error messages and fixes on the [Error Messages WIKI page](https://github.com/erew123/alltalk_tts/wiki/Error-Messages-List)<br>

    ### 🟧 Showing Your Support
    > If AllTalk has been helpful to you, consider showing your support through a donation on my **[Ko-fi page](https://ko-fi.com/erew123)**. Your support is greatly appreciated and helps ensure the continued development and improvement of AllTalk.
    """
    
    welcome3 = """     
    ### 🟧 SillyTavern users
    > There is an updated ST extension and instructions for installation are in the [SillyTavern Extension WIKI page](https://github.com/erew123/alltalk_tts/wiki/SillyTavern-Extension).    
    
    ### 🟧 DeepSpeed on Linux
    > If you need DeepSpeed for Linux, please see the [DeepSpeed Releases page](https://github.com/erew123/alltalk_tts/releases/tag/DeepSpeed-14.2-Linux).
    
    ### 🟧 Text-generation-webui Remote Extension
    > AllTalk now ALSO has a TGWUI remote server extension, allowing you to install AllTalk outside of TGWUI's Python environment. Please see [TGWUI Remote Extension WIKI page](https://github.com/erew123/alltalk_tts/wiki/Text%E2%80%90generation%E2%80%90webui-Remote-Extension).
    
    ### 🟧 Future features/Feature Requests
    > The current list is available here [Feature Request List page](https://github.com/erew123/alltalk_tts/discussions/74)
    
    ### 🟧 Can I add other TTS engines?
    Yes if you have a bit of coding experience. All the Coqui supported engines should be pretty easy to set up now, I just havnt gotten around it yet. Other engines like Piper and Parler have been done of course, showing other engines can be imported. Rough instructions are in the `system/tts_engines/template` folder.
    """
    with gr.Tab("AllTalk v2 BETA"): 
        gr.Markdown(welcome)
        with gr.Row():
            with gr.Column(): gr.Markdown(welcome2)
            with gr.Column(): gr.Markdown(welcome3)        
        with gr.Row():
            output = gr.Textbox(label="Output")
            update_btn = gr.Button("Dont show me this screen again!")
            update_btn.click(fn=modify_config, inputs=None, outputs=output)