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

def alltalk_welcome():
    welcome = """
    ## 🟧 Welcome the the AllTalk v2 BETA
    Thanks for trying out the BETA. Ive no doubt there will be bugs and quirks here and there. Most errors/issues should have outputs at the command prompt/console.
    
    I have validated this BETA to run on Windows/Linux, because I dont have a Mac. With Python 3.11.9, Pytorch 2.2.1 with CUDA 12.1 extensions installed. Thats not to say it wont work on Mac or other versions of Python and Pytorch, its just there would be some extra testing/figuring things out to make it work. Obviously, I am kind of a bit limited to do that on Mac's. Certainly Mac's dont need the Nvidia lines in the requirements file and I cannot say how certain engines will/wont work/perform on Mac's.
    
    ### 🟧 Important Requirements
    > - Please ensure that you have installed Espeak-ng for your platform (you should get and error message at the command prompt/terminal if you havnt).<br>
    > - It is possible I have forgotten some requirements, but hopefully not!
    > - Linux users, you should have installed `libaio-dev` or `libaio-devl` (depending on your Linux flavour), otherwise DeepSpeed will fail e.g.<br>
    > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Debian-based systems** `sudo apt install libaio-dev`<br>
    > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**RPM-based systems** `sudo yum install libaio-devel`
    
    ### 🟧 TLDR what you need to do
    > - To download models, you can go to the `TTS Engines Settings`, choose the engine you want and there are downloads there, along with additional instructions/information about each engine, its specific settings, how to work with it etc.
    > - RVC when enabled will download about 800MB of models that it needs to work. If you want more information on RVC, please read the Documentation. 
    > - Talking of which, Gradio isnt great at documentation. Ive found that the `Light/Dark Mode` button is handy, along with sometime using your browsers zoom function.
    > - **Check the console** for errors/warnings if you have problems. I have **tried** to cover most issues/possible errors.
    > - The documentation may be a bit rough and ready. I havnt had time to go through it all step by step.
    
    ### 🟧 DeepSpeed on Linux
    > I've not managed to update the documentation, but if you are **NOT** using AllTalk as a standalone on Linux, take a look here and this should ease the setup of DeepSpeed https://github.com/erew123/alltalk_tts/releases/tag/DeepSpeed-14.2-Linux
    
    ### 🟧 SillyTavern users
    > There is an updated ST extension in the system folder. Instructions are inside the folder.
    
    ### 🟧 Text-generation-webui users
    > There is a TGWUI remote extension, if you want to run AllTalk on a computer that is NOT your AllTalk system. Please look in the system folder. Instructions are inside the folder.
    
    ### 🟧 Why is X feature no in the BETA yet? Or I have a great idea!
    Mostly because of time restrains. V2 has been a massive re-write of code in the 100's of hours range to get it where it is. Please do check the Feature Requests list here https://github.com/erew123/alltalk_tts/discussions/74. I will get to them when I can. People are welcome to work on code/test things out with this version if they wish. I will set up a page for feeback on the **Github Discussions Board** https://github.com/erew123/alltalk_tts/discussions
    
    ### 🟧 Can I add other TTS engines?
    Yes if you have a bit of coding experience. All the Coqui supported engines should be pretty easy to set up now, I just havnt gotten around it yet. Other engines like Piper and Parler have been done of course, showing other engines can be imported. Rough instructions are in the `system/tts_engines/template` folder.
    
    ### 🟧 Showing Your Support
    > If AllTalk has been helpful to you, consider showing your support through a donation on my **[Ko-fi page](https://ko-fi.com/erew123)**. Your support is greatly appreciated and helps ensure the continued development and improvement of AllTalk.
    """
    with gr.Tab("AllTalk v2 BETA"): 
        gr.Markdown(welcome)
        with gr.Row():
            output = gr.Textbox(label="Output")
            update_btn = gr.Button("Dont show me this screen again!")
            update_btn.click(fn=modify_config, inputs=None, outputs=output)