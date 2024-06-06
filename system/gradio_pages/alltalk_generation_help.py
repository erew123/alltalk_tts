import gradio as gr

def alltalk_generation_help():
    generation_blurb = """
    The Generation Screen is the main interface where you can input text to generate speech, as well as swap **and set the default** Text-to-Speech (TTS) engine and model. Below is a detailed description of each setting and its functionality.
    """
    generation_help = """
    ### 🟨 TTS Engine and Model<br>
    **Swap TTS Engine**<br>
    > This button allows you to change and set the current TTS engine. Swapping the engine will also set the new engine as the default for future sessions. When you swap the engine, the settings on the page will refresh to match the capabilities of the selected engine. Some settings may become unavailable or newly available depending on the chosen engine.<br>
    
    **Load Different Model**<br>
    > This dropdown lists the available models for the selected TTS engine. These models are located in the models folder and are specific to each TTS engine. Changing the model will also set it as the default for future sessions and refresh the settings on the page.<br>

    ### 🟨 RVC Pipeline<br>
    **RVC Character Voice**<br>
    > This setting is part of the RVC pipeline. If RVC is not enabled (which can be done in the RVC tab), "Disabled" will be the only option. Once RVC is enabled and the required models (approximately 700MB) are downloaded, any voices stored in the `/models/rvc_voices/{yourmodelhere}` folder will be available for selection. Selecting a voice will apply it during the TTS generation process. You can select "Disabled" at any time to bypass the RVC pipeline even when it is globally enabled.<br>
    
    **RVC Narrator Voice**<br>
    > Similar to the RVC Character Voice, this setting applies to the narrator voice in the RVC pipeline.<br>

    ### 🟨 Generation Settings<br>
    **Generation Mode**<br>
    > Choose the mode of TTS generation, such as Standard or Streaming. Note that Gradio may have difficulty with streaming, so you might receive standard audio as a file even when streaming is selected. If the engine doesn't support Streaming, only Standard mode will be shown.<br>
    
    **Language/Model not multi-language**<br>
    > Select the language for TTS generation here. Some models are **not** multi-language capable and the language it generates in is built into the model itself. In such cases, this option will change to `Model not multi-language` and it will no longer be selectable when the model doesn't support multiple languages.<br>
   
    **Narrator Enabled/Disabled**<br>
    > Enable or disable the narrator feature. When enabled, the TTS engine processes three types of text (Refer to the "Narrator Function" help tab for more details):<br>
    > **Narrated Text**  : Text enclosed in asterisks *Narrated text*.<br>
    > **Character Text** : Text enclosed in double quotes "Character text".<br>
    > **Text-Not-Inside**: Any text not enclosed in asterisks or double quotes. The handling of this text is customizable.<br>
    
    **Narrator Text-not-inside**<br>
    > Configures how text that is not enclosed in asterisks or double quotes is handled. Detailed explanations are available in the "Narrator Function" help tab.<br>
    
    **Auto-Stop Current Generation**<br>
    > If the loaded model or generation method supports stopping the current TTS generation when sending a new request, this setting will attempt to stop the current process before starting a new one.
    
    ### 🟨 Voice Selection<br>
    **Character (Main TTS) Voice**<br>
    > Select the character voice from the available voices within the currently loaded TTS engine and model.<br>
    
    **Narrator Voice**<br>
    > Choose the narrator voice from the available voices within the currently loaded TTS engine and model.<br>
    
    """
    generation_help2 = """
    ### 🟨 Additional Settings
    **Text Filtering**<br>
    > **None**: No filtering. Raw text is sent to the TTS engine, which may result in odd sounds with some special characters.<br>
    > **HTML**: For HTML content, using HTML entities like emojis or text characters.<br>
    > **Standard**: Human-readable text with basic filtering to clean up special characters.<br>
    
    **Play Locally or Remotely**<br>
    > Choose to play the generated speech locally or on the terminal/console where the server is running.<br>
    
    **Remote Play Volume**<br>
    > Adjust the volume for remote playback.<br>
    
    **Output File Name**<br>
    > Specify the name for the output file.<br>
    
    **Include Timestamp**<br>
    > If enabled, each generated file will include a timestamp to ensure uniqueness. If disabled, the output filename will be used, and the file will be overwritten with each generation.<br>
    
    **Speed**<br>
    > Adjust the speed of the generated speech. This option is not available if the selected TTS engine doesn't support it.<br>
    
    **Pitch**<br>
    >Adjust the pitch of the generated speech. This option is not available if the selected TTS engine doesn't support it.<br>
    
    **Temperature**<br>
    > Control the randomness of the generation process. Higher values result in more varied output. This option is not available if the selected TTS engine doesn't support it.<br>
    
    **Repetition Penalty**<br>
    > Set the penalty for repetitive sequences in the generated speech. This option is not available if the selected TTS engine doesn't support it.<br>

    ### 🟨 Output and Controls
    **TTS Result**<br>
    > View the result of the TTS generation process here.<br>
    
    **Playback Controls**<br>
    > Use these controls to play, pause, and navigate through the generated speech.<br>
    
    **Light/Dark Mode**<br>
    > Toggle between light and dark mode for the interface.<br>
    
    **Refresh Server Settings**<br>
    >Refresh the server settings to apply any changes made. This is useful if you have reloaded the engine elsewhere or added new models/voices to your folders. If refreshing doesn't update your voices, you can use the "Swap TTS Engine" button to force a full refresh.<br>
    
    **Interrupt TTS Generation**<br>
    > Manually stop the current TTS generation process, similar to the "Auto-Stop Current Generation" setting but applied manually.<br>
    """
    gr.Markdown(generation_blurb)
    with gr.Row():
        gr.Markdown(generation_help)
        gr.Markdown(generation_help2)