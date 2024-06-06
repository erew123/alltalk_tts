import gradio as gr

def api_documentation():
    best_practices = """
    ### 🟠 Best Practices for writing an extension

    You will need to store the IP address and Port (minimum) that your extension will be connecting to. Potentially also http/https if you wish to use AllTalk over tunnels. Additionally, you may also wish to store a connection timeout value.<br>

    ```
    {
        "api_alltalk_protocol": "http://",
        "api_alltalk_ip_port": "127.0.0.1:7851",
        "api_connection_timeout": 5
    }
    ```
    When your extension starts up you will need to connect to the IP/Port and check the **/api/ready** end point for a 'Ready' status. You may wish to do this on a loop for the period of the connection timeout.<br>

    If the timeout is reached without a response, you can set voices, models available, models loaded, basically whatever values you are going to store/present to the user as "Server Offline" or whatever terminology you wish.<br>

    If the connection is available, then you would:<br>

    > 1) Pull the current settings and populate those into a paramaters variabele. **/api/currentsettings**<br>
    2) Pull the current voices and populate those into a paramaters variabele. **/api/voices**<br>
    3) Pull the current rvcvoices and populate those into a paramaters variabele. **/api/rvcvoices**<br>

    You can now present those to the user interface, choosing if you are just going to present the voices (narrator or not), RVC voices, model loaded, models available etc.<br>

    If you are going to present a multitude of options, please abide by the **xxxxxx_capable** settings provided by **/api/currentsettings**, disabling or enabling options in your user interface as the loaded in TTS engine does or doesnt allow for e.g. if the current TTS engine doesnt support changing your generation audio speed, it will send **"generationspeed_capable": false** over when you use **/api/currentsettings**. As such, you should disable the option to change the generation speed within your interface, however, ultimately, even if not disabled, if the current TTS engine loaded isnt capable of speed generation, changing the generation speed will have no effect on the resulting TTS generation, but you may end up with users asking why!<br>

    It is also advisable to have a refresh button to pull make AllTalk update its settings **/api/reload_config** and then you can pull current settings & voices again. This ensures the use always has a way to reconnect to the AllTalk server if its unavailable when your extension starts, or in the scenario that something else was changed in the Gradio interface and now the TTS Engine loaded in is different from when your extension started up.<br>

    You can obviously choose if you will give the user access to change the model loaded, TTS engine etc. Please read the documentation on the API endpoints for more information.<br>

    If you are presenting all the variables/options to the user, then when you send off a generation request to **/api/tts-generate** send all of the variable/options that are stored within your interface.<br>

    If you are NOT presenting all the variables/options to the user, then you have 2x options when sending to **/api/tts-generate**:<br>

    > 1) Send all the values/options but hard code some settings in your outbound generation request. This will ensure you always get the TTS generation result you want. Please read the **/api/tts-generate** section for details.<br>
    
    OR<br>
    
    >2) Dont send all the values/options in the generation request. In this scenario, the AllTalk server will populate missing settings from the global API Default Settings and the current TTS Engine's settings that are loaded in. e.g. if you dont provide a character voice, the default set character voice of whatever TTS engine is loaded in will be used. The same goes for all the other settings, except the actual text of the generation request.<br>

    Finally, depending on how you are storing your settings, for your extension, you may wish to save the settings each time there is a successful generation request. This ensures that the settings are valid and the connected IP address/port are correct.
    """

    settings_endpoints = """
    ### 🟠 Ready Endpoint<br>
    Check if the Text-to-Speech (TTS) engine has started and is ready to accept requests. This endpoint is useful when performing reload model/engine requests as it will return "Unloaded" while the engine is restarting. You can poll this endpoint every second or so to confirm a "Ready" status, indicating AllTalk is ready to process requests. Note that if there is an issue loading the selected model (e.g., missing from the file system), the endpoint will still return "Ready". This status only indicates that the engine has loaded.<br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/ready`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `GET`<br> 
    
    > `curl -X GET "http://127.0.0.1:7851/api/ready"`

    Response: `Ready` or `Unloaded`

    ### 🟠 Standard Voices List Endpoint<br>
    Retrieve a list of available voices for generating speech that the currently loaded TTS engine and model supports. If the currently loaded TTS engine does not load a model directly (e.g., Piper), the models themselves will be displayed as voices.

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/voices`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `GET`<br>

    > `curl -X GET "http://127.0.0.1:7851/api/voices"`

    ```
    {
    "status": "success",
    "voices": ["voice1", "voice2", "voice3"]
    }

    ```

    ### 🟠 RVC Voices List Endpoint<br>
    Retrieve a list of available RVC voices for further processing your TTS with the RVC pipeline. `Disabled` will always be included in the list. When `Disabled` is used, the RVC pipeline will not be utilized. If the RVC pipeline is globally disabled in AllTalk, `Disabled` will be the only item in the list. Index files matching their RVC voices will not be displayed; if an index file exists for an RVC voice, it will be automatically selected during generation. If multiple index files exist and it is impossible to automatically select one, a message will be output in the console.

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/rvcvoices`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `GET`<br>

    > `curl -X GET "http://127.0.0.1:7851/api/rvcvoices"`

    ```
        {
    "status": "success",
    "voices": ["Disabled", "folder1\\voice1.pth", "folder2\\voice2.pth", "folder3\\voice3.pth"]
    }
    ```
    
    ### 🟠 Current Settings Endpoint<br>
    Retrieve the current settings of the TTS engine.

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/currentsettings`<br> 
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `GET`<br>

    > `curl -X GET "http://127.0.0.1:7851/api/currentsettings"`
    

    ```
    {
    "engines_available": ["parler", "piper", "vits", "xtts"],
    "current_engine_loaded": "xtts",
    "models_available": [
        {"name": "xtts - xttsv2_2.0.2"},
        {"name": "apitts - xttsv2_2.0.2"},
        {"name": "xtts - xttsv2_2.0.3"},
        {"name": "apitts - xttsv2_2.0.3"}
    ],
    "current_model_loaded": "xtts - xttsv2_2.0.3",
    "manufacturer_name": "Coqui",
    "audio_format": "wav",
    "deepspeed_capable": true,
    "deepspeed_available": true,
    "deepspeed_enabled": true,
    "generationspeed_capable": true,
    "generationspeed_set": 1,
    "lowvram_capable": true,
    "lowvram_enabled": false,
    "pitch_capable": false,
    "pitch_set": 0,
    "repetitionpenalty_capable": true,
    "repetitionpenalty_set": 10,
    "streaming_capable": true,
    "temperature_capable": true,
    "temperature_set": 0.75,
    "ttsengines_installed": true,
    "languages_capable": true,
    "multivoice_capable": true,
    "multimodel_capable": true
    }
    ```

    #### **engines_available**
    > Lists the currently available TTS engines that can be loaded.

    #### **current_engine_loaded**
    > Shows the currently loaded TTS engine.
    These can be used together with the **/api/enginereload** endpoint to determine and change the currently loaded engine. Note that swapping the engine using this method can cause the main script process of AllTalk to lose sync with the subprocess. This results in a message indicating that script.py lost control of the subprocess when exiting AllTalk.

    #### **models_available**
    > Lists all the models available for the currently loaded TTS engine. Displayed as {engine-name - model-name} from models_available.

    #### **current_model_loaded**
    > Shows the currently loaded model.
    You can swap models using the **/api/reload** endpoint by specifying the full {enginename - modelname}.

    #### **audio_format**
    > The primary format in which the currently loaded engine produces audio output. This does not account for any global transcoding settings in the AllTalk Settings page but is used by the transcoding process to determine if transcoding is necessary.

    #### **deepspeed_capable**
    > Indicates if the current TTS engine supports DeepSpeed and can handle DeepSpeed enable/disable requests.

    #### **deepspeed_available**
    > If the current TTS engine is DeepSpeed capable, shows whether DeepSpeed was detected and loaded for use.

    #### **deepspeed_enabled**
    > Indicates whether DeepSpeed is currently enabled or disabled for the current TTS engine, assuming the above two are true.

    #### **generationspeed_capable**
    > Indicates if the current TTS engine supports generating TTS at different speeds.

    #### **generationspeed_set**
    > Shows the current default speed set in the TTS engine's configuration settings.

    #### **lowvram_capable**
    > Indicates if the current TTS engine allows moving the TTS model between VRAM and System RAM, which is useful in low VRAM situations, such as when using AllTalk with an LLM that fills up VRAM. This helps keep models contiguous within VRAM and speeds up processing.

    #### **lowvram_enabled**
    > If the model is lowvram_capable, setting this to true allows the model to move between VRAM and System RAM as necessary.

    #### **pitch_capable**
    > Indicates if the current TTS engine supports generating TTS at different pitches.

    #### **pitch_set**
    > Shows the current default audio pitch set in the TTS engine's configuration settings.

    #### **repetitionpenalty_capable**
    > Indicates if the current TTS engine supports different repetition penalties for its TTS models.

    #### **repetitionpenalty_set**
    > Shows the current default repetition penalty set in the TTS engine's configuration settings.

    #### **temperature_capable**
    > Indicates if the current TTS engine supports different temperature settings for its TTS models.

    #### **temperature_set**
    > Shows the current default temperature set in the TTS engine's configuration settings.

    #### **languages_capable**
    > Indicates if the models within the current TTS engine support multiple languages. This can be used to disable language selection in your interface if necessary.

    #### **multivoice_capable**
    > Indicates if the current model loaded into the TTS engine supports multiple voices or just one voice. If a model is not multi-voice capable, you can display 'Default' as the only available voice.

    #### **multimodel_capable**
    > Some TTS engines, such as Piper, use models as voices. In such cases, you don't need to load different models but present the models as voices. If a model is not multimodel_capable, it indicates this status in your interface.

    """

    openapi_endpoint = """
    ### 🟠 OpenAI Speech API Compatible Endpoint<br>
    Validate and generate audio based on text input using an OpenAI Speech v1 API compatible endpoint, as per the documentation at [OpenAI Text-to-Speech Guide](https://platform.openai.com/docs/guides/text-to-speech) and [OpenAI API Reference](https://platform.openai.com/docs/api-reference/audio).

    Voices are mapped on a one-to-one basis within the AllTalk Gradio interface on a TTS engine by TTS engine basis. If RVC is globally enabled in AllTalk settings and a voice other than "Disabled" is selected for the character voice, the chosen RVC voice will be applied after the TTS is generated and before the audio is transcoded and sent back out.

    Note that there is no capability within the OpenAI API to specify a language, so the response will be in whatever language the currently loaded TTS engine and model support.

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/v1/audio/speech`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `POST`<br>

    `curl -X POST "http://127.0.0.1:7851/v1/audio/speech" -H "Content-Type: application/json" -d "{\"model\":\"any_model_name\",\"input\":\"Hello, this is a test.\",\"voice\":\"nova\",\"response_format\":\"wav\",\"speed\":1.0}"`

    The request body must be a JSON object with the following fields:

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**model** (string): The TTS model to use. Currently ignored, but is required in the request.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**input** (string): The text to generate audio for. Maximum length is 4096 characters.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**voice** (string): The voice to use when generating the audio. Supported voices are ["alloy", "echo", "fable", "nova", "onyx", "shimmer"].<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**response_format** (string, optional): The format of the audio. Audio will be transcoded to the requested format.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**speed** (float, optional): The speed of the generated audio. Must be between 0.25 and 4.0. Default is 1.0. Id the TTS engine loaded doesnt support different speeds, then the audio generated will not abide by this speed request.<br><br>

    ```
    {
    "model": "any_model_name",
    "input": "Hello, this is a test.",
    "voice": "nova",
    "response_format": "wav",
    "speed": 1.0
    }
    ```
    
    #### Python Example
    ```
    import requests
    import json

    # Define the endpoint URL
    url = "http://127.0.0.1:7851/v1/audio/speech"

    # Define the request payload
    payload = {
        "model": "any_model_name",
        "input": "Hello, this is a test.",
        "voice": "nova",
        "response_format": "wav",
        "speed": 1.0
    }

    # Set the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Send the POST request
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    # Check the response
    if response.status_code == 200:
        with open("output.wav", "wb") as f:
            f.write(response.content)
        print("Audio file saved as output.wav")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```
    #### Javascript Example
    ```
    // Define the endpoint URL
    const url = "http://127.0.0.1:7851/v1/audio/speech";

    // Define the request payload
    const payload = {
        model: "any_model_name",
        input: "Hello, this is a test.",
        voice: "nova",
        response_format: "wav",
        speed: 1.0
    };

    // Set the headers
    const headers = {
        "Content-Type": "application/json"
    };

    // Send the POST request
    fetch(url, {
        method: "POST",
        headers: headers,
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (response.ok) {
            return response.blob();
        } else {
            return response.text().then(text => { throw new Error(text); });
        }
    })
    .then(blob => {
        // Create a link element
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'output.wav';

        // Append the link to the body
        document.body.appendChild(a);

        // Programmatically click the link to trigger the download
        a.click();

        // Remove the link from the document
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        console.log("Audio file saved as output.wav");
    })
    .catch(error => {
        console.error("Error:", error);
    });
    ```
    
    """

    interaction_endpoints = """
    ### 🟠 Stop Generation Endpoint<br>
    Interrupt the current TTS generation process. When this endpoint is called, it sets tts_stop_generation in the model_engine to True, which can be used to interrupt the current TTS generation. Stop requests will only be honored if the current TTS engine is capable of handling and can honor a stop request partway through generation.

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/stop-generation`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `PUT`<br>

    > `curl -X PUT "http://127.0.0.1:7851/api/stop-generation"`

    ```
    {
    "message": "Cancelling current TTS generation"
    }
    ```

    ### 🟠 Reload Configuration Endpoint<br>
    Reload the TTS engine's configuration and scan for new voices and models. When this endpoint is called, it reloads all current settings from the configuration files and scans for new voices and models that are available. This ensures that subsequent calls to /api/currentsettings, /api/voices, and /api/rvcvoices return an up-to-date set of settings and information about the current state of the TTS engine.

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/reload_config`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `GET`<br>

    > `curl -X GET "http://127.0.0.1:7851/api/reload_config"`

    `Config file reloaded successfully`
    
    ### 🟠 Reload/Swap model<br>
    Can be used to load in/swap to one of the models presented by /api/currentsettings in the `models_available` list.

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/reload?tts_method=x`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `POST`<br>

    > `curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=xtts%20-%20xttsv2_2.0.2"`

    {"status": "model-success"} or {"status": "model-failure"}
    
    ### 🟠 Switch DeepSpeed Endpoint<br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/deepspeed`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `POST`<br>

    > `curl -X POST "http://127.0.0.1:7851/api/deepspeed?new_deepspeed_value=True"`

    Replace True with False to disable DeepSpeed mode.

    ```
    {
        "status": "deepspeed-success"
    }
    ```

    ### 🟠 Switching Low VRAM Endpoint<br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/lowvramsetting`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method: `POST`<br>

    > `curl -X POST "http://127.0.0.1:7851/api/lowvramsetting?new_low_vram_value=True"`

    Replace True with False to disable Low VRAM mode.

    ```
    {
        "status": "lowvram-success"
    }
    ```
    """

    standardtts_endpoint = """
    ### 🟠 TTS Generation Endpoint (Standard Generation)<br>
    Generate TTS audio based on text input. This endpoint supports both character and narrator speech generation. All global settings for the API endpoint can be configured within the AllTalk interface under Global Settings > AllTalk API Defaults. TTS engine-specific settings, such as voices to use or engine parameters, can be set on an engine-by-engine basis in TTS Engine Settings > TTS Engine of your choice.<br>

    It's important to note that although you can send all these variables/settings, the loaded TTS engine will only support them if it is capable e.g. you can request a TTS generation in Russian, but if the TTS model that is loaded in only support English, it's only going to generate english sounding text to speech.<br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/tts-generate`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: `POST`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Content-Type**: application/x-www-form-urlencoded<br>
    
    ### 🟠 Example Command Lines (Standard Generation)
    Standard TTS generation supports narration and generates a WAV file/blob.

    > **Standard TTS Speech Example** - Generate a time-stamped file for standard text:<br><br>
    `curl -X POST "http://127.0.0.1:7851/api/tts-generate" -d "text_input=All of this is text spoken by the character. This is text not inside quotes, though that doesnt matter in the slightest" -d "text_filtering=standard" -d "character_voice_gen=female_01.wav" -d "narrator_enabled=false" -d "narrator_voice_gen=male_01.wav" -d "text_not_inside=character" -d "language=en" -d "output_file_name=myoutputfile" -d "output_file_timestamp=true" -d "autoplay=true" -d "autoplay_volume=0.8"`

    > **Narrator Example** - Generate a time-stamped file for text with narrator and character speech:<br><br>
    `curl -X POST "http://127.0.0.1:7851/api/tts-generate" -d "text_input=*This is text spoken by the narrator* \\\"This is text spoken by the character\\\;". This is text not inside quotes." -d "text_filtering=standard" -d "character_voice_gen=female_01.wav" -d "narrator_enabled=true" -d "narrator_voice_gen=male_01.wav" -d "text_not_inside=character" -d "language=en" -d "output_file_name=myoutputfile" -d "output_file_timestamp=true" -d "autoplay=true" -d "autoplay_volume=0.8"`<br>

    > If your text contains double quotes, escape them with &#92;" (see the narrator example).<br>
    Voices sent in the request have to match the voices available within the TTS engine loaded. Generation requests where the voices dont match, will result nothing being generated and possibly an error message.

    Because of the way AllTalk works you can send a request with any mix of settings you wish e.g. 

    > `curl -X POST "http://127.0.0.1:7851/api/tts-generate" -d "text_input=All of this is text spoken by the character. This is text not inside quotes, though that doesnt matter in the slightest"`

    Would generate the TTS for whatver engine is curretly loaded in. It will use the default API settings and default TTS engine settings to populate any missing fields, which gives you flexibility on how you want to send API requests to AllTalk.

    ### 🟠 Request Parameters
    #### **text_input**
    > The text you want the TTS engine to produce. Use escaped double quotes for character speech and asterisks for narrator speech if using the narrator function. Example:<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "text_input=*This is text spoken by the narrator* &#92;"This is text spoken by the character&#92;". This is text not inside quotes."

    #### **text_filtering**
    > Filter for text. Options:<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**none**: No filtering. Raw text is sent to the TTS engine.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**standard**: Basic level of filtering to clean up special characters.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**html**: For HTML content with entities like &quot;.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "text_filtering=none"<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "text_filtering=standard"<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "text_filtering=html"<br>

    #### **character_voice_gen**
    > The name of the character's voice file (WAV format).<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "character_voice_gen=female_01.wav"

    #### **rvccharacter_voice_gen**
    > The name of the RVC voice file for the character. Should be in the format `folder\\file.pth` or the word `Disabled`. When Disabled is sent and RVC is globally enabled, the RVC pipeline will not be used for the character/main voice.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "rvccharacter_voice_gen=folder\\voice1.pth"

    #### **narrator_enabled**
    > Enable or disable the narrator function. If true, minimum text filtering is set to standard.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "narrator_enabled=true"<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "narrator_enabled=false"

    #### **narrator_voice_gen**
    > The name of the narrator's voice file (WAV format).<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "narrator_voice_gen=male_01.wav"

    #### **rvcnarrator_voice_gen**
    > The name of the RVC voice file for the narrator. Should be in the format folder\\file.pth or the word Disabled.  When Disabled is sent and RVC is globally enabled, the RVC pipeline will not be used for the character/main voice.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "rvcnarrator_voice_gen=folder\\voice2.pth"

    #### **text_not_inside**
    > Specify the handling of lines not inside double quotes or asterisks for the narrator feature. Options:<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**character**: Treat as character speech.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**narrator**: Treat as narrator speech.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**silent**: Ignore text not inside quotes or asterisks.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "text_not_inside=character"<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "text_not_inside=narrator"<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "text_not_inside=silent"<br>

    #### **language** 
    > Choose the language for TTS. Options:<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**ar**: Arabic<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**zh-cn**: Chinese (Simplified)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**cs**: Czech<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**nl**: Dutch<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**en**: English<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**fr**: French<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**de**: German<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**hi**: Hindi (limited support)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**hu**: Hungarian<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**it**: Italian<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**ja**: Japanese<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**ko**: Korean<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**pl**: Polish<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**pt**: Portuguese<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**ru**: Russian<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**es**: Spanish<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**tr**: Turkish<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d "language=en"

    #### **output_file_name***
    > The name of the output file (excluding the .wav extension).<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "output_file_name=myoutputfile"

    #### **output_file_timestamp**
    > Add a timestamp to the output file name. If true, each file will have a unique timestamp; otherwise, the same file name will be overwritten each time you generate TTS.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "output_file_timestamp=true"<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "output_file_timestamp=false"

    #### **autoplay**
    > Enable or disable playing the generated TTS to your standard sound output device at the time of TTS generation.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "autoplay=true"<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "autoplay=false"
    
    #### **autoplay_volume**
    > Set the autoplay volume. Should be between 0.1 and 1.0.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "autoplay_volume=0.8"

    #### **speed**
    > Set the speed of the generated audio. Should be between 0.25 and 2.0.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "speed=1.0"

    #### **pitch**
    > Set the pitch of the generated audio. Should be between -10 and 10.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "pitch=0"

    #### **temperature**
    > Set the temperature for the TTS engine. Should be between 0.1 and 1.0.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "temperature=0.75"

    #### **repetition_penalty**
    > Set the repetition penalty for the TTS engine. Should be between 1.0 and 20.0.<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--d "repetition_penalty=10"

    ### 🟠 TTS Generation Response
    > The API returns a JSON object with the following properties:<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**status**: Indicates whether the generation was successful (generate-success) or failed (generate-failure).<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**output_file_path**: The on-disk location of the generated WAV file.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**output_file_url**: The HTTP location for accessing the generated WAV file for browser playback.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**output_cache_url**: The HTTP location for accessing the generated WAV file as a pushed download.<br>

    ```
    {
        "status": "generate-success",
        "output_file_path": "C:\\text-generation-webui\\extensions\\alltalk_tts\\outputs\\myoutputfile_1704141936.wav",
        "output_file_url": "/audio/myoutputfile_1704141936.wav",
        "output_cache_url": "/audiocache/myoutputfile_1704141936.wav"
    }
    ```
    <br>
    Note: The response no longer includes the IP address and port number. You will need to add these in your own software/extension.
    """

    streamingtts_endpoint = """
    ### 🟠 TTS Generation Endpoint (Streaming Generation)
    Generate TTS audio as a stream. This endpoint does not support narration and will generate an audio stream, not a file. It also does not support the RVC pipeline. Generate and stream TTS audio directly for real-time playback.

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **URL**: `http://{ipaddress}:{port}/api/tts-generate-streaming`<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Method**: POST<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Content-Type**: application/x-www-form-urlencoded<br>

    > `http://127.0.0.1:7851/api/tts-generate-streaming?text=Here%20is%20some%20text&voice=female_01.wav&language=en&output_file=stream_output.wav`

    Example JavaScript for Streaming Playback:

    ```
    const text = "Here is some text";
    const voice = "female_01.wav";
    const language = "en";
    const outputFile = "stream_output.wav";
    const encodedText = encodeURIComponent(text);
    const streamingUrl = `http://localhost:7851/api/tts-generate-streaming?text=${encodedText}&voice=${voice}&language=${language}&output_file=${outputFile}`;
    const audioElement = new Audio(streamingUrl);
    audioElement.play();
    ```

    Response: A StreamingResponse for the audio stream.

    Request Parameters:

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**text**: The text to convert to speech.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**voice**: The voice type to use.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**language**: The language for the TTS.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**output_file**: The name of the output file.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Response**: A JSON object indicating the status of the request.

    ```
    {
        "output_file_path": "stream_output.wav"
    }
    ```
    """
    
    with gr.Tab("API Endpoints & Dev"):
        with gr.Tab("Best Development Practices"): gr.Markdown(best_practices)
        with gr.Tab("Standard TTS Generation"): gr.Markdown(standardtts_endpoint)
        with gr.Tab("Streaming TTS Generation"): gr.Markdown(streamingtts_endpoint)
        with gr.Tab("Settings API's"): gr.Markdown(settings_endpoints)
        with gr.Tab("Interaction API's"): gr.Markdown(interaction_endpoints)
        with gr.Tab("OpenAPI Compatible Endpoint"): gr.Markdown(openapi_endpoint)