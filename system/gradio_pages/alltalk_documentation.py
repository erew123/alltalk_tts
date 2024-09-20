import gradio as gr

def alltalk_documentation():
    low_vram = """
    The Low VRAM option is a crucial feature designed to enhance performance under constrained (VRAM) conditions, as TTS models **can** require 2GB-3GB of VRAM to run effectively. This feature strategically manages the relocation of the Text-to-Speech (TTS) model between your system's Random Access Memory (RAM) and VRAM, moving it between the two on the fly. Obviously, this is very useful for people who have smaller graphics cards or people who's LLM has filled their VRAM, but you want to keep up performance.

    When you don't have enough VRAM free after loading your LLM model into your VRAM (Normal Mode example below), you can see that with so little working space, your GPU will have to swap in and out bits of the TTS model, which causes horrible slowdown.

    Note An Nvidia Graphics card is required for the LowVRAM option to work, as you will just be using system RAM otherwise.

    ### How It Works
    The Low VRAM mode intelligently relocates of the entire TTS model and stores the TTS model in your system RAM when not being used. When the TTS engine requires VRAM for processing, the entire model seamlessly moves into VRAM, causing your LLM to unload/displace some layers, ensuring optimal performance of the TTS engine.

    Post-TTS processing, the model moves back to system RAM, freeing up VRAM space for your Language Model (LLM) to load back in the missing layers. This adds about 1-2 seconds to both text generation by the LLM and the TTS engine.

    By transferring the entire model between RAM and VRAM, the Low VRAM option avoids model fragmentation, ensuring the TTS model remains cohesive and has all the working space it needs in your GPU, without having to just work on small bits of the TTS model at a time (which causes terrible slow down).

    This creates a TTS generation performance Boost for Low VRAM Users and is particularly beneficial for users with less than 2GB of free VRAM after loading their LLM, delivering a substantial 5-10x improvement in TTS generation speed.
    """

    silly_tavern_support = """
    #### SillyTavern/AllTalk extension version
    If you intend to use AllTalk V2 with SillyTavern, you will need to update the SillyTavern Extension from the `\alltalk_tts\system\SillyTavern Extension\` folder. Instructions are included within the folder.
    
    #### Important note for Text-generation-webui users
    You **HAVE** to disable **Enable TTS** within the Text-generation-webui AllTalk interface, otherwise Text-generation-webui will also generate TTS due to the way it sends out text. You can do this each time you start up Text-generation-webui or set it in the start-up settings at the top of this page.

    ### 🏰 Quick Tips
    > Only change DeepSpeed, Low VRAM or Model one at a time. Wait for it to say Ready before changing something else.<br>
    > You can permanently change the AllTalk startup settings or DeepSpeed, Low VRAM and Model at the top of this page.<br>
    > Different AI models use quotes and asterisks differently, so you may need to change "Text not inside" depending on model.<br>
    > Add new voice samples to the voices folder. You can Finetune a model to make it sound even closer to the original sample.<br>
    > DeepSpeed will improve processing time to TTS to be 2-3x faster.<br>
    > Low VRAM can be very beneficial if you don't have much memory left after loading your LLM.<br>

    #### 🏰 TTS Generation Methods in SilllyTavern
    You have 2 types of audio generation options, Streaming and Standard.

    The Streaming Audio Generation method is designed for speed and is best suited for situations where you just want quick audio playback. This method, however, is limited to using just one voice per TTS generation request. This means a limitation of the Streaming method is the inability to utilize the AllTalk narrator function, making it a straightforward but less nuanced option.

    On the other hand, the Standard Audio Generation method provides a richer auditory experience. It's slightly slower than the Streaming method but compensates for this with its ability to split text into multiple voices. This functionality is particularly useful in scenarios where differentiating between character dialogues and narration can enhance the storytelling and delivery. The inclusion of the AllTalk narrator functionality in the Standard method allows for a more layered and immersive experience, making it ideal for content where depth and variety in voice narration add significant value.

    In summary, the choice between Streaming and Standard methods in AllTalk TTS depends on what you want. Streaming is great for quick and simple audio generation, while Standard is preferable for a more dynamic and engaging audio experience.

    **AllTalk TTS Generation Method:**
    > Select between Standard and Streaming Audio Generation methods.<br>
    > This setting impacts the AllTalk narrator functionality.

    **Language Selection:**
    > Select your preferred TTS generation language from the "Language" dropdown.

    **Model Switching:**
    > Switch between different TTS models like API TTS, API Local, XTTSv2 Local, and optionally XTTSv2 FT if you have a finetuned model available.<br>
    > Fine-tuned model availability (XTTSv2 FT) will only show when a finetuned model is detected by AllTalk.<br>
    > See TTS Models/Methods for more information (though most people will want to stick with XTTSv2 Local).<br>

    **DeepSpeed and Low VRAM Options:**
    > Optimize performance with DeepSpeed and Low VRAM settings.<br>
    > DeepSpeed can offer a 2-3x performance boost on TTS generation. (Requires installation)<br>
    > See the relevant sections in this documentation for details.<br>

    Changing model or DeepSpeed or Low VRAM **each** take about 15 seconds so you should only change one at a time and wait for `Ready` before changing the next setting. To set these options long term you can apply the settings at the top of this page.

    #### 🏰 AllTalk Narrator
    Only available on the Standard Audo Generation method.

    **Narrator Voice Selection:**
    > Allows users to choose different narrator voices.<br>
    > Access via the "Narrator Voice" dropdown.

    **AllTalk Narrator:**
    > Toggle the AllTalk narrator feature.<br>
    > Access via "AT Narrator" dropdown with Enabled/Disabled options.

    **Text Outside Asterisks Handling:**
    > Choose how text outside asterisks is interpreted (as Narrator or Character voice).<br>
    > Managed via the "Text Not Inside * or "" dropdown.<br>
    > Note, only available when the AllTalk Narrator is enabled.<br>

    #### 🏰 Usage Notes:
    > On startup of SillyTavern, it will pull your current settings from AllTalk (Current model, DeepSpeed status, Low VRAM status and Finetuned model availability).<br>
    > Enabling the narrator automatically unchecks certain checkboxes related to text handling.<br>
    > Changes in model or settings might trigger multiple requests to the server; patience is advised.<br>

    #### 🏰 Troubleshooting:
    > If experiencing issues, use the Reload button in SillyTavern's TTS extension to reinitialize the connection to AllTalk and check if AllTalk is started correctly.
    """

    updating_alltalk = """
    ### 🟪 Updating

    Maintaining the latest version of your setup ensures access to new features and improvements. Below are the steps to update your installation, whether you're using Text-Generation-webui or running as a Standalone Application.

    <details>
    <summary>UPDATING - Text-Generation-webui</summary>
 
    The update process closely mirrors the installation steps. Follow these to ensure your setup remains current:<br>

    **Open a Command Prompt/Terminal**
    > Navigate to your Text-Generation-webui folder with:<br>
    > `cd text-generation-webui`

    **Start the Python Environment**
    > Activate the Python environment tailored for your operating system. Use the appropriate command from below based on your OS:<br>
    > **Windows**: `cmd_windows.bat`<br>
    > **Linux**: `./cmd_linux.sh`<br>
    > **macOS**: `cmd_macos.sh`<br>
    > **WSL (Windows Subsystem for Linux)**: `cmd_wsl.bat`<br>

    > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

    **Navigate to the AllTalk TTS Folder**
    > Move into your extensions and then the alltalk_tts directory:<br>
    > `cd extensions/alltalk_tts`

    **Update the Repository**
    > Fetch the latest updates from the repository with:<br>
    > `git pull`

    **Install Updated Requirements**
    > Depending on your machine's OS, install the required dependencies using pip:<br>
    > **Windows Machines**:<br>
    > `pip install -r system\\requirements\\requirements_textgen.txt`<br><br>
    > **Linux/Mac**:<br>
    > `pip install -r system/requirements/requirements_textgen.txt`

    **DeepSpeed Requirements**
    > If Text-gen-webui is using a new version of PyTorch, you **may** need to uninstall and update your DeepSpeed version.<br>
    > Use AllTalks diagnostics or start-up menu to identify your version of PyTorch.
    <br>
    </details>

    <details>
    <summary>UPDATING - Standalone Application</summary>
    <br>

    If you installed from a ZIP file, you cannot use a `git pull` to update, as noted in the Quick Setup instructions.

    For Standalone Application users, here's how to update your setup:

    1. **Open a Command Prompt/Terminal**:
    > Navigate to your AllTalk folder with<br>
    > `cd alltalk_tts`

    2. **Access the Python Environment**:
    > In a command prompt or terminal window, navigate to your `alltalk_tts` directory and start the Python environment:<br>
    > **Windows**<br>
    > `start_environment.bat`<br>
    > **Linux/macOS**<br>
    > `./start_environment.sh`<br>

    > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

    2. **Pull the Latest Updates**:
    > Retrieve the latest changes from the repository with:<br>
    > `git pull`
        
    3. **Install Updated Requirements**:
    > Depending on your machine's OS, install the required dependencies using pip:<br>
    > **Windows Machines**<br>
    > `pip install -r system\\requirements\\requirements_standalone.txt`<br>
    > **Linux/Mac**<br>
    > `pip install -r system/requirements/requirements_standalone.txt`
    <br>
    </details>

    ### 🟪 Resolving Update Issues

    If you encounter problems during or after an update, following these steps can help resolve the issue by refreshing your installation while preserving your data:

    <details>
    <summary>RESOLVING - Updates</summary><br>

    The process involves renaming your existing `alltalk_tts` directory, setting up a fresh instance, and then migrating your data:

    1. **Rename Existing Directory**:
    > First, rename your current `alltalk_tts` folder to keep it safe e.g. `alltalk_tts.old`. This preserves any existing data.

    2. **Follow the Quick Setup instructions**:
    > You will now follow the **Quick Setup** instructions, performing the `git clone https://github.com/erew123/alltalk_tts` to pull down a new copy of AllTalk and install the requirements.
        
    > If you're not familiar with Python environments, see **Understanding Python Environments Simplified** in the Help section for more info.

    3. **Migrate Your Data**:
    > **Before** starting the AllTalk, transfer the `models`, `voices`, `outputs` folders and also `confignew.json` from `alltalk_tts.old` to the new `alltalk_tts` directory. This action preserves your voice history and prevents the need to re-download the model.

    4) **Launch AllTalk**
    > You're now ready to launch AllTalk and check it works correctly.

    6. **Final Step**:
    > Once you've verified that everything is working as expected and you're satisfied with the setup, feel free to delete the `alltalk_tts.old` directory to free up space.

    </details>
    """

    narrator_guides = """
    Messages intended for the Narrator should be enclosed in asterisks `*` and those for the character inside quotation marks `"`. However, AI systems often deviate from these rules, resulting in text that is neither in quotes nor asterisks. Sometimes, text may appear with only a single asterisk, and AI models may vary their formatting mid-conversation. For example, they might use asterisks initially and then switch to unmarked text. A properly formatted line should look like this:

    `"`Hey! I'm so excited to finally meet you. I've heard so many great things about you and I'm eager to pick your brain about computers.`"` `*`She walked across the room and picked up her cup of coffee`*`

    Most narrator/character systems switch voices upon encountering an asterisk or quotation marks, which is somewhat effective. AllTalk has undergone several revisions in its sentence splitting and identification methods. While some irregularities and AI deviations in message formatting are inevitable, any line beginning or ending with an asterisk should now be recognized as Narrator dialogue. Lines enclosed in double quotes are identified as Character dialogue. For any other text, you can choose how AllTalk handles it: whether it should be interpreted as Character or Narrator dialogue (most AI systems tend to lean more towards one format when generating text not enclosed in quotes or asterisks).
    
    ### 🟦 Narrator Types of Text

    The TTS engine identifies and processes three types of text when the Narrator function is `Enabled` or `Enabled (Silent)`:

    > **Narrated Text**: Text enclosed in asterisks `*Narrated text*`.<br>
    > **Character Text**: Text enclosed in double quotes `"Character text"`.<br>
    > **Text-Not-Inside**: Any text not enclosed in asterisks or double quotes. The handling of this text is customizable.<br>

    ### 🟦 Settings for Narrated Text

    You can control how narrated text is handled with the following settings:

    > **Enabled**: Narrated text will be spoken by the TTS engine using the narrator voice.<br>
    > **Enabled (Silent)**: Narrated text is recognized but not spoken. This is useful for adding silent annotations or actions within the text.<br>
    > **Disabled**: Narrated text will be spoken using the character voice instead of the narrator voice. The narrator engine is turned off, and the Text-Not-Inside setting does not apply.<br>

    Please note, if you set `Enabled` or `Enabled (silent)` as the APi defaults, then all text will go into the narrator function unless `disabled` is sent as part of the TTS generation request.

    ### 🟦 Settings for Text-Not-Inside

    You can customize how text-not-inside is handled with these options:

    > **&quot;Character&quot;**: Text-not-inside is treated as character dialogue and will be spoken.<br>
    > **&ast;Narrator&ast;**: Text-not-inside is treated as narration and will be spoken.<br>
    > **Silent**: Text-not-inside is recognized but not spoken.<br>

    ### 🟦 Examples and Variations

    Here are some examples to illustrate how different settings affect the text processing:

    > **Example 1**: <br>
    **Narrator Enabled, Text-Not-Inside as Character**<br><br>
    > **&ast;Narrated text&ast;** will be spoken as narration.<br>
    > **&quot;Character text&quot;** will be spoken as character dialogue.<br>
    > **Any other** text will be spoken as character dialogue.<br>

    > **Example 2**: <br>
    **Narrator Disabled**<br><br>
    > **&ast;Narrated text&ast;** will be spoken using the character voice.<br>
    > **&quot;Character text&quot;** will be spoken as character dialogue.<br>
    > **Any other** text will be spoken as character dialogue.<br>

    > **Example 3**: <br>
    **Narrator Enabled (Silent), Text-Not-Inside as Silent**<br><br>
    > **&ast;Narrated text&ast;** will be recognized but not spoken.<br>
    > **&quot;Character text&quot;** will be spoken as character dialogue.<br>
    > **Any other** text will be recognized but not spoken.<br>

    ### 🟦 Potential Confusions and Clarifications

    **Silent Options Impact**

    > **Narrator Enabled (Silent)**: This setting allows for silent annotations within your text. Users can add notes or actions in the narrative without them being spoken.<br>
    > **Text-Not-Inside as Silent**: This setting enables users to include text that is recognized by the TTS engine but remains unspoken, useful for adding context or instructions within the script that are not meant to be vocalized.<br>

    **Combining Settings**

    > **Full Silence**: If both Narrator is set to `Enabled (Silent)` and Text-Not-Inside is set to `Silent`, **only character text** in quotes will be spoken.<br>
    > **Partial Silence**: You can configure the settings to have only certain parts of the text spoken while others are silent. For example, you might want all narration silent and only character dialogue spoken, or vice versa.<br>

    ### 🟦 Short Story Example

    ><span style="color:RoyalBlue">&ast;Once upon a time in a small village, there was a young girl named Lily.&ast;</span><br>
    <span style="color:LimeGreen">Lily woke up early every morning.</span><br>
    <span style="color:DarkOrange">&quot;I love exploring the forest near my home.&quot;</span><br>
    <span style="color:RoyalBlue">&ast; Lily said. One day, she found a mysterious key lying on the ground.&ast;</span><br>
    <span style="color:DarkOrange">&quot;This key looks old and rusty, but it seems to sparkle in the sunlight,&quot;</span><br>
    <span style="color:LimeGreen">She thought. Lily decided she would keep the key safe until she could discover its purpose.</span><br>
    <span style="color:RoyalBlue">&ast;Curious and excited, she decided to find out what the key opened.&ast;</span><br>

    ### Short Story Explanation

    **Narrated Text** (Asterisks surrounding the text):<br>
    > <span style="color:RoyalBlue">&ast;Once upon a time in a small village, there was a young girl named Lily.&ast;</span><br>
    > <span style="color:RoyalBlue">&ast; Lily said. One day, she found a mysterious key lying on the ground.&ast;</span><br>
    > <span style="color:RoyalBlue">&ast;Curious and excited, she decided to find out what the key opened.&ast;</span><br><br>
    > These lines are enclosed in asterisks and represent the narration. If Narrator was set to `Enabled (Silent)` these lines would not be spoken.

    **Character Text** (Double Quotes surrounding the text):<br>
    > <span style="color:DarkOrange">&quot;I love exploring the forest near my home.&quot;</span><br>
    > <span style="color:DarkOrange">&quot;This key looks old and rusty, but it seems to sparkle in the sunlight.&quot;</span><br><br>
    > These lines are enclosed in double quotes and represent the character’s spoken dialogue or thoughts.

    **Text-Not-Inside** (Neither Asterisks nor Quotes):<br>
    > <span style="color:LimeGreen">Lily woke up early every morning.</span><br>
    > <span style="color:LimeGreen">She thought. Lily decided she would keep the key safe until she could discover its purpose.</span><br><br>
    > These lines are not enclosed in asterisks or quotes and will be handled according to the `Text-Not-Inside` settings. If set to `Text-not-inside` > `Silent` these lines would not be spoken. 

    """

    rvc_guides = """
    RVC (Retrieval-based Voice Conversion) enhances TTS by replicating voice characteristics for characters or narrators, adding depth to synthesized speech. It functions as a TTS-to-TTS pipeline and can be used with any TTS engine/model. However, the closer the original TTS generation is to the voice you want to use with RVC, the better the result will be. Therefore, it is recommended to use a voice cloning TTS engine like Coqui XTTS with voice samples for optimal performance.

    When you first enable RVC on the `Global Settings > RVC Settings` tab and click the `Update RVC Settings` button, AllTalk will create the necessary folders and download any of the missing model files required for RVC to work.

    ### 🟩 Voice Model Files

    Voice models should be stored in the `/models/rvc_voices/{subfolder}` directory in their own subfolder. Typically, a voice model includes a PTH file and potentially an index file. If an index file is present, AllTalk will automatically select and use it. If multiple index files are found, none will be used, and a message will be output to the console. You can find places on the internet to download pre-generated RVC voice models e.g. https://voice-models.com/ and huggingface etc.

    ### 🟩 Purpose of the Index File

    The index file helps improve the quality of the generated audio by providing a reference during the conversion process. The FAISS index enables faster and more accurate retrieval of voice characteristics, leading to more natural and high-quality voice synthesis. The training data size setting determines how much of the index is used, impacting both quality and computation time. By indexing a larger number of data points, the conversion process can produce audio that closely matches the desired voice characteristics.

    ### 🟩 RVC Settings Page
    """
    rvc_guides2 = """
    ### Default Character Voice Model
    > This setting allows you to select the voice model used for character conversion. If "Disabled" is selected, RVC will not be applied to character voices. This option is used only if RVC is enabled and no other voice is specified in the API request. Selecting a specific model will apply RVC to all character dialogues, potentially making them sound more natural and lifelike.

    ### Default Narrator Voice Model
    > This setting allows you to select the voice model used for narrator conversion. If "Disabled" is selected, RVC will not be applied to the narrator voice. This option is used only if RVC is enabled and no other voice is specified in the API request. Applying RVC to the narrator can enhance the overall narrative experience by providing a consistent and engaging voice.

    ### Index Influence Ratio
    > Sets the influence exerted by the index file on the final output. A higher value increases the impact of the index, potentially enhancing detail but also increasing the risk of artifacts. Fine-tuning this setting helps achieve the desired balance between audio detail and artifact prevention.
    
    ### Pitch
    > This slider sets the pitch of the audio output. Increasing the value raises the pitch, while decreasing the value lowers it. Adjusting the pitch can help match the voice to specific character traits or emotional states, enhancing the overall storytelling experience.

    ### Volume Envelope
    > This setting substitutes or blends with the volume envelope of the output. A ratio closer to 1 means the output envelope is more heavily employed. This can help maintain the natural dynamics of the voice, making it sound more realistic and less robotic.

    ### Protect Voiceless Consonants/Breath Sounds
    > Prevents artifacts in voiceless consonants and breath sounds. Higher values (up to 0.5) provide stronger protection but might affect indexing. This setting helps maintain the clarity and intelligibility of speech, especially in parts of the audio where voiceless consonants and breath sounds are present.
    
    ### AutoTune
    > Enables or disables auto-tune for the generated audio. Recommended for singing conversions to ensure the output remains in tune. Auto-tune can correct pitch discrepancies and produce a more polished and harmonious sound.
    """
    rvc_guides3 = """
    ### Filter Radius
    > If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration. This setting helps reduce noise and unwanted sounds in the audio, leading to cleaner and more professional-sounding output.

    ### Training Data Size (AllTalk Specific)
    > Determines the number of training data points used to train the FAISS index. Increasing the size may improve the quality of the output but can also increase computation time. Different index files have different sizes. This setting limits the maximum amount of the index used. For example, if an index file has 70,000 points and you set the limit to 50,000, only 50,000 points will be used. A higher training data size generally results in better audio quality because more data points contribute to a more accurate voice conversion.

    ### Embedder Model
    > Select between different models for learning speaker embedding. "hubert" and "contentvec" are the available options.<br><br>
    **hubert**: A model that focuses on capturing the phonetic and linguistic content of the voice, which is useful for a wide range of voice conversion tasks.<br>
    **contentvec**: A model that captures more detailed voice characteristics and nuances, potentially offering higher fidelity in voice conversion but may require more computational resources.

    ### Split Audio
    > Splits the audio into chunks for inference to obtain better results in some cases. This can improve the quality of conversion, especially for longer audio inputs, by ensuring more consistent and accurate processing across the entire length of the audio.

    ### Pitch Extraction Algorithm
    > Choose the algorithm used for extracting the pitch (F0) during audio conversion. The available options include:<br><br>
    **crepe**: Provides high accuracy in pitch detection and is robust against various types of noise.<br>
    **crepe-tiny**: A smaller, faster version of crepe with slightly reduced accuracy.<br>
    **dio**: A fast pitch extraction algorithm that is less accurate than crepe but suitable for real-time applications.<br>
    **fcpe**: Focuses on precise pitch extraction, often used for high-fidelity voice conversion.<br>
    **harvest**: Known for producing smooth and natural pitch contours, suitable for music and singing voice conversion.<br>
    **hybrid[rmvpe+fcpe]**: Combines the strengths of rmvpe and fcpe to achieve a balance of accuracy and performance.<br>
    **pm**: A robust algorithm that works well in various conditions, offering a balance of speed and accuracy.<br>
    **rmvpe**: Recommended for most cases due to its balance of accuracy and performance, especially in TTS applications.<br>
    """



    deepspeed_guides = """
    ### 🔵🟢 DeepSpeed Installation Options
    **DeepSpeed requires an Nvidia Graphics card**
    
    ---
    ## Documentation yet to be updated. In a Nutshell though....
    > AllTalk **Standalone** setups **(Windows & Linux)** will install DeepSpeed **automatically**. You can stop reading here and you dont have to do anything.
    
    > **Windows** users, please see instructions in `atsetup.bat`
    
    > **Linux users**, please look here https://github.com/erew123/alltalk_tts/releases/tag/DeepSpeed-14.2-Linux
    ---
    ### Legacy manual build of DeepSpeed Instructions
    #### 🔵 Linux Installation
    DeepSpeed requires access to the **Nvidia CUDA Development Toolkit** to compile on a Linux system. It's important to note that this toolkit is distinct and unrealted to your graphics card driver or the CUDA version the Python environment uses. 

    <details>
        <summary>Linux DeepSpeed - Text-generation-webui</summary>

    ### DeepSpeed Installation for Text generation webUI

    1. **Nvidia CUDA Development Toolkit Installation**:
    - The toolkit is crucial for DeepSpeed to compile/build for your version of Linux and requires around 3GB's of disk space.
    - Install using your package manager **(Recommended)** e.g. **CUDA Toolkit 11.8** or download directly from [Nvidia CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) (choose 11.8 or 12.1 for Linux).

    2. **Open a Terminal Console**:
    - After Nvidia CUDA Development Toolkit installation, access your terminal console.

    3. **Install libaio-dev**:
    - Use your Linux distribution's package manager.
        
    - - `sudo apt install libaio-dev` for Debian-based systems
        - `sudo yum install libaio-devel` for RPM-based systems.

    4. **Navigate to Text generation webUI Folder**:
    - Change directory to your Text generation webUI folder with `cd text-generation-webui`.
    
    5. **Activate Text generation webUI Custom Conda Environment**:
    - Run `./cmd_linux.sh` to start the environment.<br><br>
    
    > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

    6. **Set `CUDA_HOME` Environment Variable**:
    - DeepSpeed locates the Nvidia toolkit using the `CUDA_HOME` environment variable.
    - You will only set this temporarily as Text generation webUI sets up its own CUDA_HOME environment each time you use `./cmd_linux.sh` or `./start_linux.sh`

    7. **Temporarily Configuring `CUDA_HOME`**:
    - When the Text generation webUI Python environment is active **(step 5)**, set `CUDA_HOME`.
        
    - - `export CUDA_HOME=/usr/local/cuda`
        - `export PATH=${CUDA_HOME}/bin:${PATH}`
        - `export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH`

    - You can confirm the path is set correctly and working by running the command `nvcc --version` should confirm `Cuda compilation tools, release 11.8.`.
    - Incorrect path settings may lead to errors. If you encounter path issues or receive errors like `[Errno 2] No such file or directory` when you run the next step, confirm the path correctness or adjust as necessary.

    8. **DeepSpeed Installation**:
    - Install DeepSpeed using `pip install deepspeed`.

    9. **Troubleshooting**:
    - Troubleshooting steps for DeepSpeed installation can be located down below.
    - **NOTE**: You **DO NOT** need to set Text-generation-webUI's **--deepspeed** setting for AllTalk to be able to use DeepSpeed. These are two completely separate things and incorrectly setting that on Text-generation-webUI may cause other complications.
    </details>
    <details>
        <summary>Linux DeepSpeed - Standalone Installation</summary>

    ### DeepSpeed Installation for Standalone AllTalk

    1. **Nvidia CUDA Development Toolkit Installation**:
    - The toolkit is crucial for DeepSpeed to compile/build for your version of Linux and requires around 3GB's of disk space.
    - Install using your package manager **(Recommended)** e.g. **CUDA Toolkit 11.8** or download directly from [Nvidia CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) (choose 11.8 or 12.1 for Linux).

    2. **Open a Terminal Console**:
    - After Nvidia CUDA Development Toolkit installation, access your terminal console.
    
    3. **Install libaio-dev**:
    - Use your Linux distribution's package manager.
        
    - - `sudo apt install libaio-dev` for Debian-based systems
        - `sudo yum install libaio-devel` for RPM-based systems.

    4. **Navigate to AllTalk TTS Folder**:
    - Change directory to your AllTalk TTS folder with `cd alltalk_tts`.

    5. **Activate AllTalk Custom Conda Environment**:
    - Run `./start_environment.sh` to start the AllTalk Python environment.
    - This command will start the custom Python environment that was installed with `./atsetup.sh`.<br><br>
    
    > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

    6. **Set `CUDA_HOME` Environment Variable**:
    - The DeepSpeed installation routine locates the Nvidia toolkit using the `CUDA_HOME` environment variable. This can be set temporarily for a session or permanently, depending on other requirements you may have for other Python/System environments.
    - For temporary use, proceed to **step 8**. For a permanent solution, see [Conda's manual on setting environment variables](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#set-env-vars).

    7. **(Optional) Permanent `CUDA_HOME` Setup**:
    - If you choose to set `CUDA_HOME` permanently, follow the instructions in the provided Conda manual link above.

    8. **Configuring `CUDA_HOME`**:
    - When your Python environment is active **(step 5)**, set `CUDA_HOME`.
        
    - - `export CUDA_HOME=/usr/local/cuda`
        - `export PATH=${CUDA_HOME}/bin:${PATH}`
        - `export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH`

    - You can confirm the path is set correctly and working by running the command `nvcc --version` should confirm `Cuda compilation tools, release 11.8.`.
    - Incorrect path settings may lead to errors. If you encounter path issues or receive errors like `[Errno 2] No such file or directory` when you run the next step, confirm the path correctness or adjust as necessary.

    9. **DeepSpeed Installation**:
    - Install DeepSpeed using `pip install deepspeed`.

    10. **Starting AllTalk TTS WebUI**:
        - Launch the AllTalk TTS interface with `./start_alltalk.sh` and enable DeepSpeed.

    ### Troubleshooting

    - If setting `CUDA_HOME` results in path duplication errors (e.g., `.../bin/bin/nvcc`), you can correct this by unsetting `CUDA_HOME` with `unset CUDA_HOME` and then adding the correct path to your system's PATH variable.
    - Always verify paths and compatibility with other CUDA-dependent applications to avoid conflicts.
    - If you have multiple versions of the Nvidia CUDA Development Toolkit installed, you will have to specify the version number in step 8 for the CUDA_HOME path.
    - If it became necessary to uninstall DeepSpeed, you can do so by start the Python enviroment and then running `pip uninstall deepspeed`<br><br>

    </details>

    #### 🟢 Windows Installation
    You have 2x options for how to setup DeepSpeed on Windows. Pre-compiled wheel files for specific Python, CUDA and Pytorch builds, or manually compiling DeepSpeed.

    <details>
        <summary>Windows DeepSpeed - Pre-Compiled Wheels (Quick and Easy)</summary>

    ### DeepSpeed Installation with Pre-compiled Wheels

    1. **Introduction to Pre-compiled Wheels**:
    - The `atsetup.bat` utility simplifies the installation of DeepSpeed by automatically downloading and installing pre-compiled wheel files. These files are tailored for **specific** versions of Python, CUDA, and PyTorch, ensuring compatibility with both the **Standalone Installation** and a standard build of **Text-generation-webui**.

    2. **Manual Installation of Pre-compiled Wheels**:
    - If needed, pre-compiled DeepSpeed wheel files that I have built are available on the [Releases Page](https://github.com/erew123/alltalk_tts/releases). You can manually install or uninstall these wheels using the following commands:
        - Installation: `pip install {deep-speed-wheel-file-name-here}`
        - Uninstallation: `pip uninstall deepspeed`
    
    3. **Using `atsetup.bat` for Simplified Management**:
    - For those running the Standalone Installation or a standard build of Text-generation-webui, the `atsetup.bat` utility offers the simplest and most efficient way to manage DeepSpeed installations on Windows.

    </details>

    <details>
        <summary>Windows DeepSpeed - Manual Compilation</summary>

    ### Manual DeepSpeed Wheel Compilation

    1. **Preparation for Manual Compilation**:
    - Manual compilation of DeepSpeed wheels is an advanced process that requires:
        - **1-2 hours** of your time for initial setup and compilation.
        - **6-10GB** of disk space on your computer.
        - A solid technical understanding of Windows environments and Python.

    2. **Understanding Wheel Compatibility**:
    - A compiled DeepSpeed wheel is uniquely tied to the specific versions of Python, PyTorch, and CUDA used during its compilation. If any of these versions are changed, you will need to compile a new DeepSpeed wheel to ensure compatibility.

    3. **Compiling DeepSpee Resources**:
    - Myself and [@S95Sedan](https://github.com/S95Sedan) have worked to simplify the compilation process. [@S95Sedan](https://github.com/S95Sedan) has notably improved the process for later versions of DeepSpeed, ensuring ease of build on Windows.
    - Because [@S95Sedan](https://github.com/S95Sedan) is now maintaining the instructions for compiling DeepSpeed on Windows, please visit [@S95Sedan](https://github.com/S95Sedan)'s<br>[DeepSpeed GitHub page](https://github.com/S95Sedan/Deepspeed-Windows).

    </details>
    """

    help_with_problems = """
    ### 🆘 Support Requests, Troubleshooting & Feature requests
    I'm thrilled to see the enthusiasm and engagement with AllTalk! Your feedback and questions are invaluable, helping to make this project even better. To ensure everyone gets the help they need efficiently, please consider the following before submitting a support request:

    **Consult the Documentation:** A comprehensive guide and FAQ sections (below) are available to help you navigate AllTalk. Many common questions and troubleshooting steps are covered here.

    **Search Past Discussions:** Your issue or question might already have been addressed in the discussions area or [closed issues](https://github.com/erew123/alltalk_tts/issues?q=is%3Aissue+is%3Aclosed). Please use the search function to see if there's an existing solution or advice that applies to your situation.

    **Bug Reports:** If you've encountered what you believe is a bug, please first check the [Updates & Bug Fixes List](https://github.com/erew123/alltalk_tts/issues/25) to see if it's a known issue or one that's already been resolved. If not, I encourage you to report it by raising a bug report in the [Issues section](https://github.com/erew123/alltalk_tts/issues), providing as much detail as possible to help identify and fix the issue.

    **Feature Requests:** The current Feature request list can be [found here](https://github.com/erew123/alltalk_tts/discussions/74). I love hearing your ideas for new features! While I can't promise to implement every suggestion, I do consider all feedback carefully. Please share your thoughts in the [Discussions area](https://github.com/erew123/alltalk_tts/discussions) or via a Feature Request in the [Issues section](https://github.com/erew123/alltalk_tts/issues). 


    ### 🟨 Help with problems

    #### &nbsp;&nbsp;&nbsp;&nbsp; 🔄 **Minor updates/bug fixes list** can be found [here](https://github.com/erew123/alltalk_tts/issues/25)

    <details>
    <summary>🟨 How to make a diagnostics report file</summary><br>

    If you are on a Windows machine or a Linux machine, you should be able to use the `atsetup.bat` or `./atsetup.sh` utility to create a diagnositcs file. If you are unable to use the `atsetup` utility, please follow the instructions below.

    <details>
        <summary><strong>Manually making a diagnostics report file</strong></summary><br>

    1) Open a command prompt window and start the Python environment. Depending on your setup (Text-generation-webui or Standalone AllTalk), the steps to start the Python environment vary:<br>

    - **For Text-generation-webui Users**:
    - Navigate to the Text-generation-webui directory:
        - `cd text-generation-webui`
    - Start the Python environment suitable for your OS:
        - Windows: `cmd_windows.bat`
        - Linux: `./cmd_linux.sh`
        - macOS: `cmd_macos.sh`
        - WSL (Windows Subsystem for Linux): `cmd_wsl.bat`
    - Move into the AllTalk directory:
        - `cd extensions/alltalk_tts`

    - **For Standalone AllTalk Users**:
    - Navigate to the `alltalk_tts` folder:
        - `cd alltalk_tts`
    - Start the Python environment:
        - Windows: `start_environment.bat`
        - Linux: `./start_environment.sh`<br><br>
        
    > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

    2) Run the diagnostics and select the requirements file name you installed AllTalk with:<br>
    - `python diagnostics.py`

    3) You will have an on screen output showing your environment setttings, file versions request vs whats installed and details of your graphics card (if Nvidia). This will also create a file called `diagnostics.log` in the `alltalk_tts` folder, that you can upload if you need to create a support ticket on here.<br><br>

    ![image](https://github.com/erew123/alltalk_tts/assets/35898566/81b9a6e1-c54b-4da0-b85d-3c6fde566d6a)
    <br><br></details></details>

    ### Installation and Setup Issues

    <details>
        <summary>🟨 Understanding Python Environments Simplified</summary><br>
        
    Think of Python environments like different rooms in your house, each designed for a specific purpose. Just as you wouldn't cook in the bathroom or sleep in the kitchen, different Python applications need their own "spaces" or environments because they have unique requirements. Sometimes, these requirements can clash with those of other applications (imagine trying to cook a meal in a bathroom!). To avoid this, you can create separate Python environments.

    #### Why Separate Environments?
    Separate environments, like separate rooms, keep everything organized and prevent conflicts. For instance, one Python application might need a specific version of a library or dependency, while another requires a different version. Just as you wouldn't store kitchen utensils in the bathroom, you wouldn't want these conflicting requirements to interfere with each other. Each environment is tailored and customized for its application, ensuring it has everything it needs without disrupting others.

    #### How It Works in Practice:

    **Standalone AllTalk Installation:** When you install AllTalk standalone, it's akin to adding a new room to your house specifically designed for your AllTalk activities. The setup process, using the atsetup utility, constructs this custom "room" (Python environment `alltalk_environment`) with all the necessary tools and furnishings (libraries and dependencies) that AllTalk needs to function smoothly, without meddling with the rest of your "house" (computer system). The AllTalk environment is started each time you run `start_alltalk` or `start_environment` within the AllTalk folder.

    **Text-generation-webui Installation:** Similarly, installing Text-generation-webui is like setting up another specialized room. Upon installation, it automatically creates its own tailored environment, equipped with everything required for text generation, ensuring a seamless and conflict-free operation. The Text-generation-webui environment is started each time you run `start_*your-os-version*` or `cmd_*your-os-version*` within the Text-generation-webui folder.

    #### Managing Environments:
    Just as you might renovate a room or bring in new furniture, you can also update or modify Python environments as needed. Tools like Conda or venv make it easy to manage these environments, allowing you to create, duplicate, activate, or delete them much like how you might manage different rooms in your house for comfort and functionality.

    Once you're in the right environment, by activating it, installing or updating dependencies (the tools and furniture of your Python application) is straightforward. Using pip, a package installer for Python, you can easily add what you need. For example, to install all required dependencies listed in a requirements.txt file, you'd use:

    `pip install -r requirements.txt`

    This command tells pip to read the list of required packages and versions from the requirements.txt file and install them in the current environment, ensuring your application has everything it needs to operate. It's like having a shopping list for outfitting a room and ensuring you have all the right items delivered and set up.

    Remember, just as it's important to use the right tools for tasks in different rooms of your house, it's crucial to manage your Python environments and dependencies properly to ensure your applications run as intended.

    #### How do I know if I am in a Python environment?:
    When a Python environment starts up, it changes the command prompt to show the Python environment that it currently running within that terminal/console. 

    ![image](https://github.com/erew123/screenshots/blob/main/pythonenvironment.jpg)
    </details>

    <details>
        <summary>🟨 Windows & Python requirements for compiling packages <strong>(ERROR: Could not build wheels for TTS)</strong></summary><br>

    `ERROR: Microsoft Visual C++ 14.0 or greater is required` or `ERROR: Could not build wheels for TTS.` or `ModuleNotFoundError: No module named 'TTS`

    Python requires that you install C++ development tools on Windows. This is detailed on the [Python site here](https://wiki.python.org/moin/WindowsCompilers). You would need to install `MSVCv142 - VS 2019 C++ x64/x86 build tools` and `Windows 10/11 SDK` from the C++ Build tools section. 
    
    You can get hold of the **Community** edition [here](https://visualstudio.microsoft.com/downloads/) the during installation, selecting `C++ Build tools` and then `MSVCv142 - VS 2019 C++ x64/x86 build tools` and `Windows 10/11 SDK`. 

    ![image](https://github.com/erew123/screenshots/raw/main/pythonrequirementswindows.jpg)
    
    </details>
    <details>
        <summary>🟨 Standalone Install - start_{youros}.xx opens and closes instantly and AllTalk doesnt start</summary><br>

    This is more than likely caused by having a space ` ` in your folder path e.g. `c:\program files\alltalk_tts`. In this circumstance you would be best moving the folder to a path without a space e.g. `c:\myfiles\alltalk_tts`. You would have to delete the `alltalk_environment` folder and `start_alltalk.bat` or `start_alltalk.sh` and then re-run `atsetup` to re-create the environment and startup files. 
    </details>
    
    <details>
        <summary>🟨 Text-geneneration-webui & Stable-Diffusion Plugin - Load Order & stripped text</summary><br>
        
    The Stable Diffusion plugin for Text-generation-webui **strips out** some of the text, which is passed to Stable Diffusion for image/scene generation. Because this text is stripped, its important to consider the load order of the plugins to get the desired result you want. Lets assume the AI has just generated the following message `*He walks into the room with a smile on his face and says* Hello how are you?`. Depending on the load order will change what text reaches AllTalk for generation e.g.

    **SD Plugin loaded before AllTalk** - Only `Hi how are you?` is sent to AllTalk, with the `*He walks into the room with a smile on his face and says*` being sent over to SD for image generation. Narration of the scene is not possible.<br><br>
    **AllTalk loaded before SD Plugin** - `*He walks into the room with a smile on his face and says* Hello how are you?` is sent to AllTalk with the `*He walks into the room with a smile on his face and says*` being sent over to SD for image generation.<br><br>
    The load order can be changed within Text-generation-webui's `settings.yaml` file or `cmd_flags.txt` (depending on how you are managing your extensions).<br><br>
    </details>   
    
    <details>
        <summary>🟨 I think AllTalks requirements file has installed something another extension doesn't like</summary><br>
        
    Ive paid very close attention to **not** impact what Text-generation-webui is requesting on a factory install. This is one of the requirements of submitting an extension to Text-generation-webui. If you want to look at a comparison of a factory fresh text-generation-webui installed packages (with cuda 12.1, though AllTalk's requirements were set on cuda 11.8) you can find that comparison [here](https://github.com/erew123/alltalk_tts/issues/23). This comparison shows that AllTalk is requesting the same package version numbers as Text-generation-webui or even lower version numbers (meaning AllTalk will not update them to a later version). What other extensions do, I cant really account for that.

    I will note that the TTS engine downgrades Pandas data validator to 1.5.3 though its unlikely to cause any issues. You can upgrade it back to text-generation-webui default (december 2023) with `pip install pandas==2.1.4` when inside of the python environment. I have noticed no ill effects from it being a lower or higher version, as far as AllTalk goes. This is also the same behaviour as the Coqui_tts extension that comes with Text-generation-webui.

    Other people are reporting issues with extensions not starting with errors about Pydantic e.g. ```pydantic.errors.PydanticImportError: BaseSettings` has been moved to the pydantic-settings package. See https://docs.pydantic.dev/2.5/migration/#basesettings-has-moved-to-pydantic-settings for more details.```

    Im not sure if the Pydantic version has been recently updated by the Text-generation-webui installer, but this is nothing to do with AllTalk. The other extension you are having an issue with, need to be updated to work with Pydantic 2.5.x. AllTalk was updated in mid december to work with 2.5.x. I am not specifically condoning doing this, as it may have other knock on effects, but within the text-gen Python environment, you can use `pip install pydantic==2.5.0` or `pip install pydantic==1.10.13` to change the version of Pydantic installed.
    </details>
    <details>
        <summary>🟨 I am having problems getting AllTalk to start after changing settings or making a custom setup/model setup.</summary><br>
        
    I would suggest following [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and if you still have issues after that, you can raise an issue [here](https://github.com/erew123/alltalk_tts/issues)
    </details>

    ### Networking and Access Issues

    <details>
        <summary>🟨 I cannot access AllTalk from another machine on my Network</summary><br>

    You will need to change the IP address within AllTalk's settings from being 127.0.0.1, which only allows access from the local machine its installed on. To do this, please see [Changing AllTalks IP address & Accessing AllTalk over your Network](https://github.com/erew123/alltalk_tts/tree/main?tab=readme-ov-file#-changing-alltalks-ip-address--accessing-alltalk-over-your-network) at the top of this page.

    You may also need to allow access through your firewall or Antivirus package to AllTalk.
    </details>

    <details>
        <summary>🟨 I am running a Headless system and need to change the IP Address manually as I cannot reach the config page</summary><br>
        
    To do this you can edit the `confignew.json` file within the `alltalk_tts` folder. You would look for `"ip_address": "127.0.0.1",` and change the `127.0.0.1` to your chosen IP address,then save the file and start AllTalk.<br><br>

    When doing this, be careful not to impact the formatting of the JSON file. Worst case, you can re-download a fresh copy of `confignew.json` from this website and that will put you back to a factory setting.
    </details>

    ### Configuration and Usage Issues
    <details>
        <summary>🟨 I activated DeepSpeed in the settings page, but I didnt install DeepSpeed yet and now I have issues starting up</summary><br>
        
    You can either follow the [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating) and fresh install your config. Or you can edit the `confignew.json` file within the `alltalk_tts` folder. You would look for '"deepspeed_activate": true,' and change the word true to false `"deepspeed_activate": false,' ,then save the file and try starting again.

    If you want to use DeepSpeed, you need an Nvidia Graphics card and to install DeepSpeed on your system. Instructions are [here](https://github.com/erew123/alltalk_tts#-deepspeed-installation-options)
    </details>

    <details>
        <summary>🟨 I am having problems updating/some other issue where it wont start up/Im sure this is a bug</summary><br>
        
    Please see [Problems Updating](https://github.com/erew123/alltalk_tts#-problems-updating). If that doesnt help you can raise an ticket [here](https://github.com/erew123/alltalk_tts/issues). It would be handy to have any log files from the console where your error is being shown. I can only losely support custom built Python environments and give general pointers. Please create a `diagnostics.log` report file to submit with a support request.

    Also, is your text-generation-webui up to date? [instructions here](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install)
    </details>

    <details>
        <summary>🟨 Error Details: HTTPConnectionPool(host='null', port=80): Max retries exceeded with url: /api/tts-generate............ etc</summary>
        
    If you get this error, the gradio window you have open has lost connection to the AllTalk backend. This is most likely caused by AllTalk being restarted in the background and you not opening the gradio window again. Simply refresh the gradio web page and it will be connected back in and working.
    <br>
    </details>

    <details>
        <summary>🟨 I see some red "asyncio" messages</summary><br>
        
    As far as I am aware, these are to do with the chrome browser the gradio text-generation-webui in some way. I raised an issue about this on the text-generation-webui [here](https://github.com/oobabooga/text-generation-webui/issues/4788) where you can see that AllTalk is not loaded and the messages persist. Either way, this is more a warning than an actual issue, so shouldnt affect any functionality of either AllTalk or text-generation-webui, they are more just an annoyance.
    </details>

    ### Startup, Performance and Compatibility Issues

    <details>
        <summary>🟨 Understanding the AllTalk start-up screen</summary><br>

    The AllTalk start-up screen provides various bits of information about the detected Python environment and errors.
    
    ![image](https://github.com/erew123/screenshots/raw/main/alltalkstartup.jpg)

    **Config file check**<br>
    - Sometimes I need to add/remove something to your existing configuration file settings. Obviously, I don’t want to impact your existing settings, however any new features may need these settings to be created before AllTalk starts up. Ive added extra code that checks `alltalk_tts/system/config/at_configupdate.json` and `alltalk_tts/system/config/at_configdowngrade.json`, either adding or removing items to your configuration as necessary. If a change is made, you will be notified and a backup of the previous configuration file will be created in the `alltalk_tts` folder.

    **AllTalk startup Mode**<br>
    - informational. This will state if AllTalk has detected it is running as part of Text-generation-webui or as a Standalone Application.
        
    **WAV file deletion**<br>
    - If you have set deletion of old generated WAV files, this will state the time frame after which they are purged.

    **DeepSpeed version**<br>
    - What version of DeepSpeed is installed/detected. This will **not** tell you if the version of DeepSpeed is compiled for your Python, PyTorch or CUDA version. Its important to remember that DeepSpeed has to be compiled for the exact version of Python, PyTorch and CUDA that you are using, so please ensure you have the correct DeepSpeed version installed if necessary.

    **Model is available**<br>
    - AllTalk is checking if your model files exist. This is not a validity check of the actual model files, they can still be corrupted. If files are missing, AllTalk will attempt to download them from Huggingface, however, if Huggingface has an outage/issue or your internet connection has issues, its possible corrupted or incomplete files will be downloaded. Please read `RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory` if you need to confirm your model files are ok.
        
    **Current Python Version**<br>
    - Informational. Literally tells you the version of Python running in your Python environment.
        
    **Current PyTorch Version**<br>
    - Informational. Tell tells you the version of **PyTorch** running in your Python environment, however if you have an Nvidia card, you should be running a CUDA based version of Pytorch. This is indicated with a `+cXXX` after the PyTorch version e.g. `2.2.2+cu121` would be PyTorch version 2.2.2 with CUDA 12.1 extensions. If you don’t have the PyTorch CUDA extensions installed, but you do have an Nvidia card, you may need to re-install PyTorch.

    **Current CUDA Version**<br>
    - Informational. This is linked to the Current PyTorch Version, as detailed above.

    **Current TTS Version**<br>
    - Informational. The current version of the TTS engine that is running.

    **AllTalk Github updated**<br>
    - As long as you have an internet connection, this will tell you last time AllTalk was updated on Github. It is checking the [commit list](https://github.com/erew123/alltalk_tts/commits/main/) to see when the last commit was made. As such, this could be simply a documentation update, a bug fix or new features. Its simply there as a guide to let you know the last time something was changed on AllTalk's Github.

    **TTS Subprocess**<br>
    - When AllTalk reaches this stage, the subprocess that loads in the AI model is starting. This is most likely where an error could occur with loading the TTS model, just after the documentation message.

    **AllTalk Settings & Documentation: http ://x.x.x.x**<br>
    - The link where you can reach AllTalks built in settings and documentation page. The TTS model will be loading immediately after this is displayed.
    <br><br>
    </details>

    <details>
        <summary>🟨 AllTalk is only loading into CPU, but I have an Nvidia GPU so it should be loading into CUDA</summary><br>
        
    This is caused by Pytorch (Torch) not having the CUDA extensions installed (You can check by running the diagnostics). Typically this happens (on Standalone installations) because when the setup routine goes to install Pytorch with CUDA, it looks in the PIP cache and if a previous application has downloaded a version of Pytorch that **doesn't** have CUDA extensions, the PIP installer doesnt recognise this fact and just uses the cached version for installation. To resolve this:

    1) On the `atsetup` utility, on the `Standalone menu` select to `Purge the PIP cache`. This will remove cached packages from the PIP cache, meaning it will have to download fresh copies.
    2) As we need to force the upgrade to the Python environment, the easiest way to do this will be to use `atsetup` to `Delete AllTalk's custom Python environment`. This means it will have to rebuild the Python environment. **Note**, you may have to run this step twice, as it has to exit the current Python environment, then you have to re-load `atsetup` and select `Delete AllTalk's custom Python environment` again.
    3) You can now use `atsetup` to `Install AllTalk as a Standalone Application` which will download fresh copies of everything and re-install the Python environment. 
    4) Once this is done you can check if CUDA is now working with the diagnostics or starting AllTalk and checking the model loads into CUDA.

    </details>


    <details>
        <summary>🟨 RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory</summary><br>
        
    This error message is caused by the model being corrupted or damaged in some way. This error can occur if Huggingface, where the model is downloaded from, have an error (when the model is downloaded) or potentailly internet issues occuring while the model is downloaded on first start-up. 

    ```
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory

    ERROR: Application startup failed. Exiting.
    [AllTalk Startup] Warning TTS Subprocess has NOT started up yet, Will keep trying for 120 seconds maximum. Please wait.
    ```

    To resolve this, first look in your `alltalk_tts/models/xttsv2_2.0.2` (or whichever) model folder and confirm that the file sizes are correct.

    ![image](https://github.com/erew123/screenshots/raw/main/modelsfiles.jpg)

    You can delete one or more suspect files and a factory fresh copy of that file or files will be downloaded on next start-up of AllTalk.

    </details>

    <details>
        <summary>🟨 RuntimeError: Found no NVIDIA driver on your system.</summary><br>
        
    This error message is caused by DeepSpeed being enabled when you do not have a Nvidia GPU. To resolve this, edit `confignew.json` and change `"deepspeed_activate": true,` to `"deepspeed_activate": false,` then restart AllTalk.

    ```
    File "C:\alltalk_tts\alltalk_environment\env\Lib\site-packages\torch\cuda\__init__.py", line 302, in _lazy_init
        torch._C._cuda_init()
    RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx

    ERROR:    Application startup failed. Exiting.
    [AllTalk Startup] Warning TTS Subprocess has NOT started up yet, Will keep trying for 120 seconds maximum. Please wait.
    ```

    </details>

    <details>
        <summary>🟨 raise RuntimeError("PyTorch version mismatch! DeepSpeed ops were compiled and installed.</summary><br>
        
    This error message is caused by having DeepSpeed enabled, but you have a version of DeepSpeed installed that was compiled for a different version of Python, PyTorch or CUDA (or any mix of those). You will need to start your Python environment and run `pip uninstall deepspeed` to remove DeepSpeed from your Python environment and then install the correct version of DeepSpeed.

    ```
    raise RuntimeError("PyTorch version mismatch! DeepSpeed ops were compiled and installed 
    RuntimeError: PyTorch version mismatch! DeepSpeed ops were compiled and installed with a different version than what is being used at runtime. Please re-install DeepSpeed or switch torch versions. Install torch version=2.1, Runtime torch version=2.2
    ```

    </details>

    <details>
        <summary>🟨 Warning TTS Subprocess has NOT started up yet, Will keep trying for 120 seconds maximum. Please wait. It times out after 120 seconds.</summary><br>
        When the subprocess is starting 2x things are occurring:<br><br>

    **A)** Its trying to load the voice model into your graphics card VRAM (assuming you have a Nvidia Graphics card, otherwise its your system RAM)<br>
    **B)** Its trying to start up the mini-webserver and send the "ready" signal back to the main process.

    Before giving other possibilities a go, some people with **old machines** are finding their startup times are **very** slow 2-3 minutes. Ive extended the allowed time within the script from 1 minute to 2 minutes. **If you have an older machine** and wish to try extending this further, you can do so by editing `script.py` and changing `startup_wait_time = 120` (120 seconds, aka 2 minutes) at the top of the script.py file, to a larger value e.g `startup_wait_time = 240` (240 seconds aka 4 minutes).

    **Note:** If you need to create a support ticket, please create a `diagnostics.log` report file to submit with a support request. Details on doing this are above.

    Other possibilities for this issue are:

    1) You are starting AllTalk in both your `CMD FLAG.txt` and `settings.yaml` file. The `CMD FLAG.txt` you would have manually edited and the `settings.yaml` is the one you change and save in the `session` tab of text-generation-webui and you can `Save UI defaults to settings.yaml`. Please only have one of those two starting up AllTalk.

    2) You are not starting text-generation-webui with its normal Python environment. Please start it with start_{your OS version} as detailed [here](https://github.com/oobabooga/text-generation-webui#how-to-install) (`start_windows.bat`,`./start_linux.sh`, `start_macos.sh` or `start_wsl.bat`) OR (`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat` and then `python server.py`).
    
    3) You have installed the wrong version of DeepSpeed on your system, for the wrong version of Python/Text-generation-webui. You can go to your text-generation-webui folder in a terminal/command prompt and run the correct cmd version for your OS e.g. (`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`) and then you can type `pip uninstall deepspeed` then try loading it again. If that works, please see here for the correct instructions for installing DeepSpeed [here](https://github.com/erew123/alltalk_tts#-deepspeed-installation-options). 

    4) You have an old version of text-generation-webui (pre Dec 2023) I have not tested on older versions of text-generation-webui, so cannot confirm viability on older versions. For instructions on updating the text-generation-webui, please look [here](https://github.com/oobabooga/text-generation-webui#how-to-install) (`update_linux.sh`, `update_windows.bat`, `update_macos.sh`, or `update_wsl.bat`).

    5) You already have something running on port 7851 on your computer, so the mini-webserver cant start on that port. You can change this port number by editing the `confignew.json` file and changing `"port_number": "7851"` to `"port_number": "7602"` or any port number you wish that isn’t reserved. Only change the number and save the file, do not change the formatting of the document. This will at least discount that you have something else clashing on the same port number.

    6) You have antivirus/firewalling that is blocking that port from being accessed. If you had to do something to allow text-generation-webui through your antivirus/firewall, you will have to do that for this too.

    7) You have quite old graphics drivers and may need to update them.

    8) Something within text-generation-webui is not playing nicely for some reason. You can go to your text-generation-webui folder in a terminal/command prompt and run the correct cmd version for your OS e.g. (`cmd_windows.bat`, `./cmd_linux.sh`, `cmd_macos.sh` or `cmd_wsl.bat`) and then you can type `python extensions\alltalk_tts\script.py` and see if AllTalk starts up correctly. If it does then something else is interfering. 

    9) Something else is already loaded into your VRAM or there is a crashed python process. Either check your task manager for erroneous Python processes or restart your machine and try again.

    10) You are running DeepSpeed on a Linux machine and although you are starting with `./start_linux.sh` AllTalk is failing there on starting. This is because text-generation-webui will overwrite some environment variables when it loads its python environment. To see if this is the problem, from a terminal go into your text-generation-webui folder and `./cmd_linux.sh` then set your environment variable again e.g. `export CUDA_HOME=/usr/local/cuda` (this may vary depending on your OS, but this is the standard one for Linux, and assuming you have installed the CUDA toolkit), then `python server.py` and see if it starts up. If you want to edit the environment permanently you can do so, I have not managed to write full instructions yet, but here is the conda guide [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#set-env-vars).

    11) You have built yourself a custom Python environment and something is funky with it. This is very hard to diagnose as its not a standard environment. You may want to updating text-generation-webui and re installing its requirements file (whichever one you use that comes down with text-generation-webui).
    </details>

    <details>
        <summary>🟨 I have multiple GPU's and I have problems running Finetuning</summary><br>
        
    Finetuning pulls in various other scripts and some of those scripts can have issues with multiple Nvidia GPU's being present. Until the people that created those other scripts fix up their code, there is a workaround to temporarily tell your system to only use the 1x of your Nvidia GPU's. To do this:

    - **Windows** - You will start the script with `set CUDA_VISIBLE_DEVICES=0 && python finetune.py`<br>
    After you have completed training, you can reset back with `set CUDA_VISIBLE_DEVICES=`<br>
    
    - **Linux** - You will start the script with `CUDA_VISIBLE_DEVICES=0 python finetune.py`<br>
    After you have completed training, you can reset back with `unset CUDA_VISIBLE_DEVICES`<br>

    Rebooting your system will also unset this. The setting is only applied temporarily.

    Depending on which of your Nvidia GPU's is the more powerful one, you can change the `0` to `1` or whichever of your GPU's is the most powerful.

    </details>

    <details>
        <summary>🟨 Firefox - Streaming Audio doesnt work on Firefox</summary><br>
        
    This is a long standing issue with Mozilla & Firefox and one I am unable to resolve as Mozilla have not resolved the issue with Firefox. The solution is to use another web browser if you want to use Streaming audio. For details of my prior invesitigation please look at this [ticket](https://github.com/erew123/alltalk_tts/issues/143)
    </details>

    ### Application Specific Issues

    <details>
        <summary>🟨 SillyTavern - I changed my IP address and now SillyTavern wont connect with AllTalk</summary><br>
    SillyTavern checks the IP address when loading extensions, saving the IP to its configuration only if the check succeeds. For whatever reason, SillyTavern's checks dont always allow changing its IP address a second time.<br><br>

    To manually change the IP address:

    1) Navigate to the SillyTavern Public folder located at `/sillytavern/public/`.
    2) Open the `settings.json` file.
    3) Look for the AllTalk section and find the `provider_endpoint` entry.
    3) Replace `localhost` with your desired IP address, for example, `192.168.1.64`.

    ![image](https://github.com/SillyTavern/SillyTavern/assets/35898566/144e4ac4-87dc-4a2b-8a73-39314abed1ca)
    </details>

    ### TTS Generation Issues & Questions

    <details>
        <summary>🟨 XTTS - Does the XTTS AI Model Support Emotion Control or Singing?</summary><br>
        
    No, the XTTS AI model does not currently support direct control over emotions or singing capabilities. While XTTS infuses generated speech with a degree of emotional intonation based on the context of the text, users cannot explicitly control this aspect. It's worth noting that regenerating the same line of TTS may yield slightly different emotional inflections, but there is no way to directly control it with XTTS.
    </details>
    <details>
        <summary>🟨 XTTS - Skips, repeats or pronunciation Issues</summary><br>
        
    Firstly, it's important to clarify that the development and maintenance of the XTTS AI models and core scripts are handled by [Coqui](https://docs.coqui.ai/en/latest/index.html), with additional scripts and libraries from entities like [huggingface](https://huggingface.co/docs/transformers/en/index) among many other Python scripts and libraries used by AllTalk. 

    AllTalk is designed to be a straightforward interface that simplifies setup and interaction with AI TTS models like XTTS. Currently, AllTalk supports the XTTS model, with plans to include more models in the future. Please understand that the deep inner workings of XTTS, including reasons why it may skip, repeat, or mispronounce, along with 3rd party scripts and libraries utilized, are ultimately outside my control.

    Although I ensure the text processed through AllTalk is accurately relayed to the XTTS model speech generation process, and I have aimed to mitigate as many issues as much as possible; skips, repeats and bad pronounciation can still occur.

    Certain aspects I have not been able to investigate due to my own time limitations, are:<br>

    - The impact of DeepSpeed on TTS quality. Is this more likely to cause skips or repetition?
    - Comparative performance between different XTTS model versions (e.g., 2.0.3 vs. 2.0.2) regarding audio quality and consistency.

    **From my experience and anecdotally gained knowledge:**<br>

    - Lower quality voice samples tend to produce more anomalies in generated speech.
    - Model finetuning with high-quality voice samples significantly reduces such issues, enhancing overall speech quality.
    - Unused/Excessive punctuation causes issues e.g. asterisks `*`, hashes `#`, brackets `(` `)` etc. Many of these AllTalk will filter out.

    So for example, the `female_01.wav` file that is provided with AllTalk is a studio quality voice sample, which the XTTS model was trained on. Typically you will find it unlikely that anomolies occur with TTS generation when using this voice sample. Hence good quality samples and finetuning, generally improve results with XTTS.

    If you wish to try out the XTTS version 2.0.3 model and see if it works better, you can download it from [here](https://huggingface.co/coqui/XTTS-v2/tree/v2.0.3), replacing all the files within your `/alltalk_tts/models/xttsv2_2.0.2` folder. This is on my list to both test version 2.0.3 more, but also build a more flexible TTS models downloader, that will not only accomdating other XTTS models, but also other TTS engines. If you try the XTTS version 2.0.3 model and gleen any insights, please let me know.
    </details>
    """

    finetuning_a_model = """
    ### ⚫ Finetuning a model
    If you have a voice that the model doesnt quite reproduce correctly, or indeed you just want to improve the reproduced voice, then finetuning is a way to train your "XTTSv2 local" model **(stored in `/alltalk_tts/models/xxxxx/`)** on a specific voice. For this you will need:

    - An Nvidia graphics card. (Please see the help section [note](https://github.com/erew123/alltalk_tts/edit/main/README.md#performance-and-compatibility-issues) if you have multiple Nvidia GPU's).
    - 18GB of disk space free (most of this is used temporarily)
    - At least 2 minutes of good quality speech from your chosen speaker in mp3, wav or flacc format, in one or more files (have tested as far as 20 minutes worth of audio).
    - As a side note, many people seem to think that the Whisper v2 model (used on Step 1) is giving better results at generating training datasets, so you may prefer to try that, as opposed to the Whisper 3 model.

    #### ⚫ How will this work/How complicated is it?
    Everything has been done to make this as simple as possible. At its simplest, you can literally just download a large chunk of audio from an interview, and tell the finetuning to strip through it, find spoken parts and build your dataset. You can literally click 4 buttons, then copy a few files and you are done. At it's more complicated end you will clean up the audio a little beforehand, but its still only 4x buttons and copying a few files.

    #### ⚫ The audio you will use
    I would suggest that if its in an interview format, you cut out the interviewer speaking in audacity or your chosen audio editing package. You dont have to worry about being perfect with your cuts, the finetuning Step 1 will go and find spoken audio and cut it out for you. Is there is music over the spoken parts, for best quality you would cut out those parts, though its not 100% necessary. As always, try to avoid bad quality audio with noises in it (humming sounds, hiss etc). You can try something like [Audioenhancer](https://audioenhancer.ai/) to try clean up noisier audio. There is no need to down-sample any of the audio, all of that is handled for you. Just give the finetuning some good quality audio to work with. 

    #### ⚫ Can I Finetune a model more than once on more than one voice
    Yes you can. You would do these as multiple finetuning's, but its absolutely possible and fine to do. Finetuning the XTTS model does not restrict it to only being able to reproduce that 1x voice you trained it on. Finetuning is generally nudging the model in a direction to learn the ability to sound a bit more like a voice its not heard before. 

    #### ⚫ A note about anonymous training Telemetry information & disabling it
    Portions of Coqui's TTS trainer scripts gather anonymous training information which you can disable. Their statement on this is listed [here](https://github.com/coqui-ai/Trainer?tab=readme-ov-file#anonymized-telemetry). If you start AllTalk Finetuning with `start_finetuning.bat` or `./start_finetuning.sh` telemetry will be disabled. If you manually want to disable it, please expand the below:

    <details>
        <summary>Manually disable telemetry</summary><br>
        
    Before starting finetuning, run the following in your terminal/command prompt:

    - On Windows by typing `set TRAINER_TELEMETRY=0`
    - On Linux & Mac by typing `export TRAINER_TELEMETRY=0`

    Before you start `finetune.py`. You will now be able to finetune offline and no anonymous training data will be sent.
    </details>
    """

    finetuning_setup = """
    #### ⚫ Prerequisites for Fine-tuning with Nvidia CUDA Development Toolkit 11.8

    All the requirements for Finetuning will be installed by using the atsetup utility and installing your correct requirements (Standalone or for Text-generation-webui). The legacy manual instructions are stored below, however these shouldnt be required.

    <details>
        <summary>Legacy manual instructions for installing Nvidia CUDA Development Toolkit 11.8</summary><br>
    - To perform fine-tuning, a specific portion of the **Nvidia CUDA Development Toolkit v11.8** must be installed. This is crucial for step 1 of fine-tuning. The objective is to minimize the installation footprint by installing only the essential components.
    - The **Nvidia CUDA Development Toolkit v11.8** operates independently from your graphics card drivers and the CUDA version utilized by your Python environment.
    - This installation process aims to keep the download and install size as minimal as possible, however a full install of the tookit requires 3GB's of disk space.
    - When running Finetuning it will require upto 20GB's of temporary disk space, so please ensure you have this space available and preferably use a SSD or NVME drive.

    1. **Download the Toolkit**:
    - Obtain the **network install** version of the Nvidia CUDA Development Toolkit 11.8 from [Nvidia's Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).

    2. **Run the Installer**:
    - Choose **Custom (Advanced)** installation.
    - Deselect all options initially.
    - Select the following components:
        - `CUDA` > `Development` > `Compiler` > `nvcc`
        - `CUDA` > `Development` > `Libraries` > `CUBLAS` (**both** development and runtime)

    3. **Configure Environment Search Path**:
    - It's essential that `nvcc` and CUDA 11.8 library files are discoverable in your environment's search path. Adjustments can be reverted post-fine-tuning if desired.

        **For Windows**:
        - Edit the `Path` environment variable to include `C:&#92;Program Files&#92;NVIDIA GPU Computing Toolkit&#92;CUDA&#92;v11.8&#92;bin`.
        - Add `CUDA_HOME` and set its path to `C:&#92;Program Files&#92;NVIDIA GPU Computing Toolkit&#92;CUDA&#92;v11.8.`

        **For Linux**:
        - The path may vary by Linux distribution. Here's a generic setup:
        - `export CUDA_HOME=/usr/local/cuda`
        - `export PATH=${CUDA_HOME}/bin:${PATH}`
        - `export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH`
        
        - Consider adding these to your `~/.bashrc` for permanence, or apply temporarily for the current session by running the above commands each time you start your Python environment.

        **Note**: If using Text-generation-webui, its best to set these temporarily.

    4. **Verify Installation**:
    - Open a **new** terminal/command prompt to refresh the search paths.
    - In a terminal or command prompt, execute `nvcc --version`.
    - Success is indicated by a response of `Cuda compilation tools, release 11.8.` Specifically, ensure it is version 11.8.

    5. **Troubleshooting**:
    - If the correct version isn't reported, recheck your environment path settings for accuracy and potential conflicts with other CUDA versions.
    </details>

    #### Additional Note on Torch and Torchaudio:
    - Ensure Torch and Torchaudio are CUDA-enabled (any version), which is separate from the CUDA Toolkit installation. CUDA 11.8 corresponds to `cu118` and CUDA 12.1 to `cu121` in AllTalk diagnostics.
    - Failure to install CUDA for Torch and Torchaudio will result in Step 2 of fine-tuning failing. These requirements are distinct from the CUDA Toolkit installation, so avoid conflating the two.<br>


    #### ⚫ Starting Fine-tuning

    **NOTE:** Ensure AllTalk has been launched at least once after any updates to download necessary files for fine-tuning.

    1. **Close Resource-Intensive Applications**:
    - Terminate any applications that are using your GPU/VRAM to ensure enough resources for fine-tuning.

    2. **Organize Voice Samples**:
    - Place your audio samples into the following directory:
        `/alltalk_tts/finetune/put-voice-samples-in-here/`

    Depending on your setup (Text-generation-webui or Standalone AllTalk), the steps to start the Python environment vary:

    ### **For Standalone AllTalk Users**:
    Navigate to the `alltalk_tts` folder:
    > `cd alltalk_tts`
    
    Start the Python environment:
    > **Windows**: `start_finetune.bat`<br>
    **Linux**: `./start_finetune.sh`
        
    ### **For Text-generation-webui Users**:
    Navigate to the Text-generation-webui directory:
    >`cd text-generation-webui`
    
    Start the Python environment suitable for your OS:<br>
    **Windows**: `cmd_windows.bat`<br>
    **Linux**: `./cmd_linux.sh`<br>
    **macOS**: `cmd_macos.sh`<br>
    **WSL (Windows Subsystem for Linux)**: `cmd_wsl.bat`
    
    Move into the AllTalk directory:
    >`cd extensions/alltalk_tts`
    
    **Linux** users only need to run this command:
    `
    export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
    `
    Start the fine-tuning process with the command:
    > `python finetune.py`<br><br>

    > If you're unfamiliar with Python environments and wish to learn more, consider reviewing **Understanding Python Environments Simplified** in the Help section.

    ### Both Standalone & Text-generation-webui users
    3. **Pre-Flight Checklist**:
    > Go through the pre-flight checklist to ensure readiness. Address any issues flagged as "Fail".

    4. **Post Fine-tuning Actions**:
    > Upon completing fine-tuning, the final tab will guide you on managing your files and relocating your newly trained model to the appropriate directory.
    """
    
    finetuning_details = """
    #### ⚫ How many Epochs etc is the right amount?

    In finetuning the suggested/recommended amount of epochs, batch size, evaluation percent etc is already set. However, there is no absolutely correct answer to what the settings should be, it all depends on what you are doing. 

    - If you just want to train a normal human voice that is in an existing language, for most people’s needs, the base settings would work fine. You may choose to increase the epochs up to maybe 20, or run a second round of training if needed.
    - If you were training an entirely new language, you would need a huge amount of training data and it requires around 1000 epochs (based on things I can find around the internet of people who tried this).
    - If you are training a cartoon style voice in an existing language, it may need well upwards of 40 epochs until it can reproduce that voice with some success.

    There are no absolute correct settings, as there are too many variables, ranging from the amount of samples you are using (5 minutes worth? 4 hours worth? etc), if they are similar samples to what the AI model already understands, so on and so forth. Coqui whom originally trained the model usually say something along the lines of, once you’ve trained it X amount, if it sounds good then you are done and if it doesn’t, train it more.

    #### ⚫ Evaluation Data Percentage
    In the process of finetuning, it's crucial to balance the data used for training the model against the data reserved for evaluating its performance. Typically, a portion of the dataset is set aside as an 'evaluation set' to assess the model's capabilities in dealing with unseen data. On Step 1 of finetuning you have the option to adjust this evaluation data percentage, offering more control over your model training process.<br><br>
    **Why Adjust the Evaluation Percentage?**<br><br>
    Adjusting the evaluation percentage **can** be beneficial in scenarios with limited voice samples. When dealing with a smaller dataset, allocating a slightly larger portion to training could enhance the model's ability to learn from these scarce samples. Conversely, with abundant data, a higher evaluation percentage might be more appropriate to rigorously test the model's performance. There are currently no absolutely optimal split percentages as it varies by dataset.
    - **Default Setting:** The default evaluation percentage is set at 15%, which is a balanced choice for most datasets.
    - **Adjustable Range:** Users can now adjust this percentage, but it’s generally recommend keeping it between 5% and 30%.
    - **Lower Bound:** A minimum of 5% ensures that there's enough data to evaluate model performance.
    - **Upper Bound:** Its suggested not exceeding 30% for evaluation to avoid limiting the amount of data available for training.

    - **Understanding the Impact:** Before adjusting this setting, it's important to understand its impact on model training and evaluation. Incorrect adjustments can lead to suboptimal model performance.
    - **Gradual Adjustments:** For those unfamiliar with the process, we recommend reading up on training data and training sets, then making small, incremental changes and observing their effects.
    - **Data Quality:** Regardless of the split, the quality of the audio data is paramount. Ensure that your datasets are built from good quality audio with enough data within them.

    #### ⚫ Using a Finetuned model in Text-generation-webui

    At the end of the finetune process, you will have an option to `Compact and move model to /trainedmodel/` this will compact the raw training file and move it to `/model/trainedmodel/`. When AllTalk starts up within Text-generation-webui, if it finds a model in this location a new loader will appear in the interface for `XTTSv2 FT` and you can use this to load your finetuned model. <br><br>**Be careful** not to train a new model from the base model, then overwrite your current `/model/trainedmodel/` **if** you want a seperately trained model. This is why there is an `OPTION B` to move your just trained model to `/models/lastfinetuned/`.

    #### ⚫ Training one model with multiple voices

    At the end of the finetune process, you will have an option to `Compact and move model to /trainedmodel/` this will compact the raw training file and move it to `/model/trainedmodel/`. This model will become available when you start up finetuning. You will have a choice to train the Base Model or the `Existing finetuned model` (which is the one in `/model/trainedmodel/`). So you can use this to keep further training this model with additional voices, then copying it back to `/model/trainedmodel/` at the end of training.

    #### ⚫ Do I need to keep the raw training data/model?

    If you've compacted and moved your model, its highly unlikely you would want to keep that data, however the choice is there to keep it if you wish. It will be between 5-10GB in size, so most people will want to delete it.

    #### ⚫ I have deeper questions about training the XTTS model, where can I find more information?

    If you have deeper questions about the XTTS model, its capabilites, the training process etc, anything thats not covered within the above text or the interface of `finetune.py`, please use the following links to research Coqui's documentation on the XTTS model. 

    - https://docs.coqui.ai/en/latest/models/xtts.html
    - https://github.com/coqui-ai/TTS
    - https://github.com/coqui-ai/TTS/discussions
    
    """

    tts_generator = """
    ### ⬜ AllTalk TTS Generator
    AllTalk TTS Generator is the solution for converting large volumes of text into speech using the voice of your choice. Whether you're creating audio content or just want to hear text read aloud, the TTS Generator is equipped to handle it all efficiently. Please see here for a quick [demo](https://www.youtube.com/watch?v=hunvXn0mLzc)<br><br>The link to open the TTS generator can be found on the built-in Settings and Documentation page.<br><br>**DeepSpeed** is **highly** recommended to speed up generation. **Low VRAM** would be best turned off and your LLM model unloaded from your GPU VRAM (unload your model). **No Playback** will reduce memory overhead on very large generations (15,000 words or more). Splitting **Export to Wav** into smaller groups will also reduce memory overhead at the point of exporting your wav files (so good for low memory systems). 

    #### ⬜ Estimated Throughput
    This will vary by system for a multitude of reasons, however, while generating a 58,000 word document to TTS, with DeepSpeed enabled, LowVram disabled, splitting size 2 and on an Nvidia RTX 4070, throughput was around 1,000 words per minute. Meaning, this took 1 hour to generate the TTS. Exporting to combined wavs took about 2-3 minutes total.

    #### ⬜ Quick Start
    > **Text Input:** Enter the text you wish to convert into speech in the 'Text Input' box.<br>
    > **Generate TTS:** Hit this to start the text-to-speech conversion.<br>
    > **Pause/Resume:** Used to pause and resume the playback of the initial generation of wavs or the stream.<br>
    > **Stop Playback:** This will stop the current audio playing back. It does not stop the text from being generated however.
    
    Once you have sent text off to be generated, either as a stream or wav file generation, the TTS server will remain busy until this process has competed. As such, think carefully as to how much you want to send to the server. <br>
    If you are generating wav files and populating the queue, you can generate one lot of text to speech, then input your next lot of text and it will continue adding to the list.
    
    #### ⬜ Customization and Preferences
    > **Character Voice:** Choose the voice that will read your text.<br>
    > **Language:** Select the language of your text.<br>
    > **Chunk Sizes:** Decide the size of text chunks for generation. Smaller sizes are recommended for better TTS quality.
    
    #### ⬜ Interface and Accessibility
    > **Dark/Light Mode:** Switch between themes for your visual comfort.<br>
    > **Word Count and Generation Queue:** Keep track of the word count and the generation progress.
    
    #### ⬜ TTS Generation Modes
    > **Wav Chunks:** Perfect for creating audio books, or anything you want to keep long term. Breaks down your text into manageable wav files and queues them up. Generation begins automatically, and playback will start after a few chunks have been prepared ahead. You can set the volume to 0 if you don’t want to hear playback. With Wav chunks, you can edit and/or regenerate portions of the TTS as needed.<br>
    
    > **Streaming:** For immediate playback without the ability to save. Ideal for on-the-fly speech generation and listening. This will not generate wav files and it will play back through your browser. You cannot stop the server generating the TTS once it has been sent.<br>
    
    With wav chunks you can either playback “In Browser” which is the web page you are on, or “On Server” which is through the console/terminal where AllTalk is running from, or "No Playback". Only generation “In Browser” can play back smoothly and populate the Generated TTS List. Setting the Volume will affect the volume level played back both “In Browser” and “On Server”.<br>
    
    For generating **large amounts of TTS**, it's recommended to select the **No Playback** option. This setting minimizes the memory usage in your web browser by avoiding the loading and playing of audio files directly within the browser, which is particularly beneficial for handling extensive audio generations. The definition of large will vary depending on your system RAM availability (will update when I have more information as to guidelines). Once the audio is generated, you can export your list to JSON (for safety) and use the **Play List** option to play back your audio.
    
    #### ⬜ Playback and List Management
    > **Playback Controls:** Utilize 'Play List' to start from the beginning or 'Stop Playback' to halt at any time.<br>
    > **Custom Start:** Jump into your list at a specific ID to hear a particular section.<br>
    > **Regeneration and Editing:** If a chunk isn't quite right, you can opt to regenerate it or edit the text directly. Click off the text to save changes and hit regenerate for the specific line.<br>
    > **Export/Import List:** Save your TTS list as a JSON file or import one. Note: Existing wav files are needed for playback. Exporting is handy if you want to take your files away into another program and have a list of which wav is which, or if you keep your audio files, but want to come back at a later date, edit one or two lines, regenerate the speech and re-combine the wav’s into one new long wav.<br>
    
    #### ⬜ Exporting Your Audio
    > **Export to WAV:** Combine all generated TTS from the list, into one single WAV file for easy download and distribution. Its always recommended to export your list to a JSON before exporting, so that you have a backup, should something go wrong. You can simply re-import the list and try exporting again.<br>
    
    When exporting, there is a file size limit of 1GB and as such you have the option to choose how many files to include in each block of audio exported. 600 is just on the limit of 1GB, depending on the average file size, so 500 or less is a good amount to work with. You can combine the generated files after if you wish, in Audacity or similar.<br>
    
    Additionally, lower export batches will lower the memory requirements, so if your system is low on memory (maybe 8 or 16GB system), you can use smaller export batches to keep the memory requirement down.
    
    #### ⬜ Exporting Subtitles (SRT file)
    > **Export SRT:** This will scan through all wav files in your list and generate a subtitles file that will match your exported wav file.
    
    #### ⬜ Analyzing generated TTS for errors
    > **Analyze TTS:** This will scan through all wav files comparing each ID's orignal text with the TTS generated for that ID and then flag up inconsistences. Its important to understand this is a **best effort** process and **not 100% perfect**, for example:<br><br>
    
    > Your text may have the word `their` and the automated routine that listens to your generated TTS interprets the word as `there`, aka a spelling difference.<br>
    > Your text may have `Examples are:` (note the colon) and the automated routine that listens to your generated TTS interprets the word as `Examples are` (note NO colon as you cannot sound out a colon in TTS), aka a punctuation difference.<br>
    > Your text may have `There are 100 items` and the automated routine that listens to your generated TTS interprets the word as `There are one hundred items`, aka numbers vs the number written out in words.<br>
    
    There will be other examples such as double quotes. As I say, please remember this is a **best effort** to help you identify issues.<br>

    As such, there is a `% Accuracy` setting. This uses a couple of methods to try find things that are similar e.g. taking the `their` and `there` example from above, it would identify that they both sound the same, so even if the text says `their` and the AI listening to the generated TTS interprets the word as `there`, it will realise that both sound the same/are similar so there is no need to flag that as an error. However, there are limits to this and some things may slip through or get picked up when you would prefer them not to be flagged.

    The higher the accuracy you choose, the more things it will flag up, however you may get more unwanted detections. The lower the less detections. Based on my few tests, accuracy settings between 96 to 98 seem to generally give the best results. However, I would highly recommend you test out a small 10-20 line text and test out the **Analyze TTS** button to get a feel for how it responds to different settings, as well as things it flags up.

    You will be able to see the ID's and Text (orignal and as interpreted) by looking at the terminal/command prompt window.

    The Analyze TTS feature uses the Whisper Larger-v2 AI engine, which will download on first use if necessary. This will require about 2.5GB's of disk space and could take a few minutes to download, depending on your internet connection.

    You can use this feature on systems that do not have an Nvidia GPU, however, unless you have a very powerful CPU, expect it to be slow.

    #### ⬜ Tricks to get the model to say things correctly
    Sometimes the AI model won’t say something the way that you want it to. It could be because it’s a new word, an acronym or just something it’s not good at for whatever reason. There are some tricks you can use to improve the chances of it saying something correctly.

    **Adding pauses**<br>
    You can use semi-colons ";" and colons ":" to create a pause, similar to a period "." which can be helpful with some splitting issues.

    **Acronyms**<br>
    Not all acronyms are going to be pronounced correctly. Let’s work with the word `ChatGPT`. We know it is pronounced `"Chat G P T"` but when presented to the model, it doesn’t know how to break it down correctly. So, there are a few ways we could get it to break out "Chat" and the G P and T. e.g.<br>

    > `Chat G P T.`<br>
    > `Chat G,P,T.`<br>
    > `Chat G.P.T.`<br>
    > `Chat G-P-T.`<br>
    > `Chat gee pee tea`<br>

    All bar the last one are using ways within the English language to split out "Chat" into one word being pronounced and then split the G, P and T into individual letters. The final example, which is to use Phonetics will sound perfectly fine, but clearly would look wrong as far as human readable text goes. The phonetics method is very useful in edge cases where pronunciation difficult.

    #### ⬜ Notes on Usage
    > For seamless TTS generation, it's advised to keep text chunks under 250 characters, which you can control with the Chunk sizes.<br>
    > Generated audio can be played back from the list, which also highlights the currently playing chunk.<br>
    > The TTS Generator remembers your settings, so you can pick up where you left off even after refreshing the page.<br>
    """
  
    with gr.Tab("Documentation and Help"):
        with gr.Tab("Narrator Function"): gr.Markdown(narrator_guides)
        with gr.Tab("RVC"): 
            gr.Markdown(rvc_guides)
            with gr.Row():
                with gr.Column(): gr.Markdown(rvc_guides2)
                with gr.Column(): gr.Markdown(rvc_guides3)
        with gr.Tab("Low VRAM"): gr.Markdown(low_vram)
        with gr.Tab("SillyTavern"): gr.Markdown(silly_tavern_support)
        with gr.Tab("Updating"): gr.Markdown(updating_alltalk)
        with gr.Tab("DeepSpeed"): gr.Markdown(deepspeed_guides)
        with gr.Tab("Help with Problems"): gr.Markdown(help_with_problems)
        with gr.Tab("XTTS Finetuning"): 
            with gr.Tab("XTTS Finetuning basics"): gr.Markdown(finetuning_a_model)
            with gr.Tab("XTTS Finetuning Setup"): gr.Markdown(finetuning_setup)
            with gr.Tab("XTTS Finetuning Detailed"): gr.Markdown(finetuning_details)
        with gr.Tab("TTS Generator"): gr.Markdown(tts_generator)