# AllTalk TTS v2
**Documentation/WIKI:** please refer to the built in documentation or the [Wiki here ](https://github.com/erew123/alltalk_tts/wiki)

**Known Errors List:** [Known Errors page is here](https://github.com/erew123/alltalk_tts/wiki/Error-Messages-List)

**Issues/Bugs/Support:** support tickets can be opened [here in the issues area](https://github.com/erew123/alltalk_tts/issues)

**Github discussions:** on AllTalk V2 are [here in the discussion board](https://github.com/erew123/alltalk_tts/discussions/245)

**Please note, my available time has become VERY limited due to unexpected family commitments [Please read here for details](https://github.com/erew123/alltalk_tts/issues/377). So please DYOR, look at the [Wiki here ](https://github.com/erew123/alltalk_tts/wiki) and refer to the TTS manufacturer (Links in the Gradio interface) for issues specific to THEIR TTS engines.**

I would also like to say a big thank you to anyone whom has contributed to this project either with PR's (list [here](https://github.com/erew123/alltalk_tts/graphs/contributors)) or in any discussion forums/issues tickets. Your input, guidance, help and thoughts are greatly appreciated.

#### [💖 Sponsor this Project on Ko-fi](https://ko-fi.com/erew123)

---

## AllTalk V2 Core Functionality
- Comprehensive setup utilities for Windows & Linux
- Multiple TTS engine support
    - Coqui XTTS TTS
    - Coqui VITS TTS
    - Piper TTS
    - Parler TTS
    - F5 TTS
    - Other TTS engines can be coded in
- Retrieval-based Voice Conversion (RVC) pipeline
- Easy integration of new TTS engines (some coding required)
- Customizable settings for each TTS engine
- In-app model downloads for each engine
- Gradio web interface for easy management
- Standalone application or integration with Text-generation-webui, SillyTavern, KoboldCPP, HomeAssistant
- Narrator function for using different voices for characters and narration
- Audio Transcoding to multiple formats (mp3, opus, etc.)
- About 50 Gradio interface themes
- Custom start-up settings
- Clear console output for troubleshooting
- Binds to all available IP addresses (0.0.0.0)
- Fully customizable Global API settings

#### Documentation and Support
- Built-in documentation with web interface
- GitHub Wiki documentation

#### Performance and Optimization (Depending on TTS engine used)
- DeepSpeed integration for 2-3x performance boost
- Low VRAM mode for systems with limited GPU memory 

#### Voice Customization and Enhancement
- Model Finetuning for improved voice reproduction
- XTTS Multiple audio sample TTS generation for better voice reproduction

#### Bulk Operations and Management
- Bulk TTS Generator/Editor for large-scale audio production

#### API and Integration
- Comprehensive API Suite
- OpenAI-compatible endpoint/API for broader software compatibility
- JSON call support for third-party applications

#### Experimental Features
- Multi Engine Manager (MEM) for running multiple TTS instances simultaneously & queuing requests between them.

#### Screenshots
Screenshots are available [here](https://github.com/erew123/alltalk_tts/discussions/237)

---

### 🟥 Platform-Specific Notes

#### Windows
- Requires Git, Microsoft C++ Build Tools, and Windows SDK.
- DeepSpeed works with NVIDIA GPUs.

#### Linux
- Requires specific packages based on your distribution (Debian-based or RPM-based).
- DeepSpeed works with NVIDIA GPUs.
- Limited experimental support for AMD GPUs (mainly for XTTS).

#### Mac (Theoretical)
- Installation process is untested and theoretical.
- No GPU acceleration for AllTalk TTS engines.
- Some TTS engines may have limited or no support on Mac.

For more details on Mac support limitations, please refer to the [Mac Support Disclaimer](https://github.com/erew123/alltalk_tts/wiki/Install-%E2%80%90-Manual-Installation-Guide).

#### GPU Support
GPU support is provided by the developer of the individual TTS engine. If their TTS engine support's X GPU, then I can support X GPU, if it doesnt support X GPU, then I cannot support X GPU. Most of the engines will run on CPU, but some may be very slow on CPU.

- NVIDIA GPUs: Full support on Windows and Linux.
- AMD GPUs: Limited experimental support on Linux (mainly for XTTS).
- Intel ARC GPUs: No specific support currently.
- Apple Silicon (M1/M2): No GPU acceleration for AllTalk TTS engines currently.

---

### 🟩 Quick Setup (Recommended for most users)

For a fast and straightforward installation and recommended:

- [Standalone Installation](https://github.com/erew123/alltalk_tts/wiki/Install-%E2%80%90-Standalone-Installation)
  - For users who want to run AllTalk TTS as a standalone application.
  - Available for Windows and Linux.

- [Text-generation-webui Installation](https://github.com/erew123/alltalk_tts/wiki/Install-%E2%80%90-Text%E2%80%90generation%E2%80%90webui-Installation)
  - For users who want to integrate AllTalk TTS as a part of [Text-generation-webui](https://github.com/oobabooga/text-generation-webui).
  - Available for Windows and Linux.
  - Includes information on the optional/alternative TGWUI Remote Extension.

These methods use scripts that automate most of the installation process, making it easier for users to get started quickly.

### 🟩 Manual Installation

For users who prefer more control over the installation process or need to troubleshoot:

- [Manual Installation Guide](https://github.com/erew123/alltalk_tts/wiki/Install-%E2%80%90-Manual-Installation-Guide)
  - Detailed step-by-step instructions for manual installation.
  - Available for Windows, Linux, and Mac (theoretical).
  - Recommended for advanced users or those with specific setup requirements.

### 🟩 Google Colab Installation

For users who want to run AllTalk TTS in a cloud environment:

- [Google Colab Installation](https://github.com/erew123/alltalk_tts/wiki/Google-COLAB)
  - Instructions for setting up and running AllTalk TTS in Google Colab.
  - Ideal for users who want to try AllTalk TTS without installing it locally.

---

### 🟪 Updating

As long as you did the `git clone` method to setup initially, you will be able to go into the folder and use `git pull` to download updates.

---

### 🟨 Diagnostics Help with Issues/Start-up problems etc.

If you are having issues with starting AllTalk, it may well be because some of the 3rd Party packages versions have changed, or something is not right in your Python environment. 

- **Known Errors List:** [Wiki page is here](https://github.com/erew123/alltalk_tts/wiki/Error-Messages-List)
- **Diagnostics tool Instructions:** [Wiki page is here](https://github.com/erew123/alltalk_tts/wiki/AllTalk-Diagnostics-Tool)

Whilst its impossible to constantly ensure that everything is going to work perfectly, after installation, you can use the diagnostics tool to:

1) Generate a `diagnostics.log` file which contains information about your Python environment setup and performs various checks to ensure everything is installed.
2) Identify possible issues by comparing **your** `diagnostics.log` file to the **AllTalk base** `basediagnostics.log` stored in the `alltalk_tts/system/config/` folder.
3) Provide some semi-automated repair of your Python environment.

---

### 🆘 Support Requests, Troubleshooting, BETA Discussions & Feature requests
**Documentation & Known Error codes**, please refer to the built in documentation or the [Wiki here ](https://github.com/erew123/alltalk_tts/wiki).

If you wish to code something yourself though, thats perfectly to do and youre welcome to discuss that with me if needed.

General discussions on the BETA should be [here in the discussion board](https://github.com/erew123/alltalk_tts/discussions/245)

If you have a specifc technical problem, please open an issue ticket [here in the issues area](https://github.com/erew123/alltalk_tts/issues).

**Please note, my available time has become VERY limited due to unexpected family commitments. So please DYOR, look at the [Wiki here ](https://github.com/erew123/alltalk_tts/wiki) and refer to the TTS manufacturer (Links in the Gradio interface) for issues specific to THEIR TTS engines.**

---
