@echo off 
cd /D "C:\AI\text-generation-webui\extensions\alltalk_tts\" 
set CONDA_ROOT_PREFIX=C:\AI\text-generation-webui\extensions\alltalk_tts\alltalk_environment\conda 
set INSTALL_ENV_DIR=C:\AI\text-generation-webui\extensions\alltalk_tts\alltalk_environment\env 
call "C:\AI\text-generation-webui\extensions\alltalk_tts\alltalk_environment\conda\condabin\conda.bat" activate "C:\AI\text-generation-webui\extensions\alltalk_tts\alltalk_environment\env" 
call python diagnostics.py 
