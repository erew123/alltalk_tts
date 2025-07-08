@echo off 
cd /D "H:\alltalk_tts\" 
set CONDA_ROOT_PREFIX=H:\alltalk_tts\alltalk_environment\conda 
set INSTALL_ENV_DIR=H:\alltalk_tts\alltalk_environment\env 
call "H:\alltalk_tts\alltalk_environment\conda\condabin\conda.bat" activate "H:\alltalk_tts\alltalk_environment\env" 
call python script.py 
