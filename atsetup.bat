@echo off
cd /D "%~dp0"

set PATH=%PATH%;%SystemRoot%\system32

:: Generate the ESC character
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

:: Standard Colors
set "BLACK=%ESC%[30m"
set "RED=%ESC%[31m"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "BLUE=%ESC%[34m"
set "MAGENTA=%ESC%[35m"
set "CYAN=%ESC%[36m"
set "WHITE=%ESC%[37m"
set "L_BLACK=%ESC%[90m"
set "L_RED=%ESC%[91m"
set "L_GREEN=%ESC%[92m"
set "L_YELLOW=%ESC%[93m"
set "L_BLUE=%ESC%[94m"
set "L_MAGENTA=%ESC%[95m"
set "L_CYAN=%ESC%[96m"
set "L_WHITE=%ESC%[97m"
set "RESET=%ESC%[0m"

@rem Check if curl is available
curl --version >nul 2>&1
if "%ERRORLEVEL%" NEQ "0" (
    echo curl is not available on this system. Please install curl then re-run the script https://curl.se/ 
	echo or perform a manual installation of a Conda Python environment.
    goto end
)

:MainMenu
cls
echo.
echo    %L_BLUE%ALLTALK WINDOWS SETUP UTILITY%RESET%
echo.
echo    INSTALLATION TYPE
echo    1) I am using AllTalk as part of %L_GREEN%Text-generation-webui%RESET%
echo    2) I am using AllTalk as a %L_GREEN%Standalone Application%RESET%
echo.
echo    9) %L_RED%Exit/Quit%RESET%
echo.
set /p UserOption="    Enter your choice: "

if "%UserOption%"=="1" goto WebUIMenu
if "%UserOption%"=="2" goto StandaloneMenu
if "%UserOption%"=="9" goto End
goto MainMenu

:WebUIMenu
cls
echo.
echo    %L_BLUE%TEXT-GENERATION-WEBUI SETUP%RESET%
echo.
echo    Please ensure you have started your Text-generation-webui Python 
echo    environment. If you have NOT done this, please run %L_GREEN%cmd_windows.bat%RESET% 
echo    in the %L_GREEN%text-generation-webui%RESET% folder and then re-run this script.
echo.
echo    BASE REQUIREMENTS
echo    1) Install the requirements for a %L_GREEN%Nvidia machine%RESET%.
echo    2) Install the requirements for a %L_GREEN%AMD or MAC machine%RESET%.
echo.
echo    OPTIONAL
echo    3) Install the Finetuning requirements.
echo    4) Install DeepSpeed v11.2 for CUDA %L_GREEN%11.8%RESET% and Python-3.11.x.
echo    5) Install DeepSpeed v11.2 for CUDA %L_GREEN%12.1%RESET% and Python-3.11.x.
echo    6) Uninstall DeepSpeed.
echo.
echo.   OTHER
echo    7) Generate a diagnostics file.
echo.
echo    9) %L_RED%Exit/Quit%RESET%
echo.
set /p WebUIOption="Enter your choice: "
if "%WebUIOption%"=="1" goto InstallNvidiaTextGen
if "%WebUIOption%"=="2" goto InstallOtherTextGen
if "%WebUIOption%"=="3" goto InstallFinetuneTextGen
if "%WebUIOption%"=="4" goto InstallDeepSpeed118TextGen
if "%WebUIOption%"=="5" goto InstallDeepSpeed121TextGen
if "%WebUIOption%"=="6" goto UnInstallDeepSpeed
if "%WebUIOption%"=="7" goto GenerateDiagsTextGen
if "%WebUIOption%"=="9" goto End
goto WebUIMenu

:StandaloneMenu
cls
echo.
echo    %L_BLUE%ALLTALK STANDALONE APPLICATION SETUP%RESET%
echo.
echo    BASE REQUIREMENTS
echo    1) Install AllTalk as a Standalone Application
echo.
echo    OPTIONAL
echo    2) Delete AllTalk's custom Python environment
echo    3) Install the Finetuning requirements
echo.
echo.   OTHER
echo    4) Generate a diagnostics file
echo.
echo    9) %L_RED%Exit/Quit%RESET%
echo.
set /p StandaloneOption="Enter your choice: "
if "%StandaloneOption%"=="1" goto InstallCustomStandalone
if "%StandaloneOption%"=="2" goto DeleteCustomStandalone
if "%StandaloneOption%"=="3" goto InstallFinetuneStandalone
if "%StandaloneOption%"=="4" goto GenerateDiagsStandalone
if "%StandaloneOption%"=="9" goto End
goto StandaloneMenu

:InstallNvidiaTextGen
pip install -r requirements_nvidia.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo    There was an error installing the Nvidia requirements.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto WebUIMenu
)
Echo.
echo.
Echo    Nvidia machine requirements installed successfully.
Echo. 
pause
goto WebUIMenu

:InstallOtherTextGen
pip install -r requirements_other.txt
if %ERRORLEVEL% neq 0 (
    echo. 
    echo    There was an error installing the Other requirements.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto WebUIMenu
)
Echo.
echo.
Echo    Other machine requirements installed successfully.
Echo. 
pause
goto WebUIMenu

:InstallFinetuneTextGen
pip install -r requirements_finetune.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo    There was an error installing the Finetune requirements.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto WebUIMenu
)
Echo.
Echo    Finetune requirements installed successfully. Please ensure you have installed CUDA 11.8.
Echo    Finetune needs access to %L_GREEN%cublas64_11.dll%RESET% and will error if this is not setup correctly.
Echo    Please see here for details %L_GREEN%https://github.com/erew123/alltalk_tts#-finetuning-a-model%RESET%
Echo.
pause
goto WebUIMenu

:InstallDeepSpeed118TextGen
echo Downloading DeepSpeed...
curl -LO https://github.com/erew123/alltalk_tts/releases/download/deepspeed/deepspeed-0.11.2+cuda118-cp311-cp311-win_amd64.whl
if %ERRORLEVEL% neq 0 (
    echo.
    echo    Failed to download DeepSpeed wheel file.
    echo    Please check your internet connection or try again later.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto WebUIMenu
)
echo DeepSpeed wheel file downloaded successfully.
echo Installing DeepSpeed...
pip install deepspeed-0.11.2+cuda118-cp311-cp311-win_amd64.whl
if %ERRORLEVEL% neq 0 (
    echo. 
    echo    Failed to install DeepSpeed.
    echo    Please check if the wheel file is compatible with your system.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto WebUIMenu
)
Echo.
echo.
Echo    DeepSpeed installed successfully.
Echo. 
del deepspeed-0.11.2+cuda118-cp311-cp311-win_amd64.whl
pause
goto WebUIMenu

:InstallDeepSpeed121TextGen
echo Downloading DeepSpeed...
curl -LO https://github.com/erew123/alltalk_tts/releases/download/deepspeed/deepspeed-0.11.2+cuda121-cp311-cp311-win_amd64.whl
if %ERRORLEVEL% neq 0 (
    echo.
    echo    Failed to download DeepSpeed wheel file.
    echo    Please check your internet connection or try again later.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto WebUIMenu
)
echo    DeepSpeed wheel file downloaded successfully.
echo    Installing DeepSpeed...
pip install deepspeed-0.11.2+cuda121-cp311-cp311-win_amd64.whl
if %ERRORLEVEL% neq 0 (
    echo. 
    echo    Failed to install DeepSpeed.
    echo    Please check if the wheel file is compatible with your system.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto WebUIMenu
)
Echo.
echo.
Echo    DeepSpeed installed successfully.
Echo. 
del deepspeed-0.11.2+cuda121-cp311-cp311-win_amd64.whl
pause
goto WebUIMenu

:UnInstallDeepSpeed
pip uninstall deepspeed
if %ERRORLEVEL% neq 0 (
    echo. 
    echo    There was an error uninstalling DeepSpeed.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto WebUIMenu
)
echo.
Echo    DeepSpeed uninstalled successfully.
Echo. 
pause
goto WebUIMenu

:GenerateDiagsTextGen
Python diagnostics.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo    There was an error running diagnostics. Have you correctly started your 
    echo    Text-generation-webui Python environment with %L_GREEN%cmd_windows.bat%RESET%?
    echo.
    pause
    goto WebUIMenu
)
Echo.
echo.
Echo    Diagnostics.log generated. Please scroll up to look over the log.
Echo. 
pause
goto WebUIMenu

:InstallCustomStandalone
set PATH=%PATH%;%SystemRoot%\system32

@rem Check if curl is available
curl --version >nul 2>&1
if "%ERRORLEVEL%" NEQ "0" (
    echo curl is not available on this system. Please install curl then re-run the script https://curl.se/ 
	echo or perform a manual installation of a Conda Python environment.
    goto end
)
echo "%CD%"| findstr /C:" " >nul && echo This script relies on Miniconda which can not be silently installed under a path with spaces. Please correct your folder names and re-try installation. && goto exit

@rem Check for special characters in installation path
set "SPCHARMESSAGE=WARNING: Special characters were detected in the installation path! This can cause the installation to fail!"
echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~]" >nul && (
    call :PrintBigMessage %SPCHARMESSAGE%
)
set SPCHARMESSAGE=

@rem fix failed install when installing to a separate drive
set TMP=%cd%\alltalk_environment
set TEMP=%cd%\alltalk_environment

@rem deactivate existing conda envs as needed to avoid conflicts
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul

@rem config
set INSTALL_DIR=%cd%\alltalk_environment
set CONDA_ROOT_PREFIX=%cd%\alltalk_environment\conda
set INSTALL_ENV_DIR=%cd%\alltalk_environment\env
set MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe
set conda_exists=F

@rem figure out whether git and conda need to be installed
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" (
    set conda_exists=T
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
    goto RunScript
)

@rem download and install conda if not installed
echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL% to %INSTALL_DIR%\miniconda_installer.exe
mkdir "%INSTALL_DIR%"
call curl -Lk "%MINICONDA_DOWNLOAD_URL%" > "%INSTALL_DIR%\miniconda_installer.exe" || ( echo. && echo Miniconda failed to download. && goto end )
echo Installing Miniconda to %CONDA_ROOT_PREFIX%
start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%
echo Miniconda version:
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version || ( echo. && echo Miniconda not found. && goto end )

@rem create the installer env
echo Packages to install: %PACKAGES_TO_INSTALL%
call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.11 || ( echo. && echo Conda environment creation failed. && goto end )

@rem check if conda environment was actually created
if not exist "%INSTALL_ENV_DIR%\python.exe" ( echo. && echo Conda environment is empty. && goto end )

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

@rem Ask user for the type of requirements to install
echo.
echo    %L_BLUE%Choose the type of requirements to install%RESET%
echo.
echo    1. Nvidia graphics card machines
echo    2. Other machines (mac, amd, etc)
echo.
set /p UserChoice="Enter your choice (1 or 2): "

@rem Install requirements based on user choice
if "%UserChoice%" == "1" (
    echo.
    echo Installing Nvidia requirements...
    echo Downloading Pytorch with CUDA..
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu121
    pip install numpy==1.24.4
    pip install typing-extensions>=4.8.0
    pip install networkx<3.0.0,>=2.5.0
    pip install fsspec>=2023.6.0
    pip install soundfile==0.12.1
    pip install uvicorn==0.24.0.post1
    pip install TTS==0.21.3
    pip install fastapi==0.104.1
    pip install Jinja2==3.1.2
    pip install requests==2.31.0
    pip install tqdm==4.66.1
    pip install importlib-metadata==4.8.1
    pip install packaging==23.2
    pip install pydantic==1.10.13
    pip install sounddevice==0.4.6
    pip install python-multipart==0.0.6
    pip install cutlet>=0.3.0
    pip install unidic-lite>=1.0.8
    echo Downloading DeepSpeed...
    curl -LO https://github.com/erew123/alltalk_tts/releases/download/DeepSpeed-12.7/deepspeed-0.12.7+d058d4b-cp311-cp311-win_amd64.whl
    echo Installing DeepSpeed...
    pip install deepspeed-0.12.7+d058d4b-cp311-cp311-win_amd64.whl
    del deepspeed-0.12.7+d058d4b-cp311-cp311-win_amd64.whl
) else (
    echo Installing requirements for other machines...
    pip install -r requirements_other.txt
)

@rem Create start_environment.bat to run AllTalk environment
echo @echo off > start_environment.bat
echo cd /D "%~dp0" >> start_environment.bat
echo set CONDA_ROOT_PREFIX=%cd%\alltalk_environment\conda >> start_environment.bat
echo set INSTALL_ENV_DIR=%cd%\alltalk_environment\env >> start_environment.bat
echo call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" >> start_environment.bat
echo exit >> start_environment.bat
@rem Create start_alltalk.bat to run AllTalk
echo @echo off > start_alltalk.bat
echo cd /D "%~dp0" >> start_alltalk.bat
echo set CONDA_ROOT_PREFIX=%cd%\alltalk_environment\conda >> start_alltalk.bat
echo set INSTALL_ENV_DIR=%cd%\alltalk_environment\env >> start_alltalk.bat
echo call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" >> start_alltalk.bat
echo call python script.py >> start_alltalk.bat
echo exit >> start_alltalk.bat
Echo.
Echo    start_alltalk.bat has been created.
Echo    You can now start AllTalk with %L_YELLOW%start_alltalk.bat%RESET%
Echo.
pause
goto StandaloneMenu

:DeleteCustomStandalone
@rem Check if the alltalk_environment directory exists
if not exist "%cd%\alltalk_environment\" (
    echo.
    echo    %L_GREEN%alltalk_environment%RESET% directory does not exist. No need to delete.
    echo.
    pause
    goto StandaloneMenu
)
@rem Check if a Conda environment is active
if not defined CONDA_PREFIX goto NoCondaEnvDeleteCustomStandalone
@rem Deactivate Conda environment if it's active
Echo    Exiting the Conda Environment. Please run %L_GREEN%atsetup.bat%RESET% again and delete the environment.
conda deactivate
:NoCondaEnvDeleteCustomStandalone
echo Deleting "alltalk_environment". Please wait.
rd /s /q "alltalk_environment"
del start_alltalk.bat
del start_environment.bat
if %ERRORLEVEL% neq 0 (
    echo.
    echo    Failed to delete alltalk_environment folder.
    echo    Please make sure it is not in use and try again.
    echo.
    pause
    goto StandaloneMenu
)
echo.
Echo.
echo    Environment %L_GREEN%alltalk_environment%RESET% deleted. Please set up the environment again.
Echo.
pause
goto StandaloneMenu

:InstallFinetuneStandalone
cd /D "%~dp0"
set CONDA_ROOT_PREFIX=%cd%\alltalk_environment\conda
set INSTALL_ENV_DIR=%cd%\alltalk_environment\env
@rem Check if the Conda environment exists
if not exist "%INSTALL_ENV_DIR%\python.exe" (
    echo.
    echo    The Conda environment at "%INSTALL_ENV_DIR%" does not exist.
    echo    Please install the environment before proceeding.
    echo.
    pause
    goto StandaloneMenu
)
@rem Attempt to activate the Conda environment
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
if errorlevel 1 (
    echo. 
    echo    Failed to activate the Conda environment.
    echo    Please check your installation and try again.
    echo.
    pause
    goto StandaloneMenu
)
@rem Proceed with installing requirements
pip install -r requirements_finetune.txt
if %ERRORLEVEL% neq 0 (
    echo. 
    echo    There was an error installing the Finetune requirements.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto StandaloneMenu
)
Echo.
Echo    Finetune requirements installed successfully. Please ensure you have installed CUDA 11.8.
Echo    Finetune needs access to %L_GREEN%cublas64_11.dll%RESET% and will error if this is not setup correctly.
Echo    Please see here for details %L_GREEN%https://github.com/erew123/alltalk_tts#-finetuning-a-model%RESET%
Echo.
pause
goto StandaloneMenu

:GenerateDiagsStandalone
cd /D "%~dp0"
set CONDA_ROOT_PREFIX=%cd%\alltalk_environment\conda
set INSTALL_ENV_DIR=%cd%\alltalk_environment\env
@rem Check if the Conda environment exists
if not exist "%INSTALL_ENV_DIR%\python.exe" (
    echo.
    echo    The Conda environment at "%INSTALL_ENV_DIR%" does not exist.
    echo    Please install the environment before proceeding.
    echo. 
    pause
    goto StandaloneMenu
)
@rem Attempt to activate the Conda environment
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
if errorlevel 1 (
    echo. 
    echo    Failed to activate the Conda environment.
    echo    Please check your installation and try again.
    echo.
    pause
    goto StandaloneMenu
)
@rem Run diagnostics
Python diagnostics.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo    There was an error running diagnostics.
    echo    Press any key to return to the menu.
    echo.
    pause
    goto StandaloneMenu
)
Echo.
Echo.
Echo    Diagnostics.log generated. Please scroll up to look over the log.
Echo.
pause
goto StandaloneMenu

:End
echo Exiting AllTalk Setup Utility...
exit /b
