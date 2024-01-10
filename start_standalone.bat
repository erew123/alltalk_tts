@echo off
cd /D "%~dp0"

@rem Check for a previously stored user choice
if exist "alltalk_settings.txt" (
    set /p UserWebUIChoice=<alltalk_settings.txt
    @rem Trim any potential whitespace from the user's choice
    set UserWebUIChoice=%UserWebUIChoice: =%
    echo Detected user choice as: %UserWebUIChoice%
    goto CheckChoice
) else (
    echo Is this installation part of an existing Text-generation-webui installation? [y/n]
	echo.
    echo If you are installing AllTalk as a Standalone Application say "n"
    echo.
    set /p UserWebUIChoice="Enter your choice (y for Yes, n for No): "
    echo %UserWebUIChoice%>alltalk_settings.txt
)

:CheckChoice
@rem Perform the action based on the environment variable
if /I "%UserWebUIChoice%"=="y" (
    echo Make sure you have started Text-generation-webui's Python environment using cmd_windows.bat.
    echo This step is required every time before starting AllTalk. If you have NOT done this
    echo you can press Ctrl+C now to exit, then go run cmd_windows.bat
	echo. 
	echo This script installed DeepSpeed for CUDA 12.1. If you have CUDA 11.8, Ctrl+C now and manually
	echo Install DeepSpeed and start AllTalk through Text-generation-webui or Python script.py
    pause > nul
    goto InstallRequirements
) else if /I "%UserWebUIChoice%"=="n" (
    echo Detected standalone installation.
    goto BeginInstall
)

echo Invalid choice detected or an error occurred.
goto end

:BeginInstall
set PATH=%PATH%;%SystemRoot%\system32

@rem Check if curl is available
curl --version >nul 2>&1
if "%ERRORLEVEL%" NEQ "0" (
    echo curl is not available on this system. Please install curl then re-run the script https://curl.se/ 
	echo or perform a manual installation of a Conda Python environment.
    goto end
)
echo "%CD%"| findstr /C:" " >nul && echo This script relies on Miniconda which can not be silently installed under a path with spaces. && goto end

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

:InstallRequirements
@rem Ask user for the type of requirements to install
echo Choose the type of requirements to install (this will take 10-20 minutes to install):
echo.
echo 1. Nvidia graphics card machines
echo 2. Other machines (mac, amd, etc)
echo.
set /p UserChoice="Enter your choice (1 or 2): "

@rem Install requirements based on user choice
if "%UserChoice%" == "1" (
    echo Installing Nvidia requirements...
    pip install -r requirements_nvidia.txt
	echo Downloading Pytorch with CUDA..
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo Downloading DeepSpeed...
    curl -LO https://github.com/erew123/alltalk_tts/releases/download/deepspeed/deepspeed-0.11.2+cuda121-cp311-cp311-win_amd64.whl
    echo Installing DeepSpeed...
    pip install deepspeed-0.11.2+cuda121-cp311-cp311-win_amd64.whl
    del deepspeed-0.11.2+cuda121-cp311-cp311-win_amd64.whl
) else (
    echo Installing requirements for other machines...
    pip install -r requirements_other.txt
)

:RunScript
@rem setup installer env
call python script.py %*

@rem below are functions for the script
goto end

:PrintBigMessage
echo. && echo.
echo *******************************************************************
for %%M in (%*) do echo * %%~M
echo *******************************************************************
echo. && echo.
exit /b

:end
pause
