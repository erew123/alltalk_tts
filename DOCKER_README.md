# Docker
The Docker image currently works on Windows and Linux, optionally supporting NVIDIA GPUs.

## General Remarks
- The resulting Docker image is 22 GB in size. Building might require even more disk space temporarily.
- Build time depends on your hardware and internet connection. Expect at least 10min to be normal.
- The Docker build:
  - Downloads XTTS as default TTS engine
  - Enables RVC by default
  - Downloads all supported RVC models
  - Enables deepspeed by default
- Starting the Docker image should only a few seconds due to all the steps that were already executed during build.

## Docker for Linux

### Ubuntu Specific Setup for GPUs
1. Make sure the latest nvidia drivers are installed: `sudo ubuntu-drivers install`
1. Install Docker your preferred way. One way to do it is to follow the official documentation [here](https://docs.docker.com/engine/install/ubuntu/#uninstall-old-versions).
    - Start by uninstalling the old versions
    - Follow the "apt" repository installation method
    - Check that everything is working with the "hello-world" container
1. If, when launching the docker contain, you have an error message saying that the GPU cannot be used, you might have to install [Nvidia Docker Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
    - Install with the "apt" method
    - Run the docker configuration command
      ```sudo nvidia-ctk runtime configure --runtime=docker```
    - Restart docker

## Docker for Windows (WSL2)
### Windows Specific Setup for GPUs
> Make sure your Nvidia drivers are up to date: https://www.nvidia.com/download/index.aspx
1. Install WSL2 in PowerShell with `wsl --install` and restart
2. Open PowerShell, type and enter ```ubuntu```.  It should now load you into wsl2
3. Remove the original nvidia cache key: `sudo apt-key del 7fa2af80`
4. Download CUDA toolkit keyring: `wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb`
5. Install keyring: `sudo dpkg -i cuda-keyring_1.1-1_all.deb`
6. Update package list: `sudo apt-get update`
7. Install CUDA toolkit: `sudo apt-get -y install cuda-toolkit-12-4`
8. Install Docker Desktop using WSL2 as the backend
9. Restart
10. If you wish to monitor the terminal remotely via SSH, follow [this guide](https://www.hanselman.com/blog/how-to-ssh-into-wsl2-on-windows-10-from-an-external-machine).
11. Open PowerShell, type ```ubuntu```, [then follow below](#building-and-running-in-docker)

## Building and Running in Docker

1. Open a terminal (or Ubuntu WSL) and go where you cloned the repo
3. Build the image with `./docker-build.sh`
4. Start the container with `./docker-start.sh`
5. Visit `http://localhost:7851/` or remotely with `http://<ip>:7851`

## Arguments for building and starting docker
There are various arguments to customize the build and start of the docker image.

### Arguments for `docker-build.sh`
- `--tts_model` allows to choose the TTS model that is used by default. Valid values are `piper`, `vits`, `xtts`. Defaults to `xtts`.
  - Example: `docker-build.sh --tts_model piper`
- `--tag` allows to choose the docker tag. Defaults to `latest`.
  - Example: `docker-build.sh --tag mytag`

### Arguments for `docker-start.sh`
- `--config` lets you choose a config JSON file which can subset of `confignew.json`. This allows you to change only 
  few values and leave the rest as defined in the default `confignew.json` file.
  - Example: `docker-build.sh --config /my/config/file.json` with content `{"branding": "My Brand "}` will just change
    the branding in `confignew.json`.
- `--voices` lets you add voices for the TTS engine in WAV format. You have to specify the folder containing all
  voice files.
  - Example: `docker-build.sh --voices /my/voices/dir`
- `--rvc_voices` similar to voices, this option lets you pick the folder containing the RVC models.
  - Example: `docker-build.sh --rvc_vices /my/rvc/voices/dir`
- `--no_ui` allows you to not expose port 7852 for the gradio interface. Note that you still have to set `launch_gradio`
  to `false` via JSON file passed to `--config`. 
- Since the above commands only address the most important options, you might pass additional arbitrary docker commands
    to the `docker-start.sh`.
