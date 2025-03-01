# Docker
The Docker image currently works on Windows and Linux supporting NVIDIA GPUs.

## System requirements
- AMD compatible Windows or Linux system
- CUDA 12.6
- Nvidia Docker Container Toolkit
- At least 25GB of free disk space

## Quickstart
Use `docker-start.sh` to pull the latest docker image using the XTTS model from Docker Hub and 
visit `http://localhost:7851/`. See instructions below for passing more arguments to the start script.

## Installing Prerequisites
### Docker for Linux
#### Ubuntu Specific Setup for GPUs
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

### Docker for Windows (WSL2)
#### Windows Specific Setup for GPUs
> Make sure your Nvidia drivers are up to date: https://www.nvidia.com/download/index.aspx
1. Install WSL2 in PowerShell with `wsl --install` and restart
2. Open PowerShell, type and enter ```ubuntu```.  It should now load you into wsl2
3. Remove the original nvidia cache key: `sudo apt-key del 7fa2af80`
4. Download CUDA toolkit keyring: `wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb`
5. Install keyring: `sudo dpkg -i cuda-keyring_1.1-1_all.deb`
6. Update package list: `sudo apt-get update`
7. Install CUDA toolkit: `sudo apt-get -y install cuda-toolkit-12-6`
8. Install Docker Desktop using WSL2 as the backend
9. Restart
10. If you wish to monitor the terminal remotely via SSH, follow [this guide](https://www.hanselman.com/blog/how-to-ssh-into-wsl2-on-windows-10-from-an-external-machine).
11. Open PowerShell, type ```ubuntu```, [then follow below](#quickstart)

## Arguments for starting the Docker container
To make it as simple as possible to use Docker, the script `docker-start.sh` can be used. It provides the 
following optional arguments:

- `--config` lets you choose a config JSON file which can subset of `confignew.json`. This allows you to change only
  few values and leave the rest as defined in the default `confignew.json` file.
    - Example: `docker-start.sh --config /my/config/file.json` with content `{"branding": "My Brand "}` will just change
      the branding in `confignew.json`.
- `--voices` lets you add voices for the TTS engine in WAV format. You have to specify the folder containing all
  voice files.
    - Example: `docker-start.sh --voices /my/voices/dir`
- `--rvc-voices` similar to voices, this option lets you pick the folder containing the [RVC models](https://github.com/erew123/alltalk_tts/wiki/RVC-(Retrieval%E2%80%90based-Voice-Conversion)).
    - Example: `docker-start.sh --rvc_voices /my/rvc/voices/dir`
- `--no-ui` allows you to not expose port 7852 for the gradio interface and sets `launch_gradio` to `false`.
- `--with-multi-engine-manager` enables the use of the [multi engine manager (MEM)](https://github.com/erew123/alltalk_tts/wiki/Multi-Engine-Manager)
    which allows for more parallel requests. By default, one TTS engine is started. Optionally, you can pass the
    file path of a JSON file which can be a subset of `mem_config.json` with more fine-grained configuration options.
  - Example: `docker-start.sh --with-multi_engine_manager` to use MEM with default settings or 
    `docker-start.sh --with-multi_engine_manager /my/config/file.json` to pass a JSON file containing more settings.
- `--tag` allows to choose the docker tag of the image to run. Defaults to `latest-xtts`.
    - Example: `docker-start.sh --tag mytag`
- `--docker-repository` allows to choose another Docker repository for pulling the image from. Use an empty 
  string for the local repo.
- Since the above commands only address the most important options, you might pass additional arbitrary docker arguments
  to the `docker-start.sh`.

Of course, like any other Docker image, you can also directly use `docker run` directly and use `docker-start.sh` 
as an inspiration.


## Building your own images (advanced)
> Be aware that building the images requires good hardware and bandwidth. Expect this process to require >
> 40 GB of disk space and > 30min execution time depending on your hardware.

Under normal circumstances, there should be no need to build the Docker images locally. However, if needed for some
reason, you may want to use `docker-build.sh` with the following arguments:

- `--tts_model` allows to choose the TTS model that is used by default. Valid values are `piper`, `vits`, `xtts`. Defaults to `xtts`.
    - Example: `docker-build.sh --tts_model piper`
- `--tag` allows to choose the docker tag. Defaults to `latest-xtts`.
    - Example: `docker-build.sh --tag mytag`
- `--docker-repository` allows to choose another Docker repository for tagging the image from. Use an empty
  string for the local repo.
- `--local-deepspeed-build` should only be used in rare cases where DeepSpeed needs to be rebuilt. Defaults to `false`.
- `--clean` allows remove existing dependency build like conda environment or DeepSpeed. Defaults to `false`.
    - Example: `docker-build.sh --clean`


