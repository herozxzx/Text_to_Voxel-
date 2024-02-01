# Text_to_Voxel

This requires Kaolin(https://github.com/NVIDIAGameWorks/kaolin) pytorch packages.

Environments: https://github.com/herozxzx/Text_to_Voxel/blob/master/setup.txt

# Setting up Kaolin Environment on Ubuntu 20.04 with Docker, NVIDIA, and Kaolin

This guide helps you set up a Kaolin environment on Ubuntu 20.04 using Docker, NVIDIA tools, and Kaolin library for 3D deep learning.

## Prerequisites

- Docker installed with NVIDIA support
- NVIDIA driver and toolkit installed
- X server configured for GUI in Docker

## Docker Image

Use the provided Docker image for Kaolin environment:

```bash
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/ShareFiles:/mnt/ShareFiles -e DISPLAY=unix$DISPLAY --net=host --gpus all --cap-add=SYS_PTRACE --security-opt seccomp=unconfined nvidia/cudagl:10.1-devel /bin/bash
