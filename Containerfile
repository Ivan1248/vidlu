FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04
#FROM cupy/cupy

# ENV PATH=/root/.local/bin:$PATH

# Green prompt in .bashrc
RUN echo 'PS1="\[\e[32m\]\u@\h:\w\$\[\e[0m\] "' >> ~/.bashrc

# Install programs and libraries
RUN sed -i 's/htt[p|ps]:\/\/archive.ubuntu.com\/ubuntu\//mirror:\/\/mirrors.ubuntu.com\/mirrors.txt/g' /etc/apt/sources.list
#RUN DEBIAN_FRONTEND=noninteractive apt-get update
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common
#RUN add-apt-repository ppa:apt-fast/stable && \
#    apt-get update && \
#    DEBIAN_FRONTEND=noninteractive apt-get install -y apt-fast

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install libjpeg-dev zlib1g-dev \
      x11-apps  \
      ffmpeg libsm6 libxext6 \
      python3 python3-pip git curl vim wget \
      libxcb-cursor-dev \
      gedit meld bat && \
    apt-get clean

# Install Chromium from Debian repos (Ubuntu 24.04 only has snap version)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:xtradeb/apps -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y chromium && \
    apt-get clean && \
    ln -s /usr/bin/python3 /usr/bin/python

# Create wrapper for chromium to run with --no-sandbox (required in Docker).
# Use printf rather than echo: the default /bin/sh used by RUN does not
# interpret \n in echo, which would write a literal "\n" and break the shebang.
RUN mv /usr/bin/chromium /usr/bin/chromium-real && \
    printf '#!/bin/bash\nexec /usr/bin/chromium-real --no-sandbox "$@"\n' > /usr/bin/chromium && \
    chmod +x /usr/bin/chromium


# Install Python requirements
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt .
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir -r requirements.txt
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir pudb

RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir bitsandbytes accelerate peft trl triton \
    --extra-index-url https://download.pytorch.org/whl/cu128
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir --no-deps cut_cross_entropy
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir sentencepiece protobuf datasets huggingface_hub hf_transfer
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir plotly
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir streamlit
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir regex

RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir psutil qwen-vl-utils
# Re-install debian-managed packages under pip ownership so vllm can upgrade them
RUN pip install --no-cache-dir --ignore-installed PyJWT
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir vllm --extra-index-url https://download.pytorch.org/whl/cu128
RUN MAX_JOBS=8 DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir flash-attn
# Install transformers after vllm so it gets the latest version (vllm would otherwise pin it)
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir transformers rich
RUN python - <<'PY'
import re
import subprocess
import torch

nvcc_output = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release (\d+\.\d+)", nvcc_output)
if match is None:
    raise RuntimeError(f"Could not parse CUDA toolkit version from nvcc output:\n{nvcc_output}")

cuda_toolkit_version = match.group(1)
torch_cuda_version = torch.version.cuda

print(f"CUDA toolkit version from nvcc: {cuda_toolkit_version}")
print(f"CUDA version used to build torch: {torch_cuda_version}")

if torch_cuda_version is None:
    raise RuntimeError("Installed torch build has no CUDA support.")

if cuda_toolkit_version.split(".")[:2] != torch_cuda_version.split(".")[:2]:
    raise RuntimeError(
        "CUDA toolkit and torch CUDA versions do not match. "
        f"nvcc={cuda_toolkit_version}, torch={torch_cuda_version}"
    )
PY

# IRAP-Vietnam data preparation: Excel + parquet I/O
RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir pandas openpyxl xlrd pyarrow 

#ENV CUDA_HOME=/usr/local/cuda
#ENV PATH=${CUDA_HOME}/bin:${PATH}
#ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
#RUN DEBIAN_FRONTEND=noninteractive pip install --no-cache-dir flash_attn

# docker run -it --rm --user $(id -u) sameuser:sameuser command

#RUN git clone https://github.com/Ivan1248/shell-configs.git
# Copy the project files to the working directory
# COPY . .
#COPY ~/.bash* /root

#RUN echo 'alias python=python3' >> ~/.bashrc

#RUN --mount=type=secret,id=wandb,target=/root/.wandb/api.key wandb login `cat /root/.wandb/api.key`

# Expose a port if your app requires it (optional)
# EXPOSE 8080

#WORKDIR /app
#WORKDIR /app/scripts

#RUN --mount=type=secret,id=wandb,target=/root/.wandb/api.key wandb login `cat /root/.wandb/api.key`

# Command to run when the container is started with no specified command
CMD ["bash", "-i"]

## docker build -t vidlu_image .
## docker run -it -v ~/data:/data -v ~/projects:/projects vidlu_image /bin/bash
## docker run -v ~/data:/data -v ~/app/vidlu:/app/vidlu vidlu_image python run.py
## docker run -it -v ~/data:/app/data -v ~/projects/vidlu:/app/vidlu -w /app/vidlu/scripts -e <(env | grep -E 'CUDA_.*|VIDLU_.*') vidlu_image bash $command