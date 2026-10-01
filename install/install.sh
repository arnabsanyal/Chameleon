#!/bin/bash

echo ""
echo "#############################################################"
echo "#                                                           #"
echo "#   ####  #   #   ##   #   #  ####  #     ####   ###  #  #  #"
echo "#  #      #   #  #  #  ## ##  #     #     #     #   # ## #  #"
echo "#  #      #####  ####  # # #  ###   #     ###   #   # # ##  #"
echo "#  #      #   #  #  #  #   #  #     #     #     #   # #  #  #"
echo "#   ####  #   #  #  #  #   #  ####  ####  ####   ###  #  #  #"
echo "#                                                           #"
echo "#############################################################"

# Author: Arnab Sanyal
# Email: sanyal@utexas.edu
# SWARM LAB
# University of Texas at Austin
# Spring 2026
# Paper: "Chameleon: Dynamic Format Adapter for Efficient Diffusion" (https://arxiv.org/abs/2609.33496)
echo ""
echo "Author: Arnab Sanyal (sanyal@utexas.edu)"
echo "SWARM Lab - University of Texas at Austin"
echo "Spring 2026"
echo "Paper: \"Chameleon: Dynamic Format Adapter for Efficient Diffusion\" (https://arxiv.org/abs/2609.33496)"
echo ""
echo "MIT License"
echo ""
echo "Copyright (c) 2026 Arnab Sanyal, SWARM Lab, University of Texas at Austin"
echo ""
echo "Permission is hereby granted, free of charge, to any person obtaining a copy"
echo "of this software and associated documentation files (the \"Software\"), to deal"
echo "in the Software without restriction, including without limitation the rights"
echo "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell"
echo "copies of the Software, and to permit persons to whom the Software is"
echo "furnished to do so, subject to the following conditions:"
echo ""
echo "The above copyright notice and this permission notice shall be included in all"
echo "copies or substantial portions of the Software."
echo ""
echo "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR"
echo "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,"
echo "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE"
echo "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER"
echo "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,"
echo "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE"
echo "SOFTWARE."
echo ""
# Prompt user to agree to the MIT License terms before proceeding
echo "============================================================="
echo "Do you agree to the terms of the MIT License? (yes/no)"
echo "============================================================="
read -r response

# Convert response to lowercase for case-insensitive comparison
response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

# Check if user agreed to the license
if [[ "$response" != "yes" && "$response" != "y" ]]; then
    echo "License not accepted. Exiting installation."
    exit 1
fi

echo "License accepted. Proceeding with installation..."
echo ""

# Check if the 'chameleon' conda environment already exists
# - 'conda info --envs' lists all conda environments
# - 'grep -q' searches for 'chameleon' quietly (no output, just exit code)
if conda info --envs | grep -q 'chameleon'; then
    # Environment found - inform user and skip creation
    echo "Conda environment 'chameleon' already exists."
else
    # Environment not found - create it with Python 3.12
    # - '--name chameleon' sets the environment name
    # - 'python=3.12' specifies Python version
    # - '-y' auto-accepts prompts
    echo "Creating conda environment 'chameleon' with Python 3.12..."
    conda create --name chameleon python=3.12 -y
fi

# Install CUDA toolkit and related NVIDIA libraries into the 'chameleon' conda environment
# - cuda-toolkit: NVIDIA CUDA compiler and development tools
# - cudnn: NVIDIA CUDA Deep Neural Network library for GPU-accelerated deep learning
# - nccl: NVIDIA Collective Communications Library for multi-GPU communication
# - cuda-nvtx-dev: NVIDIA Tools Extension library for performance profiling
# The -y flag automatically accepts all prompts during installation
conda install --name chameleon -c nvidia cuda-toolkit cudnn nccl cuda-nvtx-dev -y

# Install Python dependencies from the requirements file
# Uses conda environment named 'chameleon' to run pip install command
conda run --name chameleon python3 -m pip install -r ./requirements.txt

# Install PyTorch and associated optimization/quantization libraries into the 'chameleon' conda environment
# - torch: Core PyTorch deep learning framework
# - bitsandbytes: Library for 8-bit optimizers and quantization
# - transformer-engine[pytorch]: NVIDIA's library for optimizing transformer models with FP8 precision support
conda run --name chameleon python3 -m pip install torch bitsandbytes "transformer-engine[pytorch]"

# Install the PyTorch AO (Architecture Optimization) package from GitHub
# Uses conda to run pip installation within the 'chameleon' environment
# --no-build-isolation: Disables build isolation to use system/environment dependencies
# This allows the build process to access pre-installed packages in the conda environment
# Installs directly from the GitHub repository's main branch
conda run --name chameleon python -m pip install --no-build-isolation git+https://github.com/pytorch/ao.git

# Get the site-packages directory path from the 'chameleon' conda environment
# This directory contains all installed Python packages for that environment
# Then use sed to replace all instances of 'logger.warning' with 'print' in the torchao_quantizer.py file
# Finally, use sed to replace all instances of 'logger.debug' with 'print' in the same file
# This effectively converts logging calls to standard print statements for debugging purposes
SITE_PACKAGES=$(conda run --name chameleon python -c "import site; print(site.getsitepackages()[0])")
sed -i 's/logger.warning/print/g' "$SITE_PACKAGES/diffusers/quantizers/torchao/torchao_quantizer.py"
sed -i 's/logger.debug/print/g' "$SITE_PACKAGES/diffusers/quantizers/torchao/torchao_quantizer.py"

conda run --name chameleon python -m pip install tiktoken sentencepiece beautifulsoup4 ftfy
# MixDQ CUDA kernels — build from source for Python 3.12 compatibility.
# Pre-built wheels (mixdq-extension on PyPI) only support Python 3.8–3.10.
# Build process follows: https://github.com/A-suozhang/MixDQ/blob/master/README.md
rm -rf /tmp/MixDQ
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/A-suozhang/MixDQ.git /tmp/MixDQ
# Make the file writable
chmod +w /tmp/MixDQ/kernels/third_party/nvidia-cutlass/include/cutlass/cuda_host_adapter.hpp
# Inject dummy typedefs to satisfy the compiler
python3 -c '
path = "/tmp/MixDQ/kernels/third_party/nvidia-cutlass/include/cutlass/cuda_host_adapter.hpp"
with open(path, "r") as f: code = f.read()
code = "#include <cuda.h>\ntypedef CUresult (*PFN_cuTensorMapEncodeTiled)(...);\ntypedef CUresult (*PFN_cuTensorMapEncodeIm2col)(...);\n" + code
with open(path, "w") as f: f.write(code)
'
# Restrict to Ampere, Ada, and Hopper to speed up build and avoid SM100+ bleeding-edge bugs
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# Regular (non-editable) install: the built package is copied into the env, so /tmp can be cleared afterwards
conda run --name chameleon python -m pip install --no-build-isolation /tmp/MixDQ/kernels
rm -rf /tmp/MixDQ
