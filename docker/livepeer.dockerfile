# ComfyStream BYOC (Bring Your Own Container) Dockerfile
# Based on the BYOC documentation for Livepeer integration

FROM nvcr.io/nvidia/pytorch:25.05-py3

ENV TensorRT_ROOT=/opt/TensorRT-10.9.0.34 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt update && apt install -yqq --no-install-recommends \
    git \
    wget \
    nano \
    socat \
    libsndfile1 \
    build-essential \
    \
    llvm \
    tk-dev \
    libglvnd-dev \
    cmake \
    swig \
    libprotobuf-dev \
    protobuf-compiler \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#enable opengl support with nvidia gpu
RUN printf '%s\n' \
  '{' \
  '    "file_format_version" : "1.0.0",' \
  '    "ICD" : {' \
  '        "library_path" : "libEGL_nvidia.so.0"' \
  '    }' \
  '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# TensorRT SDK
WORKDIR /opt
RUN wget --progress=dot:giga \
    https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz \
    && tar -xzf TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz \
    && rm TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz

# Link libraries and update linker cache
RUN echo "${TensorRT_ROOT}/lib" > /etc/ld.so.conf.d/tensorrt.conf \
    && ldconfig

# Install matching TensorRT Python bindings for CPython 3.11
RUN pip install --no-cache-dir \
    ${TensorRT_ROOT}/python/tensorrt-10.9.0.34-cp311-none-linux_x86_64.whl

# Create workspace directory
RUN mkdir -p /workspace/comfystream

# Clone ComfyUI
RUN git clone --branch v0.3.27 --depth 1 https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
#RUN git clone --branch master --single-branch --depth 1 https://github.com/hiddenswitch/ComfyUI.git /workspace/comfystream/external

# Set working directory
WORKDIR /workspace/comfystream

# Copy only files needed for setup
COPY ./src/comfystream/scripts /workspace/comfystream/src/comfystream/scripts
COPY ./configs /workspace/comfystream/configs

# Copy ComfyStream files into ComfyUI
COPY . /workspace/comfystream

# Copy comfystream and example workflows to ComfyUI
COPY ./workflows/comfyui/* /workspace/ComfyUI/user/default/workflows/
COPY ./test/example-512x512.png /workspace/ComfyUI/input

# Install ComfyUI requirements
RUN pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt

# Install ComfyStream requirements
RUN ln -s /workspace/comfystream /workspace/ComfyUI/custom_nodes/comfystream
RUN pip install --no-cache-dir -e .
RUN python install.py --workspace /workspace/ComfyUI

# Install additional dependencies
RUN pip install --no-cache-dir --upgrade tensorrt-cu12-bindings==10.9.0.34 tensorrt-cu12-libs==10.9.0.34

# Install additional dependencies for BYOC server
RUN pip install --no-cache-dir \
    aiohttp \
    aiohttp-cors \
    asyncio

# Setup opencv with CUDA support
# RUN bash /workspace/comfystream/docker/entrypoint.sh --opencv-cuda

# Accept a build-arg that lets CI force-invalidate setup_nodes.py
ARG CACHEBUST=static
ENV CACHEBUST=${CACHEBUST}

# NOTE: We skip running setup_nodes as per requirements

# Expose the BYOC server port
EXPOSE 5000

# Environment variables
ENV PYTHONPATH=/workspace/comfystream
ENV WORKSPACE=/workspace

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command to run the BYOC server
CMD ["python", "example_byoc_server.py", "--workspace", "/workspace", "--host", "0.0.0.0", "--port", "5000"]
