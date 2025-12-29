FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    ca-certificates \
    libwebp-dev \
    zlib1g-dev \
    libeigen3-dev \
    nlohmann-json3-dev \
    libabsl-dev \
    libomp-dev \
    gdb \
    clang-format \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y gcc-12 g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CUDA_PATH=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /workspaces/splat-transform

CMD ["/bin/bash"]