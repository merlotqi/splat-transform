FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
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
    clang-tidy \
    ccache \
    doxygen \
    graphviz \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y gcc-12 g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && update-alternatives --set gcc /usr/bin/gcc-12 \
    && update-alternatives --set g++ /usr/bin/g++-12

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CUDA_PATH=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:/usr/lib/ccache:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV CCACHE_DIR=/workspaces/SplatLib/.ccache

WORKDIR /workspaces/SplatLib

CMD ["/bin/bash"]
