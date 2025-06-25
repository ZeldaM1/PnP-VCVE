FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
# FROM bit:5000/qic_emseg_torch1.8_v1.1
# FROM bit:5000/zhm-py36-torch12
MAINTAINER zenghuimin (zenghuimin@mail.ustc.edu.cn)
WORKDIR /ghome/zenghm

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV DEBIAN_FRONTEND=noninteractive


RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
# RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g'/etc/apt/sources.list > /tmp/tmpfile420 && \
#     cat /tmp/tmpfile420 > /etc/apt/sources.list\
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        ffmpeg \
        git \
        vim \
        htop \
        tmux \
        openssh-client \
        openssh-server \
        libboost-dev \
        libboost-thread-dev \
        libboost-filesystem-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3-dev openslide-tools python3-openslide imagemagick screen zip psmisc \
        libgl1-mesa-glx libgl1-mesa-dev

RUN useradd -m zenghm -u 1722 -s /bin/bash && \
    echo "zenghm:admin!" | chpasswd && \
    echo "zenghm ALL=(ALL) ALL" >> /etc/sudoers
USER zenghm
COPY miniconda.sh /home/zenghm/
# ==================================================================
# Miniconda3
# ------------------------------------------------------------------
RUN /bin/bash /home/zenghm/miniconda.sh -b -p /home/zenghm/anaconda3 && \
        rm /home/zenghm/miniconda.sh
        #echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc && \
ENV PATH=/home/zenghm/anaconda3/bin:$PATH
# ==================================================================
# conda
# ------------------------------------------------------------------
RUN conda config --set show_channel_urls yes && \
    conda config --add channels conda-forge && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    conda install -y \
        python=3.7 \
        cudatoolkit=11.1 \
        cudnn
        
RUN mkdir /home/zenghm/.pip && \
    echo "[global]" > /home/zenghm/.pip/pip.conf && \
    # sed -i -e "\$atrusted-host=mirrors.aliyun.com" /home/zenghm/.pip/pip.conf && \
    # sed -i -e "\$aindex-url=http://mirrors.aliyun.com/pypi/simple/" /home/zenghm/.pip/pip.conf && \
    sed -i -e "\$atrusted-host=pypi.doubanio.com/" /home/zenghm/.pip/pip.conf && \
    sed -i -e "\$aindex-url=https://pypi.doubanio.com/simple/" /home/zenghm/.pip/pip.conf && \
    pip install --upgrade pip && \
    pip install pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html \
    # pip install pip install torch==1.8.0  torchvision==0.9.0 torchaudio==0.8.0   \
    numpy \
    cupy-cuda111 \
	imageio  \
	matplotlib  \
	scikit-image  \
	easydict  \
	Pillow \
	opencv-python  \
	opencv-contrib-python  \
	tensorboardX  \
	tensorboard \
	future \
	lmdb \
	pyarrow \
	thop \
    scipy \
    plyfile \
    yacs \
    tqdm \
    argparse \
    sacred \
    progressbar2 \
    path.py \
    h5py \
    pyyaml \
    cython \
    progressbar \
    torchsummary waterz malis   \
    wandb lmdb pynvml einops \
    omegaconf==2.1.0 \
    torch-fidelity==0.3.0 \
    pytorch-lightning==1.4.4 \
    torchmetrics==0.5.0 \
    av==8.0.3  \
    facexlib \
    mmcv-full  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html     \
    codecov \
    flake8 \
    interrogate \
    isort==5.10.1 \
    onnxruntime \
    pytest \
    pytest-runner \
    yapf


EXPOSE 6006

