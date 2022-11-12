#FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
#FROM gaetanlandreau/pytorch3d:v0.1.0
FROM yuewuust/pytorch3d_v2:latest

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv
#RUN conda install git

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

# Install required python packages via pip - you may adapt the requirements.txt to your needs
RUN python -m pip install --user --upgrade pip
RUN python -m pip install --user --upgrade pip setuptools wheel
RUN python -m pip install --user -r requirements.txt
#RUN python -m pip uninstall torchvision
RUN python -m pip install --user torchvision==0.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html

#RUN python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
#RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
#RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

COPY --chown=algorithm:algorithm models/ /opt/algorithm/models/
COPY --chown=algorithm:algorithm tryme/ /opt/algorithm/tryme/
COPY --chown=algorithm:algorithm optimization/ /opt/algorithm/optimization/
COPY --chown=algorithm:algorithm output/ /opt/algorithm/output/

# these are only for testing
COPY --chown=algorithm:algorithm dummy/ /opt/algorithm/dummy/
# COPY --chown=algorithm:algorithm test/ /opt/algorithm/test/
# COPY --chown=algorithm:algorithm output/ /opt/algorithm/output/

COPY --chown=algorithm:algorithm p2ilf_test.py /opt/algorithm/
# Entrypoint to your python code - executes process.py as a script
ENTRYPOINT python -m p2ilf_test $0 $@

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=16G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=6.0
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=8G
