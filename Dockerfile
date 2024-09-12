FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=en_US.utf8
ENV TZ=Asia/Seoul

RUN apt update && apt upgrade -y \
	&& apt install git vim tig tree dialog locales tzdata libgl1-mesa-glx libglib2.0-0 language-pack-en -y \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt
