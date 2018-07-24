FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y git

RUN mkdir /model
WORKDIR /model
ADD . /model

WORKDIR /model

CMD bash
