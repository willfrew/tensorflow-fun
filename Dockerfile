FROM tensorflow/tensorflow:latest

RUN mkdir /src /build
WORKDIR /build

ADD build.py load.py /src/
