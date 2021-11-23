FROM tensorflow/tensorflow:latest

RUN mkdir /src /build
WORKDIR /build

ADD . /src/
