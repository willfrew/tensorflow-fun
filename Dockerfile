FROM tensorflow/tensorflow:latest

RUN python3 -m pip install --upgrade pillow

RUN mkdir /src /build /cache

ADD . /src/
