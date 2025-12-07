FROM ubuntu:22.04
LABEL authors="Wontae Lee"

RUN apt-get update -yq && \
    apt-get install -yq cmake build-essential libtbb-dev pkg-config libglew-dev mesa-common-dev libglu1-mesa-dev freeglut3-dev libglfw3-dev

ADD . /app

WORKDIR /app/build
RUN cmake .. && \
    make -j`nproc` && \
    make install
WORKDIR /

ENTRYPOINT ["top", "-b"]