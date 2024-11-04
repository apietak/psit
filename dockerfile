FROM ubuntu:latest


# Python stuff
RUN mkdir /app
COPY psit.py /app
COPY psitcli.py /app
COPY find_idx_wrapper.py /app
COPY math_helper.py /app

RUN apt -y update && apt -y upgrade
RUN apt install -y python3
RUN apt install -y pip
RUN apt install -y build-essential
RUN pip --version
RUN python3 --version
RUN pip install --upgrade pip
RUN pip --version

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

# SZ3 stuff
RUN apt install -y git
RUN apt install -y cmake
RUN git clone "https://github.com/szcompressor/SZ3.git" /app/SZ3
WORKDIR /app/SZ3
RUN mkdir build && mkdir install
WORKDIR /app/SZ3/build
RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/app/SZ3/install ..
RUN make && make install


# The find_idx stuff
COPY find_idx.cpp /app
WORKDIR /app
RUN g++ -shared -o find_idx.so -fPIC find_idx.cpp

RUN apt install -y libopenjp2-7


COPY coords_small.csv /app

ENTRYPOINT [ "python3", "/app/psitcli.py" ]


ENV LD_LIBRARY_PATH="/app/SZ3/install/lib/"
