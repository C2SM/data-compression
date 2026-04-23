FROM python:3.13

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    libeccodes-dev \
    libnetcdf-dev \
    libhdf5-dev \
    libglib2.0-0 \
    libegl1 \
    libgl1 \
    libdbus-1-3 \
    libxkbcommon0 \
    libfontconfig1 \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/C2SM/data-compression.git /opt/data-compression

WORKDIR /opt/data-compression

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN bash install_dc_toolkit.sh

ENTRYPOINT ["dc_toolkit"]
CMD ["--help"]
