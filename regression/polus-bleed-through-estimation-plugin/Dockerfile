FROM python:3.9-slim-buster

# Change to .ome.zarr to save output images as zarr files.
ENV POLUS_EXT=".ome.tif"

# Change to WARNING for fewer logs, or DEBUG for debugging.
ENV POLUS_LOG="INFO"

ARG DATA_DIR="/data"
ARG EXEC_DIR="/opt/executables"

COPY requirements.txt /

RUN pip3 install -r requirements.txt --no-cache-dir && \
    mkdir -p ${DATA_DIR}/{inputs, outputs} && \
    mkdir -p ${EXEC_DIR}/{bleed_through_estimation, utils}

COPY src/main.py ${EXEC_DIR}/
COPY src/utils ${EXEC_DIR}/utils/
COPY src/bleed_through_estimation ${EXEC_DIR}/bleed_through_estimation/
COPY VERSION /

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]