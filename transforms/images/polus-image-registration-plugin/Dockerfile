
FROM polusai/bfio:2.1.9
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

RUN apt-get update && \
    apt-get -y install libglib2.0-0

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

COPY VERSION /

ENTRYPOINT ["python3", "/opt/executables/main.py"]