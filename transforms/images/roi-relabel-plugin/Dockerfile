FROM polusai/bfio:2.1.9

ENV POLUS_EXT=".ome.tif"
ENV POLUS_LOG="INFO"
ENV DATA_DIR="/data"

RUN pip3 install --upgrade pip setuptools wheel

ENV PLUGIN_DIR="roi-relabel-plugin"

RUN mkdir -p ${PLUGIN_DIR}/src/polus/transforms/images/roi_relabel && \
    touch ${PLUGIN_DIR}/src/polus/transforms/images/roi_relabel/__init__.py

COPY pyproject.toml ${PLUGIN_DIR}/pyproject.toml
COPY README.md ${PLUGIN_DIR}/README.md

RUN pip3 install -e ${PLUGIN_DIR} --no-cache-dir

COPY src ${PLUGIN_DIR}/src

ENTRYPOINT ["python3", "-m", "polus.transforms.images.roi_relabel"]
