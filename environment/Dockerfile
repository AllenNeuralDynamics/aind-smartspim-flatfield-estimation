# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda create -n flatfield_estimation python=3.9 -y
SHELL ["conda", "run", "-n", "flatfield_estimation", "/bin/bash", "-c"]
# RUN conda install -c conda-forge awscli

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    dask[distributed] \
    numpy==1.24.2 \
    BaSiCPy==1.1.0 \
    jax==0.4.23 \
    jaxlib==0.4.23 \
    zarr==2.18.2 \
    matplotlib \
    natsort==8.4.0 \
    PyWavelets==1.6.0
