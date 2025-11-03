FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=false
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --upgrade pip setuptools wheel
RUN pip install jupyterlab git+https://github.com/MarkusHaak/trizod.git

WORKDIR /app/peptonebench
COPY . .
RUN pip install -e .

ENV PEPTONEDB_PATH=/app/peptonebench/datasets
CMD ["PeptoneBench"]
