FROM mambaorg/micromamba:2.1.1

USER root
ARG DEBIAN_FRONTEND=noninteracive
WORKDIR /app
RUN apt-get -y -qq update && apt-get -y -qq install vim nano git build-essential curl wget unzip && \
    wget https://spin.niddk.nih.gov/bax/software/SPARTA+/sparta+.tar.Z && \
    tar xf sparta+.tar.Z && \
    ln -s /app/SPARTA+/bin/SPARTA+.static.linux /app/SPARTA+/bin/sparta+ && \
    cd /usr/local/src && \
    git clone https://github.com/rlabduke/reduce.git && \
    cd reduce && \
    make && \
    make install && \
    mkdir /app/pepsi && \
    cd /app/pepsi && \
    wget https://files.inria.fr/NanoDFiles/Website/Software/Pepsi-SAXS/Linux/3.0/Pepsi-SAXS-Linux.zip && \
    unzip Pepsi-SAXS-Linux.zip && \
    rm Pepsi-SAXS-Linux.zip

COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /app
COPY models.tgz /tmp/models.tgz
RUN git clone https://github.com/THGLab/CSpred.git && \
    cd CSpred && \
    git checkout 0b0ae3f08e42a13225024f081df0867713b54aaf && \
    mkdir models && \
    cd models && \
    tar xzf /tmp/models.tgz && \
    rm /tmp/models.tgz

RUN ln -s /app/CSpred/bins/mTM-align/mTM-align /usr/local/bin/mTM-align && \
    ln -s /app/mambauser/CSpred/bins/mkdssp /usr/local/bin/mkdssp && \
    ln -s /app/mambauser/CSpred/bins/ncbi-blast-2.9.0+/bin/blastp /usr/local/bin/blastp && \
    ln -s /app/mambauser/CSpred/bins/ncbi-blast-2.9.0+/bin/makeblastdb /usr/local/bin/makeblastdb

WORKDIR /app/peptonebench/datasets
RUN wget https://zenodo.org/record/17306061/files/PeptoneDBs.tar.gz && \
    tar -xzvf PeptoneDBs.tar.gz ./PeptoneDB-CS/PeptoneDB-CS.csv ./PeptoneDB-Integrative ./PeptoneDB-SAXS && \
    rm PeptoneDBs.tar.gz

WORKDIR /app/peptonebench
ENV PYTHONPATH=/app/CSpred \
    SPARTAP_DIR=/app/SPARTA+ \
    PATH=/app/SPARTA+/bin:/app/pepsi:$PATH


COPY *.py /app/peptonebench

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/app/peptonebench/run_forward_model.py"]