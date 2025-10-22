FROM mambaorg/micromamba:2.1.1

USER root
ARG DEBIAN_FRONTEND=noninteracive
RUN apt-get -y -qq update && apt-get -y -qq install vim nano git build-essential curl wget unzip && \
    cd /opt && \
    wget https://spin.niddk.nih.gov/bax/software/SPARTA+/sparta+.tar.Z && \
    tar xf sparta+.tar.Z && \
    ln -s /opt/SPARTA+/bin/SPARTA+.static.linux /opt/SPARTA+/bin/sparta+ && \
    cd /usr/local/src && \
    git clone https://github.com/rlabduke/reduce.git && \
    cd reduce && \
    make && \
    make install && \
    cd /home/$MAMBA_USER && \
    wget https://files.inria.fr/NanoDFiles/Website/Software/Pepsi-SAXS/Linux/3.0/Pepsi-SAXS-Linux.zip && \
    unzip Pepsi-SAXS-Linux.zip && \
    rm Pepsi-SAXS-Linux.zip && \
    chown $MAMBA_USER:$MAMBA_USER Pepsi-SAXS

USER $MAMBA_USER

ENV SPARTAP_DIR=/opt/SPARTA+ \
    PATH=/opt/SPARTA+/bin:$PATH

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /home/mambauser
COPY --chown=$MAMBA_USER:$MAMBA_USER models.tgz /tmp/models.tgz
RUN git clone https://github.com/THGLab/CSpred.git && \
    cd CSpred && \
    git checkout 0b0ae3f08e42a13225024f081df0867713b54aaf && \
    mkdir models && \
    cd models && \
    tar xzf /tmp/models.tgz && \
    rm /tmp/models.tgz

USER root
RUN ln -s /home/mambauser/CSpred/bins/mTM-align/mTM-align /usr/local/bin/mTM-align && \
    ln -s /home/mambauser/CSpred/bins/mkdssp /usr/local/bin/mkdssp && \
    ln -s /home/mambauser/CSpred/bins/ncbi-blast-2.9.0+/bin/blastp /usr/local/bin/blastp && \
    ln -s /home/mambauser/CSpred/bins/ncbi-blast-2.9.0+/bin/makeblastdb /usr/local/bin/makeblastdb && \
    chmod -R a+r /opt/SPARTA+
USER $MAMBA_USER

WORKDIR /app/peptonebench/datasets
RUN wget https://zenodo.org/record/17306061/files/PeptoneDBs.tar.gz && \
    tar -xzvf PeptoneDBs.tar.gz ./PeptoneDB-CS/PeptoneDB-CS.csv ./PeptoneDB-Integrative/PeptoneDB-Integrative.csv ./PeptoneDB-SAXS && \
    rm PeptoneDBs.tar.gz

WORKDIR /home/mambauser
ENV PYTHONPATH=/home/mambauser/CSpred
COPY --chown=$MAMBA_USER:$MAMBA_USER *.py /home/mambauser/
