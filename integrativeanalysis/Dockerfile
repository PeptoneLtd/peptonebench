FROM mambaorg/micromamba:2.1.1

USER root
ARG DEBIAN_FRONTEND=noninteracive
WORKDIR /app
RUN apt-get -y -qq update && apt-get -y -qq install vim nano git build-essential curl wget unzip && \
    wget https://spin.niddk.nih.gov/bax/software/PALES/pales.linux.tar.Z && \
    tar xf pales.linux.tar.Z && \
    rm pales.linux.tar.Z

COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes


WORKDIR /app/peptonebench/datasets
RUN wget https://zenodo.org/record/17306061/files/PeptoneDBs.tar.gz && \
    tar -xzvf PeptoneDBs.tar.gz ./PeptoneDB-CS/PeptoneDB-CS.csv ./PeptoneDB-Integrative ./PeptoneDB-SAXS/PeptoneDB-SAXS.csv && \
    rm PeptoneDBs.tar.gz

WORKDIR /app/integrativeanalysis
ENV PATH=/app/pales/linux:$PATH

COPY . /app/integrativeanalysis

WORKDIR /app
EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0"]
