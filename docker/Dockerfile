FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20240123.v1

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

RUN apt-get update --fix-missing && apt-get install -y \
    wget bzip2 ca-certificates curl unzip git zip \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the actual CPU-focused environment file
COPY azureml-environment-setup/mutated_conda_dependencies.yml .

# Create environment (CPU only)
RUN conda env create -p /azureml-envs/bert-env -f mutated_conda_dependencies.yml && \
    conda clean -afy

ENV PATH=/azureml-envs/bert-env/bin:$PATH
ENV CONDA_DEFAULT_ENV=bert-env

# Optional kernel setup for Jupyter
RUN python -m ipykernel install --user --name bert-env --display-name "Python (bert-env)"

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

