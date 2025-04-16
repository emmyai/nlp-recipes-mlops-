FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y \
    wget bzip2 ca-certificates curl unzip git zip \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (Python 3 version)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda clean -tipsy

# Create working directory
WORKDIR /workspace

# Optional: clone your repo if needed (or mount it)
# COPY . /workspace
# RUN git clone https://github.com/emmyai/nlp-recipes-mlops-.git /workspace

# Copy and install environment from custom conda file (mutated_conda_dependencies.yml)
COPY mutated_conda_dependencies.yml .
RUN conda env create -n nlp_gpu -f mutated_conda_dependencies.yml && \
    conda clean -afy

# Activate environment and set it as default
ENV CONDA_DEFAULT_ENV=nlp_gpu
RUN echo "conda activate nlp_gpu" >> ~/.bashrc
ENV PATH=$CONDA_DIR/envs/nlp_gpu/bin:$PATH

# Set up Jupyter kernel (optional)
RUN python -m ipykernel install --user --name nlp_gpu --display-name "Python (nlp_gpu)"

# Jupyter notebook exposed
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

