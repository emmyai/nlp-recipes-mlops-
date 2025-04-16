FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20240123.v1

# Set conda environment name and location
ENV CONDA_ENV_NAME=bert-env
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Create cache directory
RUN mkdir -p $HOME/.cache

# Copy environment file into container
COPY azureml-environment-setup/mutated_conda_dependencies.yml .

# Build conda environment
RUN conda env create -p /azureml-envs/$CONDA_ENV_NAME -f mutated_conda_dependencies.yml && \
    rm -rf "$HOME/.cache/pip" && \
    conda clean -aqy && \
    CONDA_ROOT_DIR=$(conda info --root) && \
    rm -rf "$CONDA_ROOT_DIR/pkgs" && \
    find "$CONDA_ROOT_DIR" -type d -name __pycache__ -exec rm -rf {} + && \
    ldconfig

# Set the new environment as default
ENV PATH=/azureml-envs/$CONDA_ENV_NAME/bin:$PATH
ENV CONDA_DEFAULT_ENV=$CONDA_ENV_NAME

# Jupyter kernel for this environment
RUN python -m ipykernel install --user --name $CONDA_ENV_NAME --display-name "Python ($CONDA_ENV_NAME)"

# Expose port for Jupyter (optional)
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
