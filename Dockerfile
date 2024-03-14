# Use an official CUDA-enabled base image for building
FROM nvidia/cuda:12.3.2-base-ubuntu22.04 AS builder

# Set the working directory
WORKDIR /app

# Install CUDA-related packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch to the Python 3.10 image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy CUDA-related files from the builder stage
COPY --from=builder /app /app

# Install additional packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy your application code
COPY . /app

# Create directories for the files
#RUN mkdir -p /app/model_repo/enhancer_stage2/ds/G/default
# Download files using wget
#RUN wget -O /app/model_repo/enhancer_stage2/hparams.yaml "https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/#hparams.yaml?download=true" && \
#    wget -O /app/model_repo/enhancer_stage2/ds/G/latest "https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/ds/G/latest?download=true" && \
#    wget -O /app/model_repo/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt "https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt?download=true"

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --force-reinstall soundfile
RUN pip install --force-reinstall tensorflow[and-cuda]
# Set up your entrypoint and expose necessary ports
EXPOSE 80
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

