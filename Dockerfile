# # Use Python 3.10 as the base image
# FROM python:3.10-slim

# # Set working directory inside the container
# WORKDIR /app

# # Copy requirements file
# COPY ./requirements.txt /app/requirements.txt 

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     python3-dev \
#     libssl-dev

# # Install system dependencies and Python packages
# RUN pip install --no-cache-dir -r /app/requirements.txt 
# # RUN apt-get update && \
# #     apt-get install -y --no-install-recommends gcc build-essential && \
# #     pip install --no-cache-dir -r requirements.txt && \
# #     apt-get clean && \
# #     rm -rf /var/lib/apt/lists/*

# # Copy the project files
# COPY src/ /app/src/
# COPY app.py /app/
# COPY main.py /app/
# COPY models/ /app/models
# COPY start.sh /app/

# # Expose port for FastAPI
# EXPOSE 5000

# # Command to start the FastAPI server
# # Use uvicorn to serve the FastAPI application
# # CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

# # Make the start.sh script executable
# RUN chmod +x /app/start.sh

# # Start FastAPI and ML pipeline in parallel
# CMD ["/app/start.sh"]

# # Use Python 3.10 as the base image
# FROM python:3.10-slim

# # Set working directory inside the container
# WORKDIR /app

# # Copy requirements file
# COPY ./requirements.txt /app/requirements.txt 

# # Install system dependencies and Python packages
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     python3-dev \
#     libssl-dev && \
#     pip install --no-cache-dir -r /app/requirements.txt 

# # Copy the project files
# COPY src/ /app/src/
# COPY app.py /app/
# COPY main.py /app/
# COPY models/ /app/models
# COPY start.sh /app/

# ENV HOPSWORKS_API_KEY=""

# # Expose port for FastAPI
# EXPOSE 5000

# # Make the start.sh script executable
# RUN chmod +x /app/start.sh

# # Start FastAPI and ML pipeline in parallel
# CMD ["/app/start.sh"]


FROM python:3.10-slim AS builder

# Install build dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    gcc \
    g++ \
    make \
    python3-dev && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Copy only dependency files for better caching
COPY pyproject.toml /app/

# Force installation of specific numpy version before other dependencies
# This ensures TensorFlow gets a compatible NumPy version
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "numpy>=1.23.0,<2.0.0" && \
    uv pip install --system -e .

# Copy the project files
COPY src/ /app/src/
COPY app.py /app/
COPY main.py /app/
COPY models/ /app/models/
COPY start.sh /app/

# Ensure the start.sh script has Unix line endings and is executable
RUN apt-get update && apt-get install -y dos2unix && \
    dos2unix /app/start.sh && \
    chmod +x /app/start.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV HOPSWORKS_API_KEY=""

# Expose port for FastAPI
EXPOSE 5000

# Start FastAPI and ML pipeline in parallel
CMD ["/app/start.sh"]







