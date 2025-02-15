# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools>=61.0.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Create mnist directory with proper permissions
RUN mkdir -p /app/mnist && chmod 777 /app/mnist

# Create a non-root user
RUN useradd -m -u 1000 brian2user
RUN chown -R brian2user:brian2user /app
USER brian2user

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter by default
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]