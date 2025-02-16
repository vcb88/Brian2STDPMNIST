.PHONY: all install prepare-data train test jupyter docker-build docker-run docker-stop clean dataset-download dataset-prepare dataset-status dataset-clean

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
VENV_BIN := $(VENV)/bin
JUPYTER_PORT := 8888
DOCKER_IMAGE := brian2-stdp-mnist
DOCKER_TAG := latest
CONTAINER_NAME := brian2stdpmnist-notebook-1

# Colors for pretty printing
BLUE := \033[1;34m
GREEN := \033[1;32m
RED := \033[1;31m
YELLOW := \033[1;33m
NC := \033[0m # No Color

all: install prepare-data

# Initial setup
install: docker-build docker-run

# Data preparation
prepare-data: dataset-prepare

# Shortcuts for Docker commands
train: container-train

test: container-test

# Shortcut for Jupyter notebook
jupyter: docker-run
        @echo "$(GREEN)Jupyter notebook is available at http://localhost:$(JUPYTER_PORT)$(NC)"

docker-build:
        @echo "$(BLUE)Building Docker image...$(NC)"
        docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
        @echo "$(GREEN)Docker image built successfully$(NC)"

docker-run:
        @echo "$(BLUE)Starting Docker containers...$(NC)"
        docker-compose up -d
        @echo "$(GREEN)Containers started successfully$(NC)"
        @echo "$(GREEN)Jupyter notebook is available at http://localhost:$(JUPYTER_PORT)$(NC)"

docker-stop:
        @echo "$(BLUE)Stopping Docker containers...$(NC)"
        docker-compose down
        @echo "$(GREEN)Containers stopped successfully$(NC)"

# Commands for running inside container
container-train:
        @echo "$(BLUE)Starting training inside container...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 diehl_cook_spiking_mnist_brian2.py --train --data-dir /app/mnist
        @echo "$(GREEN)Training completed$(NC)"

container-test:
        @echo "$(BLUE)Starting full testing inside container...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 diehl_cook_spiking_mnist_brian2.py --test --data-dir /app/mnist
        @echo "$(GREEN)Testing completed$(NC)"

container-test-quick:
        @echo "$(BLUE)Starting quick test (1000 examples) inside container...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 diehl_cook_spiking_mnist_brian2.py --test --test-size 1000 --data-dir /app/mnist
        @echo "$(GREEN)Quick testing completed$(NC)"

container-test-random:
        @echo "$(BLUE)Starting random test (1000 examples) inside container...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 diehl_cook_spiking_mnist_brian2.py --test --test-size 1000 --random-subset --data-dir /app/mnist
        @echo "$(GREEN)Random testing completed$(NC)"

# Custom test with parameters
# Usage: make container-test-custom TEST_ARGS="--test-size 500 --random-subset"
container-test-custom:
        @echo "$(BLUE)Starting custom test inside container...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 diehl_cook_spiking_mnist_brian2.py --test --data-dir /app/mnist $(TEST_ARGS)
        @echo "$(GREEN)Custom testing completed$(NC)"

# Test with specific sample size
# Usage: make container-test-size SIZE=500
container-test-size:
        @if [ -z "$(SIZE)" ]; then \
                echo "$(RED)Error: SIZE parameter is required. Usage: make container-test-size SIZE=500$(NC)"; \
                exit 1; \
        fi
        @echo "$(BLUE)Starting test with $(SIZE) examples...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 diehl_cook_spiking_mnist_brian2.py --test --test-size $(SIZE) --data-dir /app/mnist
        @echo "$(GREEN)Testing with $(SIZE) examples completed$(NC)"

# Test with specific sample size (random subset)
# Usage: make container-test-size-random SIZE=500
container-test-size-random:
        @if [ -z "$(SIZE)" ]; then \
                echo "$(RED)Error: SIZE parameter is required. Usage: make container-test-size-random SIZE=500$(NC)"; \
                exit 1; \
        fi
        @echo "$(BLUE)Starting test with $(SIZE) random examples...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 diehl_cook_spiking_mnist_brian2.py --test --test-size $(SIZE) --random-subset --data-dir /app/mnist
        @echo "$(GREEN)Testing with $(SIZE) random examples completed$(NC)"

container-train-verbose:
        @echo "$(BLUE)Starting training inside container with verbose output...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 diehl_cook_spiking_mnist_brian2.py --train --verbose --data-dir /app/mnist
        @echo "$(GREEN)Training completed$(NC)"

container-test-verbose:
        @echo "$(BLUE)Starting testing inside container with verbose output...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 diehl_cook_spiking_mnist_brian2.py --test --verbose --data-dir /app/mnist
        @echo "$(GREEN)Testing completed$(NC)"

# Dataset management commands (for running inside container)
dataset-download:
        @echo "$(BLUE)Downloading MNIST dataset inside container...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 scripts/prepare_dataset.py --download-only
        @echo "$(GREEN)Dataset download completed$(NC)"

dataset-prepare:
        @echo "$(BLUE)Preparing MNIST dataset inside container...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 scripts/prepare_dataset.py --prepare-only
        @echo "$(GREEN)Dataset preparation completed$(NC)"

dataset-status:
        @echo "$(BLUE)Checking dataset status inside container...$(NC)"
        @docker exec -it $(CONTAINER_NAME) python3 scripts/prepare_dataset.py --status
        @echo "$(BLUE)Checking directory structure...$(NC)"
        @docker exec -it $(CONTAINER_NAME) ls -la /app/mnist/
        @echo "$(GREEN)Status check completed$(NC)"

dataset-clean:
        @echo "$(BLUE)Cleaning dataset inside container...$(NC)"
        @docker exec -it $(CONTAINER_NAME) rm -rf /app/mnist/*
        @echo "$(GREEN)Dataset cleaned successfully$(NC)"

# Combined commands for convenience
container-prepare-and-train: dataset-prepare container-train

container-full-test: dataset-status container-test

reset-and-test:
        @echo "$(BLUE)Stopping containers...$(NC)"
        docker-compose down
        @echo "$(BLUE)Cleaning MNIST data...$(NC)"
        rm -rf mnist/*
        @echo "$(BLUE)Pulling latest changes...$(NC)"
        git pull
        @echo "$(BLUE)Rebuilding Docker image...$(NC)"
        docker-compose build
        @echo "$(BLUE)Starting containers...$(NC)"
        docker-compose up -d
        @echo "$(BLUE)Preparing dataset...$(NC)"
        @$(MAKE) dataset-download
        @$(MAKE) dataset-prepare
        @$(MAKE) dataset-status
        @echo "$(BLUE)Running tests...$(NC)"
        @$(MAKE) container-test-size SIZE=300
        @echo "$(GREEN)Reset and test sequence completed$(NC)"

clean:
        @echo "$(BLUE)Cleaning project...$(NC)"
        rm -rf $(VENV)
        rm -rf __pycache__
        rm -rf .pytest_cache
        rm -rf *.pyc
        find . -type d -name "__pycache__" -exec rm -r {} +
        @echo "$(GREEN)Cleanup completed$(NC)"
