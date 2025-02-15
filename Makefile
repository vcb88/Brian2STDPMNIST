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

install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/$(PIP) install --upgrade pip
	$(VENV_BIN)/$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

prepare-data:
	@echo "$(BLUE)Preparing data locally...$(NC)"
	$(VENV_BIN)/$(PYTHON) scripts/prepare_dataset.py
	@echo "$(GREEN)Data preparation completed$(NC)"

train:
	@echo "$(BLUE)Starting training...$(NC)"
	$(VENV_BIN)/$(PYTHON) src/train.py --mode train

test:
	@echo "$(BLUE)Starting testing...$(NC)"
	$(VENV_BIN)/$(PYTHON) src/train.py --mode test

jupyter:
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	$(VENV_BIN)/jupyter notebook --ip=0.0.0.0 --port=$(JUPYTER_PORT) --no-browser --NotebookApp.token='' --NotebookApp.password=''

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
	@docker exec -it $(CONTAINER_NAME) python3 Diehl\&Cook_spiking_MNIST_Brian2.py --train
	@echo "$(GREEN)Training completed$(NC)"

container-test:
	@echo "$(BLUE)Starting testing inside container...$(NC)"
	@docker exec -it $(CONTAINER_NAME) python3 Diehl\&Cook_spiking_MNIST_Brian2.py --test
	@echo "$(GREEN)Testing completed$(NC)"

container-train-verbose:
	@echo "$(BLUE)Starting training inside container with verbose output...$(NC)"
	@docker exec -it $(CONTAINER_NAME) python3 Diehl\&Cook_spiking_MNIST_Brian2.py --train --verbose
	@echo "$(GREEN)Training completed$(NC)"

container-test-verbose:
	@echo "$(BLUE)Starting testing inside container with verbose output...$(NC)"
	@docker exec -it $(CONTAINER_NAME) python3 Diehl\&Cook_spiking_MNIST_Brian2.py --test --verbose
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
	@echo "$(GREEN)Status check completed$(NC)"

dataset-clean:
	@echo "$(BLUE)Cleaning dataset inside container...$(NC)"
	@docker exec -it $(CONTAINER_NAME) rm -rf /app/mnist/*
	@echo "$(GREEN)Dataset cleaned successfully$(NC)"

# Combined commands for convenience
container-prepare-and-train: dataset-prepare container-train

container-full-test: dataset-status container-test

clean:
	@echo "$(BLUE)Cleaning project...$(NC)"
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc
	find . -type d -name "__pycache__" -exec rm -r {} +
	@echo "$(GREEN)Cleanup completed$(NC)"
