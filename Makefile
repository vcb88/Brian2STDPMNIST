.PHONY: all install prepare-data train test jupyter docker-build docker-run clean

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
VENV_BIN := $(VENV)/bin
JUPYTER_PORT := 8888
DOCKER_IMAGE := brian2-stdp-mnist
DOCKER_TAG := latest

all: install prepare-data

install:
        $(PYTHON) -m venv $(VENV)
        $(VENV_BIN)/$(PIP) install --upgrade pip
        $(VENV_BIN)/$(PIP) install -r requirements.txt

prepare-data:
        $(VENV_BIN)/$(PYTHON) scripts/prepare_dataset.py

train:
        @echo "Starting training..."
        $(VENV_BIN)/$(PYTHON) src/train.py --mode train

test:
        @echo "Starting testing..."
        $(VENV_BIN)/$(PYTHON) src/train.py --mode test

jupyter:
        $(VENV_BIN)/jupyter notebook --ip=0.0.0.0 --port=$(JUPYTER_PORT) --no-browser --NotebookApp.token='' --NotebookApp.password=''

docker-build:
        docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
        docker-compose up -d

docker-stop:
        docker-compose down

clean:
        rm -rf $(VENV)
        rm -rf __pycache__
        rm -rf .pytest_cache
        rm -rf *.pyc
        find . -type d -name "__pycache__" -exec rm -r {} +