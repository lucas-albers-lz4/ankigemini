.PHONY: help install install-dev lint test test-cov clean format run venv docs typecheck

# Use bash for shell commands
SHELL := /bin/bash

# Python settings
PYTHON := python
VENV_NAME := venv

help:  ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies
	pip install -r requirements.txt

install-dev: install ## Install development dependencies
	pip install -r requirements-dev.txt
	pre-commit install

lint: ## Run all linting checks
	pre-commit run --all-files

test: ## Run tests
	$(PYTHON) -m pytest

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=. --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

typecheck: ## Run type checking
	mypy .

clean: ## Clean up Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

format: ## Format code using ruff and isort
	ruff format .
	ruff check --fix .
	isort .

run: ## Run the application
	@echo $(PYTHON) main.py -i INPUTFILE -o OUTPUTFILE

venv: ## Create a new virtual environment
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Run 'source $(VENV_NAME)/bin/activate' to activate the virtual environment"

