#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = chalcogenide-perovskite-screening
PYTHON_VERSION = 3.8.20
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies (pip)
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install Python Dependencies (uv, recommended)
.PHONY: install
install:
	uv sync --extra dev --extra notebooks

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff and black
.PHONY: lint
lint:
	ruff check tf_chpvk_pv
	black --check --config pyproject.toml tf_chpvk_pv

## Format source code with ruff and black
.PHONY: format
format:
	ruff check --fix tf_chpvk_pv
	black --config pyproject.toml tf_chpvk_pv

## Serve documentation locally (requires Python >= 3.9)
.PHONY: docs
docs:
	pip install -r docs/requirements.txt
	mkdocs serve


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) tf_chpvk_pv/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
