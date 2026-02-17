PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /usr/bin/env bash

VENV_DIR := .venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

CONFIG ?= configs/smoke.yaml

.PHONY: setup data train eval report all clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@./scripts/bootstrap_venv.sh "$(VENV_DIR)"
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt

data:
	@$(PY) -m src.data.make_data --config "$(CONFIG)"

train:
	@$(PY) -m src.train.biencoder --config "$(CONFIG)" --experiment dense_baseline
	@$(PY) -m src.train.biencoder --config "$(CONFIG)" --experiment matryoshka

eval:
	@$(PY) -m src.eval.run_eval --config "$(CONFIG)" --out artifacts/results.json

report:
	@$(PY) -m src.report.make_report --results artifacts/results.json --out artifacts/report.md

all: setup data train eval report

clean:
	@rm -rf "$(VENV_DIR)" .cache checkpoints runs reports

