# Convenience targets.  Override the interpreter with `make PY=python ...` (e.g. on Kaggle/Colab); by default
# the project venv below is used where it exists, python3 otherwise.
VENV_PY := /home/claude/.venvs/gaehep/bin/python
PY ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)
CLI := $(PY) -m gaehep.cli
# `make smoke` writes its synthetic dataset and results here (the directory is deleted first)
SCRATCH ?= /tmp/gaehep_smoke

.PHONY: setup data data-blackbox download preprocess tables splits test lint smoke baselines report train-% flow-% qn-%

setup:
	$(PY) -m pip install -e ".[dev]"

data: download preprocess tables splits

# the three LHC Olympics black boxes (test data; ~2.5 GB download and ~2.4 GB processed each): download, cluster,
# build the tables, delete each raw file once its run is verified, then make the test splits.  Rerunnable.
data-blackbox:
	for s in bb1 bb2 bb3; do \
	  $(CLI) download --sample $$s && \
	  $(CLI) preprocess --sample $$s --delete-raw && \
	  $(CLI) tables --sample $$s || exit 1; \
	done
	$(CLI) splits --sample bb1 bb2 bb3

download:
	$(CLI) download

preprocess:
	$(CLI) preprocess --sample rnd
	$(CLI) preprocess --sample qqq

tables:
	$(CLI) tables --sample rnd
	$(CLI) tables --sample qqq

splits:
	$(CLI) splits

test:
	$(PY) -m pytest -q -p no:warnings

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

# end-to-end run on a small synthetic dataset (no LHCO files needed)
smoke:
	rm -rf $(SCRATCH)
	$(PY) -m gaehep.data.fake --out $(SCRATCH) --n-bkg 8000 --n-sig 1500
	GAEHEP_DATA=$(SCRATCH) GAEHEP_RESULTS=$(SCRATCH)/results $(CLI) train configs/smoke.yaml --name smoke
	GAEHEP_DATA=$(SCRATCH) GAEHEP_RESULTS=$(SCRATCH)/results $(CLI) flow smoke
	GAEHEP_DATA=$(SCRATCH) GAEHEP_RESULTS=$(SCRATCH)/results $(CLI) qn smoke
	GAEHEP_DATA=$(SCRATCH) GAEHEP_RESULTS=$(SCRATCH)/results $(CLI) bumphunt smoke --agg prod
	GAEHEP_DATA=$(SCRATCH) GAEHEP_RESULTS=$(SCRATCH)/results $(CLI) weak smoke --n-models 2 --n-inj 100,200 --reps 1
	GAEHEP_DATA=$(SCRATCH) GAEHEP_RESULTS=$(SCRATCH)/results $(CLI) baselines
	GAEHEP_DATA=$(SCRATCH) GAEHEP_RESULTS=$(SCRATCH)/results $(CLI) report

baselines:
	$(CLI) baselines

report:
	$(CLI) report

train-%:
	$(CLI) train configs/$*.yaml

flow-%:
	$(CLI) flow $*

qn-%:
	$(CLI) qn $*
