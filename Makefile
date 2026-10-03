PY ?= python3
BASE ?= sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

.PHONY: install install-ml data test eval eval-base finetune eval-lora serve up

install:
	$(PY) -m pip install -r requirements-dev.txt

install-ml:
	$(PY) -m pip install -r requirements-ml.txt

data:
	$(PY) -m rag.data_gen

test: data
	$(PY) -m pytest -q

eval: data            ## offline baseline (hashing embedder)
	$(PY) -m rag.evaluate --tag hashing

eval-base: data       ## multilingual encoder, no fine-tuning
	$(PY) -m rag.evaluate --model $(BASE) --tag base

finetune: data        ## LoRA fine-tuning of the encoder
	$(PY) -m rag.finetune --model $(BASE) --out models/lora-plm

eval-lora:            ## same encoder + LoRA adapter
	$(PY) -m rag.evaluate --model $(BASE) --adapter models/lora-plm --tag lora

serve:
	uvicorn rag.api:app --reload --port 8000

up:
	docker compose up --build
