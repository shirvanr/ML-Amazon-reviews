PYTHON ?= python3

.PHONY: install train serve test lint latency latency-load docker-build docker-run

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

train:
	$(PYTHON) -m src.train

serve:
	$(PYTHON) -m uvicorn src.service.app:app --host 127.0.0.1 --port 8000

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

latency:
	$(PYTHON) scripts/latency_check.py

latency-load:
	$(PYTHON) scripts/latency_check.py --concurrency 50 --requests 1000

docker-build:
	docker build -t sentiment-service .

docker-run:
	docker run --rm -p 8000:8000 sentiment-service
