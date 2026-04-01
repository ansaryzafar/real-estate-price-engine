.PHONY: setup train test serve clean lint

setup:
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

train:
.venv/bin/python -m src.models.train

test:
.venv/bin/pytest tests/ -v

serve:
.venv/bin/uvicorn src.api.app:app --reload --port 8000

lint:
.venv/bin/python -m flake8 src/ tests/

clean:
rm -rf .venv __pycache__ .pytest_cache
find . -type d -name __pycache__ -exec rm -rf {} +

docker:
docker build -t $(shell basename $(CURDIR)) .
docker run -p 8000:8000 $(shell basename $(CURDIR))
