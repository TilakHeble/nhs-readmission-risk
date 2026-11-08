.PHONY: install data test lint notebooks dashboard all

install:
	pip install -e ".[dev,dashboard]"

data:
	python -m nhsread.ingest.build

test:
	pytest -q

lint:
	ruff check src tests dashboards

notebooks:
	for nb in notebooks/*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 $$nb; done

dashboard:
	streamlit run dashboards/app.py

all: data test notebooks
