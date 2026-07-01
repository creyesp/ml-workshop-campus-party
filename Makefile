.PHONY: install lint format test train compare hpo docker-build

install:
	pip install -e '.[dev]'

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest -q

train:
	python -m src.train train --model xgboost

compare:
	python -m src.train compare

hpo:
	python -m src.train hpo --model xgboost --trials 30

docker-build:
	docker build -t churn-ml .
