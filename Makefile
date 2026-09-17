# Makefile

.PHONY: all format check validate test test-cov test-integration test-all db-up db-down test-integration-local vulture complexity xenon bandit pyright fix reformat-ruff fix-ruff

# Default target: runs format and check
all: validate test

# Format the code using ruff
format:
	uv run ruff format --check --diff .

reformat-ruff:
	uv run ruff format .

# Check the code using ruff
check:
	uv run ruff check .

fix-ruff:
	uv run ruff check . --fix

fix: reformat-ruff fix-ruff
	@echo "Updated code."

test:
	uv run pytest tests -m "not integration"

test-cov:
	uv run pytest tests -m "not integration" --cov --cov-report=xml --cov-report=term-missing

test-integration:
	uv run pytest tests -v -m integration --timeout=120

test-all: test-cov
	uv run pytest tests -v -m integration --timeout=120

# Integration test helpers
db-up:
	docker compose up -d elasticsearch mongodb
	@echo "Waiting for Elasticsearch to be ready..."
	@bash -c 'for i in $$(seq 1 90); do curl -sf http://localhost:9200/_cluster/health >/dev/null 2>&1 && exit 0; sleep 1; done; exit 1' || { echo "Timeout waiting for Elasticsearch"; exit 1; }
	@echo "Elasticsearch is ready!"
	@echo "Waiting for MongoDB to be ready..."
	@bash -c 'for i in $$(seq 1 30); do nc -z localhost 27017 2>/dev/null && exit 0; sleep 1; done; exit 1' || { echo "Timeout waiting for MongoDB"; exit 1; }
	@echo "MongoDB is ready!"

db-down:
	docker compose down

test-integration-local: db-up
	uv run pytest tests -v -m integration --timeout=120; status=$$?; $(MAKE) db-down; exit $$status

vulture:
	uv run vulture . --exclude .venv,tests,docs --make-whitelist

complexity:
	uv run radon cc . -a -nc

xenon:
	uv run xenon -b D -m B -a B .

bandit:
	uv run bandit -c pyproject.toml -r .

pyright:
	uv run pyright

# Validate the code (format + check)
validate: format check complexity bandit pyright vulture
	@echo "Validation passed. Your code is ready to push."
