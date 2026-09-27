CONFIG ?= config.yaml
.PHONY: setup check test index doctor run

setup:
	uv sync

check:
	uv run ruff check .
	uv run pytest -q

test: check

index:
	uv run xh index --config $(CONFIG)

doctor:
	uv run xh check --config $(CONFIG)

# make run Q="Which datasets do the RAG papers use?"
run:
	uv run xh run --config $(CONFIG) --question "$(Q)"
