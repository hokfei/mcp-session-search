# Session Search MCP - development tasks.
#
# Portable across platforms: no rm/rmdir/for-loop syntax, and PYTHON can be
# overridden for a virtualenv interpreter.
#
#   make PYTHON=.venv/Scripts/python.exe test     (Windows)
#   make test                                     (POSIX, with an active venv)
#
# Run `make help` for the list.

PYTHON ?= python
VENV   ?= .venv

.PHONY: help install install-gpu test test-unit test-contract check lint format \
        typecheck run verify calibrate index health clean

help:
	@echo "Session Search MCP"
	@echo ""
	@echo "  make install      Install dependencies"
	@echo "  make install-gpu  Install with CUDA support"
	@echo ""
	@echo "  make test         Run all tests"
	@echo "  make test-unit    Unit tests only (fast, no subprocesses)"
	@echo "  make test-contract Contract tests (spawn a real server)"
	@echo ""
	@echo "  make lint         Lint with ruff"
	@echo "  make format       Auto-fix lint and apply ruff formatting"
	@echo "  make typecheck    Type-check with mypy"
	@echo "  make check        Lint, typecheck, and unit tests"
	@echo ""
	@echo "  make verify       Check the mcp.json wiring end to end"
	@echo "  make calibrate    Re-measure the similarity threshold"
	@echo "  make index        Rebuild the search index from scratch"
	@echo "  make health       Print a health report"
	@echo "  make clean        Remove caches and build artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

install-gpu:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,gpu]"

test:
	$(PYTHON) -m pytest tests -q

test-unit:
	$(PYTHON) -m pytest tests/unit -q

test-contract:
	$(PYTHON) -m pytest tests/contract -q

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

typecheck:
	$(PYTHON) -m mypy src

check: lint typecheck test-unit

# Serves over stdio; expects to wait for input.
run:
	PYTHONPATH=src $(PYTHON) -m session_search

# Confirms a client can actually reach the server as configured.
verify:
	$(PYTHON) verify_mcp_config.py

calibrate:
	$(PYTHON) calibrate.py

index:
	PYTHONPATH=src $(PYTHON) -m session_search reindex

health:
	PYTHONPATH=src $(PYTHON) -m session_search health

# Done in Python so it behaves the same on every platform.
clean:
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in map(pathlib.Path, ['build','dist','htmlcov','.pytest_cache','.mypy_cache','.ruff_cache'])]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; pathlib.Path('.coverage').unlink(missing_ok=True)"