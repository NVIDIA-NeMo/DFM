# Makefile for Sphinx documentation

# Default target shows help
.DEFAULT_GOAL := help

.PHONY: help docs-html docs-clean docs-live docs-publish ensure-docs-env check-uv

# Help target
help: ## Show this help message
	@echo ""
	@echo "📚 Documentation Build System"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  make docs-html      Build HTML documentation"
	@echo "  make docs-live      Start live-reload server"
	@echo "  make docs-publish   Build for publication (fail on warnings)"
	@echo "  make docs-clean     Clean built documentation"
	@echo ""
	@echo "Note: Environment is automatically set up on first run."
	@echo ""

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    VENV_PYTHON = ../.venv-docs/Scripts/python.exe
    VENV_ACTIVATE = ..\\.venv-docs\\Scripts\\activate
    VENV_ACTIVATE_PS = ..\\.venv-docs\\Scripts\\Activate.ps1
    RM_CMD = if exist _build rmdir /s /q _build
    ECHO_BLANK = @echo.
else
    VENV_PYTHON = ../.venv-docs/bin/python
    VENV_ACTIVATE = source ../.venv-docs/bin/activate
    RM_CMD = rm -rf _build
    ECHO_BLANK = @echo ""
endif

# Check if uv is installed
check-uv:
ifeq ($(OS),Windows_NT)
	@where uv >nul 2>&1 || ( \
		echo. && \
		echo ❌ uv is not installed or not in PATH && \
		echo. && \
		echo Please install uv: https://docs.astral.sh/uv/getting-started/installation/ && \
		exit 1 \
	)
else
	@command -v uv >/dev/null 2>&1 || ( \
		echo ""; \
		echo "❌ uv is not installed or not in PATH"; \
		echo ""; \
		echo "Please install uv: https://docs.astral.sh/uv/getting-started/installation/"; \
		echo ""; \
		exit 1; \
	)
endif

# Ensure docs environment exists and is up to date
ensure-docs-env: check-uv
	@if [ ! -f "$(VENV_PYTHON)" ]; then \
		echo "📦 Setting up docs environment with uv..."; \
		cd .. && uv venv .venv-docs && uv pip install --group docs --python .venv-docs; \
		echo "✅ Environment ready!"; \
	else \
		echo "🔄 Syncing docs dependencies (this ensures dependencies are up to date)..."; \
		cd .. && uv pip install --group docs --python .venv-docs; \
		echo "✅ Dependencies synced!"; \
	fi

docs-html: ensure-docs-env
	@echo "Building HTML documentation..."
	$(VENV_PYTHON) -m sphinx -b html . _build/html

docs-publish: ensure-docs-env
	@echo "Building HTML documentation for publication (fail on warnings)..."
	$(VENV_PYTHON) -m sphinx --fail-on-warning --builder html . _build/html

docs-clean:
	@echo "Cleaning built documentation..."
	$(RM_CMD)

docs-live: ensure-docs-env
	@echo "Starting live-reload server (sphinx-autobuild)..."
	$(VENV_PYTHON) -m sphinx_autobuild --port 8001 . _build/html
