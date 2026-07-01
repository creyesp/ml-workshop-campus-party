.PHONY: post-create install-extra-tools test-tools init sync lint format type-check gitignore freeze dev-tools ensure-ipykernel test train

PYTHON_VERSION = 3.13

# Post-create command: install CLIs, verify tools, init project, and sync dependencies
post-create: install-extra-tools test-tools init ensure-ipykernel

# Install additional CLIs used in this workspace
install-extra-tools:
	@echo "Installing extra CLIs..."
	@bash .devcontainer/install_extra_tools.sh > /tmp/install-extra-tools.log 2>&1
	@echo "✓ Extra CLIs installation complete (log: /tmp/install-extra-tools.log)"

# Initialize Python environment with uv and create pyproject.toml if needed
init:
	@echo "Initializing Python environment..."
	@(uv python install ${PYTHON_VERSION} && \
	if [ ! -f pyproject.toml ]; then \
		if [ "${UV_INIT_BARE}" = "true" ]; then \
			uv init --bare --python ${PYTHON_VERSION}; \
		else \
			uv init --python ${PYTHON_VERSION}; \
		fi; \
	fi) > /tmp/init.log 2>&1
	@echo "✓ Initialization complete (log: /tmp/init.log)"

# Ensure ipykernel is installed as a dev dependency (optional)
ensure-ipykernel:
	@if [ "${INSTALL_IPYKERNEL}" = "true" ]; then \
		( \
			if ! uv run python -c "import ipykernel" >/dev/null 2>&1; then \
				echo "Installing ipykernel (dev dependency)..."; \
				uv add --dev ipykernel; \
			else \
				echo "ipykernel already present; skipping installation."; \
			fi \
		) > /tmp/jupyter-kernel.log 2>&1; \
		echo "✓ ipykernel check complete (log: /tmp/jupyter-kernel.log)"; \
	else \
		echo "Skipping ipykernel installation (INSTALL_IPYKERNEL != true)"; \
	fi

# Verify installed tools
test-tools:
	@echo "Running tool verification..."
	@bash .devcontainer/test_tools.sh > /tmp/test-tools.log 2>&1
	@echo "✓ Tool verification complete (log: /tmp/test-tools.log)"

# Sync dependencies with uv
sync:
	@echo "Syncing dependencies..."
	@uv sync > /tmp/uv-sync.log 2>&1
	@echo "✓ Dependency sync complete (log: /tmp/uv-sync.log)"

# Run ruff linter
lint:
	@uv run ruff check .

# Run ruff formatter
format:
	@uv run ruff format .

# Run mypy type checker
type-check:
	@uv run mypy src

# Run test suite
test:
	@uv run pytest

# Reentrenar y congelar el modelo XGBoost ganador en artifacts/model_v1
train:
	@uv run floodit-train

# Download Python .gitignore from GitHub
gitignore:
	@if [ -f .gitignore ]; then \
		echo "⚠️  .gitignore already exists, skipping"; \
	else \
		( \
			echo "📥 Downloading Python .gitignore from GitHub..."; \
			curl -fsSL https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore -o .gitignore; \
			echo "✅ .gitignore created"; \
		) > /tmp/gitignore.log 2>&1; \
		echo "✓ .gitignore download complete (log: /tmp/gitignore.log)"; \
	fi

# Install dev tools (ruff, mypy)
dev-tools:
	@echo "Installing dev tools (ruff, mypy)..."
	@uv add --dev ruff mypy > /tmp/dev-tools.log 2>&1
	@echo "✓ Dev tools installed (log: /tmp/dev-tools.log)"

# Freeze dependencies to tmp folder
freeze:
	@echo "Freezing dependencies..."
	@echo "# Generated on $$(date)" > /tmp/requirements.txt
	@uv pip freeze >> /tmp/requirements.txt
	@echo "✓ Dependencies frozen (log: /tmp/requirements.txt)"
