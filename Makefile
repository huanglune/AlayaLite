# AlayaLite Makefile
# Usage: make help

.PHONY: help build build-debug build-release build-coverage configure test test-cpp test-py test-py-cov lint lint-tidy lint-commit-msg clean clean-all install dev-install wheel

# Configuration
BUILD_DIR         := build
TIDY_BUILD_DIR    := build-tidy
BUILD_TYPE        ?= Release
CMAKE_GENERATOR   ?= Ninja
CMAKE_FLAGS       := -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=ON
EXTRA_CMAKE_FLAGS ?=
PYTEST_FLAGS      := -v
PYTEST_COV_FLAGS  := --cov=python/src/alayalite --cov=app --cov-report=html
CTEST_FLAGS       := --output-on-failure
JOBS              ?= 60
PYTHON_VERSION    := 3.11
COMMIT_MSG        ?=

# Suppress "Entering/Leaving directory" messages from sub-makes
MAKEFLAGS     += --no-print-directory

# Colors
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(CYAN)AlayaLite Development Commands$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make build          # Build release version"
	@echo "  make test           # Run all tests"

# ============================================================================
# Build
# ============================================================================

build: BUILD_TYPE := Release
build: configure ## Build project (release mode)
	@cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) -j$(JOBS)

build-debug: BUILD_TYPE := Debug
build-debug: ## Build project in debug mode
	@cmake -G "$(CMAKE_GENERATOR)" -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FLAGS)
	@cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) -j$(JOBS)

build-release: BUILD_TYPE := Release
build-release: ## Build project in release mode
	@cmake -G "$(CMAKE_GENERATOR)" -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FLAGS)
	@cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) -j$(JOBS)

build-coverage: BUILD_TYPE := Debug
build-coverage: EXTRA_CMAKE_FLAGS := -DENABLE_COVERAGE=ON
build-coverage: ## Build with coverage instrumentation
	@cmake -G "$(CMAKE_GENERATOR)" -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FLAGS) $(EXTRA_CMAKE_FLAGS)
	@cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) -j$(JOBS)

configure: ## Only configure cmake (no build)
	@cmake -G "$(CMAKE_GENERATOR)" -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FLAGS) $(EXTRA_CMAKE_FLAGS)

# ============================================================================
# Test
# ============================================================================

test: test-cpp test-py ## Run all tests (C++ and pytest suites)

test-cpp: BUILD_TYPE := Release
test-cpp: build ## Run C++ tests with CTest
	@ctest --test-dir $(BUILD_DIR) $(CTEST_FLAGS) -C $(BUILD_TYPE) -j$(JOBS)

test-py: ## Run Python and API tests with pytest
	@uv run pytest $(PYTEST_FLAGS)

test-py-cov: ## Run Python and API tests with coverage
	@uv run pytest $(PYTEST_FLAGS) $(PYTEST_COV_FLAGS)

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run file-based pre-commit checks
	@uvx pre-commit run -a

lint-tidy: ## Run clang-tidy static analysis (clean rebuild with checks enabled)
	@cmake -G "$(CMAKE_GENERATOR)" -B $(TIDY_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS) -DENABLE_CLANG_TIDY=ON $(EXTRA_CMAKE_FLAGS)
	@cmake --build $(TIDY_BUILD_DIR) --config Release --clean-first -j$(JOBS)

lint-commit-msg: ## Validate commit message (use COMMIT_MSG="type: subject")
	@test -n "$(COMMIT_MSG)" || (echo "COMMIT_MSG is required" >&2; exit 1)
	@tmp_file=$$(mktemp); \
	trap 'rm -f "$$tmp_file"' EXIT; \
	printf '%s\n' "$(COMMIT_MSG)" > "$$tmp_file"; \
	uvx pre-commit run --hook-stage commit-msg conventional-pre-commit --commit-msg-filename "$$tmp_file"

# ============================================================================
# Install & Package
# ============================================================================

install: ## Install package without dev dependencies (non-editable)
	@uv sync --no-dev --no-editable

dev-install: ## Install package with dev dependencies
	@uv sync

wheel: ## Build wheel package (use PYTHON_VERSION=3.x to specify)
	@uvx --python $(PYTHON_VERSION) --from build pyproject-build

# ============================================================================
# Clean
# ============================================================================

clean: ## Remove build artifacts
	@rm -rf $(BUILD_DIR)
	@rm -rf $(TIDY_BUILD_DIR)
	@rm -rf dist *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

clean-all: clean ## Remove build artifacts and dependencies
	@rm -rf .venv
	@echo "Cleaned all."
