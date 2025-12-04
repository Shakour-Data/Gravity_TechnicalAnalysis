# Makefile for Gravity Technical Analysis
# Quick commands for common development tasks

.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
UVICORN := uvicorn
BLACK := black
RUFF := ruff
MYPY := mypy
ISORT := isort

SRC_DIR := src
TEST_DIR := tests
APP_MODULE := src.gravity_tech.api.main:app

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m

##@ Help

.PHONY: help
help: ## نمایش این راهنما
	@echo "$(COLOR_BOLD)Gravity Technical Analysis - Makefile Commands$(COLOR_RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(COLOR_BOLD)Usage:$(COLOR_RESET)\n  make $(COLOR_BLUE)<target>$(COLOR_RESET)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(COLOR_BLUE)%-20s$(COLOR_RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(COLOR_BOLD)%s$(COLOR_RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation & Setup

.PHONY: install
install: ## نصب تمام dependencies
	@echo "$(COLOR_GREEN)Installing dependencies...$(COLOR_RESET)"
	$(PIP) install -e ".[dev,ml,enterprise]"

.PHONY: install-prod
install-prod: ## نصب فقط dependencies production
	@echo "$(COLOR_GREEN)Installing production dependencies...$(COLOR_RESET)"
	$(PIP) install -e "."

.PHONY: install-dev
install-dev: ## نصب فقط dependencies development
	@echo "$(COLOR_GREEN)Installing development dependencies...$(COLOR_RESET)"
	$(PIP) install -e ".[dev]"

.PHONY: setup-db
setup-db: ## راه‌اندازی دیتابیس
	@echo "$(COLOR_GREEN)Setting up database...$(COLOR_RESET)"
	$(PYTHON) setup_database.py

##@ Development

.PHONY: run
run: ## اجرای development server
	@echo "$(COLOR_GREEN)Starting development server...$(COLOR_RESET)"
	$(UVICORN) $(APP_MODULE) --reload --host 0.0.0.0 --port 8000

.PHONY: run-prod
run-prod: ## اجرای production server
	@echo "$(COLOR_GREEN)Starting production server...$(COLOR_RESET)"
	$(UVICORN) $(APP_MODULE) --host 0.0.0.0 --port 8000 --workers 8

##@ Testing

.PHONY: test
test: ## اجرای تمام تست‌ها
	@echo "$(COLOR_GREEN)Running all tests...$(COLOR_RESET)"
	$(PYTEST) $(TEST_DIR) -v

.PHONY: test-unit
test-unit: ## اجرای unit tests
	@echo "$(COLOR_GREEN)Running unit tests...$(COLOR_RESET)"
	$(PYTEST) $(TEST_DIR)/unit -v

.PHONY: test-integration
test-integration: ## اجرای integration tests
	@echo "$(COLOR_GREEN)Running integration tests...$(COLOR_RESET)"
	$(PYTEST) $(TEST_DIR)/integration -v

.PHONY: test-cov
test-cov: ## اجرای تست‌ها با coverage report
	@echo "$(COLOR_GREEN)Running tests with coverage...$(COLOR_RESET)"
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR)/gravity_tech --cov-report=html --cov-report=term-missing

.PHONY: test-watch
test-watch: ## اجرای تست‌ها در حالت watch
	@echo "$(COLOR_GREEN)Running tests in watch mode...$(COLOR_RESET)"
	$(PYTEST) $(TEST_DIR) -v --looponfail

##@ Code Quality

.PHONY: lint
lint: ## بررسی کد با linters
	@echo "$(COLOR_GREEN)Running linters...$(COLOR_RESET)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)
	$(MYPY) $(SRC_DIR)

.PHONY: lint-fix
lint-fix: ## رفع خودکار مشکلات lint
	@echo "$(COLOR_GREEN)Fixing linting issues...$(COLOR_RESET)"
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR)

.PHONY: format
format: ## فرمت کردن کد
	@echo "$(COLOR_GREEN)Formatting code...$(COLOR_RESET)"
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(ISORT) $(SRC_DIR) $(TEST_DIR)

.PHONY: format-check
format-check: ## بررسی فرمت کد بدون تغییر
	@echo "$(COLOR_GREEN)Checking code format...$(COLOR_RESET)"
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)

.PHONY: type-check
type-check: ## بررسی type hints
	@echo "$(COLOR_GREEN)Running type checker...$(COLOR_RESET)"
	$(MYPY) $(SRC_DIR)

.PHONY: quality
quality: format lint test ## اجرای تمام بررسی‌های کیفیت
	@echo "$(COLOR_GREEN)✓ All quality checks passed!$(COLOR_RESET)"

##@ Cleaning

.PHONY: clean
clean: ## پاکسازی فایل‌های موقت
	@echo "$(COLOR_YELLOW)Cleaning temporary files...$(COLOR_RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build

.PHONY: clean-all
clean-all: clean ## پاکسازی همه چیز (شامل venv)
	@echo "$(COLOR_YELLOW)Cleaning everything...$(COLOR_RESET)"
	rm -rf venv
	rm -rf .venv
	rm -rf ml_models/*.pkl
	rm -rf data/processed/*

##@ Docker

.PHONY: docker-build
docker-build: ## ساخت Docker image
	@echo "$(COLOR_GREEN)Building Docker image...$(COLOR_RESET)"
	docker build -t gravity-tech-analysis:latest .

.PHONY: docker-run
docker-run: ## اجرای Docker container
	@echo "$(COLOR_GREEN)Running Docker container...$(COLOR_RESET)"
	docker run -p 8000:8000 gravity-tech-analysis:latest

.PHONY: docker-compose-up
docker-compose-up: ## اجرای docker-compose
	@echo "$(COLOR_GREEN)Starting services with docker-compose...$(COLOR_RESET)"
	docker-compose up -d

.PHONY: docker-compose-down
docker-compose-down: ## متوقف کردن docker-compose
	@echo "$(COLOR_YELLOW)Stopping docker-compose services...$(COLOR_RESET)"
	docker-compose down

##@ Documentation

.PHONY: docs
docs: ## ایجاد مستندات
	@echo "$(COLOR_GREEN)Building documentation...$(COLOR_RESET)"
	cd docs && $(MAKE) html

.PHONY: docs-serve
docs-serve: ## سرو کردن مستندات
	@echo "$(COLOR_GREEN)Serving documentation...$(COLOR_RESET)"
	cd docs/_build/html && $(PYTHON) -m http.server 8080

##@ Database

.PHONY: db-migrate
db-migrate: ## اجرای migrations
	@echo "$(COLOR_GREEN)Running database migrations...$(COLOR_RESET)"
	alembic upgrade head

.PHONY: db-rollback
db-rollback: ## برگشت به migration قبلی
	@echo "$(COLOR_YELLOW)Rolling back database migration...$(COLOR_RESET)"
	alembic downgrade -1

.PHONY: db-reset
db-reset: ## ریست کامل دیتابیس
	@echo "$(COLOR_YELLOW)Resetting database...$(COLOR_RESET)"
	alembic downgrade base
	alembic upgrade head

##@ Utilities

.PHONY: version
version: ## نمایش ورژن
	@echo "$(COLOR_BLUE)Gravity Technical Analysis$(COLOR_RESET)"
	@cat VERSION

.PHONY: check
check: format-check lint type-check test ## بررسی کامل پروژه
	@echo "$(COLOR_GREEN)✓ All checks passed!$(COLOR_RESET)"

.PHONY: init
init: install setup-db ## راه‌اندازی اولیه پروژه
	@echo "$(COLOR_GREEN)✓ Project initialized successfully!$(COLOR_RESET)"
	@echo "Run 'make run' to start the server"

.PHONY: update
update: ## آپدیت dependencies
	@echo "$(COLOR_GREEN)Updating dependencies...$(COLOR_RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -e ".[dev,ml,enterprise]"

##@ CI/CD

.PHONY: ci
ci: lint test ## اجرای CI pipeline
	@echo "$(COLOR_GREEN)✓ CI checks passed!$(COLOR_RESET)"

.PHONY: pre-commit
pre-commit: format lint ## اجرای بررسی‌های pre-commit
	@echo "$(COLOR_GREEN)✓ Pre-commit checks passed!$(COLOR_RESET)"
