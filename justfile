# Justfile for iplot-mpl
# Install just: https://github.com/casey/just#installation
# Windows: winget install Casey.Just

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

venv := ".venv"
python := venv / "Scripts" / "python.exe"
pip := venv / "Scripts" / "pip.exe"

# Default recipe - show help
default:
    @just --list

# Create virtual environment if it doesn't exist
venv:
    @if (!(Test-Path "{{venv}}")) { \
        Write-Host "Creating virtual environment..." -ForegroundColor Green; \
        python -m venv {{venv}} \
    } else { \
        Write-Host "Virtual environment already exists" -ForegroundColor Yellow \
    }

# Install package with dev dependencies
install: venv
    {{pip}} install -e ".[dev]"

# Run tests with coverage
test: install
    {{python}} -m pytest -v --cov=interactive_figure --cov-report=term-missing

# Run tests without coverage (faster)
test-fast: install
    {{python}} -m pytest -v

# Run linter checks
lint: install
    {{python}} -m ruff check .
    {{python}} -m ruff format --check .

# Format code
format: install
    {{python}} -m ruff format .
    {{python}} -m ruff check --fix .

# Run type checker
typecheck: install
    {{python}} -m mypy interactive_figure

# Run all CI checks (lint + typecheck + test)
ci: lint typecheck test

# Clean build artifacts
clean:
    if (Test-Path "{{venv}}") { Remove-Item -Recurse -Force "{{venv}}" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "*.egg-info") { Remove-Item -Recurse -Force "*.egg-info" }
    if (Test-Path ".pytest_cache") { Remove-Item -Recurse -Force ".pytest_cache" }
    if (Test-Path ".mypy_cache") { Remove-Item -Recurse -Force ".mypy_cache" }
    if (Test-Path ".coverage") { Remove-Item -Force ".coverage" }
    if (Test-Path "htmlcov") { Remove-Item -Recurse -Force "htmlcov" }
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Build package
build: clean
    {{python}} -m pip install build
    {{python}} -m build
