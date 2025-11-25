#!/bin/bash
# Setup script for databricks-langchain-agent project using Poetry

set -e

echo "Setting up databricks-langchain-agent project with Poetry..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo ""
    echo "Please add Poetry to your PATH or restart your terminal, then run this script again."
    echo "Or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 1
fi

echo "Poetry is installed. Version:"
poetry --version

# Configure Poetry to create virtual environment in project directory
poetry config virtualenvs.in-project true

# Install dependencies
echo "Installing dependencies with Poetry..."
poetry install

echo ""
echo "Setup complete! To activate the virtual environment, run:"
echo "  poetry shell"
echo ""
echo "Or to run commands within the Poetry environment:"
echo "  poetry run python agent.py"
echo ""
echo "To add new dependencies:"
echo "  poetry add <package-name>"
echo ""
echo "To add development dependencies:"
echo "  poetry add --group dev <package-name>"
