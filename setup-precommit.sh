#!/bin/bash
# Setup script for pre-commit hooks

echo "🔧 Setting up pre-commit hooks for NIDS-Verify research repository..."

# Add pre-commit to dev dependencies if not already present
echo "📦 Adding pre-commit to dev dependencies..."
uv add --dev pre-commit

# Install the pre-commit hooks using uv
echo "🪝 Installing pre-commit hooks..."
uv run pre-commit install

# Run pre-commit on all files (optional - can be slow)
read -p "🚀 Run pre-commit on all existing files? This may take a while and make many changes. (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔍 Running pre-commit on all files..."
    uv run pre-commit run --all-files
    echo "✅ Pre-commit setup complete!"
else
    echo "✅ Pre-commit hooks installed! They will run on your next commit."
fi

echo ""
echo "📋 Quick reference:"
echo "  - Run manually: uv run pre-commit run --all-files"
echo "  - Skip hooks: git commit --no-verify"
echo "  - Update hooks: uv run pre-commit autoupdate"
echo ""
echo "🎯 The configuration is research-friendly with:"
echo "  - Ruff for fast Python formatting and linting"
echo "  - Jupyter notebook cleaning and formatting"
echo "  - Security scanning with Bandit"
echo "  - Documentation checks (lenient 50% threshold)"
echo "  - Automatic exclusion of data/models/logs directories"
