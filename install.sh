#!/bin/bash

echo "🚀 Setting up DensityDependentMetricModel environment..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install JAX with Metal support for M1 Macs
echo "Installing JAX with Metal support..."
pip install jax==0.4.26 jaxlib==0.4.26 --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
pip install --no-deps jax-metal==0.1.1

echo "✅ Project environment is ready."
echo "Virtual environment is activated at: .venv"
