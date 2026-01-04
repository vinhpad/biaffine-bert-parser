#!/bin/bash

# Setup script for Biaffine BERT Dependency Parser

echo "Setting up Biaffine BERT Dependency Parser..."

# Create data directory and copy files
echo "Creating data directory..."
mkdir -p data

echo "Copying UD Vietnamese files..."
cp ../vi_vtb-ud-train.conllu data/ 2>/dev/null || echo "Warning: vi_vtb-ud-train.conllu not found"
cp ../vi_vtb-ud-dev.conllu data/ 2>/dev/null || echo "Warning: vi_vtb-ud-dev.conllu not found"
cp ../vi_vtb-ud-test.conllu data/ 2>/dev/null || echo "Warning: vi_vtb-ud-test.conllu not found"

# Create config directory
echo "Creating config directory..."
mkdir -p config

# Create experiments directory
echo "Creating experiments directory..."
mkdir -p experiments

echo "Setup completed!"
echo ""
echo "To install dependencies, run:"
echo "  pip install -r requirements.txt"
echo ""
echo "To train the model, run:"
echo "  cd src/training"
echo "  python train.py --config ../../config/default.yaml"
echo ""
echo "To run a quick demo, run:"
echo "  python demo.py"