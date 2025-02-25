#!/usr/bin/env bash
# exit on error
set -o errexit

# Use pip caching
export PIP_CACHE_DIR="/root/.cache/pip"

# Install Python dependencies with optimizations
pip install --no-cache-dir --compile -r requirements.txt

# Download spaCy model in the background while creating directories
python -m spacy download en_core_web_sm &
SPACY_PID=$!

# Create necessary directories
mkdir -p uploads

# Wait for spaCy download to complete
wait $SPACY_PID
