#!/bin/bash
# setup_env.sh
# Professional environment setup for Flax Stable Diffusion fine-tuning on TPU/GPU

set -e  # Better error handling

echo "=========================================="
echo " Flax Stable Diffusion - Environment Setup"
echo "=========================================="

# Use current user instead of hardcoded name
CURRENT_USER="${USER:-$(whoami)}"
echo "Running as user: $CURRENT_USER"

# Ensure required env vars are set
: "${SAVE_DIR:?Error: SAVE_DIR environment variable is not set. Please export SAVE_DIR=gs://your-bucket/path}"

echo "===== Testing GCS Connection ====="
mkdir -p ./sd-full-finetuned

# Test write
echo "GCS connection test - $(date)" > ./sd-full-finetuned/connection_test.txt
gsutil cp ./sd-full-finetuned/connection_test.txt "$SAVE_DIR/full/test/"
echo "✓ GCS write test passed"

# Download original weights (assuming they were uploaded once)
echo "===== Downloading Pretrained Weights from GCS ====="
if gsutil -m ls "$SAVE_DIR/full/" &> /dev/null; then
    gsutil -m cp -r "$SAVE_DIR/full/*" ./sd-full-finetuned/
    echo "✓ Weights downloaded"
else
    echo "Warning: No weights found at $SAVE_DIR/original_weights/"
    echo "    Please upload your base model weights first."
fi

echo "===== Installing Python Dependencies ====="
pip install --quiet --upgrade pip

pip install --quiet --upgrade \
    jax[tpu] \
    flax \
    optax \
    transformers==4.57.3 \
    datasets \
    diffusers==0.36.0 \
    torch torchvision \
    Pillow \
    matplotlib \
    grain \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "✓ Dependencies installed"
