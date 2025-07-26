#!/bin/bash
# build.sh

# This tells the script to exit immediately if any command fails
set -o errexit

echo "--- Build Script Started ---"

# 1. Install all Python packages
echo "--- Installing Python Dependencies ---"
pip install -r requirements.txt

# 2. Download the model files using your proven wget method
echo "--- Downloading Model Files ---"
echo "Downloading model.mdl..."
wget -q --show-progress -O model.mdl "https://drive.google.com/uc?export=download&id=1T9tSdIm-8AEz0c6mJuFfLyBL75_lnsTU"

echo "Downloading model.mdl.wv.vectors.npy..."
wget -q --show-progress -O model.mdl.wv.vectors.npy "https://drive.google.com/uc?export=download&id=1z5n9L-2oS_YEh3qf-nkz3ugMM8oqpxGZ"

echo "Downloading model.mdl.syn1neg.npy..."
wget -q --show-progress -O model.mdl.syn1neg.npy "https://drive.google.com/uc?export=download&id=1uhu7bevYhCYZNLPdvupSuw_4X42_-smY"

# 3. Confirm files were downloaded
echo "--- Verifying Downloads ---"
ls -lh

# 4. Create the local data directory
mkdir -p data

echo "--- Build Script Finished Successfully ---"
