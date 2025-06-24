#!/bin/bash

set -e

DATA_DIR="data"

mkdir -p $DATA_DIR

BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for file in "${FILES[@]}"; do
    echo " > Downloading $file"
    curl "$BASE_URL/$file" -o "$DATA_DIR/$file" --no-progress-meter
    gunzip -f "$DATA_DIR/$file"
done
