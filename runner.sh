#!/bin/bash
for file in data/optimize/*/*.csv; do
    basename=$(basename "$file")
    echo "Running $basename..."
    python 2.py "$file" > "results/2/$basename"
done