#!/bin/bash
find data/optimize/*/ -name "*.csv" | \
    parallel --jobs 50% --load 80% --progress \
    'echo "Running {/}..." && python 2.py {} > results/2/{/}'

# for file in data/optimize/process/*.csv; do
#     basename=$(basename "$file")
#     echo "Running $basename..."
#     python 2.py "$file" > "results/2/$basename"
# done