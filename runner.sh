#!/bin/bash
find data/optimize/*/ -name "*.csv" | \
    parallel --jobs 50% --load 80% --progress \
    'echo "Running {/}..." && python 3.py {} > results/3/{/}'

# for file in data/optimize/misc/*.csv; do
#     basename=$(basename "$file")
#     echo "Running $basename..."
#     python 3.py "$file" > "results/3/$basename"
# done