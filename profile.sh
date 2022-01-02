#!/usr/bin/env bash

# Prepare results directory and paths
ts=$(date "+%Y%m%d_%H%M%S")
echo $ts
mkdir -p "./profiling/$ts"
stats_path="$(pwd)/profiling/$ts/output.pstats"
png_path="$(pwd)/profiling/$ts/output.png"
echo "Storing results in $(pwd)/profiling/$ts"

# Set profile timeout, default 60 seconds
profile_timeout=${1:-60}
echo "Running for $profile_timeout seconds"

# Profile 
./venv/bin/python -m cProfile -o $stats_path ./main.py & (sleep $profile_timeout && kill -s INT $!)

# Visualise
sleep 1 && ./venv/bin/gprof2dot -f pstats $stats_path | dot -Tpng -o $png_path

open $png_path