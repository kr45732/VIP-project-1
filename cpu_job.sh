#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH -N1 -n8
#SBATCH --time=00:30:00

module load anaconda3
source activate my_env

pip install transformers pillow torch pandas opencv-python matplotlib psutil

echo "Starting Benchmark..."
start=$(date +%s)
python main.py
end=$(date +%s)

echo "Execution Time: $(($end - $start)) seconds"
