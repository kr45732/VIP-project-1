#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH -N1 -n8 --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:59:99

module load anaconda3
source activate my_env

pip install transformers pillow torch pandas opencv-python matplotlib psutil pynvml

echo "Starting Benchmark..."
start=$(date +%s)
python main_single_gpu.py
end=$(date +%s)

echo "Execution Time: $(($end - $start)) seconds"
