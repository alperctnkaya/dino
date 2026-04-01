#!/bin/bash
#SBATCH --job-name=dino_vit_tiny_hotels-large-gpu      # Job name
#SBATCH -p large-gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes=2
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/dino_vit_tiny_hotels_%j.out      # STDOUT log
#SBATCH --error=logs/dino_vit_tiny_hotels_%j.err       # STDERR log

# Load modules
module load python3/3.13.3

# Activate virtualenv
source /SEAS/home/g45307115/venvs/transformers/bin/activate

# Move to working directory
cd /SEAS/home/g45307115/dino

export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH

python run_with_submitit.py --nodes 2 --ngpus 4 --arch vit_tiny --num_workers 16 --batch_size_per_gpu 64 --epochs 100 --output_dir /SEAS/home/g45307115/dino/vit_tiny_logs


torchrun --nproc_per_node=4 main_dino.py --arch vit_tiny --num_workers 16 --batch_size_per_gpu 64 --epochs 100 --output_dir /SEAS/home/g45307115/dino/vit_tiny_logs

python run_with_submitit.py --nodes 2 --ngpus 4 --arch vit_tiny --num_workers 16 --batch_size_per_gpu 64 --epochs 100 --output_dir /SEAS/home/g45307115/dino/vit_tiny_logs --partition large-gpu