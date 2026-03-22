#!/bin/bash
#SBATCH --job-name=jiayan_infer
#SBATCH --output=logs/jiayan_infer_%j.out
#SBATCH --error=logs/jiayan_infer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=compute

# Thay đổi env dựa trên cluster của bạn
# module load miniconda
# conda activate jiayan_env

# Chuyển directory theo project path thực tế nếu cần (như $HOME/WorkSpace/sinonom-ss/jiayan_inference)
# cd /path/to/jiayan_inference

mkdir -p logs

echo "Starting Jiayan Inference Job..."

# Chạy inference script và có thể test một data cụ thể được override
python infer.py --config config.yaml --test_path test.parquet

echo "Inference Finished."
