#!/bin/bash

#SBATCH --job-name=resnet22_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=./out/resnet22_1.out

module purge
module load python/intel/3.8.6

## activate
source ./bin/activate

## install required packages
pip install -qqq -r requirements.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python main.py --model=resnet22_1 --num-workers=2 --optim=adadelta
