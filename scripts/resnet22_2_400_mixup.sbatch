#!/bin/bash

#SBATCH --job-name=resnet22_2_400_mixup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=./out/resnet22_2_400_mixup.out

module purge
module load python/intel/3.8.6

## activate
source ./bin/activate

## install required packages
pip install -qqq -r requirements.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python main.py --model=resnet22_2 --num-workers=2 --optim=adadelta --batch-size=64 --lr=0.1 --epochs=400 --with-mixup=True --summary=True
