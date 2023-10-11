#!/bin/bash

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x.lan@student.tue.nl
#SBATCH --partition=bme.gpustudent.q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:0
#SBATCH --job-name=eval
#SBATCH --output=/home/bme001/20225898/eval_out.out

module load cuda11.6/toolkit

cd /home/bme001/20225898/
srun python eval_server.py