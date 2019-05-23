#!/bin/bash
#SBATCH --array=0-3
#SBATCH -p tornado-k40
#SBATCH -J kfac_parallel
#SBATCH -o /home/ikntcit2/kell/out/job_%j_%a.out 
#SBATCH -e /home/ikntcit2/kell/err/job_%j_%a.err
module purge

module load python/3.6.5
module load nvidia/cuda/9.2
module load nvidia/cudnn/7.3.1

source $HOME/thesisvenv/bin/activate

for i in {0..1}; do

CUDA_VISIBLE_DEVICES=${i} L_FILENAME=tasks/task${SLURM_ARRAY_TASK_ID}_${i}.pkl python qfac_agent.py > $HOME/out/${SLURM_ARRAY_TASK_ID}_${i}.out 2>$HOME/err/${SLURM_ARRAY_TASK_ID}_${i}.err &

done

wait

