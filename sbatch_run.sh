#!/bin/sh
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=120:00:00
#SBATCH --output=./feature_logs.out
#SBATCH --job-name="features_extractions"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module load singularity
module load pytorch
ls
ml

USER=haitham.mohameda
echo "LOAD CONDA MODULE & ENV"
module load conda
conda activate ingest

python3 OME_Feature_Organization.py
