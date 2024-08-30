#!/bin/bash
#PBS -P ny83
#PBS -q gpuvolta
#PBS -l walltime=48:00:00
#PBS -l mem=24GB
#PBS -l jobfs=0
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -M david.timms@anu.edu.au
#PBS -l storage=scratch/sj53
#PBS -o out_TD30.txt
#PBS -e err_TD30.txt
#PBS -l software=python
module load python3/3.9.2
module load pytorch/1.9.0

python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PER_Study/Per_Proportional/TD3/alpha_075/beta_075/TD30_1 --patient_id 0 --replay_buffer_type per_proportional --replay_buffer_alpha 0.75 --replay_buffer_beta 0.75  --seed 1 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PER_Study/Per_Proportional/TD3/alpha_075/beta_075/TD30_2 --patient_id 0 --replay_buffer_type per_proportional --replay_buffer_alpha 0.75 --replay_buffer_beta 0.75  --seed 2 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PER_Study/Per_Proportional/TD3/alpha_075/beta_075/TD30_3 --patient_id 0 --replay_buffer_type per_proportional --replay_buffer_alpha 0.75 --replay_buffer_beta 0.75  --seed 3 --debug 0
wait
