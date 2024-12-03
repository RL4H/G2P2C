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
#PBS -o out_TD32.txt
#PBS -e err_TD32.txt
#PBS -l software=python
module load python3/3.9.2
module load pytorch/1.9.0

python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PER_Study/Per_Proportional/TD3/alpha_05/beta_0/TD32_1 --patient_id 2 --replay_buffer_type per_proportional --replay_buffer_alpha 0.5 --replay_buffer_beta 0.0  --seed 1 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PER_Study/Per_Proportional/TD3/alpha_05/beta_0/TD32_2 --patient_id 2 --replay_buffer_type per_proportional --replay_buffer_alpha 0.5 --replay_buffer_beta 0.0  --seed 2 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PER_Study/Per_Proportional/TD3/alpha_05/beta_0/TD32_3 --patient_id 2 --replay_buffer_type per_proportional --replay_buffer_alpha 0.5 --replay_buffer_beta 0.0  --seed 3 --debug 0
wait
