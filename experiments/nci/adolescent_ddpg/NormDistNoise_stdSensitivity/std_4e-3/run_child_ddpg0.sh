#!/bin/bash
#PBS -P sj53
#PBS -q gpuvolta
#PBS -l walltime=24:00:00
#PBS -l mem=24GB
#PBS -l jobfs=0
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -M david.timms@anu.edu.au
#PBS -l storage=scratch/sj53
#PBS -o out_ddpg0.txt
#PBS -e err_ddpg0.txt
#PBS -l software=python
module load python3/3.9.2
module load pytorch/1.9.0

python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent ddpg --folder_id DDPG/NOISE_NORMALDISTRIBUTION/STD_4e-3/DDPG0_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --noise_std 0.004 --seed 1 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent ddpg --folder_id DDPG/NOISE_NORMALDISTRIBUTION/STD_4e-3/DDPG0_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --noise_std 0.004 --seed 2 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent ddpg --folder_id DDPG/NOISE_NORMALDISTRIBUTION/STD_4e-3/DDPG0_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --noise_std 0.004 --seed 3 --debug 0

wait
