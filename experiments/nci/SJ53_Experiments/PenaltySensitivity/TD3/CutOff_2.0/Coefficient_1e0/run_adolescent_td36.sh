#!/bin/bash
#PBS -P sj53
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

python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PenaltyTermSensitivity/TD3/cutoff_20/coefficient1e0/TD36_1 --patient_id 6 --return_type average --action_type exponential --device cuda --pi_lr 1e-3 --vf_lr 1e-3 --soft_tau 0.005 --noise_application 1 --noise_model ou_noise --mu_penalty 1 --noise_std 1e0 --action_penalty_limit 2.0 --action_penalty_coef 1e0 --seed 1 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PenaltyTermSensitivity/TD3/cutoff_20/coefficient1e0/TD36_2 --patient_id 6 --return_type average --action_type exponential --device cuda --pi_lr 1e-3 --vf_lr 1e-3 --soft_tau 0.005 --noise_application 1 --noise_model ou_noise --mu_penalty 1 --noise_std 1e0 --action_penalty_limit 2.0 --action_penalty_coef 1e0 --seed 2 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PenaltyTermSensitivity/TD3/cutoff_20/coefficient1e0/TD36_3 --patient_id 6 --return_type average --action_type exponential --device cuda --pi_lr 1e-3 --vf_lr 1e-3 --soft_tau 0.005 --noise_application 1 --noise_model ou_noise --mu_penalty 1 --noise_std 1e0 --action_penalty_limit 2.0 --action_penalty_coef 1e0 --seed 3 --debug 0
wait
