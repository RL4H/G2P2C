#!/bin/bash
#PBS -P eu59
#PBS -q gpuvolta
#PBS -l walltime=48:00:00
#PBS -l mem=24GB
#PBS -l jobfs=0
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -M david.timms@anu.edu.au
#PBS -l storage=scratch/sj53
#PBS -o out_TD36.txt
#PBS -e err_TD36.txt
#PBS -l software=python
module load python3/3.9.2
module load pytorch/1.9.0

python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id TD3/model1/normal_dist/with_penalty/std2e-1/TD36_1 --patient_id 6 --return_type average --action_type exponential --device cuda --pi_lr 1e-4 --vf_lr 1e-3 --soft_tau 0.001 --noise_application 1 --noise_model normal_dist --mu_penalty 1 --noise_std 2e-1 --seed 1 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id TD3/model1/normal_dist/with_penalty/std2e-1/TD36_2 --patient_id 6 --return_type average --action_type exponential --device cuda --pi_lr 1e-4 --vf_lr 1e-3 --soft_tau 0.001 --noise_application 1 --noise_model normal_dist --mu_penalty 1 --noise_std 2e-1 --seed 2 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id TD3/model1/normal_dist/with_penalty/std2e-1/TD36_3 --patient_id 6 --return_type average --action_type exponential --device cuda --pi_lr 1e-4 --vf_lr 1e-3 --soft_tau 0.001 --noise_application 1 --noise_model normal_dist --mu_penalty 1 --noise_std 2e-1 --seed 3 --debug 0
wait
