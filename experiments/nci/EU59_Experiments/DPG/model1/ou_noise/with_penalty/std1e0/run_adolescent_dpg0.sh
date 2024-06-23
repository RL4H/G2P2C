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
#PBS -o out_DPG0.txt
#PBS -e err_DPG0.txt
#PBS -l software=python
module load python3/3.9.2
module load pytorch/1.9.0

python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent dpg --folder_id DPG/model1/ou_noise/with_penalty/std1e0/DPG0_1 --patient_id 0 --return_type average --action_type exponential --device cuda --pi_lr 1e-4 --vf_lr 1e-3 --soft_tau 0.001 --noise_application 1 --noise_model ou_noise --mu_penalty 1 --noise_std 1e0 --seed 1 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent dpg --folder_id DPG/model1/ou_noise/with_penalty/std1e0/DPG0_2 --patient_id 0 --return_type average --action_type exponential --device cuda --pi_lr 1e-4 --vf_lr 1e-3 --soft_tau 0.001 --noise_application 1 --noise_model ou_noise --mu_penalty 1 --noise_std 1e0 --seed 2 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent dpg --folder_id DPG/model1/ou_noise/with_penalty/std1e0/DPG0_3 --patient_id 0 --return_type average --action_type exponential --device cuda --pi_lr 1e-4 --vf_lr 1e-3 --soft_tau 0.001 --noise_application 1 --noise_model ou_noise --mu_penalty 1 --noise_std 1e0 --seed 3 --debug 0
wait
