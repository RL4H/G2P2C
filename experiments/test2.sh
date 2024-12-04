# batch
python run_RL_agent.py --agent ppo --folder_id mppo/MP0_001_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.001 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP0_001_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.001 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP0_001_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.001 --variance_type current &

python run_RL_agent.py --agent ppo --folder_id mppo/MP24_001_1 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.001 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP24_001_2 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.001 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP24_001_3 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.001 --variance_type current &

python run_RL_agent.py --agent ppo --folder_id mppo/MP0_01_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.01 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP0_01_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.01 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP0_01_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.01 --variance_type current &

python run_RL_agent.py --agent ppo --folder_id mppo/MP24_01_1 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.01 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP24_01_2 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.01 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP24_01_3 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.01 --variance_type current &

python run_RL_agent.py --agent ppo --folder_id mppo/MP0_1_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.1 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP0_1_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.1 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP0_1_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.1 --variance_type current &

python run_RL_agent.py --agent ppo --folder_id mppo/MP24_1_1 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.1 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP24_1_2 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.1 --variance_type current &
python run_RL_agent.py --agent ppo --folder_id mppo/MP24_1_3 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.1 --variance_type current &

# total
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_001_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.001 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_001_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.001 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_001_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.001 --variance_type total &

python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_001_1 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.001 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_001_2 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.001 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_001_3 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.001 --variance_type total &

python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_01_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.01 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_01_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.01 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_01_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.01 --variance_type total &

python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_01_1 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.01 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_01_2 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.01 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_01_3 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.01 --variance_type total &

python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_1_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.1 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_1_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.1 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_1_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.1 --variance_type total &

python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_1_1 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.1 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_1_2 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.1 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_1_3 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.1 --variance_type total &

python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_5_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.5 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_5_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.5 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP0_5_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.5 --variance_type total &

python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_5_1 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 --variance_weight 0.5 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_5_2 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 --variance_weight 0.5 --variance_type total &
python run_RL_agent.py --agent ppo --folder_id mppo_t/MP24_5_3 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 --variance_weight 0.5 --variance_type total &

wait
