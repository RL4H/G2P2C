python run_RL_agent.py --agent multi_ppo --folder_id mppo/MP0_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 &
python run_RL_agent.py --agent multi_ppo --folder_id mppo/MP0_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 &
python run_RL_agent.py --agent multi_ppo --folder_id mppo/MP0_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 &

python run_RL_agent.py --agent ppo --folder_id ppo/P0_1 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 &
python run_RL_agent.py --agent ppo --folder_id ppo/P0_2 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 &
python run_RL_agent.py --agent ppo --folder_id ppo/P0_3 --patient_id 0 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 &

python run_RL_agent.py --agent multi_ppo --folder_id mppo/MP24_1 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 &
python run_RL_agent.py --agent multi_ppo --folder_id mppo/MP24_2 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 &
python run_RL_agent.py --agent multi_ppo --folder_id mppo/MP24_3 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 &

python run_RL_agent.py --agent ppo --folder_id ppo/P24_1 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 1 --debug 0 &
python run_RL_agent.py --agent ppo --folder_id ppo/P24_2 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 2 --debug 0 &
python run_RL_agent.py --agent ppo --folder_id ppo/P24_3 --patient_id 24 --return_type average --action_type exponential --device cuda:0 --seed 3 --debug 0 &

wait
