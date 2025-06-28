# experiments/run_extended_dmms_training.py

import os
from pathlib import Path
import sys
import torch
import json
import shutil
import random
import numpy as np
import traceback
import argparse
from pprint import pprint
from decouple import config
import uvicorn
import threading
import time
import datetime

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.core import set_logger, get_patient_env
from utils.options import Options
from Sim_CLI.main_finetune import app as fastapi_app_finetune
from Sim_CLI.main_finetune import setup_app_state_for_finetuning

import warnings
warnings.simplefilter('ignore', Warning)

def setup_extended_training_folders(args):
    """Setup folders for extended training with unique naming based on episode count"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = MAIN_PATH + "/results/" + f"extended_dmms_{args.extended_episodes}ep_{timestamp}"
    
    if os.path.isdir(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    
    os.makedirs(LOG_DIR + '/checkpoints')
    os.makedirs(LOG_DIR + '/training/data')
    os.makedirs(LOG_DIR + '/training/plots')
    os.makedirs(LOG_DIR + '/testing/data')
    os.makedirs(LOG_DIR + '/testing/plots')
    os.makedirs(LOG_DIR + '/code')
    os.makedirs(LOG_DIR + '/evaluation')
    
    args.experiment_dir = LOG_DIR
    set_logger(LOG_DIR)
    
    print(f"INFO: Extended training folder created: {LOG_DIR}")
    return args

def copy_folder(src, dst):
    for folders, subfolders, filenames in os.walk(src):
        for filename in filenames:
            shutil.copy(os.path.join(folders, filename), dst)

def setup_dmms_dirs(args):
    """Create directories for DMMS-based training runs."""
    dmms_root = Path(args.main_dir) / "results" / "dmms_runs"
    dmms_root.mkdir(parents=True, exist_ok=True)
    args.dmms_io_root = str(dmms_root)
    return args

def set_extended_training_parameters(args):
    """Set parameters optimized for extended DMMS training"""
    agent = None
    device = args.device

    if args.agent == 'g2p2c':
        from agents.g2p2c.g2p2c import G2P2C
        from agents.g2p2c.parameters import set_args, set_args_dmms_debug

        args.aux_mode = getattr(args, 'aux_mode', 'dual')
        args.use_planning = getattr(args, 'use_planning', 'yes')
        
        # Force DMMS debug mode for extended training
        print("INFO: Using DMMS.R debug hyperparameters for extended episode training")
        args = set_args_dmms_debug(args)
        
        # Override with custom hyperparameters if provided
        if hasattr(args, 'custom_pi_lr'):
            args.pi_lr = args.custom_pi_lr
            print(f"INFO: Overriding policy learning rate to: {args.pi_lr}")
        if hasattr(args, 'custom_vf_lr'):
            args.vf_lr = args.custom_vf_lr
            print(f"INFO: Overriding value function learning rate to: {args.vf_lr}")
        if hasattr(args, 'custom_batch_size'):
            args.batch_size = args.custom_batch_size
            print(f"INFO: Overriding batch size to: {args.batch_size}")
        if hasattr(args, 'custom_n_step'):
            args.n_step = args.custom_n_step
            print(f"INFO: Overriding n_step to: {args.n_step}")
        if hasattr(args, 'custom_grad_clip'):
            args.grad_clip = args.custom_grad_clip
            print(f"INFO: Overriding gradient clipping to: {args.grad_clip}")
        
        # Override with extended training specific parameters
        args.n_training_episodes = args.extended_episodes
        args.n_testing_episodes = 1  # Test once at the end
        args.save_freq = max(10, args.extended_episodes // 10)  # Save every 10% of episodes
        
        # Setup folders after parameter configuration
        args = setup_extended_training_folders(args)

        # Load from checkpoint for fine-tuning
        if args.fine_tune_from_checkpoint and args.fine_tune_from_checkpoint > 0:
            print(f"INFO: Loading G2P2C agent from episode {args.fine_tune_from_checkpoint} for extended training.")
            actor_checkpoint_path = os.path.join(MAIN_PATH, 'results', 'test', 'checkpoints', f'episode_{args.fine_tune_from_checkpoint}_Actor.pth')
            critic_checkpoint_path = os.path.join(MAIN_PATH, 'results', 'test', 'checkpoints', f'episode_{args.fine_tune_from_checkpoint}_Critic.pth')

            if not os.path.exists(actor_checkpoint_path):
                raise FileNotFoundError(f"Actor checkpoint not found: {actor_checkpoint_path}")
            if not os.path.exists(critic_checkpoint_path):
                raise FileNotFoundError(f"Critic checkpoint not found: {critic_checkpoint_path}")

            print(f"INFO: Loading Actor from: {actor_checkpoint_path}")
            print(f"INFO: Loading Critic from: {critic_checkpoint_path}")
            
            agent = G2P2C(args, device, load=True, path1=actor_checkpoint_path, path2=critic_checkpoint_path)
            print(f"INFO: Successfully loaded G2P2C agent for extended training.")
        else:
            print(f"INFO: Initializing new G2P2C agent for extended training from scratch.")
            agent = G2P2C(args, device, load=False, path1='', path2='')
        
        # Copy agent code
        dst = os.path.join(args.experiment_dir, 'code', 'g2p2c_agent_code')
        os.makedirs(dst, exist_ok=True)
        copy_folder(os.path.join(MAIN_PATH, 'agents', 'g2p2c'), dst)
        try:
            shutil.copy2(os.path.join(MAIN_PATH, 'agents', 'g2p2c', 'parameters.py'), os.path.join(dst, 'parameters.py'))
        except FileNotFoundError:
            print(f"WARNING: agents/g2p2c/parameters.py not found for copying.")
            
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")

    return args, agent, device

class UvicornServer(threading.Thread):
    def __init__(self, app, host="127.0.0.1", port=5000, log_level="info"):
        super().__init__(daemon=True)
        self.config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
        self.server = uvicorn.Server(self.config)

    def run(self):
        print(f"INFO: [UvicornThread] Starting Uvicorn server on http://{self.config.host}:{self.config.port}")
        try:
            self.server.run()
        except Exception as e:
            print(f"ERROR: [UvicornThread] Uvicorn server crashed: {e}")

def save_baseline_performance(args, agent):
    """Save baseline performance before extended training"""
    baseline_file = os.path.join(args.experiment_dir, 'evaluation', 'baseline_performance.json')
    
    baseline_data = {
        "initial_checkpoint": args.fine_tune_from_checkpoint,
        "extended_episodes_planned": args.extended_episodes,
        "training_start_time": datetime.datetime.now().isoformat(),
        "agent_parameters": {
            "pi_lr": args.pi_lr,
            "vf_lr": args.vf_lr,
            "n_step": args.n_step,
            "n_training_workers": args.n_training_workers,
            "normalize_reward": args.normalize_reward
        }
    }
    
    os.makedirs(os.path.dirname(baseline_file), exist_ok=True)
    with open(baseline_file, 'w') as f:
        json.dump(baseline_data, f, indent=4)
    
    print(f"INFO: Baseline performance data saved to {baseline_file}")

def main():
    # Parse custom arguments for extended training
    parser = argparse.ArgumentParser(description='Extended DMMS Training with Increased Episodes')
    parser.add_argument('--extended_episodes', type=int, default=30, 
                       help='Number of episodes for extended training (default: 30)')
    parser.add_argument('--fine_tune_from_checkpoint', type=int, default=195,
                       help='Episode number of checkpoint to load (default: 195)')
    parser.add_argument('--dmms_debug_mode', action='store_true',
                       help='Enable DMMS debug mode (recommended for extended training)')
    parser.add_argument('--pi_lr', type=float, default=1e-5,
                       help='Policy learning rate (default: 1e-5)')
    parser.add_argument('--vf_lr', type=float, default=1e-5,
                       help='Value function learning rate (default: 1e-5)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for training (default: 512)')
    parser.add_argument('--dmms_cfg', type=str,
                       help='Path to DMMS configuration file')
    
    # Parse known args first, then pass the rest to Options
    extended_args, remaining_args = parser.parse_known_args()
    
    # Create a modified sys.argv for Options().parse()
    original_argv = sys.argv
    
    # Use provided dmms_cfg if specified, otherwise use default
    dmms_cfg_path = extended_args.dmms_cfg if extended_args.dmms_cfg else 'C:\\Users\\user\\Documents\\DMMS.R\\config\\Sim_CLI_1.2.xml'
    
    sys.argv = [sys.argv[0]] + remaining_args + [
        '--agent', 'g2p2c',
        '--sim', 'dmms',
        '--dmms_debug_mode',
        '--fine_tune_from_checkpoint', str(extended_args.fine_tune_from_checkpoint),
        '--folder_id', f'extended_dmms_{extended_args.extended_episodes}ep',
        '--dmms_exe', 'C:\\Program Files\\The Epsilon Group\\DMMS.R\\simulator\\DMMS.R.exe',
        '--dmms_cfg', dmms_cfg_path,
        '--device', 'cpu',
        '--debug', '1',
        '--return_type', 'average',
        '--action_type', 'exponential'
    ]
    
    try:
        args = Options().parse()
        
        # Add extended training specific parameters
        args.extended_episodes = extended_args.extended_episodes
        args.dmms_debug_mode = True  # Force enable for extended training
        
        # Add custom hyperparameters from command line for DMMS.R fine-tuning
        args.custom_pi_lr = extended_args.pi_lr
        args.custom_vf_lr = extended_args.vf_lr
        args.custom_batch_size = extended_args.batch_size
        
        # Ensure main_dir is set correctly
        if not hasattr(args, 'main_dir'):
            args.main_dir = MAIN_PATH
        
        print(f"INFO: Extended Training Configuration:")
        print(f"  - Episodes to train: {args.extended_episodes}")
        print(f"  - Policy learning rate: {args.custom_pi_lr}")
        print(f"  - Value function learning rate: {args.custom_vf_lr}")
        print(f"  - Batch size: {args.custom_batch_size}")
        print(f"  - DMMS config file: {dmms_cfg_path}")
        print(f"  - Starting from checkpoint: episode_{args.fine_tune_from_checkpoint}")
        print(f"  - DMMS debug mode: {args.dmms_debug_mode}")
        print(f"  - Main directory: {args.main_dir}")
        print(f"  - Device: {args.device}")
        
    finally:
        sys.argv = original_argv

    # 1. Initialize agent with extended training parameters
    print("--- Initializing Agent for Extended Training ---")
    args, agent, device = set_extended_training_parameters(args)
    if agent is None:
        print("CRITICAL_ERROR: Agent could not be initialized. Exiting.")
        return
    print(f"--- Agent Initialized (Device: {device}) ---")

    # 2. Save baseline performance data
    save_baseline_performance(args, agent)

    # 3. Setup FastAPI app state
    current_finetune_session_start_episode = getattr(args, 'start_finetune_session_at_episode', 1)
    print(f"--- Setting up FastAPI app state for extended training ---")
    setup_app_state_for_finetuning(
        agent_instance=agent, 
        agent_args_for_statespace=agent.args,
        initial_episode_number=current_finetune_session_start_episode,
        experiment_dir=Path(args.experiment_dir)
    )
    print("--- FastAPI app state setup complete ---")

    # 4. Start Uvicorn server
    server_host = getattr(args, 'dmms_server_host', "127.0.0.1") 
    server_port = getattr(args, 'dmms_server_port', 5000)
    
    print(f"--- Starting FastAPI server for extended training on http://{server_host}:{server_port} ---")
    uvicorn_server_thread = UvicornServer(app=fastapi_app_finetune, host=server_host, port=server_port)
    uvicorn_server_thread.start()

    print("INFO: Waiting for Uvicorn server to start...")
    time.sleep(5)
    print("--- FastAPI server ready for extended training ---")

    # 5. Setup DMMS directories
    if args.sim == 'dmms':
        print("--- Setting up DMMS directories for extended training ---")
        args = setup_dmms_dirs(args)
        print("--- DMMS directories setup complete ---")

    # 6. Save extended training configuration
    args_json_path = os.path.join(args.experiment_dir, 'extended_training_config.json')
    print(f"--- Saving extended training configuration to {args_json_path} ---")
    try:
        with open(args_json_path, 'w') as fp:
            serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            json.dump(serializable_args, fp, indent=4)
        print("--- Extended training configuration saved ---")
    except Exception as e:
        print(f"ERROR: Could not save configuration: {e}")

    # 7. Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"--- Random seeds set (Seed: {args.seed}) ---")

    # 8. Get patient environment
    print("--- Getting patient environment for extended training ---")
    patients, env_ids = get_patient_env() 
    print(f"--- Patient env info: patients={patients}, env_ids={env_ids} ---")

    # 9. Start extended training
    print(f"=== STARTING EXTENDED TRAINING ({args.extended_episodes} episodes) ===")
    print(f"Training will save checkpoints every {args.save_freq} episodes")
    print(f"Results will be saved to: {args.experiment_dir}")
    
    try:
        start_time = time.time()
        agent.run(args, patients, env_ids, args.seed)
        end_time = time.time()
        
        # Save training completion data
        completion_data = {
            "training_end_time": datetime.datetime.now().isoformat(),
            "total_training_time_seconds": end_time - start_time,
            "episodes_completed": args.extended_episodes,
            "final_checkpoint_available": True
        }
        
        completion_file = os.path.join(args.experiment_dir, 'evaluation', 'training_completion.json')
        with open(completion_file, 'w') as f:
            json.dump(completion_data, f, indent=4)
        print(f"Total training time: {(end_time - start_time)/3600:.2f} hours")
        
        
        print(f"=== EXTENDED TRAINING COMPLETED ===")
        print(f"Training completion data saved to: {completion_file}")
        
    except KeyboardInterrupt:
        print("\nINFO: KeyboardInterrupt received. Shutting down extended training...")
    except Exception as e:
        print(f"CRITICAL_ERROR: Exception during extended training: {e}")
        traceback.print_exc()
    finally:
        print("--- Extended training finished or interrupted ---")
        print("INFO: Extended training script finished.")

if __name__ == '__main__':
    main()