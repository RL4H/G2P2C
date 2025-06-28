# experiments/run_multi_scenario_dmms_training.py

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

def setup_scenario_training_folders(args, scenario_name):
    """Setup folders for scenario-specific training"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = MAIN_PATH + "/results/" + f"multi_scenario_{scenario_name}_{args.extended_episodes}ep_{timestamp}"
    
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
    
    print(f"INFO: Scenario training folder created: {LOG_DIR}")
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

def set_scenario_training_parameters(args, scenario_name, checkpoint_dir_override=None, checkpoint_episode_override=None):
    """Set parameters optimized for scenario-specific DMMS training"""
    agent = None
    device = args.device

    if args.agent == 'g2p2c':
        from agents.g2p2c.g2p2c import G2P2C
        from agents.g2p2c.parameters import set_args, set_args_dmms_debug

        args.aux_mode = getattr(args, 'aux_mode', 'dual')
        args.use_planning = getattr(args, 'use_planning', 'yes')
        
        # Force DMMS debug mode for scenario training
        print(f"INFO: Using DMMS.R debug hyperparameters for scenario {scenario_name}")
        args = set_args_dmms_debug(args)
        
        # Add missing G2P2C specific parameters
        args.n_bgp_steps = getattr(args, 'n_bgp_steps', 6)  # Default planning steps
        args.planning_lr = getattr(args, 'planning_lr', 5e-5)  # Planning learning rate
        
        # Apply optimized hyperparameters (from successful 100ep experiment)
        args.pi_lr = args.custom_pi_lr
        args.vf_lr = args.custom_vf_lr
        args.batch_size = args.custom_batch_size
        print(f"INFO: Using optimized hyperparameters - pi_lr: {args.pi_lr}, vf_lr: {args.vf_lr}, batch_size: {args.batch_size}")
        
        # Override with scenario training specific parameters
        args.n_training_episodes = args.extended_episodes
        args.n_testing_episodes = 1  # Test once at the end
        args.save_freq = max(10, args.extended_episodes // 10)  # Save every 10% of episodes
        
        # Setup folders after parameter configuration
        args = setup_scenario_training_folders(args, scenario_name)

        # Use override checkpoint if provided (for continuous training)
        checkpoint_dir = checkpoint_dir_override if checkpoint_dir_override else args.checkpoint_dir
        checkpoint_episode = checkpoint_episode_override if checkpoint_episode_override is not None else args.checkpoint_episode
        
        # Load from the specified checkpoint
        if checkpoint_dir and checkpoint_episode is not None:
            print(f"DEBUG: Using checkpoint_dir = {checkpoint_dir}")
            print(f"DEBUG: Using checkpoint_episode = {checkpoint_episode}")
            print(f"INFO: Loading G2P2C agent from {checkpoint_dir} episode {checkpoint_episode} for scenario {scenario_name}")
            actor_checkpoint_path = os.path.join(checkpoint_dir, f'episode_{checkpoint_episode}_Actor.pth')
            critic_checkpoint_path = os.path.join(checkpoint_dir, f'episode_{checkpoint_episode}_Critic.pth')

            if not os.path.exists(actor_checkpoint_path):
                raise FileNotFoundError(f"Actor checkpoint not found: {actor_checkpoint_path}")
            if not os.path.exists(critic_checkpoint_path):
                raise FileNotFoundError(f"Critic checkpoint not found: {critic_checkpoint_path}")

            print(f"INFO: Loading Actor from: {actor_checkpoint_path}")
            print(f"INFO: Loading Critic from: {critic_checkpoint_path}")
            
            agent = G2P2C(args, device, load=True, path1=actor_checkpoint_path, path2=critic_checkpoint_path)
            print(f"INFO: Successfully loaded G2P2C agent for scenario {scenario_name}")
        else:
            raise ValueError("Checkpoint directory and episode must be specified for scenario training")
        
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

def save_scenario_baseline_performance(args, agent, scenario_name):
    """Save baseline performance before scenario training"""
    baseline_file = os.path.join(args.experiment_dir, 'evaluation', 'scenario_baseline_performance.json')
    
    baseline_data = {
        "scenario_name": scenario_name,
        "base_checkpoint": f"{args.checkpoint_dir}/episode_{args.checkpoint_episode}",
        "extended_episodes_planned": args.extended_episodes,
        "training_start_time": datetime.datetime.now().isoformat(),
        "agent_parameters": {
            "pi_lr": args.pi_lr,
            "vf_lr": args.vf_lr,
            "batch_size": args.batch_size,
            "n_step": args.n_step,
            "n_training_workers": args.n_training_workers,
            "normalize_reward": args.normalize_reward
        }
    }
    
    os.makedirs(os.path.dirname(baseline_file), exist_ok=True)
    with open(baseline_file, 'w') as f:
        json.dump(baseline_data, f, indent=4)
    
    print(f"INFO: Scenario baseline performance data saved to {baseline_file}")

def get_scenario_config_path(scenario_id):
    """Get the XML config path for a given scenario ID"""
    config_base_dir = "C:\\Users\\user\\Desktop\\G2P2C\\config"
    
    if scenario_id == "average":
        config_filename = "RL_scenario_1_07_13_19_50_adult_average.xml"
    else:
        config_filename = f"RL_scenario_1_07_13_19_50_adult{scenario_id}.xml"
    
    config_path = os.path.join(config_base_dir, config_filename)
    return config_path

def train_single_scenario(scenario_id, args_template, current_checkpoint_dir=None, current_checkpoint_episode=None):
    """Train a single scenario and return results"""
    scenario_name = f"adult{scenario_id}" if scenario_id != "average" else "adult_average"
    config_path = get_scenario_config_path(scenario_id)
    
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING FOR SCENARIO: {scenario_name}")
    print(f"Config file: {config_path}")
    print(f"Episodes: {args_template.extended_episodes}")
    print(f"{'='*60}")
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": "Config file not found",
            "config_path": config_path
        }
    
    try:
        # Create proper args using Options().parse() like the original script
        original_argv = sys.argv
        sys.argv = [sys.argv[0]] + [
            '--agent', 'g2p2c',
            '--sim', 'dmms',
            '--dmms_debug_mode',
            '--folder_id', f'multi_scenario_{scenario_name}',
            '--dmms_exe', 'C:\\Program Files\\The Epsilon Group\\DMMS.R\\simulator\\DMMS.R.exe',
            '--dmms_cfg', config_path,
            '--device', 'cpu',
            '--debug', '1',
            '--return_type', 'average',
            '--action_type', 'exponential'
        ]
        
        try:
            args = Options().parse()
            
            # Add extended training specific parameters
            args.extended_episodes = args_template.extended_episodes
            args.dmms_debug_mode = True
            # Use updated checkpoint if available (for continuous training)
            args.checkpoint_dir = current_checkpoint_dir if current_checkpoint_dir else args_template.checkpoint_dir
            args.checkpoint_episode = current_checkpoint_episode if current_checkpoint_episode else args_template.checkpoint_episode
            args.custom_pi_lr = args_template.custom_pi_lr
            args.custom_vf_lr = args_template.custom_vf_lr
            args.custom_batch_size = args_template.custom_batch_size
            args.main_dir = MAIN_PATH
            
        finally:
            sys.argv = original_argv
        
        # 1. Initialize agent with scenario training parameters
        print(f"--- Initializing Agent for Scenario {scenario_name} ---")
        args, agent, device = set_scenario_training_parameters(args, scenario_name, current_checkpoint_dir, current_checkpoint_episode)
        if agent is None:
            raise Exception("Agent could not be initialized")
        print(f"--- Agent Initialized (Device: {device}) ---")

        # 2. Save baseline performance data
        save_scenario_baseline_performance(args, agent, scenario_name)

        # 3. Setup FastAPI app state
        current_finetune_session_start_episode = getattr(args, 'start_finetune_session_at_episode', 1)
        print(f"--- Setting up FastAPI app state for scenario {scenario_name} ---")
        setup_app_state_for_finetuning(
            agent_instance=agent, 
            agent_args_for_statespace=agent.args,
            initial_episode_number=current_finetune_session_start_episode,
            experiment_dir=Path(args.experiment_dir)
        )
        print("--- FastAPI app state setup complete ---")

        # 4. Start Uvicorn server
        server_host = "127.0.0.1"
        server_port = 5000
        
        print(f"--- Starting FastAPI server for scenario {scenario_name} on http://{server_host}:{server_port} ---")
        uvicorn_server_thread = UvicornServer(app=fastapi_app_finetune, host=server_host, port=server_port)
        uvicorn_server_thread.start()

        print("INFO: Waiting for Uvicorn server to start...")
        time.sleep(5)
        print("--- FastAPI server ready for scenario training ---")

        # 5. Setup DMMS directories
        if args.sim == 'dmms':
            print("--- Setting up DMMS directories for scenario training ---")
            args = setup_dmms_dirs(args)
            print("--- DMMS directories setup complete ---")

        # 6. Save scenario training configuration
        args_json_path = os.path.join(args.experiment_dir, 'scenario_training_config.json')
        print(f"--- Saving scenario training configuration to {args_json_path} ---")
        try:
            with open(args_json_path, 'w') as fp:
                serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                json.dump(serializable_args, fp, indent=4)
            print("--- Scenario training configuration saved ---")
        except Exception as e:
            print(f"ERROR: Could not save configuration: {e}")

        # 7. Set random seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"--- Random seeds set (Seed: {args.seed}) ---")

        # 8. Get patient environment
        print("--- Getting patient environment for scenario training ---")
        patients, env_ids = get_patient_env() 
        print(f"--- Patient env info: patients={patients}, env_ids={env_ids} ---")

        # 9. Start scenario training
        print(f"=== STARTING SCENARIO TRAINING: {scenario_name} ({args.extended_episodes} episodes) ===")
        print(f"Training will save checkpoints every {args.save_freq} episodes")
        print(f"Results will be saved to: {args.experiment_dir}")
        
        start_time = time.time()
        agent.run(args, patients, env_ids, args.seed)
        end_time = time.time()
        
        # Save training completion data
        # G2P2C saves episodes starting from 0, so final episode is (extended_episodes - 1)
        final_episode = args.extended_episodes - 1
        
        completion_data = {
            "scenario_name": scenario_name,
            "training_end_time": datetime.datetime.now().isoformat(),
            "total_training_time_seconds": end_time - start_time,
            "episodes_completed": args.extended_episodes,
            "start_episode": args.checkpoint_episode,
            "final_episode": final_episode,
            "note": f"G2P2C internal: trained {args.extended_episodes} episodes, saved as 0 to {final_episode}",
            "final_checkpoint_available": True,
            "experiment_dir": args.experiment_dir,
            "final_checkpoint_dir": os.path.join(args.experiment_dir, 'checkpoints')
        }
        
        completion_file = os.path.join(args.experiment_dir, 'evaluation', 'scenario_training_completion.json')
        with open(completion_file, 'w') as f:
            json.dump(completion_data, f, indent=4)
        
        training_time_hours = (end_time - start_time) / 3600
        print(f"Scenario {scenario_name} training time: {training_time_hours:.2f} hours")
        print(f"=== SCENARIO TRAINING COMPLETED: {scenario_name} ===")
        
        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "status": "completed",
            "training_time_hours": training_time_hours,
            "experiment_dir": args.experiment_dir,
            "completion_file": completion_file,
            "config_path": config_path,
            "final_episode": final_episode,
            "final_checkpoint_dir": os.path.join(args.experiment_dir, 'checkpoints')
        }
        
    except KeyboardInterrupt:
        print(f"\nINFO: KeyboardInterrupt received during scenario {scenario_name} training.")
        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "status": "interrupted",
            "error": "KeyboardInterrupt",
            "config_path": config_path
        }
    except Exception as e:
        print(f"CRITICAL_ERROR: Exception during scenario {scenario_name} training: {e}")
        traceback.print_exc()
        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
            "config_path": config_path
        }

def generate_final_summary_report(results, args):
    """Generate a comprehensive summary report of all scenario training"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.join(MAIN_PATH, "results", f"multi_scenario_summary_{timestamp}")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_file = os.path.join(summary_dir, "MULTI_SCENARIO_SUMMARY.md")
    
    with open(summary_file, 'w') as f:
        f.write(f"# Multi-Scenario DMMS Training Summary\n\n")
        f.write(f"**Training Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Base Checkpoint**: {args.checkpoint_dir}/episode_{args.checkpoint_episode}\n")
        f.write(f"**Episodes per Scenario**: {args.extended_episodes}\n")
        f.write(f"**Optimized Hyperparameters**: pi_lr={args.pi_lr}, vf_lr={args.vf_lr}, batch_size={args.batch_size}\n\n")
        
        # Summary statistics
        completed = [r for r in results if r["status"] == "completed"]
        failed = [r for r in results if r["status"] == "failed"]
        interrupted = [r for r in results if r["status"] == "interrupted"]
        
        f.write(f"## Summary Statistics\n\n")
        f.write(f"- **Total Scenarios**: {len(results)}\n")
        f.write(f"- **Completed**: {len(completed)}\n")
        f.write(f"- **Failed**: {len(failed)}\n")
        f.write(f"- **Interrupted**: {len(interrupted)}\n\n")
        
        if completed:
            total_time = sum([r.get("training_time_hours", 0) for r in completed])
            avg_time = total_time / len(completed)
            f.write(f"- **Total Training Time**: {total_time:.2f} hours\n")
            f.write(f"- **Average Time per Scenario**: {avg_time:.2f} hours\n\n")
        
        # Detailed results
        f.write(f"## Detailed Results\n\n")
        
        for result in results:
            f.write(f"### Scenario: {result['scenario_name']}\n\n")
            f.write(f"- **Status**: {result['status']}\n")
            f.write(f"- **Config**: `{result['config_path']}`\n")
            
            if result["status"] == "completed":
                f.write(f"- **Training Time**: {result['training_time_hours']:.2f} hours\n")
                f.write(f"- **Results Directory**: `{result['experiment_dir']}`\n")
                f.write(f"- **Completion File**: `{result['completion_file']}`\n")
            elif result["status"] == "failed":
                f.write(f"- **Error**: {result['error']}\n")
            
            f.write(f"\n")
        
        # Next steps
        f.write(f"## Next Steps\n\n")
        f.write(f"1. **Performance Analysis**: Compare RMSE and clinical metrics across scenarios\n")
        f.write(f"2. **Model Evaluation**: Test each trained model on validation scenarios\n")
        f.write(f"3. **Cross-Patient Generalization**: Evaluate models trained on one patient with other patients\n\n")
        
        # Evaluation commands
        if completed:
            f.write(f"## Evaluation Commands\n\n")
            f.write(f"```bash\n")
            for result in completed:
                if result["status"] == "completed":
                    exp_dir = os.path.basename(result["experiment_dir"])
                    f.write(f"# Evaluate {result['scenario_name']}\n")
                    f.write(f"python experiments/evaluate_dmms_performance.py --folder_id {exp_dir}\n\n")
            f.write(f"```\n\n")
    
    print(f"INFO: Multi-scenario summary report generated: {summary_file}")
    return summary_file

def main():
    # Parse custom arguments for multi-scenario training
    parser = argparse.ArgumentParser(description='Multi-Scenario DMMS Training with Optimized Hyperparameters')
    parser.add_argument('--scenarios', type=str, required=True,
                       help='Comma-separated list of scenario IDs (e.g., "2,3,8,average")')
    parser.add_argument('--extended_episodes', type=int, default=50, 
                       help='Number of episodes for each scenario training (default: 50)')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='C:\\Users\\user\\Desktop\\G2P2C\\results\\extended_dmms_100ep_20250607_183216\\checkpoints',
                       help='Directory containing the base checkpoint to load from')
    parser.add_argument('--checkpoint_episode', type=int, default=99,
                       help='Episode number of checkpoint to load (default: 99)')
    parser.add_argument('--pi_lr', type=float, default=3e-5,
                       help='Policy learning rate (default: 3e-5, optimized)')
    parser.add_argument('--vf_lr', type=float, default=3e-5,
                       help='Value function learning rate (default: 3e-5, optimized)')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training (default: 1024, optimized)')
    parser.add_argument('--delay_between_scenarios', type=int, default=10,
                       help='Delay in seconds between scenario training sessions (default: 10)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Parse scenario list
    scenario_list = [s.strip() for s in args.scenarios.split(',')]
    print(f"INFO: Selected scenarios for training: {scenario_list}")
    
    # Validate checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        return
    
    actor_checkpoint = os.path.join(args.checkpoint_dir, f'episode_{args.checkpoint_episode}_Actor.pth')
    critic_checkpoint = os.path.join(args.checkpoint_dir, f'episode_{args.checkpoint_episode}_Critic.pth')
    
    if not os.path.exists(actor_checkpoint) or not os.path.exists(critic_checkpoint):
        print(f"ERROR: Checkpoint files not found:")
        print(f"  Actor: {actor_checkpoint}")
        print(f"  Critic: {critic_checkpoint}")
        return
    
    print(f"INFO: Using base checkpoint from: {args.checkpoint_dir}/episode_{args.checkpoint_episode}")
    
    # Setup base arguments template
    args_template = argparse.Namespace()
    args_template.agent = 'g2p2c'
    args_template.sim = 'dmms'
    args_template.dmms_debug_mode = True
    args_template.extended_episodes = args.extended_episodes
    args_template.checkpoint_dir = args.checkpoint_dir
    args_template.checkpoint_episode = args.checkpoint_episode
    args_template.custom_pi_lr = args.pi_lr
    args_template.custom_vf_lr = args.vf_lr
    args_template.custom_batch_size = args.batch_size
    args_template.device = 'cpu'
    args_template.debug = 1
    args_template.return_type = 'average'
    args_template.action_type = 'exponential'
    args_template.seed = 42
    args_template.main_dir = MAIN_PATH
    
    print(f"\nINFO: Multi-Scenario Training Configuration:")
    print(f"  - Scenarios to train: {scenario_list}")
    print(f"  - Episodes per scenario: {args.extended_episodes}")
    print(f"  - Base checkpoint: episode_{args.checkpoint_episode}")
    print(f"  - Optimized hyperparameters: pi_lr={args.pi_lr}, vf_lr={args.vf_lr}, batch_size={args.batch_size}")
    print(f"  - Delay between scenarios: {args.delay_between_scenarios}s")
    
    # Train each scenario continuously
    results = []
    current_checkpoint_dir = args.checkpoint_dir
    current_checkpoint_episode = args.checkpoint_episode
    
    for i, scenario_id in enumerate(scenario_list):
        print(f"\n{'#'*80}")
        print(f"TRAINING SCENARIO {i+1}/{len(scenario_list)}: {scenario_id}")
        print(f"Starting from checkpoint: episode {current_checkpoint_episode}")
        print(f"Will train to episode: {current_checkpoint_episode + args.extended_episodes}")
        print(f"{'#'*80}")
        
        result = train_single_scenario(scenario_id, args_template, current_checkpoint_dir, current_checkpoint_episode)
        results.append(result)
        
        print(f"\nScenario {scenario_id} result: {result['status']}")
        
        # Update checkpoint for next scenario (continuous training)
        if result['status'] == 'completed':
            current_checkpoint_dir = result['final_checkpoint_dir']
            current_checkpoint_episode = result['final_episode']  # This is the G2P2C internal episode number
            print(f"INFO: Updated checkpoint for next scenario - using episode {current_checkpoint_episode} from {current_checkpoint_dir}")
        else:
            print(f"WARNING: Scenario {scenario_id} failed, using previous checkpoint for next scenario")
        
        # Add delay between scenarios (except for the last one)
        if i < len(scenario_list) - 1:
            print(f"INFO: Waiting {args.delay_between_scenarios} seconds before next scenario...")
            time.sleep(args.delay_between_scenarios)
    
    # Generate final summary report
    print(f"\n{'='*80}")
    print(f"MULTI-SCENARIO TRAINING COMPLETED")
    print(f"{'='*80}")
    
    summary_file = generate_final_summary_report(results, args)
    
    # Print final summary
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]
    interrupted = [r for r in results if r["status"] == "interrupted"]
    
    print(f"\nFINAL SUMMARY:")
    print(f"  - Total scenarios: {len(results)}")
    print(f"  - Completed: {len(completed)} {[r['scenario_name'] for r in completed]}")
    print(f"  - Failed: {len(failed)} {[r['scenario_name'] for r in failed]}")
    print(f"  - Interrupted: {len(interrupted)} {[r['scenario_name'] for r in interrupted]}")
    
    if completed:
        total_time = sum([r.get("training_time_hours", 0) for r in completed])
        print(f"  - Total training time: {total_time:.2f} hours")
    
    print(f"\nSummary report saved to: {summary_file}")
    print(f"Multi-scenario training session completed!")

if __name__ == '__main__':
    main()