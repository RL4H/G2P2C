#!/usr/bin/env python3
# run_extended_training_simple.py
# Simplified extended training script that directly uses run_RL_agent_finetune.py

import subprocess
import sys
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run Extended DMMS Training (Simplified)')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes (default: 50)')
    parser.add_argument('--checkpoint', type=int, default=195, help='Checkpoint episode (default: 195)')
    
    args = parser.parse_args()
    
    # Generate folder ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_id = f"extended_dmms_{args.episodes}ep_{timestamp}"
    
    print(f"=== Extended DMMS Training ===")
    print(f"Episodes: {args.episodes}")
    print(f"Checkpoint: episode_{args.checkpoint}")
    print(f"Folder ID: {folder_id}")
    print(f"Start time: {datetime.now()}")
    print("")
    
    # Build command
    cmd = [
        sys.executable, 
        "experiments/run_RL_agent_finetune.py",
        "--agent", "g2p2c",
        "--folder_id", folder_id,
        "--sim", "dmms",
        "--dmms_exe", "C:\\Program Files\\The Epsilon Group\\DMMS.R\\simulator\\DMMS.R.exe",
        "--dmms_cfg", "C:\\Users\\user\\Documents\\DMMS.R\\config\\Sim_CLI_1.2.xml",
        "--fine_tune_from_checkpoint", str(args.checkpoint),
        "--device", "cpu",
        "--debug", "1",
        "--dmms_debug_mode",
        "--return_type", "average",
        "--action_type", "exponential",
        "--seed", "42"
    ]
    
    # Modify training episodes in parameters.py if needed
    print("Command to execute:")
    print(" ".join(cmd))
    print("")
    
    try:
        # Execute the command
        result = subprocess.run(cmd, check=True, text=True)
        
        print(f"\n=== Training Completed Successfully ===")
        print(f"End time: {datetime.now()}")
        print(f"Results saved to: results/{folder_id}/")
        
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed with exit code {e.returncode}")
        print("Check the logs for more details")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nINFO: Training interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()