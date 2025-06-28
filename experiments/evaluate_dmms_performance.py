# experiments/evaluate_dmms_performance.py

import os
from pathlib import Path
import sys
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from decouple import config
import uvicorn
import threading
import time
import argparse

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.core import set_logger, get_patient_env
from utils.options import Options
from Sim_CLI.main_finetune import app as fastapi_app_finetune
from Sim_CLI.main_finetune import setup_app_state_for_finetuning

import warnings
warnings.simplefilter('ignore', Warning)

class PerformanceEvaluator:
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        self.evaluation_dir = self.experiment_dir / 'evaluation'
        self.evaluation_dir.mkdir(exist_ok=True)
        
    def load_training_logs(self):
        """Load training logs and metrics from the experiment directory"""
        logs = {}
        
        # Load model logs
        model_log_path = self.experiment_dir / 'model_log.csv'
        if model_log_path.exists():
            logs['model_log'] = pd.read_csv(model_log_path)
            print(f"INFO: Loaded model log with {len(logs['model_log'])} entries")
        
        # Load evaluation logs
        eval_log_path = self.experiment_dir / 'evaluation_log.csv'
        if eval_log_path.exists():
            logs['evaluation_log'] = pd.read_csv(eval_log_path)
            print(f"INFO: Loaded evaluation log with {len(logs['evaluation_log'])} entries")
        
        # Load experiment summary
        summary_path = self.experiment_dir / 'experiment_summary.csv'
        if summary_path.exists():
            logs['experiment_summary'] = pd.read_csv(summary_path)
            print(f"INFO: Loaded experiment summary with {len(logs['experiment_summary'])} entries")
            
        return logs
    
    def calculate_glucose_metrics(self, glucose_values):
        """Calculate key glucose control metrics"""
        glucose_array = np.array(glucose_values)
        
        # Time in Range (TIR) - 70-180 mg/dL
        tir = np.sum((glucose_array >= 70) & (glucose_array <= 180)) / len(glucose_array) * 100
        
        # Time below range (TBR) - <70 mg/dL
        tbr = np.sum(glucose_array < 70) / len(glucose_array) * 100
        
        # Time above range (TAR) - >180 mg/dL
        tar = np.sum(glucose_array > 180) / len(glucose_array) * 100
        
        # Mean glucose
        mean_glucose = np.mean(glucose_array)
        
        # Glucose variability (CV)
        cv = np.std(glucose_array) / mean_glucose * 100
        
        # Risk indices
        hypo_risk = np.sum(glucose_array < 54) / len(glucose_array) * 100  # Severe hypoglycemia
        hyper_risk = np.sum(glucose_array > 250) / len(glucose_array) * 100  # Severe hyperglycemia
        
        return {
            'TIR': tir,
            'TBR': tbr,
            'TAR': tar,
            'mean_glucose': mean_glucose,
            'glucose_cv': cv,
            'severe_hypo_risk': hypo_risk,
            'severe_hyper_risk': hyper_risk
        }
    
    def analyze_learning_progress(self, logs):
        """Analyze learning progress over episodes"""
        if 'model_log' not in logs:
            print("WARNING: No model log found for learning progress analysis")
            return None
            
        model_log = logs['model_log']
        
        analysis = {
            'total_episodes': len(model_log),
            'initial_metrics': {},
            'final_metrics': {},
            'improvement': {}
        }
        
        if len(model_log) > 0:
            # Initial performance (first 10% of episodes)
            initial_episodes = max(1, len(model_log) // 10)
            initial_data = model_log.head(initial_episodes)
            
            # Final performance (last 10% of episodes)
            final_episodes = max(1, len(model_log) // 10)
            final_data = model_log.tail(final_episodes)
            
            # Calculate metrics for initial and final periods
            for period, data in [('initial', initial_data), ('final', final_data)]:
                metrics = {}
                
                if 'reward_mean' in data.columns:
                    metrics['reward_mean'] = data['reward_mean'].mean()
                if 'val_loss' in data.columns:
                    metrics['val_loss'] = data['val_loss'].mean()
                if 'policy_loss' in data.columns:
                    metrics['policy_loss'] = data['policy_loss'].mean()
                if 'explained_variance' in data.columns:
                    metrics['explained_variance'] = data['explained_variance'].mean()
                
                analysis[f'{period}_metrics'] = metrics
            
            # Calculate improvement
            for metric in analysis['initial_metrics'].keys():
                if metric in analysis['final_metrics']:
                    initial_val = analysis['initial_metrics'][metric]
                    final_val = analysis['final_metrics'][metric]
                    
                    if metric in ['val_loss', 'policy_loss']:  # Lower is better
                        improvement = (initial_val - final_val) / initial_val * 100
                    else:  # Higher is better
                        improvement = (final_val - initial_val) / abs(initial_val) * 100 if initial_val != 0 else 0
                    
                    analysis['improvement'][metric] = improvement
        
        return analysis
    
    def generate_performance_report(self, logs, baseline_checkpoint=None, final_checkpoint=None):
        """Generate comprehensive performance report"""
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'experiment_directory': str(self.experiment_dir),
            'baseline_checkpoint': baseline_checkpoint,
            'final_checkpoint': final_checkpoint
        }
        
        # Learning progress analysis
        learning_analysis = self.analyze_learning_progress(logs)
        if learning_analysis:
            report['learning_progress'] = learning_analysis
        
        # Generate plots
        self.generate_performance_plots(logs)
        
        # Save report
        report_path = self.evaluation_dir / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        print(f"INFO: Performance report saved to {report_path}")
        return report
    
    def generate_performance_plots(self, logs):
        """Generate visualization plots for performance analysis"""
        plt.style.use('seaborn-v0_8')
        
        if 'model_log' in logs:
            model_log = logs['model_log']
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Extended Training Performance Analysis', fontsize=16)
            
            # Plot 1: Reward progression
            if 'reward_mean' in model_log.columns:
                axes[0, 0].plot(model_log.index, model_log['reward_mean'])
                axes[0, 0].set_title('Reward Mean Over Episodes')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward Mean')
                axes[0, 0].grid(True)
            
            # Plot 2: Value loss progression
            if 'val_loss' in model_log.columns:
                axes[0, 1].plot(model_log.index, model_log['val_loss'], color='red')
                axes[0, 1].set_title('Value Loss Over Episodes')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Value Loss')
                axes[0, 1].grid(True)
            
            # Plot 3: Policy loss progression
            if 'policy_loss' in model_log.columns:
                axes[1, 0].plot(model_log.index, model_log['policy_loss'], color='green')
                axes[1, 0].set_title('Policy Loss Over Episodes')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Policy Loss')
                axes[1, 0].grid(True)
            
            # Plot 4: Explained variance progression
            if 'explained_variance' in model_log.columns:
                axes[1, 1].plot(model_log.index, model_log['explained_variance'], color='purple')
                axes[1, 1].set_title('Explained Variance Over Episodes')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Explained Variance')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = self.evaluation_dir / 'training_progress.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"INFO: Training progress plot saved to {plot_path}")

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

def load_agent_for_evaluation(checkpoint_path, args, device):
    """Load agent from checkpoint for evaluation"""
    from agents.g2p2c.g2p2c import G2P2C
    from agents.g2p2c.parameters import set_args_dmms_debug
    
    # Use DMMS debug parameters for consistency
    args = set_args_dmms_debug(args)
    
    if os.path.exists(checkpoint_path):
        # Extract episode number from checkpoint path
        checkpoint_name = os.path.basename(checkpoint_path)
        if 'Actor' in checkpoint_name:
            episode_num = checkpoint_name.replace('_Actor.pth', '').replace('episode_', '')
            critic_path = checkpoint_path.replace('_Actor.pth', '_Critic.pth')
        elif 'Critic' in checkpoint_name:
            episode_num = checkpoint_name.replace('_Critic.pth', '').replace('episode_', '')
            actor_path = checkpoint_path.replace('_Critic.pth', '_Actor.pth')
            checkpoint_path = actor_path
            critic_path = checkpoint_path.replace('_Actor.pth', '_Critic.pth')
        else:
            raise ValueError(f"Invalid checkpoint path format: {checkpoint_path}")
        
        if not os.path.exists(critic_path):
            raise FileNotFoundError(f"Critic checkpoint not found: {critic_path}")
        
        print(f"INFO: Loading agent from episode {episode_num}")
        print(f"INFO: Actor: {checkpoint_path}")
        print(f"INFO: Critic: {critic_path}")
        
        agent = G2P2C(args, device, load=True, path1=checkpoint_path, path2=critic_path)
        return agent, episode_num
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

def run_evaluation_episode(agent, args):
    """Run a single evaluation episode with the given agent"""
    print("INFO: Starting evaluation episode...")
    
    # Setup FastAPI for evaluation
    setup_app_state_for_finetuning(
        agent_instance=agent,
        agent_args_for_statespace=agent.args,
        initial_episode_number=1,
        experiment_dir=Path(args.experiment_dir) / 'evaluation_run'
    )
    
    # Start server
    server_thread = UvicornServer(app=fastapi_app_finetune, host="127.0.0.1", port=5001)
    server_thread.start()
    time.sleep(3)
    
    print("INFO: Evaluation setup complete. Agent ready for DMMS.R evaluation.")
    print("INFO: Please run DMMS.R simulation now for evaluation.")
    print("INFO: Press Enter when evaluation episode is complete...")
    input()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Evaluate DMMS Training Performance')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory with training results')
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                       help='Path to baseline checkpoint for comparison')
    parser.add_argument('--final_checkpoint', type=str, default=None,
                       help='Path to final trained checkpoint')
    parser.add_argument('--run_evaluation', action='store_true',
                       help='Run evaluation episodes with loaded agents')
    
    args = parser.parse_args()
    
    # Validate experiment directory
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"ERROR: Experiment directory not found: {experiment_dir}")
        return
    
    print(f"INFO: Evaluating experiment: {experiment_dir}")
    
    # Initialize evaluator
    evaluator = PerformanceEvaluator(experiment_dir)
    
    # Load training logs
    print("INFO: Loading training logs...")
    logs = evaluator.load_training_logs()
    
    if not logs:
        print("WARNING: No training logs found in experiment directory")
        return
    
    # Find checkpoints if not provided
    checkpoints_dir = experiment_dir / 'checkpoints'
    available_checkpoints = []
    
    if checkpoints_dir.exists():
        for checkpoint_file in checkpoints_dir.glob('episode_*_Actor.pth'):
            episode_num = int(checkpoint_file.stem.split('_')[1])
            available_checkpoints.append((episode_num, str(checkpoint_file)))
        
        available_checkpoints.sort()
        print(f"INFO: Found {len(available_checkpoints)} checkpoints")
        
        if available_checkpoints:
            if args.baseline_checkpoint is None:
                # Use first checkpoint as baseline
                args.baseline_checkpoint = available_checkpoints[0][1]
                print(f"INFO: Using first checkpoint as baseline: episode_{available_checkpoints[0][0]}")
            
            if args.final_checkpoint is None:
                # Use last checkpoint as final
                args.final_checkpoint = available_checkpoints[-1][1]
                print(f"INFO: Using last checkpoint as final: episode_{available_checkpoints[-1][0]}")
    
    # Generate performance report
    print("INFO: Generating performance report...")
    report = evaluator.generate_performance_report(
        logs, 
        baseline_checkpoint=args.baseline_checkpoint,
        final_checkpoint=args.final_checkpoint
    )
    
    # Print summary
    if 'learning_progress' in report:
        progress = report['learning_progress']
        print("\n=== TRAINING PERFORMANCE SUMMARY ===")
        print(f"Total episodes: {progress['total_episodes']}")
        
        if 'improvement' in progress:
            print("\nPerformance Improvements:")
            for metric, improvement in progress['improvement'].items():
                print(f"  {metric}: {improvement:+.2f}%")
        
        if 'final_metrics' in progress:
            print("\nFinal Performance Metrics:")
            for metric, value in progress['final_metrics'].items():
                print(f"  {metric}: {value:.4f}")
    
    # Optional: Run evaluation episodes
    if args.run_evaluation and args.baseline_checkpoint and args.final_checkpoint:
        print("\n=== RUNNING EVALUATION EPISODES ===")
        
        # Setup basic args for agent loading
        original_argv = sys.argv
        sys.argv = [sys.argv[0], '--agent', 'g2p2c', '--sim', 'dmms']
        
        try:
            eval_args = Options().parse()
            eval_args.experiment_dir = str(experiment_dir)
            device = eval_args.device
            
            # Evaluate baseline
            print("\n--- Evaluating Baseline Agent ---")
            baseline_agent, baseline_episode = load_agent_for_evaluation(
                args.baseline_checkpoint, eval_args, device)
            run_evaluation_episode(baseline_agent, eval_args)
            
            # Evaluate final trained model
            print("\n--- Evaluating Final Trained Agent ---")
            final_agent, final_episode = load_agent_for_evaluation(
                args.final_checkpoint, eval_args, device)
            run_evaluation_episode(final_agent, eval_args)
            
        finally:
            sys.argv = original_argv
    
    print(f"\n=== EVALUATION COMPLETE ===")
    print(f"Results saved to: {evaluator.evaluation_dir}")

if __name__ == '__main__':
    main()