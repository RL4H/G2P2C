# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements reinforcement learning-based Artificial Pancreas Systems (APS) for Type 1 Diabetes glucose control. The core objective is **online fine-tuning of pre-trained G2P2C agents from Simglucose simulation to the more realistic DMMS.R environment**.

**Current Status**: The technical pipeline for online fine-tuning is successfully implemented with a single-process architecture that enables real-time policy updates. The focus is now on debugging and improving the RL performance during DMMS.R fine-tuning.

## Development Commands

### Environment Setup
```bash
# Create Python 3.10.12 virtual environment and install requirements
pip install -r requirements.txt

# Install simglucose environment in editable mode
cd environments/simglucose
pip install -e .
cd ../..

# Create .env file with MAIN_PATH=path-to-this-project at project root
```

### Core Workflows

**Initial RL Training (Simglucose):**
```bash
cd experiments
python run_RL_agent.py --agent g2p2c --folder_id initial_training --patient_id 0 --return_type average --action_type exponential --device cpu --seed 3
```

**DMMS.R Fine-tuning (Primary Workflow):**
```bash
cd experiments
python run_RL_agent_finetune.py --agent g2p2c --folder_id finetune_test --sim dmms --dmms_exe path/to/DMMS.R.exe --dmms_cfg path/to/config.xml --fine_tune_from_checkpoint episode_195 --device cpu
```

**DMMS.R Evaluation (Static Agent):**
```bash
# For evaluating fixed pre-trained agents
python Sim_CLI/run_dmms_cli.py --dmms_exe path/to/DMMS.R.exe --dmms_cfg path/to/config.xml --folder_id dmms_eval
```

**Clinical Baselines:**
```bash
cd experiments  
python run_clinical_treatment.py --folder_id clinical_baseline --carb_estimation_method real
```

## DMMS.R Fine-tuning Architecture (Current Implementation)

### Single-Process Architecture Overview

The project implements a sophisticated **single-process architecture** for online fine-tuning that enables real-time policy updates:

**Core Pipeline Flow**:
1. `experiments/run_RL_agent_finetune.py` loads pre-trained G2P2C agent (`episode=195`)
2. Same script launches `Sim_CLI/main_finetune.py` FastAPI server in background thread
3. Live agent instance is "injected" into FastAPI server's `app_state`
4. DMMS.R simulator communicates with FastAPI server via JavaScript plugin
5. RL training loop updates agent policy in real-time
6. Updated policy immediately available to DMMS.R environment

### Key Components

**Fine-tuning Controller** (`experiments/run_RL_agent_finetune.py`):
- Main orchestrator for fine-tuning process
- Loads pre-trained checkpoints and manages RL training loop
- Injects live agent instance into FastAPI server state
- Runs Uvicorn server programmatically in background thread

**DMMS.R Communication Bridge** (`Sim_CLI/main_finetune.py`):
- Dedicated FastAPI server for fine-tuning (separate from `main.py`)
- Receives injected agent instance instead of loading fixed checkpoint
- Handles `/env_step`, `/get_state` API endpoints from JavaScript plugin
- Logs real-time data and episode experiences

**Environment Wrapper** (`Sim_CLI/dmms_env.py`):
- Gym-compatible interface for DMMS.R simulator
- Manages DMMS.R subprocess and HTTP communication
- Used by `agents/g2p2c/worker.py` for environment interaction
- Bridges between RL training loop and external simulator

**DMMS.R Integration** (`Sim_CLI/RL_Agent_Plugin_v1.0.js`):
- JavaScript plugin embedded in DMMS.R simulator
- Makes HTTP requests to FastAPI server for agent decisions
- Sends glucose/insulin state data and receives insulin actions

### Environment Selection Logic

**Dual Environment Support** (`utils/core.py`):
- `get_env()` returns appropriate environment based on `args.sim` parameter
- `args.sim == 'simglucose'`: Returns extended T1DSimEnv wrapper
- `args.sim == 'dmms'`: Returns DmmsEnv with `args.dmms_exe` and `args.dmms_cfg`
- Unified Gym interface enables seamless environment switching

**Patient Environment Handling**:
- `get_patient_env()` currently returns Simglucose patient lists regardless of `args.sim`
- For DMMS.R fine-tuning, all workers use single scenario specified by `args.dmms_cfg`
- This design choice supports focused fine-tuning on specific patient scenarios

## Current Development Focus: RL Performance Debugging

**Technical Pipeline Status**: ✅ **COMPLETED**
- Single-process architecture successfully implemented
- Real-time agent policy injection working (confirmed via object ID logging)
- DMMS.R communication pipeline operational

**Current Challenge**: 🔧 **RL Performance Optimization**
The focus has shifted from technical integration to improving the actual reinforcement learning performance during DMMS.R fine-tuning.

### Key Debugging Areas

**1. RL Metrics Monitoring** (Priority: High):
- Episode/rollout cumulative rewards from DMMS.R environment
- G2P2C loss functions (actor, critic, entropy) in `agents/g2p2c/g2p2c.py:update()`
- Glucose management metrics: TIR, hypoglycemia/hyperglycemia events
- Action distribution changes over time

**2. Hyperparameter Sensitivity** (Priority: High):
- Learning rates (`args.pi_lr`, `args.vf_lr`) may need adjustment for DMMS.R
- Rollout length (`args.n_step`) optimization for 5-minute interaction cycles
- Entropy coefficient (`args.entropy_coef`) for exploration balance

**3. Environment-Specific Adaptations**:
- Reward function scaling in `utils/reward_func.py:composite_reward()`
- State feature analysis from `utils/statespace.py:StateSpace`
- Comparison of feature distributions between Simglucose and DMMS.R

### Critical Files for Debugging

**Core RL Algorithm**:
- `agents/g2p2c/g2p2c.py`: Main algorithm, `update()` method for loss logging
- `agents/g2p2c/worker.py`: Environment interaction, `rollout()` method
- `agents/g2p2c/parameters.py`: Hyperparameter configuration

**Environment Interface**:
- `utils/reward_func.py`: Reward function analysis and potential scaling
- `utils/statespace.py`: State representation debugging
- `Sim_CLI/main_finetune.py`: API request/response logging for debugging

**Execution Control**:
- `experiments/run_RL_agent_finetune.py`: Main fine-tuning script
- `utils/options.py`: Command-line argument additions for experiments

### Development Workflow for Performance Improvement

**1. Systematic Debugging Approach**:
- Change one component at a time (hyperparameters, reward scaling, etc.)
- Use structured logging for quantitative analysis
- Compare against baseline metrics from Simglucose training

**2. Experiment Design**:
- Start with simplified conditions (shorter episodes, reduced complexity)
- Gradually increase complexity as performance improves
- Document all changes with Git commits and experiment logs

**3. Success Metrics**:
- Define concrete targets (e.g., "TIR > 70% after 50 episodes")
- Track both RL metrics and clinical glucose management outcomes
- Monitor for signs of learning (reward trends, policy convergence)

### Results and Logging Structure

**Experiment Results**: `/results/{folder_id}/`
- `checkpoints/`: Model checkpoints by episode
- `training/`: RL training logs and metrics
- `testing/`: Evaluation results
- `code/`: Code snapshot for reproducibility

**DMMS.R Sessions**: `/results/dmms_runs/{folder_id}/`
- Real-time logs from DMMS.R simulator interactions
- Episode experience data in JSON format

**Important Notes**:
- Always use `--device cpu` to avoid CUDA driver issues
- Pre-trained checkpoint `episode=195` is the standard starting point
- Results show initial positive signs: "glucose control ability improved slightly" in early tests