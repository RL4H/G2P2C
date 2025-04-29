"""
Module: Sim_CLI/g2p2c_agent_api.py
Purpose: Utility functions to load a G2P2C agent and perform inference
"""
# Add repository root to path so that `utils` and `agents` can be imported
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import importlib.util
from typing import Optional

import numpy as np
import torch

from utils.options import Options
from agents.g2p2c.g2p2c import G2P2C


def load_agent(
    args_json_path: str,
    params_py_path: str,
    device: Optional[torch.device] = None,
    episode: Optional[int] = None,
) -> G2P2C:
    """
    Load G2P2C agent from saved arguments and checkpoints.
    - args_json_path: path to args.json used in training
    - params_py_path: path to parameters.py for setting args
    - device: torch.device (default: cpu)
    - episode: if provided, load corresponding episode Actor/Critic; else pick latest
    Returns: initialized G2P2C agent in eval mode
    """
    # 1. Load args (and override experiment_dir for local environment)
    with open(args_json_path, 'r') as f:
        args_dict = json.load(f)
    args = Options().parse()
    for k, v in args_dict.items():
        setattr(args, k, v)
    # Use local directory of args_json as experiment_dir
    args.experiment_dir = os.path.abspath(os.path.dirname(args_json_path))
    # Ensure experiment_dir exists (for log files)
    os.makedirs(args.experiment_dir, exist_ok=True)

    # 2. Load parameters.py
    spec = importlib.util.spec_from_file_location('parameters', params_py_path)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    params.set_args(args)

    # 3. Determine device
    device = device or torch.device('cpu')

    # 4. Initialize agent (no load inside)
    agent = G2P2C(args, device, load=False, path1=None, path2=None)

    # 5. Locate checkpoint files
    ckpt_dir = os.path.dirname(args.experiment_dir + '/checkpoints/')
    # pick Actor/Critic paths
    if episode is None:
        # find all Actor.pth and choose highest episode
        candidates = []
        for fname in os.listdir(ckpt_dir):
            if fname.endswith('_Actor.pth'):
                parts = fname.split('_')
                try:
                    num = int(parts[1])
                    candidates.append((num, fname))
                except Exception:
                    continue
        if not candidates:
            raise FileNotFoundError('No Actor checkpoint found in ' + ckpt_dir)
        episode = max(candidates, key=lambda x: x[0])[0]
    actor_path = os.path.join(ckpt_dir, f'episode_{episode}_Actor.pth')
    critic_path = os.path.join(ckpt_dir, f'episode_{episode}_Critic.pth')

    # 6. Load model objects
    agent.policy.Actor = torch.load(actor_path, map_location=device)
    agent.policy.Critic = torch.load(critic_path, map_location=device)
    agent.policy.Actor.eval()
    agent.policy.Critic.eval()
    return agent


def infer_action(
    agent: G2P2C,
    history: np.ndarray,
    feat: Optional[np.ndarray] = None,
) -> float:
    """
    Perform inference using loaded agent.
    - history: np.ndarray shape (12, n_features)
    - feat: np.ndarray shape (1, n_handcrafted_features)
    Returns clipped float action within [args.insulin_min, args.insulin_max]
    """
    # 1. Validate history shape and values
    if not isinstance(history, np.ndarray):
        history = np.array(history, dtype=np.float32)
    if history.ndim != 2:
        raise ValueError(f'history must be 2D array, got ndim={history.ndim}, shape={history.shape}')
    expected_shape = (agent.args.feature_history, agent.args.n_features)
    if history.shape != expected_shape:
        raise ValueError(f'history shape must be {expected_shape}, got {history.shape}')
    if not np.all(np.isfinite(history)):
        raise ValueError('history contains NaN or Inf')
    history_tensor = history.astype(np.float32)

    # 2. Prepare feat
    if feat is None:
        n_hf = agent.args.n_handcrafted_features
        feat = np.zeros((1, n_hf), dtype=np.float32)
    if not isinstance(feat, np.ndarray):
        feat = np.array(feat, dtype=np.float32)
    feat_tensor = feat.astype(np.float32)

    # 3. Inference
    out = agent.policy.get_action(history_tensor, feat_tensor)
    action = float(out['action'][0])

    # 4. Clip
    action_clipped = float(np.clip(action, agent.args.insulin_min, agent.args.insulin_max))
    return action_clipped