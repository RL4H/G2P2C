"""
Module: Sim_CLI/g2p2c_agent_api.py
Purpose: Utility functions to load a G2P2C agent and perform inference
"""
import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # pathlib 사용 시 불필요할 수 있음
import json
import importlib.util
from typing import Optional
from pathlib import Path # pathlib 임포트

import numpy as np
import torch

# utils와 agents를 임포트하기 위해 프로젝트 루트를 sys.path에 추가 (필요시 유지 또는 다른 방식 고려)
_current_script_dir = Path(__file__).resolve().parent
_project_root = _current_script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 이제 utils 및 agents 임포트 가능
from utils.options import Options
from agents.ppo.ppo import PPO

# 프로젝트 루트 기준으로 기본 경로 설정 (load_agent 함수 외부에서도 사용 가능하도록)
_DEFAULT_RESULTS_SUBDIR = "results_ppo/test" #!!!!!!!!!!!수정!!!!!!!!!!!!!!!!!!!!
_DEFAULT_ARGS_JSON_REL_PATH = f"{_DEFAULT_RESULTS_SUBDIR}/args.json"
_DEFAULT_PARAMS_PY_REL_PATH = f"{_DEFAULT_RESULTS_SUBDIR}/code/parameters.py"
_DEFAULT_CHECKPOINTS_REL_DIR = f"{_DEFAULT_RESULTS_SUBDIR}/checkpoints"



def load_agent(
    # 경로 인자 제거 또는 기본값으로만 사용 (외부에서 거의 지정할 일 없음)
    # args_json_path: str, # 제거됨
    # params_py_path: str, # 제거됨
    device: Optional[torch.device] = None,
    episode: Optional[int] = None,
):
    """
    Load G2P2C agent from saved arguments and checkpoints using fixed relative paths
    from the project root.
    - device: torch.device (default: cpu)
    - episode: if provided, load corresponding episode Actor/Critic; else pick latest
    Returns: initialized G2P2C agent in eval mode
    """
    # --- 경로 계산 (스크립트 위치 기준) ---
    args_json_path = (_project_root / _DEFAULT_ARGS_JSON_REL_PATH).resolve()
    params_py_path = (_project_root / _DEFAULT_PARAMS_PY_REL_PATH).resolve()
    checkpoints_dir_abs = (_project_root / _DEFAULT_CHECKPOINTS_REL_DIR).resolve()

    # 파일 존재 여부 확인 (오류 메시지 명확화)
    if not args_json_path.is_file():
        raise FileNotFoundError(f"args.json not found at expected path: {args_json_path}")
    if not params_py_path.is_file():
        raise FileNotFoundError(f"parameters.py not found at expected path: {params_py_path}")
    if not checkpoints_dir_abs.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found at expected path: {checkpoints_dir_abs}")

    # 1. Load args (Use absolute path)
    with open(args_json_path, 'r') as f:
        args_dict = json.load(f)

    # Options().parse() 호출 시 빈 리스트 [] 전달!
    args = Options().parse([]) # <--- 수정된 호출 방식!

    # args_dict의 내용으로 args 객체 업데이트
    for k, v in args_dict.items():
        setattr(args, k, v)

    # experiment_dir 설정 (로드된 args.json 위치 기준으로 설정)
    # 주의: args_dict에 experiment_dir이 있으면 그것을 사용할지,
    # 아니면 args_json 위치 기준으로 재정의할지 정책 결정 필요.
    # 여기서는 args_json 위치 기준으로 재정의
    args.experiment_dir = str(args_json_path.parent.parent) # results/test 디렉토리의 절대 경로
    # Ensure experiment_dir exists (이미 위에서 확인했지만 안전하게)
    os.makedirs(args.experiment_dir, exist_ok=True)

    # 2. Load parameters.py (Use absolute path)
    spec = importlib.util.spec_from_file_location('parameters', str(params_py_path))
    if spec is None or spec.loader is None:
         raise ImportError(f"Could not load spec for parameters.py from {params_py_path}")
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    params.set_args(args)

    # 3. Determine device (기존 로직 유지)
    device = device or torch.device('cpu')

    # 4. Initialize agent (기존 로직 유지)
    agent = PPO(args, device, load=False, path1=None, path2=None) # 새로운 PPO 객체 생성

    # 5. Locate checkpoint files (Use absolute path)
    # ckpt_dir 변수 이름을 checkpoints_dir_abs로 변경했으므로 맞춰줌
    # pick Actor/Critic paths
    if episode is None:
        candidates = []
        for fname in os.listdir(checkpoints_dir_abs): # 절대 경로 사용
            if fname.endswith('_Actor.pth'):
                parts = fname.split('_')
                try:
                    num = int(parts[1])
                    candidates.append((num, fname))
                except Exception:
                    continue
        if not candidates:
            raise FileNotFoundError(f'No Actor checkpoint found in {checkpoints_dir_abs}')
        episode = max(candidates, key=lambda x: x[0])[0]

    # 경로 생성 시 pathlib 사용
    actor_path = checkpoints_dir_abs / f'episode_{episode}_Actor.pth'
    critic_path = checkpoints_dir_abs / f'episode_{episode}_Critic.pth'

    # 체크포인트 파일 존재 확인 추가
    if not actor_path.is_file():
        raise FileNotFoundError(f"Actor checkpoint file not found: {actor_path}")
    if not critic_path.is_file():
         raise FileNotFoundError(f"Critic checkpoint file not found: {critic_path}")

    # 6. Load model objects (Use absolute path)
    # 보안 경고는 여전히 유효함: weights_only=False 사용 시 주의
    agent.policy.Actor = torch.load(str(actor_path), map_location=device, weights_only=False)
    agent.policy.Critic = torch.load(str(critic_path), map_location=device, weights_only=False)
    agent.policy.Actor.eval()
    agent.policy.Critic.eval()
    return agent

# g2p2c_agent_api.py 의 infer_action 함수 수정 제안

def infer_action(
    agent: PPO, # 타입을 PPO로 명시 (선택 사항)
    history: np.ndarray,
    feat: Optional[np.ndarray] = None,
) -> float:
    """
    Perform inference using loaded PPO agent.
    - history: np.ndarray shape (12, n_features) # args.feature_history 사용하도록 변경
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


    # 2. Prepare feat (기존 로직 유지, agent.args 사용)
    if feat is None:
        n_hf = agent.args.n_handcrafted_features
        feat = np.zeros((1, n_hf), dtype=np.float32)
    elif not isinstance(feat, np.ndarray):
        feat = np.array(feat, dtype=np.float32)

    feat_tensor = feat.astype(np.float32)
    if feat_tensor.ndim == 1:
        feat_tensor = feat_tensor.reshape(1, -1)
    expected_feat_shape_len = agent.args.n_handcrafted_features
    if feat_tensor.shape[1] != expected_feat_shape_len:
         raise ValueError(f'feat shape must be (1, {expected_feat_shape_len}), got {feat_tensor.shape}')
    if not np.all(np.isfinite(feat_tensor)):
        raise ValueError('feat contains NaN or Inf')

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