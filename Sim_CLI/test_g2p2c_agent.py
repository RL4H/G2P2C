import os
import sys
import json
import importlib.util
import torch
import numpy as np

# 경로 세팅 (상대경로 등 환경에 맞게 수정 필요)
sys.path.extend(['.', './agents/g2p2c', './utils'])

from agents.g2p2c.g2p2c import G2P2C
from utils.options import Options
from Sim_CLI.g2p2c_agent_api import load_agent, infer_action

## (1) Agent 로드 및 초기화
g2p2c_agent = load_agent(
    args_json_path="results/test/args.json",
    params_py_path="results/test/code/parameters.py",
    device=torch.device("cpu"),
)

# --- (3) 추론 및 예외 처리 테스트 ---
def test_g2p2c_inference(verbose=True):
    # 배치 크기 1, feat는 infer_action 내에서 기본 생성

    valid_inputs = [
        np.column_stack([np.random.uniform(80, 200, 12), np.random.uniform(0, 5,
12)]).astype(np.float32),
        np.column_stack([np.full(12, 40), np.full(12, 0)]).astype(np.float32),
        np.column_stack([np.full(12, 500), np.full(12, 5)]).astype(np.float32),
        np.column_stack([np.linspace(90, 160, 12), np.linspace(0, 2, 12)]).astype(np.float32),
    ]
    invalid_inputs = [
        np.random.uniform(80, 200, (11, 2)),                  # shape 오류
        np.random.uniform(80, 200, (12, 3)),                  # 불필요한 features
        np.full((12, 2), np.nan).astype(np.float32),          # 모두 NaN
    ]

    print("\n== 정상 입력 테스트 ==")
    for idx, arr in enumerate(valid_inputs):
        try:
            # infer_action 내에서 입력 검증 및 배치 차원 처리
            action = infer_action(g2p2c_agent, arr)
            print(f"[{idx}] PASS - shape: {arr.shape}, action: {action}")
        except Exception as e:
            print(f"[{idx}] FAIL! 예외: {e}")


    print("\n== 예외 입력 테스트 ==")
    for idx, arr in enumerate(invalid_inputs):
        try:
            # invalid input → infer_action에서 예외 발생해야 함
            action = infer_action(g2p2c_agent, arr)
            print(f"ERROR: Invalid input[{idx}] did NOT raise (shape={arr.shape})! action: {action}")
        except Exception as e:
            print(f"[{idx}] 올바르게 예외 발생: {e}")

if __name__ == "__main__":
    test_g2p2c_inference(verbose=True)