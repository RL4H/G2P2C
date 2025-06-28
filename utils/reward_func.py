import math
import torch
import numpy as np
from collections import deque
from utils import core
from utils.core import custom_reward, custom_reward_traj


def composite_reward_safe(args, state=None, reward=None):
    """
    안전한 보상 함수 - NaN 값 처리 포함
    """
    MAX_GLUCOSE = 600
    MIN_GLUCOSE = 39
    TARGET_GLUCOSE = 125
    
    # 입력 검증
    if state is None:
        print("WARNING: State is None in reward function")
        return 0.0
    
    # NaN 또는 무한값 처리
    if not np.isfinite(state):
        print(f"WARNING: Invalid state value: {state}")
        return -10.0
    
    # 상태 범위 클리핑
    state = float(np.clip(state, MIN_GLUCOSE, MAX_GLUCOSE))
    
    try:
        if reward is None:
            # custom_reward 함수 안전 호출
            reward = custom_reward([state])
            if not np.isfinite(reward):
                print(f"WARNING: custom_reward returned invalid value: {reward}")
                reward = 0.0
        
        # 안전한 정규화
        x_max = 0  # 최적 상태에서의 보상
        x_min = custom_reward([MAX_GLUCOSE])
        
        if not np.isfinite(x_min):
            x_min = -10.0
            
        if abs(x_max - x_min) < 1e-8:
            normalized_reward = 0.0
        else:
            normalized_reward = (reward - x_min) / (x_max - x_min)
            
        if not np.isfinite(normalized_reward):
            normalized_reward = 0.0
            
    except Exception as e:
        print(f"WARNING: Error in reward calculation: {e}")
        normalized_reward = 0.0
    
    # 혈당 범위 기반 보상 조정
    if state <= 40:
        final_reward = -15.0
    elif state >= MAX_GLUCOSE:
        final_reward = 0.0
    else:
        final_reward = normalized_reward
    
    # 최종 안전성 검사
    if not np.isfinite(final_reward):
        final_reward = 0.0
        
    return float(final_reward)


def composite_reward_simple(args, state=None, reward=None):
    """
    매우 단순하고 안정적인 보상 함수 (디버깅용)
    """
    if state is None or not np.isfinite(state):
        return 0.0
    
    state = float(np.clip(state, 39, 600))
    
    # 혈당 목표 범위: 70-180 mg/dL
    if 70 <= state <= 180:
        # 목표 범위 내: 긍정적 보상 (0~1)
        distance_from_target = abs(state - 125)
        reward = 1.0 - (distance_from_target / 55.0)
    elif state < 70:
        # 저혈당: 강한 패널티
        reward = -2.0 - (70 - state) / 10.0
    else:  # state > 180
        # 고혈당: 중간 패널티  
        reward = -1.0 - (state - 180) / 50.0
    
    return float(np.clip(reward, -10.0, 2.0))


def composite_reward(args, state=None, reward=None):
    """
    기본 함수 - 안전한 버전으로 리다이렉트
    """
    return composite_reward_simple(args, state, reward)
