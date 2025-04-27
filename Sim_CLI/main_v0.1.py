## main.py for G2P2C FastAPI Agent Wrapper

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

# ----- [G2P2C 실제 에이전트 모델 임포트/초기화 예시] -----
# from agents.g2p2c.g2p2c import G2P2C
# (설정 및 가중치 로딩 등)
# g2p2c_agent = G2P2C(...)

# 일단 더미 함수로 구현. 
def get_action_dummy(cur_state, feat=None):
    # 1. 입력값 검증 (cur_state: 12 x 2 배열)
    if not isinstance(cur_state, np.ndarray):
        raise ValueError('cur_state must be a numpy ndarray')
    if cur_state.shape != (12, 2):
        raise ValueError(f'cur_state shape must be (12,2), got {cur_state.shape}')
    # 값 검증
    if not np.all(np.isfinite(cur_state)):
        raise ValueError('Non-finite (NaN/Inf) value in cur_state')
    # 값이 float임을 보장(엄격한 환경에서는 개별 체크도 가능)
    if not np.issubdtype(cur_state.dtype, np.floating):
        raise ValueError('cur_state elements must be float')

    # 2. (데모) 0.1~4.9 사이의 랜덤값 반환
    action = float(np.random.uniform(low=0.1, high=4.9))
    return action

# ----- FastAPI 모델 정의 -----

class StateRequest(BaseModel):
    history: List[List[float]] = Field(..., description="길이 12, 각 요소는 [혈당(mg/dL), 인슐린(U/h)]")
    feat: Optional[List[float]] = None  # 후속 확장을 위해

class ActionResponse(BaseModel):
    insulin_action_U_per_h: float = Field(..., description="0.0~5.0 U/h 사이의 실수")

# ----- FastAPI 어플리케이션 생성 -----

app = FastAPI(title="G2P2C Agent API")

# CORS 허용 (로컬 테스트/다중 origin 대비)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- 실제 Inference 핸들러 -----

@app.post("/predict_action", response_model=ActionResponse)
def predict_action(req: StateRequest):
    try:
        # 1. 입력 확인 및 변환
        hist_np = np.array(req.history)  # shape (12,2)
        if hist_np.shape != (12, 2):
            raise ValueError(f'history 형상 오류: {hist_np.shape}, (12,2) 필요함')
        bg_array = hist_np[:, 0]
        ins_array = hist_np[:, 1]
        for i, (bg, ins) in enumerate(zip(bg_array, ins_array)):
            if not np.isfinite(bg) or not np.isfinite(ins):
                raise ValueError(f'입력 NaN/Inf at idx {i}: bg={bg}, ins={ins}')
            if not (0.0 <= ins <= 5.0):  # 인슐린은 항상 0~5U/h 클리핑된 값이어야 함
                raise ValueError(f'입력 인슐린 값 비정상 at idx {i}: {ins}')

        # 2. feat vector 확장 가능 (현재 미사용)
        #   feat = req.feat if req.feat is not None else 기본값

        # 3. G2P2C agent에 데이터 전달 (예시코드, 실제 구현시 예제부분 바꿔야 함)
        cur_state = hist_np  # shape (12,2): [[bg,ins] ...]
        feat = np.zeros((1, 1), dtype=np.float32)  # feat 미사용시 placeholder

        # --- 실제 행동 추론 ----
        # 실제 배포시 아래 로직을 인공췌장 agent inference로 대체!
        # 예시: random 또는 가장 최근 BG 기반 naive 룰
        # action = g2p2c_agent.get_action(cur_state, feat)["action"][0]
        action = get_action_dummy(hist_np)

        # 4. 범위 클리핑(안전): 반드시 0.0~5.0 U/h
        action_clipped = float(np.clip(action, 0.0, 5.0))

        # 5. 응답
        return {"insulin_action_U_per_h": action_clipped}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"G2P2C agent inference error: {str(e)}")