## main.py for G2P2C FastAPI Agent Wrapper

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

# ----- [G2P2C 실제 에이전트 모델 로드 및 초기화] -----
import torch
from g2p2c_agent_api import load_agent, infer_action

# 에이전트 인스턴스 생성 (모델 로드 및 체크포인트 적용)
_AGENT = load_agent(
    args_json_path="results/test/args.json",
    params_py_path="results/test/code/parameters.py",
    device=torch.device("cpu"),
)

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

        # 3. G2P2C agent inference 호출
        cur_state = hist_np  # shape (12,2): [[bg, ins], ...]
        feat = np.array(req.feat, dtype=np.float32).reshape(1, -1) if req.feat else np.zeros((1, _AGENT.args.n_handcrafted_features), dtype=np.float32)
        action = infer_action(_AGENT, cur_state, feat)

        # 4. 범위 클리핑(안전): 반드시 0.0~5.0 U/h
        action_clipped = float(np.clip(action, 0.0, 5.0))

        # 5. 응답
        return {"insulin_action_U_per_h": action_clipped}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"G2P2C agent inference error: {str(e)}")