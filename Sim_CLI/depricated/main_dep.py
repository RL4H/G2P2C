## main.py for G2P2C FastAPI Agent Wrapper (with Startup Event)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict # Dict 추가
from contextlib import asynccontextmanager # asynccontextmanager 추가
import numpy as np
import torch
# .g2p2c_agent_api 임포트는 유지
from .g2p2c_agent_api import load_agent, infer_action # g2p2c 사용할 경우. !!!!!!!!!!!!!
# from .ppo_agent_api import load_agent, infer_action # ppo 사용할 경우 !!!!!!!!!!!!!!!!!

import traceback # traceback 모듈 추가
import sys # sys 모듈 추가

# ----- 전역 변수 대신 상태 저장용 딕셔너리 사용 -----
# 에이전트 인스턴스를 앱의 lifespan 상태에 저장
app_state: Dict = {}

# ----- FastAPI Lifespan 설정 -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행될 코드
    print("INFO:     Loading agent...")
    # load_agent를 호출하여 에이전트 로드
    # app_state["agent"] = load_agent(device=torch.device("cpu")) # !!!!!!!!!!!!!!!!!!!!!
    # app_state["agent"] = load_agent(device=torch.device("cpu"), episode=195) 
    app_state["agent"] = load_agent(device=torch.device("cpu"), episode=15) 
    print("INFO:     Agent loaded successfully.")
    yield
    # 앱 종료 시 실행될 코드 (필요시)
    print("INFO:     Application shutdown.")
    app_state.clear()

# ----- FastAPI 모델 정의 -----
class StateRequest(BaseModel):
    history: List[List[float]] = Field(..., description="길이 12, 각 요소는 [혈당(mg/dL), 인슐린(U/h)]")
    feat: Optional[List[float]] = None  # 후속 확장을 위해

class ActionResponse(BaseModel):
    insulin_action_U_per_h: float = Field(..., description="0.0~5.0 U/h 사이의 실수")

# ----- FastAPI 어플리케이션 생성 (lifespan 연결) -----
app = FastAPI(title="Agent API", lifespan=lifespan)

# CORS 허용 (기존과 동일)
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
    # 시작 시 로드된 에이전트 인스턴스 가져오기
    agent = app_state.get("agent")
    if not agent:
        raise HTTPException(status_code=500, detail="agent not loaded.")

    try:
        # 1. 입력 확인 및 변환 (기존과 동일)
        hist_np = np.array(req.history)
        if hist_np.shape != (12, 2):
             raise ValueError(f'history 형상 오류: {hist_np.shape}, (12,2) 필요함')
        # ... (NaN, 인슐린 범위 검사 등 기존 검증 로직) ...
        bg_array = hist_np[:, 0]
        ins_array = hist_np[:, 1]
        for i, (bg, ins) in enumerate(zip(bg_array, ins_array)):
            if not np.isfinite(bg) or not np.isfinite(ins):
                raise ValueError(f'입력 NaN/Inf at idx {i}: bg={bg}, ins={ins}')
            if not (0.0 <= ins <= 5.0):
                raise ValueError(f'입력 인슐린 값 비정상 at idx {i}: {ins}')

        # 2. feat vector 준비 (기존과 동일)
        feat = np.array(req.feat, dtype=np.float32).reshape(1, -1) if req.feat else np.zeros((1, agent.args.n_handcrafted_features), dtype=np.float32)

        # 3. G2P2C agent inference 호출 (로드된 agent 사용)
        cur_state = hist_np
        action = infer_action(agent, cur_state, feat) # 전역 _AGENT 대신 app_state["agent"] 사용

        # 4. 범위 클리핑 (기존과 동일)
        action_clipped = float(np.clip(action, 0.0, 5.0))

        # 5. 응답 (기존과 동일)
        return {"insulin_action_U_per_h": action_clipped}

    except Exception as e:
        # !!!!! 오류 발생 시 Traceback을 콘솔(stderr)에 직접 출력 !!!!!
        print(f"\n!!! ERROR processing /predict_action !!!", file=sys.stderr)
        traceback.print_exc() # 상세 Traceback 출력
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", file=sys.stderr)
        # !!!!! 추가된 로깅 끝 !!!!!

        # 기존 HTTPException 발생은 유지
        raise HTTPException(status_code=500, detail=f"agent inference error: {str(e)}")