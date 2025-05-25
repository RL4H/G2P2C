# Sim_CLI/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import numpy as np
import torch
import traceback
import sys
import os
from pathlib import Path
import json # JSON 로깅을 위해 추가

# --- 프로젝트 루트 경로 설정 ---
# 이 파일(main.py)의 현재 위치를 기준으로 프로젝트 루트를 계산합니다.
# Sim_CLI 폴더 내에 main.py가 위치한다고 가정합니다.
_current_script_dir = Path(__file__).resolve().parent
_project_root = _current_script_dir.parent # G2P2C 폴더가 프로젝트 루트라고 가정
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# ------------------------------

# --- 필요한 모듈 임포트 ---
from Sim_CLI.g2p2c_agent_api import load_agent, infer_action # G2P2C 에이전트용 API
from utils.statespace import StateSpace # StateSpace 클래스 임포트
# Options 클래스는 load_agent 내부에서 사용되므로 여기서 직접 임포트할 필요는 없을 수 있습니다.
# from utils.options import Options
# ---------------------------

# ----- 전역 변수 대신 상태 저장용 딕셔너리 사용 -----
app_state: Dict = {}

# ----- 로깅용 헬퍼 함수 -----
def format_log_data(data):
    """
    NumPy 배열이나 PyTorch 텐서를 로깅하기 좋은 리스트 형태로 변환합니다.
    딕셔너리나 리스트 내부의 요소들도 재귀적으로 변환합니다.
    """
    if isinstance(data, np.ndarray):
        return data.tolist() # NumPy 배열을 리스트로 변환
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist() # PyTorch 텐서를 CPU로 옮긴 후 NumPy 배열로 변환, 다시 리스트로 변환
    if isinstance(data, list):
        return [format_log_data(item) for item in data] # 리스트 내부 요소들도 재귀적으로 변환
    if isinstance(data, dict):
        return {k: format_log_data(v) for k, v in data.items()} # 딕셔너리 내부 값들도 재귀적으로 변환
    # 다른 타입은 그대로 반환 (예: int, float, str, bool)
    return data
# ---------------------------

# ----- FastAPI Lifespan 설정 -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO:     Application startup...")
    try:
        print("INFO:     [LIFESPAN_LOAD_AGENT_START] Loading G2P2C agent...")
        # 에피소드 번호는 실제 훈련된 모델에 맞게 조정 필요
        loaded_g2p2c_agent = load_agent(device=torch.device("cpu"), episode=195)
        app_state["agent"] = loaded_g2p2c_agent
        print("INFO:     [LIFESPAN_LOAD_AGENT_END] G2P2C Agent loaded successfully.")

        agent_args = loaded_g2p2c_agent.args
        app_state["state_space_instance"] = StateSpace(args=agent_args)
        app_state["agent_args_for_statespace"] = agent_args # StateSpace 초기화에 사용된 args도 저장
        
        # --- [LOG_LIFESPAN_STATESPACE_INIT] ---
        # StateSpace 초기화 시 사용된 주요 파라미터들을 로그로 남깁니다.
        print(f"INFO:     [LOG_LIFESPAN_STATESPACE_INIT] StateSpace initialized with: "
              f"feature_history={agent_args.feature_history}, "
              f"glucose_min={agent_args.glucose_min}, glucose_max={agent_args.glucose_max}, "
              f"insulin_min={agent_args.insulin_min}, insulin_max={agent_args.insulin_max}, "
              f"t_meal={agent_args.t_meal}, "
              f"use_meal_announcement={getattr(agent_args, 'use_meal_announcement', 'N/A')}, "
              f"use_tod_announcement={getattr(agent_args, 'use_tod_announcement', 'N/A')}, "
              f"use_carb_announcement={getattr(agent_args, 'use_carb_announcement', 'N/A')}")
        print(f"INFO:     [LOG_LIFESPAN_STATESPACE_INIT] StateSpace related args from agent: "
              f"n_features={agent_args.n_features}, "
              f"n_handcrafted_features={agent_args.n_handcrafted_features}, "
              f"use_handcraft={getattr(agent_args, 'use_handcraft', 'N/A')}")
        # --------------------------------------

    except FileNotFoundError as fnf_error:
        print(f"ERROR:    [LIFESPAN] Failed to load agent or StateSpace components: {fnf_error}", file=sys.stderr)
        print(f"DEBUG:    [LIFESPAN] Project root used for loading: {_project_root}", file=sys.stderr)
        print(f"DEBUG:    [LIFESPAN] Current working directory: {os.getcwd()}", file=sys.stderr)
        app_state["agent"] = None
        app_state["state_space_instance"] = None
    except Exception as e:
        print(f"ERROR:    [LIFESPAN] Unexpected error during agent/StateSpace loading: {e}", file=sys.stderr)
        traceback.print_exc()
        app_state["agent"] = None
        app_state["state_space_instance"] = None
    yield
    print("INFO:     Application shutdown.")
    app_state.clear()

# ----- FastAPI 모델 정의 -----
class StateRequest(BaseModel):
    history: List[List[float]] = Field(..., description="길이 12의 리스트, 각 요소는 [혈당(mg/dL), 이전 인슐린 액션(U/h)]. JavaScript에서 전송하는 원본 값.")
    hour: float = Field(..., description="현재 시간 (0-23 사이의 실수 또는 정수). JavaScript에서 전송해야 함.")
    meal: Optional[float] = Field(0.0, description="현재 스텝의 식사량 (탄수화물 g, 없으면 0.0). JavaScript에서 전송해야 함.")

class ActionResponse(BaseModel):
    insulin_action_U_per_h: float = Field(..., description="0.0~5.0 U/h 사이의 실수")

# ----- FastAPI 어플리케이션 생성 (lifespan 연결) -----
app = FastAPI(title="G2P2C Agent API with StateSpace Logging", lifespan=lifespan)

# CORS 허용
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
    agent = app_state.get("agent")
    state_space = app_state.get("state_space_instance")
    agent_args = app_state.get("agent_args_for_statespace")

    # --- [LOG_MAIN_REQUEST_START] ---
    # 요청 시작 시점과 받은 데이터를 JSON 형태로 로그로 남깁니다.
    try:
        request_body_for_log = json.dumps(format_log_data(req.dict()))
    except Exception: # 직렬화 실패 시 간단한 문자열로
        request_body_for_log = str(req.dict())
    print(f"INFO:     [LOG_MAIN_REQUEST_START] Received /predict_action. Body: {request_body_for_log}")
    # ---------------------------------

    if not agent or not state_space or not agent_args:
        print("ERROR:    [LOG_MAIN] Agent, StateSpace, or agent_args not loaded properly during startup.", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Agent, StateSpace, or agent_args not loaded. Check server logs.")

    try:
        # 1. JavaScript로부터 받은 원본 데이터 추출
        if not req.history or len(req.history[-1]) != 2 :
            raise ValueError(f"Invalid 'history' format or empty/malformed history received. Last element: {req.history[-1] if req.history else 'N/A'}")
        
        current_cgm_raw = req.history[-1][0]
        previous_insulin_action_raw = req.history[-1][1] # StateSpace.update의 'ins'로 들어감
        current_hour_raw = req.hour
        current_meal_raw = req.meal if req.meal is not None else 0.0

        # --- [LOG_MAIN_RAW_INPUTS_TO_STATESPACE] ---
        print(f"INFO:     [LOG_MAIN_RAW_INPUTS_TO_STATESPACE] Raw inputs for StateSpace.update(): "
              f"cgm={current_cgm_raw:.2f}, ins(prev_action)={previous_insulin_action_raw:.4f}, "
              f"hour={current_hour_raw:.1f}, meal={current_meal_raw:.1f}")
        # -------------------------------------------

        if not (np.isfinite(current_cgm_raw) and
                np.isfinite(previous_insulin_action_raw) and
                np.isfinite(current_hour_raw) and
                np.isfinite(current_meal_raw)):
            raise ValueError(f"NaN/Inf in raw inputs: CGM={current_cgm_raw}, PrevIns={previous_insulin_action_raw}, Hour={current_hour_raw}, Meal={current_meal_raw}")

        # 2. StateSpace를 사용하여 모델 입력 형태로 변환
        # StateSpace.update()는 내부적으로 정규화를 수행하고 이력을 관리합니다.
        processed_state_hist_np, processed_hc_state_list = state_space.update(
            cgm=current_cgm_raw,
            ins=previous_insulin_action_raw,
            meal=current_meal_raw,
            hour=current_hour_raw,
            meal_type=0, 
            carbs=0      
        )
        
        # --- [LOG_MAIN_STATESPACE_OUTPUT] ---
        # StateSpace.update()의 반환값을 로그로 남깁니다.
        # NumPy 배열은 너무 클 수 있으므로, shape과 마지막 몇 개 행만 출력합니다.
        print(f"INFO:     [LOG_MAIN_STATESPACE_OUTPUT] StateSpace.update() returned: \n"
              f"           processed_state_hist_np (shape {processed_state_hist_np.shape}, last 2 rows):\n{format_log_data(processed_state_hist_np[-2:])}\n"
              f"           processed_hc_state_list: {format_log_data(processed_hc_state_list)}")
        # ----------------------------------
        
        # handcraft_features (리스트)를 NumPy 배열로 변환 및 형태 조정
        processed_hc_state_np = np.array(processed_hc_state_list, dtype=np.float32).reshape(1, agent_args.n_handcrafted_features)

        # --- [LOG_MAIN_INPUTS_TO_INFER_ACTION] ---
        print(f"INFO:     [LOG_MAIN_INPUTS_TO_INFER_ACTION] Inputs to infer_action():\n"
              f"           state_hist_processed (shape {processed_state_hist_np.shape}, last 2 rows):\n{format_log_data(processed_state_hist_np[-2:])}\n"
              f"           hc_state_processed (shape {processed_hc_state_np.shape}): {format_log_data(processed_hc_state_np)}")
        # -----------------------------------------

        # 3. G2P2C agent inference 호출
        action_U_per_h = infer_action(
            agent=agent,
            state_hist_processed=processed_state_hist_np,
            hc_state_processed=processed_hc_state_np
        )

        # action_U_per_h = action_U_per_h * 10

        # --- [LOG_MAIN_FINAL_ACTION_FROM_API] ---
        print(f"INFO:     [LOG_MAIN_FINAL_ACTION_FROM_API] Action from infer_action (to be returned to JS): {action_U_per_h:.4f} U/h")
        # ----------------------------------------

        # 4. 응답
        return {"insulin_action_U_per_h": action_U_per_h}

    except ValueError as ve:
        print(f"\n!!! ValueError processing /predict_action: {ve} !!!", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Input validation error: {str(ve)}")
    except Exception as e:
        print(f"\n!!! ERROR processing /predict_action !!!", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent inference error: {str(e)}")