# Sim_CLI/main.py
"""FastAPI 서버를 구동하여 DMMS.R 시뮬레이터와 G2P2C 에이전트 사이를 중계한다.

DMMS.R 에서 JavaScript 플러그인을 통해 호출되는 ``/predict_action`` 엔드포인트를
제공하며, 에이전트 초기화와 CSV 로그 저장 기능을 담당한다. 해당 서버를 실행한
상태에서 DMMS.R 시뮬레이션을 수행하면 매 5분마다 인슐린 주입량을 예측하여
응답한다.
"""

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
import json
import csv

# --- 프로젝트 루트 경로 설정 ---
_current_script_dir = Path(__file__).resolve().parent
_project_root = _current_script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# ------------------------------

from Sim_CLI.g2p2c_agent_api import load_agent, infer_action
from utils.statespace import StateSpace
from utils.reward_func import composite_reward

# ----- CSV 로깅 설정 -----
LOG_FILE_DIR = _project_root / "results" / "dmms_realtime_logs_v2" # 로그 저장 디렉토리 (이름 변경)
LOG_FILE_NAME = "dmms_to_simglucose_log_v2.csv"
LOG_FILE_PATH = LOG_FILE_DIR / LOG_FILE_NAME

SimglucoseLogHeader = [
    'epi', 't', 'cgm', 'meal', 'ins', 'rew', 'rl_ins',
    'mu', 'sigma', 'prob', 'state_val', 'day_hour', 'day_min'
]
# ---------------------------

app_state: Dict = {}

def format_log_data(data):
    """CSV 로 기록하기 위해 NumPy/Tensor 객체를 파이썬 기본 타입으로 변환한다."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()
    if isinstance(data, list):
        return [format_log_data(item) for item in data]
    if isinstance(data, dict):
        return {k: format_log_data(v) for k, v in data.items()}
    return data

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작과 종료 시 필요한 초기화를 담당한다.

    G2P2C 에이전트를 로드하고 ``StateSpace`` 를 생성하며, CSV 로그 파일의
    헤더를 준비한다. ``yield`` 이후에는 종료 처리를 수행한다.
    """
    print("INFO:     Application startup...")
    app_state["api_call_counter"] = 0 # API 호출 횟수 카운터 (t 스텝으로 사용)
    app_state["current_episode"] = 1   # 현재 에피소드 번호 (기본값)
    app_state["experience_buffer"] = []  # 상태-행동-보상 저장용 버퍼
    app_state["prev_state"] = None
    app_state["prev_action"] = None
    app_state["last_cgm"] = None

    try:
        LOG_FILE_DIR.mkdir(parents=True, exist_ok=True)
        if not LOG_FILE_PATH.is_file() or os.path.getsize(LOG_FILE_PATH) == 0: # 파일이 없거나 비어있으면 헤더 작성
            with open(LOG_FILE_PATH, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(SimglucoseLogHeader)
            print(f"INFO:     [LIFESPAN_CSV_LOG] Created new log file with header: {LOG_FILE_PATH}")
        else:
            print(f"INFO:     [LIFESPAN_CSV_LOG] Appending to existing log file: {LOG_FILE_PATH}")
    except Exception as e:
        print(f"ERROR:    [LIFESPAN_CSV_LOG] Failed to initialize CSV log file: {e}", file=sys.stderr)
    
    try:
        print("INFO:     [LIFESPAN_LOAD_AGENT_START] Loading G2P2C agent...")
        loaded_g2p2c_agent = load_agent(device=torch.device("cpu"), episode=195)
        app_state["agent"] = loaded_g2p2c_agent
        print("INFO:     [LIFESPAN_LOAD_AGENT_END] G2P2C Agent loaded successfully.")

        agent_args = loaded_g2p2c_agent.args
        app_state["state_space_instance"] = StateSpace(args=agent_args)
        app_state["agent_args_for_statespace"] = agent_args
        
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

    except FileNotFoundError as fnf_error:
        print(f"ERROR:    [LIFESPAN] Failed to load agent or StateSpace components: {fnf_error}", file=sys.stderr)
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

class StateRequest(BaseModel):
    history: List[List[float]] = Field(
        ..., description="길이 12의 리스트, 각 요소는 [혈당(mg/dL), 이전 인슐린 액션(U/h)]."
    )
    # 'hour' 필드는 더 이상 클라이언트에서 전달하지 않는다. 서버가 내부에서 생성한다.
    meal: Optional[float] = Field(
        0.0, description="현재 스텝의 식사량 (탄수화물 g, 없으면 0.0)."
    )

class ActionResponse(BaseModel):
    insulin_action_U_per_h: float = Field(..., description="0.0~5.0 U/h 사이의 실수")

app = FastAPI(title="G2P2C Agent API with Realtime Simglucose-like Logging (v2)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_action", response_model=ActionResponse)
def predict_action(req: StateRequest):
    """에이전트에게 상태를 전달하고 인슐린 주입률을 응답받는 엔드포인트."""
    agent = app_state.get("agent")
    state_space = app_state.get("state_space_instance")
    agent_args = app_state.get("agent_args_for_statespace")

    # API 호출 카운터 증가 (이것을 't' 스텝으로 사용)
    app_state["api_call_counter"] = app_state.get("api_call_counter", 0) + 1
    current_t_step = app_state["api_call_counter"] - 1 # 0부터 시작하는 스텝

    # --- 이전 로그 로직은 유지 ---
    try:
        request_body_for_log = json.dumps(format_log_data(req.dict()))
    except Exception:
        request_body_for_log = str(req.dict())
    print(f"INFO:     [LOG_MAIN_REQUEST_START] Call #{current_t_step}. Received /predict_action. Body: {request_body_for_log}")
    # ----------------------------

    if not agent or not state_space or not agent_args:
        print("ERROR:    [LOG_MAIN] Agent, StateSpace, or agent_args not loaded properly.", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Agent, StateSpace, or agent_args not loaded.")

    try:
        if not req.history or len(req.history[-1]) != 2 :
            raise ValueError(f"Invalid 'history' format. Last element: {req.history[-1] if req.history else 'N/A'}")
        
        current_cgm_raw = req.history[-1][0]
        previous_insulin_action_raw = req.history[-1][1]
                # 시간 정보는 서버에서 생성: API 호출 순서(current_t_step)를 그대로 사용
        generated_hour = current_t_step
        current_meal_raw = req.meal if req.meal is not None else 0.0

        print(
            f"INFO:     [LOG_MAIN_RAW_INPUTS_TO_STATESPACE] Raw inputs for StateSpace.update(): "
            f"cgm={current_cgm_raw:.2f}, ins(prev_action)={previous_insulin_action_raw:.4f}, "
            f"hour_generated={generated_hour:.1f}, t_step_for_state={current_t_step}, "
            f"meal={current_meal_raw:.1f}"
        )

        if not (
            np.isfinite(current_cgm_raw)
            and np.isfinite(previous_insulin_action_raw)
            and np.isfinite(generated_hour)
            and np.isfinite(current_meal_raw)
        ):
            raise ValueError(f"NaN/Inf in raw inputs.")

        processed_state_hist_np, processed_hc_state_list = state_space.update(
            cgm=current_cgm_raw,
            ins=previous_insulin_action_raw,
            meal=current_meal_raw,
            hour=current_t_step,
            meal_type=0,
            carbs=0      
        )
        
        print(f"INFO:     [LOG_MAIN_STATESPACE_OUTPUT] StateSpace.update() returned: \n"
              f"           processed_state_hist_np (shape {processed_state_hist_np.shape}, last 2 rows):\n{format_log_data(processed_state_hist_np[-2:])}\n"
              f"           processed_hc_state_list: {format_log_data(processed_hc_state_list)}")
        
        processed_hc_state_np = np.array(processed_hc_state_list, dtype=np.float32).reshape(1, agent_args.n_handcrafted_features)

        print(f"INFO:     [LOG_MAIN_INPUTS_TO_INFER_ACTION] Inputs to infer_action():\n"
              f"           state_hist_processed (shape {processed_state_hist_np.shape}, last 2 rows):\n{format_log_data(processed_state_hist_np[-2:])}\n"
              f"           hc_state_processed (shape {processed_hc_state_np.shape}): {format_log_data(processed_hc_state_np)}")

        action_U_per_h = infer_action(
            agent=agent,
            state_hist_processed=processed_state_hist_np,
            hc_state_processed=processed_hc_state_np
        )

        # ----- 경험 버퍼 업데이트 -----
        current_state_dict = {
            "state_hist": processed_state_hist_np.tolist(),
            "hc_state": processed_hc_state_np.flatten().tolist(),
        }

        prev_state = app_state.get("prev_state")
        prev_action = app_state.get("prev_action")
        if prev_state is not None and prev_action is not None:
            reward_val = composite_reward(agent_args, state=current_cgm_raw)
            experience = {
                "state": prev_state,
                "action": prev_action,
                "reward": float(reward_val),
                "next_state": current_state_dict,
            }
            app_state["experience_buffer"].append(experience)

        app_state["prev_state"] = current_state_dict
        app_state["prev_action"] = float(action_U_per_h)
        app_state["last_cgm"] = float(current_cgm_raw)

        # !!! 위 수치가 인슐린 수치에 해당함. 
        # 아래는 디버깅 시 사용할 수 있는 코드.
        # action_U_per_h *= 0

        print(f"INFO:     [LOG_MAIN_FINAL_ACTION_FROM_API] Action from infer_action (to be returned to JS): {action_U_per_h:.4f} U/h")

        # --- Simglucose 형식 로그 기록 ---
        try:
            # 'day_min' 계산: current_t_step (5분 간격 스텝)을 사용하여 시간(분)으로 변환 후 60으로 나눈 나머지
            # JavaScript가 5분마다 API를 호출한다고 가정.
            # 예: t_step=0 -> 0분, t_step=1 -> 5분, t_step=11 -> 55분, t_step=12 -> 60분(다음 시간 0분)
            calculated_total_minutes = current_t_step * 5
            day_min_for_log = calculated_total_minutes % 60

            log_row_dict = {
                'epi': app_state.get("current_episode", 1), # 현재 에피소드 번호 사용
                't': current_t_step, # API 호출 기반 5분 간격 스텝
                'cgm': current_cgm_raw,
                'meal': current_meal_raw,
                'ins': action_U_per_h,
                'rew': np.nan,
                'rl_ins': action_U_per_h,
                'mu': np.nan,
                'sigma': np.nan,
                'prob': np.nan,
                'state_val': np.nan,
                'day_hour': int(generated_hour),  # 서버에서 생성한 시간 정보
                'day_min': day_min_for_log
            }
            with open(LOG_FILE_PATH, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=SimglucoseLogHeader)
                writer.writerow(log_row_dict)
            print(
                f"INFO:     [LOG_MAIN_CSV_WRITE] Successfully wrote to {LOG_FILE_PATH}: "
                f"(epi={log_row_dict['epi']}, t_step={log_row_dict['t']}, generated_hour={generated_hour})"
            )
            print(
                f"INFO:     [LOG_MAIN_CSV_WRITE] Successfully wrote to {LOG_FILE_PATH}: "
                f"(epi={log_row_dict['epi']}, t_step={log_row_dict['t']}, generated_hour={generated_hour})"
            )
        except Exception as csv_e:
            print(f"ERROR:    [LOG_MAIN_CSV_WRITE] Failed to write Simglucose format log: {csv_e}", file=sys.stderr)
        # ---------------------------------

        return ActionResponse(insulin_action_U_per_h=action_U_per_h)

    except ValueError as ve:
        print(f"\n!!! ValueError processing /predict_action: {ve} !!!", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Input validation error: {str(ve)}")
    except Exception as e:
        print(f"\n!!! ERROR processing /predict_action !!!", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent inference error: {str(e)}")


class EpisodeEndResponse(BaseModel):
    episode: int
    steps: int


@app.post("/episode_end", response_model=EpisodeEndResponse)
def episode_end():
    """Handle end of an episode and return collected experience."""
    exp_buffer = app_state.get("experience_buffer", [])

    prev_state = app_state.get("prev_state")
    prev_action = app_state.get("prev_action")
    if prev_state is not None and prev_action is not None:
        last_cgm = app_state.get("last_cgm")
        agent_args = app_state.get("agent_args_for_statespace")
        if last_cgm is not None and agent_args is not None:
            final_reward = composite_reward(agent_args, state=last_cgm)
        else:
            final_reward = 0.0
        exp_buffer.append({
            "state": prev_state,
            "action": prev_action,
            "reward": float(final_reward),
            "next_state": None,
        })

    episode_num = app_state.get("current_episode", 1)
    num_steps = len(exp_buffer)

    # Reset episode-related states
    app_state["experience_buffer"] = []
    app_state["prev_state"] = None
    app_state["prev_action"] = None
    app_state["last_cgm"] = None
    app_state["api_call_counter"] = 0
    app_state["current_episode"] = episode_num + 1

    return EpisodeEndResponse(episode=episode_num, steps=num_steps)

