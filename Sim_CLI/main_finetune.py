# Sim_CLI/main_finetune.py
"""
FastAPI 서버 (미세 조정 모드).
이 서버는 외부 스크립트(예: run_RL_agent_finetune.py)에 의해 실행되며,
해당 스크립트에서 생성 및 관리하는 G2P2C 에이전트 인스턴스를 주입받아 사용합니다.
DMMS.R 시뮬레이터의 JavaScript 플러그인과 통신하여 에이전트의 행동 결정을 중계하고,
실시간 로그 및 경험 데이터를 저장합니다.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import torch # format_log_data 등에서 사용될 수 있음
import traceback
import sys
import os
from pathlib import Path
import json
import csv

# --- 프로젝트 루트 경로 설정 (기존 main.py와 동일하게 유지하여 utils 등 참조) ---
_current_script_dir = Path(__file__).resolve().parent
_project_root = _current_script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# ------------------------------

# StateSpace는 utils에서 가져옵니다. G2P2C 에이전트 자체는 외부에서 주입됩니다.
from utils.statespace import StateSpace
from utils.reward_func import composite_reward
from Sim_CLI.g2p2c_agent_api import infer_action # infer_action은 계속 사용

# ----- 전역 상태 및 로깅 설정 (main.py에서 가져옴) -----
app_state: Dict[str, Any] = {} # FastAPI 앱의 상태를 저장 (주입된 에이전트 포함)

LOG_FILE_DIR = _project_root / "results" / "dmms_realtime_logs_finetune" # 로그 저장 디렉토리 (이름 변경)
LOG_FILE_NAME = "dmms_to_simglucose_log_finetune.csv"
LOG_FILE_PATH = LOG_FILE_DIR / LOG_FILE_NAME
EXPERIENCE_DIR = _project_root / "results" / "dmms_experience_finetune" # 경험 저장 디렉토리 (이름 변경)

SimglucoseLogHeader = [ # 기존 헤더 유지
    'epi', 't', 'cgm', 'meal', 'ins', 'rew', 'rl_ins',
    'mu', 'sigma', 'prob', 'state_val', 'day_hour', 'day_min'
]
# ----------------------------------------------------

def format_log_data(data: Any) -> Any:
    """CSV 로 기록하기 위해 NumPy/Tensor 객체를 파이썬 기본 타입으로 변환합니다."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist() # torch가 import 되어 있어야 함
    if isinstance(data, list):
        return [format_log_data(item) for item in data]
    if isinstance(data, dict):
        return {k: format_log_data(v) for k, v in data.items()}
    return data

def setup_app_state_for_finetuning(
    agent_instance: Any, # 실제로는 G2P2C 타입
    agent_args_for_statespace: Any, # 에이전트의 args 객체
    initial_episode_number: int = 1,
    experiment_dir: Optional[Path] = None # 결과 저장을 위해 실험 디렉토리 경로를 받을 수 있음
) -> None:
    """
    외부(훈련 스크립트)에서 주입된 에이전트와 설정을 사용하여 app_state를 초기화합니다.
    FastAPI 앱이 요청을 처리하기 전에 호출되어야 합니다.
    """
    global app_state, LOG_FILE_DIR, LOG_FILE_PATH, EXPERIENCE_DIR # 전역 변수 수정 명시
    
    print("INFO:     [MAIN_FINETUNE_SETUP] Configuring app_state with injected agent for fine-tuning.")
    app_state["agent"] = agent_instance
    app_state["agent_args_for_statespace"] = agent_args_for_statespace
    
    if experiment_dir:
        # 로그 및 경험 저장 경로를 현재 실험에 맞게 동적으로 설정
        LOG_FILE_DIR = Path(experiment_dir) / "dmms_finetune_logs"
        LOG_FILE_NAME = "dmms_to_simglucose_finetune_log.csv"
        LOG_FILE_PATH = LOG_FILE_DIR / LOG_FILE_NAME
        EXPERIENCE_DIR = Path(experiment_dir) / "dmms_finetune_experience"
        print(f"INFO:     [MAIN_FINETUNE_SETUP] Log Dir: {LOG_FILE_DIR}")
        print(f"INFO:     [MAIN_FINETUNE_SETUP] Experience Dir: {EXPERIENCE_DIR}")


    try:
        state_space_instance = StateSpace(args=agent_args_for_statespace)
        app_state["state_space_instance"] = state_space_instance
        print(f"INFO:     [MAIN_FINETUNE_SETUP] StateSpace initialized with: "
              f"feature_history={getattr(agent_args_for_statespace, 'feature_history', 'N/A')}, "
              f"glucose_min={getattr(agent_args_for_statespace, 'glucose_min', 'N/A')}, "
              f"glucose_max={getattr(agent_args_for_statespace, 'glucose_max', 'N/A')}, "
              f"insulin_min={getattr(agent_args_for_statespace, 'insulin_min', 'N/A')}, "
              f"insulin_max={getattr(agent_args_for_statespace, 'insulin_max', 'N/A')}, "
              # ... 기타 StateSpace 관련 주요 인자 로그 추가 ...
             )
    except Exception as e:
        print(f"ERROR:    [MAIN_FINETUNE_SETUP] Failed to initialize StateSpace: {e}", file=sys.stderr)
        traceback.print_exc()
        app_state["state_space_instance"] = None

    app_state["api_call_counter"] = 0
    app_state["current_episode"] = initial_episode_number
    app_state["experience_buffer"] = []
    app_state["prev_state"] = None
    app_state["prev_action"] = None
    app_state["last_cgm"] = None

    try:
        LOG_FILE_DIR.mkdir(parents=True, exist_ok=True)
        if not LOG_FILE_PATH.is_file() or os.path.getsize(LOG_FILE_PATH) == 0:
            with open(LOG_FILE_PATH, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(SimglucoseLogHeader)
            print(f"INFO:     [MAIN_FINETUNE_SETUP_CSV_LOG] Created new log file with header: {LOG_FILE_PATH}")
        else:
            # 새 미세조정 세션 시작 시 이전 로그에 이어쓰기보다는 새 파일을 만들거나, 파일명에 타임스탬프 추가 고려
            print(f"INFO:     [MAIN_FINETUNE_SETUP_CSV_LOG] Appending to existing log file: {LOG_FILE_PATH}")
    except Exception as e:
        print(f"ERROR:    [MAIN_FINETUNE_SETUP_CSV_LOG] Failed to initialize CSV log file: {e}", file=sys.stderr)
    
    print("INFO:     [MAIN_FINETUNE_SETUP] app_state configured.")
    # 디버깅: 설정된 app_state 내용 일부 출력
    print(f"DEBUG:    [MAIN_FINETUNE_SETUP] Agent in app_state (id): {id(app_state.get('agent'))}")
    print(f"DEBUG:    [MAIN_FINETUNE_SETUP] StateSpace in app_state (id): {id(app_state.get('state_space_instance'))}")

@asynccontextmanager
async def finetuning_lifespan(app_ref: FastAPI): # app_ref는 FastAPI 인스턴스 자체를 받을 수 있음 (여기선 사용 안함)
    """미세 조정 시 사용될 lifespan. 에이전트 관련 설정은 setup_app_state_for_finetuning에서 완료됨."""
    print("INFO:     [MAIN_FINETUNE_LIFESPAN] Application startup...")
    if not app_state.get("agent") or not app_state.get("state_space_instance"):
        error_msg = ("[MAIN_FINETUNE_LIFESPAN] CRITICAL_ERROR: Agent or StateSpace not found in app_state at startup! "
                     "Ensure setup_app_state_for_finetuning() was called correctly before starting the app.")
        print(error_msg, file=sys.stderr)
        # 실제 운영 시 여기서 예외를 발생시켜 서버 시작을 중단시키는 것이 좋을 수 있습니다.
        # raise RuntimeError(error_msg) 
    yield
    print("INFO:     [MAIN_FINETUNE_LIFESPAN] Application shutdown.")
    # 필요한 경우 여기서 app_state의 특정 항목 정리
    # (예: app_state["experience_buffer"] = [])
    # 단, 전체 프로세스 종료 시 Python이 메모리를 정리하므로 필수적이지 않을 수 있음

# FastAPI 앱 인스턴스 생성 (이 파일 내에서 정의)
app = FastAPI(title="G2P2C Agent API (Fine-tuning Mode)", lifespan=finetuning_lifespan)

# CORS 미들웨어 추가 (기존 main.py와 동일)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Pydantic 모델 정의 (기존 main.py와 동일) -----
class StateRequest(BaseModel):
    history: List[List[float]] = Field(
        ..., description="길이 12의 리스트, 각 요소는 [혈당(mg/dL), 이전 인슐린 액션(U/h)]."
    )
    meal: Optional[float] = Field(
        0.0, description="현재 스텝의 식사량 (탄수화물 g, 없으면 0.0) - JS 플러그인에서 오는 값의 의미 확인 필요."
    )

class ActionResponse(BaseModel):
    insulin_action_U_per_h: float = Field(..., description="0.0~5.0 U/h 사이의 실수")

class StepResponse(ActionResponse):
    cgm: Optional[float] = None
    reward: Optional[float] = None
    done: Optional[bool] = None
    info: Optional[Dict[str, Any]] = None

class StateResponse(BaseModel):
    cgm: Optional[float] = None

class EpisodeEndResponse(BaseModel):
    episode: int
    steps: int
# ----------------------------------------------------

# ----- 핵심 로직 함수 (기존 main.py와 거의 동일, app_state 사용) -----
def _handle_env_step(req: StateRequest, endpoint_name: str) -> StepResponse:
    # 이 함수는 app_state에 있는 "agent", "state_space_instance", "agent_args_for_statespace" 등을 사용
    # 해당 값들은 setup_app_state_for_finetuning 함수를 통해 주입되어 있어야 함
    
    agent = app_state.get("agent")
    state_space = app_state.get("state_space_instance")
    agent_args = app_state.get("agent_args_for_statespace")

    if not agent or not state_space or not agent_args:
        print("ERROR:    [MAIN_FINETUNE_HANDLE_STEP] Agent, StateSpace, or agent_args not loaded properly in app_state.", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Agent, StateSpace, or agent_args not loaded in app_state.")

    app_state["api_call_counter"] = app_state.get("api_call_counter", 0) + 1
    current_t_step = app_state["api_call_counter"] - 1

    try:
        request_body_for_log = json.dumps(format_log_data(req.dict()))
    except Exception:
        request_body_for_log = str(req.dict()) # fallback
    # print(f"INFO:     [MAIN_FINETUNE_HANDLE_STEP] Call #{current_t_step}. Received {endpoint_name}. Body: {request_body_for_log}") # 너무 빈번한 로그일 수 있음

    try:
        if not req.history or len(req.history[-1]) != 2: # JS에서 항상 길이 2로 보낸다고 가정
            raise ValueError(f"Invalid 'history' format. Last element: {req.history[-1] if req.history else 'N/A'}")

        current_cgm_raw = req.history[-1][0]
        previous_insulin_action_raw = req.history[-1][1] 
        # 'hour'는 JS에서 보내지만 StateRequest 모델에는 없음. StateSpace.update에서 내부 t_step 기반으로 생성.
        generated_hour_for_statespace = current_t_step # state_space.update에 전달될 시간 정보 (예시)
        current_meal_raw = req.meal if req.meal is not None else 0.0

        if not (np.isfinite(current_cgm_raw) and 
                np.isfinite(previous_insulin_action_raw) and
                np.isfinite(current_meal_raw)):
            raise ValueError(f"NaN/Inf in raw inputs: CGM={current_cgm_raw}, PrevIns={previous_insulin_action_raw}, Meal={current_meal_raw}")

        # StateSpace를 사용하여 상태 전처리
        processed_state_hist_np, processed_hc_state_list = state_space.update(
            cgm=current_cgm_raw,
            ins=previous_insulin_action_raw,
            meal=current_meal_raw,
            hour=generated_hour_for_statespace, # state_space가 사용하는 시간 정보
            meal_type=0, # 이 값들은 agent_args에 따라 사용 여부 결정됨
            carbs=0      # meal_raw가 실제 탄수화물 양이라면, 이 carbs는 중복일 수 있음. StateSpace 설계 확인.
        )
        processed_hc_state_np = np.array(processed_hc_state_list, dtype=np.float32).reshape(1, agent_args.n_handcrafted_features)
        
        # 디버깅: 전처리된 상태 값 일부 출력
        # print(f"DEBUG:    [MAIN_FINETUNE_HANDLE_STEP] Processed state hist (shape): {processed_state_hist_np.shape}")
        # print(f"DEBUG:    [MAIN_FINETUNE_HANDLE_STEP] Processed hc state (shape): {processed_hc_state_np.shape}")

        # 에이전트 추론 (주입된 에이전트 사용)
        action_U_per_h = infer_action( # Sim_CLI.g2p2c_agent_api.infer_action 사용
            agent=agent,
            state_hist_processed=processed_state_hist_np,
            hc_state_processed=processed_hc_state_np,
        )
        # print(f"DEBUG:    [MAIN_FINETUNE_HANDLE_STEP] Inferred action (U/h): {action_U_per_h}")


        # --- 경험 버퍼 및 로깅 (기존 main.py 로직과 유사) ---
        current_state_dict_for_buffer = { # 경험 버퍼에 저장될 상태 표현
            "state_hist": processed_state_hist_np.tolist(), # 정규화된 값
            "hc_state": processed_hc_state_np.flatten().tolist(), # 정규화된 값
        }
        reward_val = 0.0 # 기본값
        
        prev_state_for_buffer = app_state.get("prev_state")
        prev_action_for_buffer = app_state.get("prev_action")

        if prev_state_for_buffer is not None and prev_action_for_buffer is not None:
            # 보상 함수는 raw CGM 값을 사용할 수 있음 (StateSpace.args.glucose_norm_output 등 확인)
            # 또는 정규화된 값을 다시 원래 스케일로 변환한 값을 사용해야 할 수도 있음.
            # 여기서는 composite_reward가 raw CGM을 받는다고 가정.
            # composite_reward 함수가 agent_args와 현재 상태(raw CGM)를 받는다고 가정
            reward_val = composite_reward(agent_args, state=current_cgm_raw) # composite_reward의 인자 확인 필요
            
            experience = {
                "state": prev_state_for_buffer,
                "action": prev_action_for_buffer, # 에이전트가 실제로 선택한 행동 (스케일링/클리핑 후)
                "reward": float(reward_val),
                "next_state": current_state_dict_for_buffer,
                "done": False # done 조건은 DMMS.R 시뮬레이션 길이에 따라 결정되거나 다른 조건 필요
            }
            app_state["experience_buffer"].append(experience)

        app_state["prev_state"] = current_state_dict_for_buffer
        app_state["prev_action"] = float(action_U_per_h) # 에이전트의 최종 행동
        app_state["last_cgm"] = float(current_cgm_raw)

        # CSV 로깅
        try:
            # 로그에는 raw 값과 에이전트 관련 값들을 기록
            # day_hour, day_min 등 시간 정보는 generated_hour_for_statespace를 기반으로 계산
            total_sim_minutes = current_t_step * 5 # 각 스텝이 5분이라고 가정
            log_day_hour = (total_sim_minutes // 60) % 24 # 예시: 시뮬레이션 시작부터의 시간
            log_day_min = total_sim_minutes % 60

            log_row_dict = {
                'epi': app_state.get("current_episode", 1), 't': current_t_step,
                'cgm': current_cgm_raw, 'meal': current_meal_raw,
                'ins': action_U_per_h, # 에이전트 최종 행동
                'rew': float(reward_val) if prev_state_for_buffer is not None else np.nan,
                'rl_ins': action_U_per_h, # 에이전트 최종 행동
                'mu': np.nan, 'sigma': np.nan, 'prob': np.nan, 'state_val': np.nan, # 이 값들은 infer_action 반환값에 따라 채울 수 있음
                'day_hour': int(log_day_hour), 'day_min': int(log_day_min),
            }
            with open(LOG_FILE_PATH, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=SimglucoseLogHeader)
                if current_t_step == 0 and app_state.get("current_episode", 1) == 1 : # 첫 줄 헤더 (파일 생성 시 이미 했으므로 중복될 수 있음)
                     if os.path.getsize(LOG_FILE_PATH) <= len(",".join(SimglucoseLogHeader)) + 5: # 대략적인 헤더 크기 체크
                        pass # 헤더가 이미 있는 것으로 간주 (setup_app_state_for_finetuning 에서 처리)
                writer.writerow(log_row_dict)
        except Exception as csv_e:
            print(f"ERROR:    [MAIN_FINETUNE_HANDLE_STEP_CSV_WRITE] Failed to write Simglucose format log: {csv_e}", file=sys.stderr)

        # info 딕셔너리 구성 (Simglucose env.step() 반환 형식과 유사하게)
        step_info = {
            "sample_time": 5, # 5분 간격 가정
            "meal": current_meal_raw, # JS에서 전달된 meal 값
            # "day_hour": int(log_day_hour), # 이 정보는 Simglucose 환경에서는 env 내부에서 관리
            # "day_min": int(log_day_min),
            # ... 기타 Simglucose 환경이 반환하는 info 값들 (필요시 추가)
        }
        
        # done 플래그: DMMS.R 시뮬레이션은 보통 고정된 기간 동안 실행됨.
        # 에피소드 종료는 JS 플러그인의 cleanup 함수에서 /episode_end 호출로 처리.
        # 따라서 여기서 done은 항상 False일 수 있음. 또는 특정 조건(예: 최대 스텝 도달)에서 True.
        # 현재는 JS 플러그인이 /episode_end를 호출하므로, 여기서는 False로 둠.
        is_done = False 

        return StepResponse(
            insulin_action_U_per_h=action_U_per_h,
            cgm=float(current_cgm_raw),
            reward=float(reward_val) if prev_state_for_buffer is not None else 0.0,
            done=is_done, 
            info=step_info,
        )

    except ValueError as ve:
        print(f"\n!!! ValueError processing {endpoint_name} in MAIN_FINETUNE: {ve} !!!", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Input validation error: {str(ve)}")
    except Exception as e:
        print(f"\n!!! ERROR processing {endpoint_name} in MAIN_FINETUNE !!!", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent inference error: {str(e)}")
# -------------------------------------------------------------------

# ----- API 엔드포인트 정의 (app 객체 사용) -----
@app.post("/predict_action", response_model=StepResponse)
async def predict_action_endpoint(req: StateRequest): # async 키워드 추가 (FastAPI 권장)
    return _handle_env_step(req, "/predict_action")

@app.post("/env_step", response_model=StepResponse)
async def env_step_endpoint(req: StateRequest): # async 키워드 추가
    return _handle_env_step(req, "/env_step")

@app.get("/get_state", response_model=StateResponse)
async def get_state_endpoint(): # async 키워드 추가
    return StateResponse(cgm=app_state.get("last_cgm"))

@app.post("/episode_end", response_model=EpisodeEndResponse)
async def episode_end_endpoint(): # async 키워드 추가
    exp_buffer = app_state.get("experience_buffer", [])
    
    # 마지막 상태에 대한 최종 보상 계산 및 경험 추가 (기존 main.py 로직 참고)
    prev_state = app_state.get("prev_state")
    prev_action = app_state.get("prev_action")
    if prev_state is not None and prev_action is not None:
        last_cgm = app_state.get("last_cgm")
        agent_args = app_state.get("agent_args_for_statespace")
        if last_cgm is not None and agent_args is not None:
            final_reward = composite_reward(agent_args, state=last_cgm) # composite_reward 인자 확인
        else:
            final_reward = 0.0 
        
        exp_buffer.append({
            "state": prev_state, "action": prev_action, "reward": float(final_reward),
            "next_state": None, # 에피소드 종료이므로 next_state는 없음
            "done": True # 에피소드 종료
        })

    episode_num = app_state.get("current_episode", 1)
    num_steps = len(exp_buffer) # 또는 app_state.get("api_call_counter", 0)

    try:
        EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)
        exp_path = EXPERIENCE_DIR / f"episode_{episode_num}_exp.json" # 파일명에 _exp 추가
        # JSON 저장 시 NumPy/Tensor 객체가 있다면 format_log_data와 유사한 처리 필요
        # 여기서는 exp_buffer 내용물이 이미 Python 기본 타입이라고 가정
        with open(exp_path, "w") as f:
            json.dump(exp_buffer, f, indent=2) # indent 추가 (가독성)
        print(f"INFO:     [MAIN_FINETUNE_EPISODE_END] Experience buffer for episode {episode_num} saved to {exp_path} ({num_steps} steps)")
    except Exception as exp_e:
        print(f"ERROR:    [MAIN_FINETUNE_EPISODE_END] Failed to save experience buffer: {exp_e}", file=sys.stderr)

    # 다음 에피소드를 위한 상태 초기화 (current_episode는 훈련 스크립트에서 관리/증가시킬 수 있음)
    app_state["experience_buffer"] = []
    app_state["prev_state"] = None
    app_state["prev_action"] = None
    app_state["last_cgm"] = None
    app_state["api_call_counter"] = 0
    # app_state["current_episode"] = episode_num + 1 # 이 부분은 훈련 스크립트에서 제어하는 것이 더 적절할 수 있음

    return EpisodeEndResponse(episode=episode_num, steps=num_steps)
# ----------------------------------------------------