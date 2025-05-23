# Guidelines for Contributors

This repository contains code for training and evaluating RL-based insulin dosing agents.
The original environment is **Simglucose**, but the `Sim_CLI` package bridges a trained
agent with the external simulator **DMMS.R** via a FastAPI server and a JavaScript plugin.
The notes below summarise key conventions and design decisions that have emerged in
previous discussions.

## Folder overview

- `agents/` – RL algorithms including the G2P2C implementation.
- `Sim_CLI/` – utilities to connect a trained agent to DMMS.R. Important files are:
  - `main.py`: FastAPI server that loads the agent and exposes API endpoints.
  - `g2p2c_agent_api.py`: helper to load the G2P2C model and run inference.
  - `RL_Agent_Plugin_v1.0.js`: DMMS.R plugin that queries the server for insulin
    actions.
  - `run_dmms_cli.py`: example script showing how to start DMMS.R from Python.
- `results/` – log files written by the server during interaction with DMMS.R.

## 강화 학습 구축을 위한 구체적·검증 가능한 단계

### 시뮬레이터 제어 스크립트 작성 - 완료

기존 CLI 실행 원리에 따라 DMMS.R 실행 파일을 subprocess.run([exe, cfg, log, results_dir], check=True) 방식으로 호출한다.
실행이 완료되면 결과 폴더에 생성되는 CSV 파일을 통해 각 에피소드의 로그를 확보한다.

### JavaScript 플러그인 확장 (RL_Agent_Plugin_v1.0.js) - 완료

현재는 인슐린 결정만 받은 후 종료 시 Python에 알리지 않는다.
에피소드 종료 시점에 FastAPI 서버의 /episode_end(신규)로 POST 요청을 보내도록 함.
필요 시 매 스텝마다 reward 계산에 필요한 정보를 /env_step으로 보낼 수 있도록 상태 전송 로직을 분리한다.

### FastAPI 서버 기능 확장 (Sim_CLI/main.py)

lifespan()에서 학습용 버퍼와 에피소드 관리 변수를 초기화한다.
/env_step
플러그인에서 받은 상태와 이전 행동으로 reward를 계산한다(utils.reward_func.composite_reward).
(state_t, action_t, reward_{t+1}, state_{t+1})를 버퍼에 저장한다.
다음 행동은 infer_action()으로 계산하여 반환한다.
/episode_end
남은 경험을 정리하고 버퍼를 파일로 저장하거나 Python 학습 모듈로 전달한다.

### Gym 호환 환경 클래스 구현

Sim_CLI/dmms_env.py (신규 파일)
reset()에서 DMMS.R을 새로운 결과 폴더와 함께 실행하고 초기 상태를 받아온다.
step(action)은 위의 /env_step API를 통해 한 스텝을 진행하고 다음 상태·보상을 받아오도록 한다.
close()에서 실행 중인 DMMS.R 프로세스를 종료한다.
이 클래스는 기존 RL 알고리즘에서 사용되는 인터페이스(env.reset, env.step, env.close)를 그대로 제공한다.

### 학습 루프 통합

experiments/run_RL_agent.py의 설정 로직을 참고하여 G2P2C 또는 PPO 알고리즘을 선택적으로 초기화한다.
새로 구현한 DmmsEnv를 환경으로 사용하도록 옵션을 추가한다(--sim dmms).
경험 버퍼는 /episode_end에서 받아오거나, 학습 루프 내에서 API 호출을 통해 직접 수집한다.
학습 스크립트는 기존 agents/g2p2c.worker.Worker를 그대로 활용하여 미니배치 생성과 업데이트를 수행한다.

### 검증과 기록

results/dmms_realtime_logs_v2에 남는 CSV 로그를 활용해 실제 DMMS.R 시뮬레이션과 Python 서버 간의 상호작용을 확인한다.
각 에피소드의 성공 여부, reward 변화 등을 기록하여 학습이 정상적으로 진행되는지 모니터링한다.
이렇게 하면 이미 검증된 Simglucose 기반 학습 코드를 최대한 재활용하면서, 외부 시뮬레이터 DMMS.R과 상호작용하는 강화학습 환경을 구축할 수 있다. 필요한 부분(플러그인 수정, FastAPI 확장, 환경 래퍼 작성)을 차근차근 구현한 뒤 로그를 통해 동작을 확인하면 된다.



## RL training with DMMS.R

The current code base primarily performs inference. To train with DMMS.R
consider the following workflow:

1. Modify the JS plugin and the FastAPI server as described above to collect
   step data and handle episode boundaries.
2. Implement a Gym-style environment class that wraps the API communication.
3. Reuse the existing PPO/G2P2C training loops (`experiments/run_RL_agent.py`)
   by replacing the environment with the new DMMS.R wrapper.

These guidelines should help maintainers and contributors keep the code base
consistent while extending the project to support external simulators.
