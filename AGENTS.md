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

## 강화 학습 구축을 위한 구체적·검증 가능한 단계

### 시뮬레이터 제어 스크립트 작성 - 완료

기존 CLI 실행 원리에 따라 DMMS.R 실행 파일을 subprocess.run([exe, cfg, log, results_dir], check=True) 방식으로 호출한다.
실행이 완료되면 결과 폴더에 생성되는 CSV 파일을 통해 각 에피소드의 로그를 확보한다.

### JavaScript 플러그인 확장 (RL_Agent_Plugin_v1.0.js) - 완료

현재는 인슐린 결정만 받은 후 종료 시 Python에 알리지 않는다.
에피소드 종료 시점에 FastAPI 서버의 /episode_end(신규)로 POST 요청을 보내도록 함.
필요 시 매 스텝마다 reward 계산에 필요한 정보를 /env_step으로 보낼 수 있도록 상태 전송 로직을 분리한다.

### FastAPI 서버 기능 확장 (Sim_CLI/main.py) - 완료

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

## DMMS.R 시뮬레이터를 “코드(스크립트)”로 실행하는 기본 원리

사용자 GUI를 거치지 않고도 DMMS.R 시뮬레이션은 명령행 인터페이스(CLI) 로 직접 호출할 수 있습니다.
핵심은 GUI에서 미리 저장해 둔 설정 파일(= configuration, Rx Plan) 과 함께 DMMS.R 실행 파일을 호출하는 방식입니다.

요약
형식 : DMMS.R config.xml log.txt [results_dir]
반복 실험 : BAT / PowerShell / Python 스크립트를 통해 순차 또는 병렬 호출
충돌 방지 : 병렬 시 각각 다른 결과 폴더나 시뮬레이션 이름 사용
GUI는 한 번만 : 설정 파일을 미리 만들어 두어야 CLI가 실행 가능

이렇게 하면 DMMS.R 시뮬레이션을 “코드 수준”에서 자동화 · 대량 실행할 수 있습니다.

| 요소                                | 동작 원리                                                                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `&` (호출 연산자)                      | 따옴표로 묶인 **문자열**을 *실행 가능한 명령* 으로 변환한다. PowerShell 은 `"string"`만 놓으면 *단순 문자열* 로 간주하기 때문에, `&`가 반드시 필요하다.                   |
| `"C:\Program Files\…\DMMS.R.exe"` | 공백(Program Files) 때문에 전체를 따옴표로 감쌌다.                                                                                      |
| `"C:\…\Sim_CLI_1.2.xml"`          | **<config file>** – GUI에서 *Save Rx Plan…* 으로 만든 XML.                                                                     |
| `"log_single.txt"`                | **<log file>** – 실행·오류 메시지가 여기에 append 된다.                                                                               |
| `"\results_test"`                 | **<results dir>** – 드라이브 글자 없이 백슬래시로 시작하면 “현재 드라이브의 루트”를 의미한다. 따라서 `C:\results_test` 가 된다. (상대 경로로 쓰려면 `.\results_test`) |

DMMS.R 실행 파일은 “`프로그램 경로  +  세 개 인자`” 형태의 호출을 받으면, 매뉴얼 Appendix B 에 정의된 CLI 모드로 전환되어 시뮬레이션을 수행한다 .

---

### 2-1. Python (subprocess) 예시

```python
from subprocess import run, CalledProcessError
exe = r"C:\Program Files\The Epsilon Group\DMMS.R\simulator\DMMS.R.exe"
cfg = r"C:\Users\user\Documents\DMMS.R\config\Sim_CLI_1.2.xml"
log = r"log_single.txt"
out = r"results_test"          # 상대 폴더면 pathlib.Path.cwd()/...
try:
    run([exe, cfg, log, out], check=True)
except CalledProcessError as e:
    print("DMMS.R 실패:", e.returncode)
```


## 응답 방식

사용자의 요청에 최선을 다하여 응답하며, 한국어로 친절하고 자세히 응답해라.
없는 사실을 지어나거나, 혹은 알 수 없는 사실을 가정하지 말고, 만약 추가적으로 필요한 정보가 있다면 사용자에게 즉시 요청해라. 

### Recent Updates
- Added check in `environments/simglucose/simglucose/__init__.py` to skip Gym environment registration if Gym is not installed or if `simglucose-v0` is already registered. This prevents errors like `gym.error.Error: Cannot re-register id: simglucose-v0` when running the FastAPI server.
- `Sim_CLI/main.py` now returns an extended step response (`StepResponse`) containing `cgm`, `reward`, `done`, and `info` fields in addition to `insulin_action_U_per_h`. This helps `DmmsEnv` interact with the API like a Gym environment.
- `DmmsEnv` tracks the latest CGM value and updates it after each step. The environment also uses `args.dmms_io_root` from `utils.core.get_env()` so results are saved under `results/dmms_runs`.
- `uvicorn Sim_CLI.main:app --host 127.0.0.1 --port 500` 항상 이 코드를 사용해서 서버를 열고, 이 서버와 통신을 함.
- `python Sim_CLI/run_dmms_cli.py "C:\Program Files\The Epsilon Group\DMMS.R\simulator\DMMS.R.exe" "C:\Users\user\Documents\DMMS.R\config\Sim_CLI_1.2.xml" "log_single.txt" "\results_test" ` 이 코드를 통해서 터미널에서 DMMS.R 시뮬레이터를 항상 구동함. 위를 실행하고 난 디버깅 결과는 아래와 같다. 
  - 두 개의 디버깅 결과 제시됨.
    - -----------------------------
      DEBUG: Model raw output: -0.2924, Scaled action: 1.5524, Clipped action: 1.5524
      INFO:     127.0.0.1:59741 - "POST /env_step HTTP/1.1" 200 OK
      INFO:     [LOG_MAIN_REQUEST_START] Call #287. Received /env_step. Body: {"history": [[129.9519295262746, 0.9567926079980968], [127.74815449501149, 1.069693769968829], [125.56159826953449, 1.6721291921683878], [123.4038636966688, 1.3690522510255796], [121.2850768399388, 2.5691536979178764], [119.21212366038587, 0.44198040672057415], [117.18737828145304, 3.563908304074195], [115.20990995252951, 3.278973343266496], [113.27440556262681, 1.4103845310824195], [111.36806162494409, 1.8585728015680467], [109.47351483387143, 4.181181316271177], [107.57309510765505, 1.5523959585057883]], "meal": -287.0}
    - (venv_3.10) PS C:\Users\user\Desktop\G2P2C> python Sim_CLI/run_dmms_cli.py "C:\Program Files\The Epsilon Group\DMMS.R\simulator\DMMS.R.exe" "C:\Users\user\Documents\DMMS.R\config\Sim_CLI_1.2.xml" "log_single.txt" "\results_test" 
      Running: C:\Program Files\The Epsilon Group\DMMS.R\simulator\DMMS.R.exe C:\Users\user\Documents\DMMS.R\config\Sim_CLI_1.2.xml log_single.txt \results_test
      Generated CSV files:
      - \results_test\signalHistory.Type1 Adult.adult#001.csv
      - \results_test\signalHistory.Type2 Adult.adult#001.csv
      - \results_test\survivalInfo.csv
      Loaded \results_test\signalHistory.Type1 Adult.adult#001.csv with 1441 rows
- 위 두 개의 코드와 연관된 코드는 모두 문제가 없음
- 주로 작업은 Sim_CLI에서 수행하고, 수정도 주로 이 폴더 안에 있는 코드에서 수행함. 
- `DmmsEnv`를 통해 DMMS.R을 서브프로세스로 실행하고 HTTP API로 제어하는 기본 기능은 동작한다. 다만 학습 루프와 완전히 통합되어 있지는 않다.
- 다음 단계 제안:
  1. `experiments/run_RL_agent.py`에서 `--sim dmms` 옵션 사용 시 학습 반복이 제대로 끝났는지 확인하고, 필요하면 에피소드 종료 후 `/episode_end`를 호출하도록 개선.
- Worker 모듈의 초기 상태 및 step 결과는 `step.observation.CGM` 형태로 접근하도록 수정되었다.
- 2024-04-13: `DmmsEnv.step`이 `/env_step` 호출 시 `{"action": ...}` 형태의 잘못된 페이로드를 보내 422 오류가 발생함을 확인.
  `history` 필드와 `meal` 값을 포함하도록 수정하여 FastAPI 서버 스키마와 일치시켰다. 옵션 `--debug` 사용 시
  `DmmsEnv`가 요청과 응답을 출력한다. `utils.core.get_env()`는 이제 `debug` 값을 전달한다.
- 2025-05-26: `DmmsEnv.reset` 및 `step`이 `Step(..., **info)` 형식으로 반환하도록 수정해 `info` 딕셔너리가 올바르게 전달된다.
  `Worker` 클래스는 `env.step()` 결과를 언팩한 뒤 `Observation.CGM` 값만 사용하도록 정리하였다.
  `Pump` 모듈은 `Step`과 `Observation`을 모두 처리할 수 있도록 `_get_cgm()` 헬퍼를 도입했다.
