**[보고서] DMMS.R 시뮬레이터 - G2P2C 에이전트 연동: 진행 상황 및 현안 상세 요약**

**1. 프로젝트 목표**

* **핵심 목표:** Python 기반 G2P2C PPO 강화학습 에이전트를 Abbott의 DMMS.R (Diabetes Mellitus Metabolic Simulator - Research Version) 시뮬레이션 환경에 연동하여, 상세한 생리학적 모델 상에서 G2P2C 에이전트의 인공 췌장 제어 성능을 검증하고 개선하는 것.
* **주요 기술 과제:** 서로 다른 환경(DMMS.R은 독립 실행형 Windows 애플리케이션, G2P2C는 외부 Python 프로세스)과 언어(DMMS.R 플러그인은 JavaScript, G2P2C는 Python) 간의 실시간 데이터 교환 및 제어 명령 전달을 위한 안정적인 '연동 브릿지' 구현.

**2. 초기 분석 및 핵심 문제 식별**

* **DMMS.R 연동 방식:** DMMS.R 사용자 매뉴얼 분석 결과, 외부 로직과의 연동은 JavaScript 플러그인 컨트롤 요소(Plugin Control Element) 기능을 통해서만 가능함을 확인[cite: 155, 156, 157].
* **G2P2C 에이전트 사양:**
    * 입력: 최근 N 스텝 (보고서 기준 N=6, 현재 구현은 N=12)의 [혈당(mg/dL), 인슐린(U/h)] 시퀀스 (`cur_state`) 및 추가 특징 벡터 (`feat`).
    * 출력: 단일 인슐린 주입률(U/h) 값 (`selected_action`).
* **핵심 문제점:**
    1.  **입출력 불일치:** G2P2C는 단일 인슐린 값을 출력하지만, DMMS.R은 기저(Basal), 볼루스(Bolus), 사용자 정의(Custom) 등 다중 인슐린 입력 채널을 가짐[cite: 1160]. G2P2C의 단일 출력을 DMMS.R의 어떤 입력에 어떻게 매핑할 것인가?
    2.  **기술 스택 차이:** DMMS.R(JS 플러그인)과 G2P2C(Python) 간의 언어 및 실행 환경 차이로 인한 직접적인 연동 불가. 실시간 통신 메커니즘 필요.

**3. 솔루션 설계: 웹 서비스 기반 연동 브릿지**

* **통신 방식 채택:** 안정성, 확장성, 구현 용이성을 고려하여 **웹 서비스 (HTTP POST)** 방식을 채택. Python 측에서는 FastAPI 프레임워크를 사용하고, JavaScript 측에서는 DMMS.R 플러그인이 제공하는 `httpWebServiceInvoker` 객체 [cite: 864]를 사용하기로 결정.
* **API 사양 정의:**
    * URL: `http://<FastAPI 서버 주소>:<포트>/predict_action` (초기 `http://127.0.0.1:5000`, 이후 `8000` 등으로 수정됨)
    * Method: `POST`
    * Request Body (JSON): `{ "history": [[Number(mg/dL), Number(U/h)], ... * 12] }` (`feat` 벡터는 초기 미사용 결정)
    * Response Body (JSON): `{ "insulin_action_U_per_h": Number(U/h) }`
    * Header: `Content-Type: application/json`
* **입출력 매핑 결정:** G2P2C의 단일 인슐린 출력(U/h)을 DMMS.R 내에서 **사용자 정의 인슐린 채널(`sqCustomInsulin1`) [cite: 1160]에 적용**하기로 결정. 이는 제어 방식을 단순화하지만, 기술적 연결을 가능하게 함.
* **오류 처리:** 웹 통신 실패 또는 에이전트 응답 오류 시, 안전을 위해 인슐린 주입률을 0.0 U/h로 적용하는 Fallback 로직 구현 결정.

**4. 구현: JavaScript 플러그인 및 Python 서버**

* **JavaScript 플러그인 (`RL_Agent_Plugin_v1.0.js`):**
    * 매 시뮬레이션 스텝(1분)마다 다음을 수행:
        1.  DMMS.R로부터 CGM 혈당 값 수신 (`sensorSigArray`).
        2.  최근 12 스텝의 [CGM, 이전 스텝 적용 인슐린(U/h)] 이력을 관리 (`bgHistory`, `insulinActionHistory`).
        3.  이력 데이터를 JSON 요청 본문(`{ "history": [...] }`)으로 포맷.
        4.  `httpWebServiceInvoker.performPostRequest` [cite: 867]를 사용하여 Python FastAPI 서버에 POST 요청 전송.
        5.  서버로부터 JSON 응답 (`{ "insulin_action_U_per_h": ... }`) 수신 및 파싱.
        6.  수신된 인슐린 값(U/h)을 pmol/min 단위로 변환.
        7.  변환된 값을 `modelInputsToModObject` 객체의 `sqCustomInsulin1` 키 [cite: 1160]에 할당하여 DMMS.R 시뮬레이터에 적용 (통신 실패 시 0.0 적용).
        8.  다음 스텝의 이력 관리를 위해 현재 적용된 인슐린 값(U/h)을 `lastAppliedAction_UperH`에 저장.
    * 디버깅 및 상태 관리를 위한 로깅 기능 포함 (`debugLog`, `debugLoggingEnabled`).
* **Python FastAPI 서버 (`main.py`):**
    * FastAPI를 사용하여 `/predict_action` 엔드포인트에서 POST 요청 수신.
    * 요청 본문의 `history` 데이터를 NumPy 배열로 변환.
    * **현재 상태:** 실제 G2P2C 에이전트 대신, 수신된 `history` 데이터를 기반으로 **랜덤 인슐린 값(0.1~4.9 U/h)을 반환하는 더미 함수(`get_action_dummy`)** 사용 중.
    * 결정된 액션 값을 JSON 응답 (`{ "insulin_action_U_per_h": ... }`)으로 반환.
    * 안전 클리핑 (0.0 ~ 5.0 U/h) 로직 포함.
    * CORS 설정 완료.

**5. 통합 테스트 및 디버깅 과정 (상세)**

* **초기 설정:** DMMS.R GUI 및 XML 파일(`Sim_CLI_0.1.xml`)을 통해 시뮬레이션 환경 구성 (피험자: adult#007, 사용자 정의 인슐린: PPO\_Agent\_Insulin, 센서: IdealCgmSensor, 제어 요소: JS Plugin, 전달 장치: IdealPump 등).
* **1차 문제 발생 (JS Initialize 실패):** 초기 버전의 JS 플러그인 실행 시, 로그 파일에 `[Error] runIteration called before initialize!` 오류가 매 스텝 반복됨. 이는 `initialize` 함수가 성공적으로 완료되지 못함을 의미.
    * **디버깅:** `debugLog` 위치 변경, `try...catch` 추가 등의 시도에도 불구, `initialize` 함수 내부 로그 자체가 출력되지 않아 함수 시작 실패 또는 극초반 중단으로 추정.
    * **진단:** 정상 작동하는 샘플 스크립트(`samplePluginControlElement.js`, `initialize` 함수 비어 있음)를 동일 환경에서 실행했을 때 성공. 문제 원인이 사용자 정의 JS 스크립트 내부에 있음을 확신.
    * **해결:** `initialize` 함수 내 배열 초기화 로직(`Array.fill`)이 DMMS.R의 JS 엔진과 호환되지 않을 가능성을 의심. **`initialize` 함수를 단순화하고 배열 관리를 `runIteration`으로 옮기는 방식(수정 방안 1)**을 적용하여 `RL_Agent_Plugin_v1.0.js`로 업데이트. → **Initialize 문제 해결됨.**
* **2차 문제 발생 (HTTP 404 오류):** Initialize 문제 해결 후, JS 플러그인이 웹 서비스 호출 시 `HTTP Status: 404 Not Found` 오류 발생.
    * **진단:** JS 플러그인이 요청하는 URL (`http://127.0.0.1:5000/predict_action`)과 FastAPI 서버가 실제로 리스닝하는 주소/포트가 불일치할 가능성 제기 (FastAPI 기본 포트는 8000).
    * **해결:** 사용자가 실제 FastAPI 서버 실행 주소/포트를 확인하고, 이에 맞춰 JS 플러그인의 `AGENT_API_URL` 값을 수정. → **404 문제 해결됨.**

**6. 현재 상태**

* **DMMS.R <-> Python (G2P2C) 연동 브릿지는 기술적으로 완전히 작동합니다.**
    * JS 플러그인은 DMMS.R 상태를 성공적으로 Python 서버로 전송합니다.
    * Python 서버는 요청을 받아 (현재는 더미) 응답을 성공적으로 반환합니다.
    * JS 플러그인은 응답을 받아 DMMS.R 시뮬레이터에 인슐린 액션을 성공적으로 적용합니다 (`sqCustomInsulin1` 채널 사용).
* **주요 기술적 과제(언어 차이, 실시간 통신)는 해결되었습니다.**
* **개념적 과제(단일 출력 vs. 다중 입력)는 단일 채널 매핑 방식으로 해결되었으며, 이는 제어 방식의 단순화를 의미합니다.**
* Python 서버(`main.py`)는 아직 **실제 G2P2C PPO 에이전트 로직을 포함하고 있지 않습니다.**

**7. 다음 단계 **

* **핵심 요청:** 현재 `main.py`의 더미 함수 `get_action_dummy` 부분을 **실제 G2P2C PPO 에이전트의 모델 로딩 및 추론 로직으로 교체**해주십시오.
* **필요 정보:**
    * 에이전트가 `main.py` 내에서 호출될 때 입력으로 받게 될 `cur_state`는 **NumPy 배열 형태**이며, shape은 `(12, 2)`입니다. 각 행은 `[CGM 혈당(mg/dL), 이전 스텝 적용 인슐린(U/h)]` 입니다 (가장 마지막 행이 가장 최신 데이터).
    * `feat` 벡터는 현재 `USE_FEAT_VECTOR = false` 상태이므로 사용되지 않지만, 필요시 JS 플러그인에서 생성하여 전달 가능합니다. G2P2C 에이전트가 `feat` 벡터를 필요로 하는지, 필요하다면 어떤 값들을 어떤 순서로 요구하는지 알려주시면 JS 플러그인에 반영하겠습니다.
    * 에이전트 함수는 최종적으로 **단일 float 값 (인슐린 주입률, U/h 단위)**을 반환해야 합니다. `main.py`에서 0.0~5.0 U/h 범위로 클리핑하는 로직은 이미 포함되어 있습니다.
* **향후 계획:** 실제 에이전트 통합 후, 다양한 DMMS.R 시나리오(식사, 운동 포함) 기반 테스트, 성능 평가, 결과 분석 및 개선 작업을 진행할 예정입니다.