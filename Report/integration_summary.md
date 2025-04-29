<!---
  File: Report/integration_summary.md
  Purpose: Summarize all integration steps for G2P2C agent helper and FastAPI integration
-->
# Integration Summary: G2P2C Agent Helper & FastAPI Integration

## 1. 배경
- G2P2C 프로젝트는 강화학습 기반 인공췌장 시스템(APS)용 에이전트를 개발합니다.
- `test_g2p2c_agent.py`는 시뮬레이터 없이 에이전트 모델 로드·추론만 검증합니다.
- `main_v0.1.py` + `RL_Agent_Plugin_v1.0.js` 조합은 HTTP/JSON을 통해 **DMMS.R 시뮬레이터**와 에이전트를 연결합니다.

## 2. 목표
1. `test_g2p2c_agent.py`의 로드·추론 코드를 **모듈화**하여 재사용성 확보
2. **Helper 모듈**(`g2p2c_agent_api.py`) 생성: 에이전트 로딩·추론 함수 제공
3. 테스트 스크립트(`test_g2p2c_agent.py`) 리팩터링
4. FastAPI 서버(`main_v0.1.py`) 리팩터링: 더미 대신 실제 에이전트 호출
5. 별도 Git 브랜치(`integrate-g2p2c-agent`)로 변경사항 관리

## 3. 주요 작업 내용

### 3.1 디버깅 보고서 작성
- `Report/test_g2p2c_agent_debug_report.md` 추가: 스크립트 단계별 오류 분석 및 해결 과정을 상세 정리

### 3.2 Helper 모듈 생성
- 파일: `Sim_CLI/g2p2c_agent_api.py`
- 함수:
  - `load_agent(args_json_path, params_py_path, device, episode)`:
    1. `args.json` 로드 → `Options` 복원
    2. `parameters.py` 로드 → `params.set_args`
    3. `G2P2C` 인스턴스 생성
    4. 체크포인트 디렉터리 스캔 → 최신 episode 선택
    5. `torch.load`로 Actor/Critic 객체 불러와 `agent.policy`에 할당
    6. `eval()` 호출 후 에이전트 반환
  - `infer_action(agent, history, feat)`:
    1. 입력 검증(shape, NaN 등)
    2. `agent.policy.get_action` 호출 → 액션 추출
    3. 범위 클리핑 → float 반환

### 3.3 테스트 스크립트 리팩터링
- 파일: `Sim_CLI/test_g2p2c_agent.py`
- 변경 사항:
  - 기존의 `Options`, `torch.load`, `get_action` 호출 로직을 삭제
  - `load_agent`로 에이전트 로드
  - `infer_action`로 정상/비정상 케이스 테스트
  - 테스트 결과: 정상 4개 패스, 비정상 3개 모두 예외 발생 성공

### 3.4 FastAPI 서버 리팩터링
- 파일: `Sim_CLI/main_v0.1.py`
- 변경 사항:
  - `get_action_dummy` 제거
  - 상단에 `load_agent`, `infer_action` 임포트
  - `_AGENT = load_agent(...)` 전역 인스턴스 생성
  - `predict_action` 핸들러에서 `infer_action(_AGENT, hist_np, feat)` 호출로 대체
  - `feat` 벡터 요청 기반 혹은 기본값 자동 생성

### 3.5 에이전트 입력 처리(AI Agent Input Pipeline)
에이전트에 최종 입력으로 들어가는 데이터 흐름을 단계별로 정리합니다.
1) JS Plugin 측 (DMMS.R `runIteration`)
   - `bgHistory`, `insulinActionHistory` 전역 배열을 `FEATURE_HISTORY_LENGTH`(12)로 유지
     * 매 스텝마다 `sensorSigArray[0]` → `currentCGM`(`bgHistory.push`)  
     * `lastAppliedAction_UperH` → `insulinActionHistory.push`
     * 길이 초과 시 `shift()`로 과거 데이터 제거
   - `prepareAgentState(bgHist, insHist, ...)` 호출
     ```js
     // history: [[bg0, ins0], [bg1, ins1], ..., [bg11, ins11]]
     state = { history: historyArray };
     if (USE_FEAT_VECTOR) state.feat = featArray;
     ```
   - JSON 직렬화 후 `/predict_action`에 POST

2) FastAPI 서버 측 (`main_v0.1.py`)
   - Pydantic `StateRequest`로 JSON body 자동 변환
     ```python
     class StateRequest(BaseModel):
         history: List[List[float]]  # 길이 12 × [bg, ins]
         feat: Optional[List[float]]
     ```
   - `predict_action` 핸들러에서
     ```python
     hist_np = np.array(req.history, dtype=np.float32)
     # (12,2) 검증 및 NaN/Inf 체크
     feat = (np.array(req.feat, dtype=np.float32).reshape(1,-1)
             if req.feat else np.zeros((1, agent.args.n_handcrafted_features)))
     action = infer_action(agent, hist_np, feat)
     ```

3) Helper 모듈 측 (`g2p2c_agent_api.py`)
   - `infer_action(agent, history, feat)` 내부 로직
     1. `history`가 numpy 2D인지, shape `(12,n_features)`인지 확인
     2. `np.isfinite` 검사로 NaN/Inf 제거
     3. `feat` 없이 호출 시 default zero 벡터 생성 `(1,n_handcrafted_features)`
     4. `agent.policy.get_action(history, feat)` 호출  
        - 내부적으로 `ActorCritic.get_action` → `predict` → `ActorNetwork.forward` → `FeatureExtractor.forward`
        - `FeatureExtractor.forward(mode='forward')`에서 입력 `s` (2D) 에 `s.unsqueeze(0)` 수행하여 3D `(1,12,n_features)` LSTM 입력 생성
     5. 반환된 numpy 배열 `out['action']`에서 `float(out['action'][0])` 추출
     6. `insulin_min`~`insulin_max` 범위로 `np.clip`

### 흐름 비교
```text
JS Plugin ─(prepare history)─▶ JSON POST to FastAPI ─▶ infer_action ─▶ RL Agent ─▶ action
        ◀────────── JSON response ────────────┘                ↳ clipping → 반환
``` 
FastAPI 핸들러 → Helper → ActorCritic → FeatureExtractor → ActionModule → 반환  

### 이점
- 입력 검증(형상·NaN 등)이 일관되게 helper 모듈에 집중
- JS ↔ Python 간 JSON 명세(`history`·`feat`)만 맞추면 연동 가능

### 3.6 Git 브랜치 생성 및 커밋
- 브랜치: `integrate-g2p2c-agent`
- 커밋 메시지: `feat: integrate G2P2C agent into FastAPI and helper module`
- 원격 푸시: `git push --set-upstream origin integrate-g2p2c-agent`

## 4. 사용법 안내
1. **브랜치 체크아웃**:
   ```bash
   git fetch origin
   git checkout integrate-g2p2c-agent
   ```
2. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```
3. **테스트 스크립트 실행**:
   ```bash
   python Sim_CLI/test_g2p2c_agent.py
   ```
4. **FastAPI 서버 실행**:
   ```bash
   uvicorn Sim_CLI.main_v0.1:app --reload --port 5000
   ```
5. **DMMS.R 시뮬레이터(원격)**:
   - `RL_Agent_Plugin_v1.0.js`에서 `AGENT_API_URL`을 실제 서버 주소로 수정
   - DMMS.R 시뮬레이션을 실행하여 에이전트 연동 확인

## 5. 향후 과제
- **환경-에이전트 통합 테스트**: 원격 DMMS.R 환경에서 FastAPI 연동 검증
- **성능 최적화**: 동시 요청·GPU 사용 고려
- **추가 기능**: handcraft feature, planning 모드, 모니터링 API 등

---
이 보고서는 G2P2C 에이전트의 Helper 모듈화 및 FastAPI 통합 과정을 요약·정리합니다.
추가 문의나 업데이트 사항이 있으면 본 문서에 반영 부탁드립니다.