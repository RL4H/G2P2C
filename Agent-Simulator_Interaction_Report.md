# G2P2C: Agent–Simulator Interaction Detailed Report

본 문서는 **G2P2C 프로젝트 코드**를 기반으로, **에이전트(Agent)**와 **시뮬레이터(환경)** 간에 오고 가는 수치/시퀀스/정보 구조 및 구체 예시를 **신뢰성 있게 정리한 기술 보고서**입니다. 시뮬레이터와 에이전트 간의 **상호작용 과정**과 **각 스텝마다 주고받는 값들**의 정확한 구조와 단위를 분석하여 제공됩니다.

---

## 목차
1. [시뮬레이터 초기화: 어떤 값/시퀀스가 입력되는가?](#1)
2. [시뮬레이터 → 에이전트: 실제 전달값(구조, 단위, 유동성 포함)](#2)
3. [에이전트 → 시뮬레이터: 실제 전달값(구조, 단위, 실제 적용)](#3)
4. [각 스텝에서 오가는 값의 5단계 예시(실시간 인풋/아웃풋 흐름)](#4)
5. [세부 사항: 값과 차원의 유동성, 단위](#5)
6. [주요 코드 근거/참조](#6)

---

## 1. **시뮬레이터 초기화: 어떤 값/시퀀스가 입력되는가?**

### 코드 근거: `G2P2C/agents/g2p2c/worker.py` - `get_env()` 호출

- **사용할 환자 이름** (예: `\"adult#005\"`, `\"adolescent#002\"`)
- **환경 ID** (예: `\"simglucose-adult5-v0\"`)
- **실험 옵션 객체** (`feature_history`, 혈당·인슐린 min/max, 옵션 등)
- **실험 난수 시드**

> 이때 입력받은 값은 환경과 환자 개체, 상태 이력 버퍼(`StateSpace` 등) 생성에 활용되며, 초기 혈당/인슐린 값, feature 이력의 초기값도 아래 값으로 세팅됩니다.

---

## 2. **시뮬레이터 → 에이전트: 실제 값**

### (a) 전달값 구조 및 단위
- **`cur_state`**: shape=(feature_history, n_features), dtype=float32
  - 각 행: [혈당(mg/dL 또는 mmol/L), 인슐린(U/h)]
  - 혈당: 임상 단위(보통 mg/dL; 환경 내부 환자 설정에 따라 다를 수 있음), 예시=132.1
  - 인슐린: U/h (0.00~0.07 등, 매우 작은 값, 실제 인슐린 펌프 연속 주입 평균 단위)
  
- **`feat`**: shape=(n_handcrafted_features,), dtype=float32
  - 예시: [IOB(Insulin-on-board), time-of-day, 기타 도메인 특성]
  - 값과 구조는 환경 옵션·feature 구성에 따라 바뀔 수 있으나, **step 동안 차원 수는 고정**
  - 값은 매 스텝마다 업데이트되어 동적으로 변함

### (b) 실제 코드 예시 객체
```python
# 예시 (feature_history=6, n_features=2, n_handcrafted_features=5)
cur_state = np.array([
    [132.1, 0.00],
    [129.7, 0.00],
    [120.1, 0.02],
    [118.8, 0.01],
    [115.5, 0.04],
    [112.2, 0.03]
], dtype=np.float32)  # (6,2): 6-step window의 혈당/인슐린

feat = np.array([1.82, 0.07, 0.03, 12.5, 0.0], dtype=np.float32)  # (5,)
```

> 이 값은 `self.state_space.update(cgm=..., ins=..., meal=...)`로 매 스텝 생성됩니다.

---

## 3. **에이전트 → 시뮬레이터: 실제 값**

### (a) 구조, 단위, 실제 적용
- **`selected_action`**: float, 인슐린 투여량(U/h)
  - 에이전트(`policy.get_action(...)`)의 output
  - 예: 0.035 (U/h 단위)

- 실제 환경입력(pump 변환): pump 클래스에서 min/max 클리핑 등 적용
  - `pump_action = min(max(selected_action, insulin_min), action_scale)`
  - 실제 환자/펌프가 허용하는 단위 내로 제한됨(U/h 단위)

### (b) 전달 코드 예시
```python
selected_action = policy.get_action(cur_state, feat)['action'][0]  # float
pump_action = max(min(selected_action, max_val), min_val)
# pump_action만 환경 step()에 입력됨
```

---

## 4. **5스텝의 실제 상호작용 예시 (실제 코드구조와 논의 내용 완전 반영)**

아래 예시는 **실제 코드의 동작 방식·값 포맷·단위를 그대로 반영**하며, 각 스텝마다 어떤 값이 생성/전달/업데이트되는지 상세히 보여줍니다.

### Step 0
**시뮬레이터 → 에이전트**
```python
cur_state_0 = [
    [132.1, 0.00], [129.7, 0.00], [120.1, 0.02],
    [118.8, 0.01], [115.5, 0.04], [112.2, 0.03]
]
feat_0 = [1.82, 0.07, 0.03, 12.5, 0.0]
```

**에이전트 → 시뮬레이터**
```python
selected_action_0 = 0.035      # float, U/h
pump_action_0 = max(min(0.035, max_val), min_val)
```

**환경 step 후 반환**
```python
cgm_1 = 113.7
reward_1 = -0.5
done_1 = False
info_1 = { 'meal': 0, ... }
```

### Step 1
**`cur_state_1`**
```python
[[129.7, 0.00], [120.1, 0.02], [118.8, 0.01], 
 [115.5, 0.04], [112.2, 0.03], [113.7, 0.035]]
```

**`feat_1` (IOB, 시간 등 업데이트)**
```python
[1.74, 0.11, 0.09, 13.4, 0.0]
```

**선택 행동 및 환경 step**
```python
selected_action_1 = 0.028
pump_action_1 = ...
cgm_2 = 117.2, reward_2 = -0.42, done_2 = False, info_2={...}
```

### Step 2
`cur_state_2`: (최신값 slide in), `feat_2`: ..., `selected_action_2`, ...

### Step 3
`cur_state_3`~`selected_action_3` 과정 동일, 값만 시간 흐름에 따라 갱신

### Step 4
글로벌 규칙:  
- cur_state/feat: step 진행에 따라 shift, 값 갱신.  
- selected_action: 매 step마다 새로 산출, 펌프변환.  
- 시뮬레이터 출력: 혈당/리워드/종료/부가정보 등 반환.

---

## 5. **feat, cur_state 차원과 값의 유동성/불변성, 단위 (중요 논의 요약)**

- **cur_state**
  - 각 행 = [혈당, 인슐린] 등, 단위: mg/dL(or mmol/L), U/h
  - `feature_history`, `n_features`는 환경/옵션으로 고정. 단위 및 크기 변하지 않음.

- **feat**
  - 값은 매 스텝 변함(동적).
  - **차원수(길이)는 환경/feature 옵션에 의해 초기화 시 고정**
  - 환경 타입/옵션 바뀌거나 완전히 새 환경 접목 시만 shape이 변함  
  - 한 환경 내에서는 episode·step과 무관하게 feat shape(차원)은 불변

- **selected_action**
  - float, U/h(인슐린) 단위, min/max 지정됨.

---

## 6. 주요 코드 근거/참조
- `G2P2C/agents/g2p2c/worker.py` (rollout, 행동/상태 루프)
- `G2P2C/utils/statespace.py` (state/feat 관리, update)
- `G2P2C/agents/g2p2c/models.py` (get_action, 정책)
- `G2P2C/utils/pumpAction.py` (pump_action 변환)
- `G2P2C/environments/simglucose/simglucose/envs/simglucose_gym_env.py` (환경 내부)

---

## 결론 및 사용상의 시사점
- 에이전트와 시뮬레이터는 step별로 **주기적으로 state, feat → action → (갱신된)state, feat** 순환 구조로 상호작용.
- **shape(차원)은 환경 옵션에 따라 결정되며, 값만 시간경과에 따라 변동**
- 실험 형태/환경 변경 시, 해당 입력 데이터 shape만 신경 쓰면 타 환경에도 에이전트 재활용이 용이.

---

이 문서는 **실제 구현** 및 여러 환경/옵션의 “실제 데이터” 관점에서 **agent-simulator interface 흐름**을 체계적으로 요약합니다.
