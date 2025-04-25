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

알겠습니다! **Step 1~5 예시 부분**을 보다 **구체적으로 설명**하여 생략 없이 자세히 작성해드리겠습니다. 아래는 **시뮬레이터와 에이전트 간의 상호작용** 예시를 **각 단계별로 완전하게 작성**한 내용입니다.

---

## 4. **5스텝의 실제 상호작용 예시 (실제 코드구조와 논의 내용 완전 반영)**

아래 예시는 **실제 코드의 동작 방식·값 포맷·단위를 그대로 반영**하며, 각 스텝마다 어떤 값이 생성/전달/업데이트되는지 상세히 보여줍니다.

### Step 0

#### **시뮬레이터 → 에이전트**

시뮬레이터는 에이전트에게 **현재 상태(`cur_state`)**와 **특징 벡터(`feat`)**를 전달합니다. 

- **`cur_state_0`** (최근 6스텝치 혈당과 인슐린):
```python
cur_state_0 = [
    [132.1, 0.00],  # 혈당 132.1 mg/dL, 인슐린 0.00 U/h (6 steps ago)
    [129.7, 0.00],  # 혈당 129.7 mg/dL, 인슐린 0.00 U/h (5 steps ago)
    [120.1, 0.02],  # 혈당 120.1 mg/dL, 인슐린 0.02 U/h (4 steps ago)
    [118.8, 0.01],  # 혈당 118.8 mg/dL, 인슐린 0.01 U/h (3 steps ago)
    [115.5, 0.04],  # 혈당 115.5 mg/dL, 인슐린 0.04 U/h (2 steps ago)
    [112.2, 0.03]   # 혈당 112.2 mg/dL, 인슐린 0.03 U/h (1 step ago)
]
```
- **`feat_0`** (각종 handcrafted features):
```python
feat_0 = [1.82, 0.07, 0.03, 12.5, 0.0]  # 예시: IOB(20분), IOB(60분), IOB(120분), 시간대, 기타 특성
```

#### **에이전트 → 시뮬레이터**

에이전트는 **현재 상태(`cur_state_0`)**와 **특징 벡터(`feat_0`)**를 바탕으로 **행동(`selected_action`)**을 결정합니다. 이 값은 **인슐린 투여량**을 나타내며, **U/h** 단위로 주어집니다.

- **`selected_action_0`**:
```python
selected_action_0 = 0.035  # float, U/h 단위 인슐린 투여량
```

이 **인슐린 투여량** 값은 시뮬레이터의 **펌프 시스템**에 의해 **pump_action**으로 변환됩니다. 여기서는 **클리핑**이 적용되어, 인슐린 투여량이 시뮬레이터의 제한 범위 내로 조정됩니다.

- **`pump_action_0`** (변환 후):
```python
pump_action_0 = max(min(0.035, max_val), min_val)  # max/min 클리핑이 적용되어 실제 환경에 입력됨
```

#### **시뮬레이터 step → 환경상태 갱신**

시뮬레이터는 **`pump_action_0`**을 적용한 후, **다음 혈당**, **보상(reward)**, **done 상태** 등을 반환합니다.

- **`cgm_1`** (다음 혈당):
```python
cgm_1 = 113.7  # 5초 후 새로운 혈당 값
```

- **`reward_1`** (보상):
```python
reward_1 = -0.5  # 보상 값 (혈당 조절 정도에 따라 달라짐)
```

- **`done_1`** (게임 종료 여부):
```python
done_1 = False  # 에피소드가 끝나지 않음
```

- **`info_1`** (추가 정보):
```python
info_1 = {'meal': 0, 'exercise': 0}  # 예시: 식사나 운동 이벤트 정보
```

**다음 상태**는 **`cur_state_1`**으로 갱신되며, 1단계에 대한 새로운 **`feat_1`**도 생성됩니다.

---

### Step 1

#### **시뮬레이터 → 에이전트**

**`cur_state_1`**은 이전의 **`cur_state_0`** 데이터를 오른쪽으로 **shift**하고, **새로운 혈당**과 **인슐린** 값을 마지막에 추가합니다. 예를 들어, **`cgm_1 = 113.7`**과 **`pump_action_0 = 0.035`**이 **새로운 값**으로 추가됩니다.

```python
cur_state_1 = [
    [129.7, 0.00],  # 혈당 129.7 mg/dL, 인슐린 0.00 U/h
    [120.1, 0.02],  # 혈당 120.1 mg/dL, 인슐린 0.02 U/h
    [118.8, 0.01],  # 혈당 118.8 mg/dL, 인슐린 0.01 U/h
    [115.5, 0.04],  # 혈당 115.5 mg/dL, 인슐린 0.04 U/h
    [112.2, 0.03],  # 혈당 112.2 mg/dL, 인슐린 0.03 U/h
    [113.7, 0.035]  # 혈당 113.7 mg/dL, 인슐린 0.035 U/h (새로운 값)
]
```

- **`feat_1`**는 IOB, 시간대 등의 **handcrafted feature** 값으로 업데이트됩니다.

```python
feat_1 = [1.74, 0.11, 0.09, 13.4, 0.0]  # 예시: IOB(20분), IOB(60분), IOB(120분), 시간대, 기타 특성
```

#### **에이전트 → 시뮬레이터**

에이전트는 **`selected_action_1`**을 산출합니다.

```python
selected_action_1 = 0.028  # float, U/h 단위 인슐린 투여량
```

이 값은 시뮬레이터로 전달되며, **`pump_action_1`**으로 변환됩니다.

```python
pump_action_1 = max(min(0.028, max_val), min_val)  # 실제 환경에 적용
```

#### **시뮬레이터 step**

시뮬레이터는 **`pump_action_1`**을 바탕으로 **다음 혈당**, **보상**, **done 상태**를 반환합니다.

```python
cgm_2 = 117.2
reward_2 = -0.42
done_2 = False
info_2 = {'meal': 1, 'exercise': 0}
```

---

### Step 2

#### **시뮬레이터 → 에이전트**

**`cur_state_2`**는 이전 상태에서 **최신 혈당과 인슐린**을 추가한 값입니다.

```python
cur_state_2 = [
    [120.1, 0.02],  # 혈당 120.1 mg/dL, 인슐린 0.02 U/h
    [118.8, 0.01],  # 혈당 118.8 mg/dL, 인슐린 0.01 U/h
    [115.5, 0.04],  # 혈당 115.5 mg/dL, 인슐린 0.04 U/h
    [112.2, 0.03],  # 혈당 112.2 mg/dL, 인슐린 0.03 U/h
    [113.7, 0.035], # 혈당 113.7 mg/dL, 인슐린 0.035 U/h
    [117.2, 0.028]  # 혈당 117.2 mg/dL, 인슐린 0.028 U/h (새로운 값)
]
```

- **`feat_2`**는 **`feat_1`**에서 업데이트된 값으로, 시간에 따라 변합니다.

```python
feat_2 = [1.65, 0.15, 0.15, 14.2, 0.0]
```

#### **에이전트 → 시뮬레이터**

에이전트는 **`selected_action_2`**를 산출합니다.

```python
selected_action_2 = 0.032  # float, U/h 단위 인슐린 투여량
```

그리고 이를 **`pump_action_2`**로 변환하여 시뮬레이터로 전달합니다.

```python
pump_action_2 = max(min(0.032, max_val), min_val)
```

#### **시뮬레이터 step**

시뮬레이터는 **`pump_action_2`**에 따라 **다음 혈당**과 **보상** 등을 계산합니다.

```python
cgm_3 = 124.0
reward_3 = -0.48
done_3 = False
info_3 = {'meal': 1, 'exercise': 1}
```

---

### Step 3

#### **시뮬레이터 → 에이전트**

**`cur_state_3`**는 **`cur_state_2`**에서 **최신 혈당**과 **인슐린** 값을 업데이트한 상태입니다.

```python
cur_state_3 = [
    [118.8, 0.01],  # 혈당 118.8 mg/dL, 인슐린 0.01 U/h
    [115.5, 0.04],  # 혈당 115.5 mg/dL, 인슐린 0.04 U/h
    [112.2, 0.03],  # 혈당 112.2 mg/dL, 인슐린 0.03 U/h
    [113.7, 0.035], # 혈당 113.7 mg/dL, 인슐린 0.035 U/h
    [117.2, 0.028], # 혈당 117.2 mg/dL, 인슐린 0.028 U/h
    [124.0, 0.032]  # 혈당 124.0 mg/dL, 인슐린 0.032 U/h
]
```

- **`feat_3`**는 시간이 지나면서 업데이트된 **handcrafted features**입니다.

```python
feat_3 = [1.52, 0.17, 0.22, 15.1, 0.0]
```

#### **에이전트 → 시뮬레이터**

에이전트는 **`selected_action_3`**을 계산하고, **`pump_action_3`**으로 변환하여 시뮬레이터에 전달합니다.

```python
selected_action_3 = 0.029  # float, U/h 단위 인슐린 투여량
pump_action_3 = max(min(0.029, max_val), min_val)
```

#### **시뮬레이터 step**

시뮬레이터는 **`pump_action_3`**을 적용한 후, **다음 혈당**과 **보상**을 계산하여 반환합니다.

```python
cgm_4 = 119.8
reward_4 = -0.45
done_4 = False
info_4 = {'meal': 1, 'exercise': 1}
```

---

### Step 4

#### **시뮬레이터 → 에이전트**

**`cur_state_4`**는 **`cur_state_3`**에서 **최신 혈당**과 **인슐린** 값을 업데이트한 값입니다.

```python
cur_state_4 = [
    [115.5, 0.04],  # 혈당 115.5 mg/dL, 인슐린 0.04 U/h
    [112.2, 0.03],  # 혈당 112.2 mg/dL, 인슐린 0.03 U/h
    [113.7, 0.035], # 혈당 113.7 mg/dL, 인슐린 0.035 U/h
    [117.2, 0.028], # 혈당 117.2 mg/dL, 인슐린 0.028 U/h
    [124.0, 0.032], # 혈당 124.0 mg/dL, 인슐린 0.032 U/h
    [119.8, 0.029]  # 혈당 119.8 mg/dL, 인슐린 0.029 U/h
]
```

- **`feat_4`**는 **time-of-day**, **IOB** 등 **handcrafted features**를 바탕으로 업데이트된 값입니다.

```python
feat_4 = [1.44, 0.18, 0.23, 16.0, 0.0]
```

#### **에이전트 → 시뮬레이터**

에이전트는 **`selected_action_4`**을 산출하여 시뮬레이터로 전달합니다.

```python
selected_action_4 = 0.031  # float, U/h 단위 인슐린 투여량
pump_action_4 = max(min(0.031, max_val), min_val)
```

#### **시뮬레이터 step**

시뮬레이터는 **`pump_action_4`**을 적용한 후, **다음 혈당**과 **보상** 등을 계산하여 반환합니다.

```python
cgm_5 = 117.9
reward_5 = -0.41
done_5 = False
info_5 = {'meal': 0, 'exercise': 1}
```

---

## Step별 데이터 흐름 요약
- **cur_state**: 항상 shape (6,2), 데이터는 한 단계씩 밀려오며 최신 관측치가 마지막 row에 추가
- **feat**: shape(5,) (예시), 값은 시뮬 동작/시간 등 실시간 업데이트
- **selected_action**: float, 각 step마다 새롭게 결정됨, 실제 시뮬 입력값
- **pump_action**: clipping 등 보정 후 실제 환경 입력값
- **환경 출력**: cgm(혈당), reward, done, info 등 —> 다시 state/feat update

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
