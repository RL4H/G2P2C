## *** Begin Patch
*** Update File: G2P2C/Agent-Simulator_Interaction_Report.md
@@
-# G2P2C: Agent–Simulator Interaction Detailed Report

## -본 문서는 G2P2C 프로젝트 코드의 실제 구현을 근거로, 에이전트와 시뮬레이터(환경) 간에 오고 가는 수치/시퀀스/정보 구조 및 구체 예시를 심도 있고 신뢰성 
있게 정리한 기술 보고서입니다.

## -## 목차
-1. 초기화 단계에서 시뮬레이터에 제공되는 값 및 시퀀스
-2. 시뮬레이터가 에이전트에게 전달하는 값 및 시퀀스
-3. 에이전트가 시뮬레이터에게 전달하는 값 및 시퀀스
-4. 실제 코드 기반 상호작용 순서와 명확한 예시
-5. 실제 한 단계 전체 예시
-6. 참고: 코드 근거와 주요 위치

-----------------------------------------------------------------------------------------------------------------------------------------------------------
---

    *

## -<a id="1"></a>
-## 1. 초기화 단계: 시뮬레이터에 제공되는 값/시퀀스

-- 실제 코드 위치:

    * `G2P2C/agents/g2p2c/worker.py`
    * 함수: `self.env = get_env(self.args, patient_name=..., env_id=..., ...)`, `self.env.reset()`
        -- **주요 입력 값**
    * * `patient_name`: 환자 구분 이름 (예: `'adult#005'`, `'adolescent#001'`)
    * * `env_id`: 환경 고유 ID (예: `"simglucose-adult5-v0"`)
    * * `seed`: 시뮬레이션 난수 시드 (재현성/무작위성 제어)
    * * `args`: 실험 옵션 세트(혈당:최저/최고값, feature_history 등 실험 파라미터)
            -- **동작**:
    * 환경과 시뮬 환자, 초기 상태, 상태 시퀀스 버퍼(`StateSpace`의 deque 등)를 모두 셋업합니다.
    *

-----------------------------------------------------------------------------------------------------------------------------------------------------------
---

    *

## -<a id="2"></a>
-## 2. 시뮬레이터 → 에이전트: 시뮬레이터가 에이전트에게 보내는 값/시퀀스

-- 실제 코드 위치:

    * * `self.cur_state, self.feat = self.state_space.update(cgm=..., ins=..., meal=...)`
    * * 매 스텝마다 cur_state/feat 갱신, 에이전트의 관측값으로 사용됨
    *

-- 주요 값 설명

    * * **cur_state** (`state`):
    * * shape: `(feature_history, n_features)` (`numpy.ndarray`, `float32`)
    * * 내용: 최근 `feature_history` 길이 만큼의
    * - 혈당(CGM) 값 시퀀스
    * - 인슐린(action) 이력
    * - (옵션) 식사, 시간, 기타 부가 항목(옵션 feature에 따라 컬럼 증가)
    * * **feat** (`feature`):
    * * shape: `(n_handcrafted_features,)` (`numpy.ndarray`, `float32`)
    * * 예) IOB(Insulin-on-board), time-of-day, 기타 도메인 특성
    *

-- 예시 배열

    *
    * cur_state = np.array([
    * [132.1, 0.00],
    * [129.7, 0.00],
    * [120.1, 0.02],
    * [118.8, 0.01],
    * [115.5, 0.04],
    * [112.2, 0.03],
    * ], dtype=np.float32) # (6,2): 6-step window의 혈당/인슐린
    * feat = np.array([1.82, 0.07, 0.03, 12.5, 0.0], dtype=np.float32) # (5,)
    *
    *

-----------------------------------------------------------------------------------------------------------------------------------------------------------
---

    *

## -<a id="3"></a>
-## 3. 에이전트 → 시뮬레이터: 에이전트가 보내는 값/시퀀스

-- 실제 코드 위치

    * * `policy_step = policy.get_action(self.cur_state, self.feat)`
    * * `selected_action = policy_step['action'][0]`
    * * `state, reward, done, info = self.env.step(pump_action)`
    *

-- 주요 값 설명

    * * **selected_action** (`action`):
    * * shape: `(1,)`, 또는 float 단일값
    * * 내용: 인슐린 투여량 (단위: U/h 등, 실수형)
    * * 예) `selected_action = np.array([0.035], dtype=np.float32)` (0.035 U/h)
    * * pump 변환 뒤(클리핑 등): 실제 환경에 입력
    *
    * pump_action = min(max(selected_action, args.insulin_min), args.action_scale)
    *
    *

-----------------------------------------------------------------------------------------------------------------------------------------------------------
---

    *

## -<a id="4"></a>
-## 4. 상호작용 전체 순서(코드/데이터 흐름)

-#### 1. 초기화

    * * 시뮬레이터 인스턴스 생성
    * * 입력: `args`, `patient_name`, `env_id`, `seed`
    * * 환경 상태 버퍼(deque 등) 및 환자 초기화
    *

-#### 2. state/feat 생성 및 에이전트에 전달

    * * 최근 `feature_history`길이의 혈당, 인슐린 등 과거 시퀀스 (`cur_state`)
    * * 보조 도메인 feature 벡터 (`feat`)
    *

-#### 3. 에이전트 정책에 따라 행동(action) 산출

    * * 정책 모델 입력: (`cur_state`, `feat`)
    * * 산출: `selected_action` (float · continuous)
    *

-#### 4. (Pump 등 거쳐) 환경에 action 전달

    * * 환경에 실제로 입력되는 action: `pump_action` (float, min/max 적용)
    *

-#### 5. 환경 step: 시뮬레이션 반영

    * * 환경의 혈당상태/리워드/done/info 등 반환
    * * 다음 state/feat에 반영, 루프 반복
    *

-----------------------------------------------------------------------------------------------------------------------------------------------------------
---

    *

## -<a id="5"></a>
-## 5. 구체적 1 step 상호작용 예시

## -### (하이퍼파라미터 예)
-- feature_history = 6
-- n_features = 2 (혈당, 인슐린)
-- n_handcrafted_features = 5

## -### 실제 1 step 데이터 예

## -1) 시뮬레이터 → 에이전트

-- cur_state

    *
    * [[132.1, 0.00],
    * [129.7, 0.00],
    * [120.1, 0.02],
    * [118.8, 0.01],
    * [115.5, 0.04],
    * [112.2, 0.03]]    # 최근 6스텝치 혈당/인슐린
    *

-- feat

    *
    * [1.82, 0.07, 0.03, 12.5, 0.0]   # 예: IOB, 시간, 기타 특성
    *
    *

-2) 에이전트 → 시뮬레이터
-- selected_action:

    *
    * 0.035   # float, U/h단위 인슐린 투여량
    *

-- pump 변환/클리핑 후 실제 투여

    *
    * pump_action = max(min(0.035, max_val), min_val)
    *
    *

## -3) 시뮬레이터→에이전트(다음step)
-- 다음 혈당: 113.7
-- reward: -0.5
-- done: False
-- info: { 'meal': 0, ... }
-- next step에 state 시퀀스 slide-in

-----------------------------------------------------------------------------------------------------------------------------------------------------------
---

    *

## -<a id="6"></a>
-## 6. 주요 코드 참조

## -- G2P2C/agents/g2p2c/worker.py (시뮬레이터/상태/행동 관련 주요 흐름)
-- G2P2C/utils/statespace.py (상태 시퀀스 버퍼/관리)
-- G2P2C/agents/g2p2c/models.py (정책 get_action)
-- G2P2C/environments/simglucose/simglucose/envs/simglucose_gym_env.py (환경/step 구현)

-----------------------------------------------------------------------------------------------------------------------------------------------------------
---

    *

## -## 요약

## -- 초기화: (환자/seed/env id), 실험 파라미터 명시 입력
-- 시뮬레이터→에이전트: 최근 기록 시퀀스(cur_state) + 도메인 feature(feat) 전달 (모두 float 배열)
-- 에이전트→시뮬레이터: 인슐린 행동(selected_action, float) 1개
-- 위 값들의 자료형, shape, 예시값 및 전 과정 데이터 흐름을 실제 코드 근거로 전부 제시함

-실제 환자 유형/옵션 바뀔 때 추가 feature나 action scaling만 맞추면 동일 구조로 확장 가능.
-(이 문서는 코드 리스팅과 함께 신뢰성 있게 agent-simulator interface/프로토콜을 정리합니다.)
+# G2P2C: Agent–Simulator Interaction Report
+
+본 문서는 실제 G2P2C 코드 레퍼런스를 기반으로, 에이전트(Agent)와 시뮬레이터(환경)가 각 스텝마다 실제로 주고받는 값의 타입/구조/단위까지 명확하게 분석하고,
    논의한 구체 예시(값 채번 흐름 포함)를 바탕으로 작성한 신뢰성 있는 기술 보고서입니다.
+
+---
+## 목차
+1. 시뮬레이터 초기화: 어떤 값/시퀀스가 입력되는가?
+2. 시뮬레이터 → 에이전트: 실제 전달값(구조, 단위, 유동성 포함)
+3. 에이전트 → 시뮬레이터: 실제 전달값(구조, 단위, 실제 적용)
+4. 각 스텝에서 오가는 값의 5단계 예시(실시간 인풋/아웃풋 흐름)
+5. 세부 사항: 값과 차원의 유동성, 단위
+6. 주요 코드 근거/참조
+
+---
+## 1. 시뮬레이터 초기화: 어떤 값/시퀀스가 입력되는가?
+코드 근거: G2P2C/agents/g2p2c/worker.py - get_env() 호출
+- 사용할 환자 이름 (예: "adult#005", "adolescent#002")
+- 환경 ID (예: "simglucose-adult5-v0")
+- 실험 옵션 객체 args (feature_history, 혈당·인슐린 min/max, 옵션 등)
+- 실험 난수 시드
+
+> 이때 입력받은 값은 환경과 환자개체, 상태이력버퍼(StateSpace 등) 생성에 활용되며,

    * 초기 혈당/인슐린 값, feature 이력의 초기값도 아래 값으로 세팅됩니다.
    *

+---
+## 2. 시뮬레이터 → 에이전트: 실제 값
+### (a) 전달값 구조 및 단위
+- cur_state: shape=(feature_history, n_features), dtype=float32

    * * 각 행: [혈당(mg/dL 또는 mmol/L), 인슐린(U/h)]
    * * 혈당: 임상 단위(보통 mg/dL; 환경 내부 환자 설정에 따라 다를 수 있음), 예시=132.1
    * * 인슐린: U/h (0.00~0.07 등, 매우 작은 값, 실제 인슐린 펌프 연속 주입 평균 단위)

+- feat: shape=(n_handcrafted_features,), dtype=float32

    * * 예시: [IOB_20min, IOB_60min, IOB_120min, time-of-day, ...]
    * * 값과 구조는 환경옵션·feature구성에 따라 바뀔 수 있으나, **step동안 차원 수는 고정**
    * * 값은 매 스텝마다 업데이트되어 동적으로 변함
    *

+### (b) 실제 코드 예시 객체
+```python
+# 예시 (feature_history=6, n_features=2, n_handcrafted_features=5)
+cur_state = np.array([

    * [132.1, 0.00],
    * [129.7, 0.00],
    * [120.1, 0.02],
    * [118.8, 0.01],
    * [115.5, 0.04],
    * [112.2, 0.03]

+], dtype=np.float32)
+feat = np.array([1.82, 0.07, 0.03, 12.5, 0.0], dtype=np.float32)
+```
+> 이 값은 self.state_space.update(cgm=..., ins=..., meal=...)로 매 스텝 생성됩니다.
+
+---
+## 3. 에이전트 → 시뮬레이터: 실제 값
+### (a) 구조, 단위, 실제 적용
+- selected_action: float, 인슐린 투여량(U/h)

    * * 에이전트(policy.get_action(...))의 output
    * * 예: 0.035 (U/h 단위)

+- 실제 환경입력(pump 변환): pump 클래스에서 min/max 클리핑 등 적용

    * * pump_action = min(max(selected_action, insulin_min), action_scale)
    * * 실제 환자/펌프가 허용하는 단위 내로 제한됨(U/h 단위)
    *

+### (b) 전달 코드 예시
+python +selected_action = policy.get_action(cur_state, feat)['action'][0]  # float +pump_action = max(min(selected_action, max_val), min_val) +# 
pump_action만 환경 step()에 입력됨 +
+---
+## 4. 5스텝의 실제 상호작용 예시 (실제 코드구조와 논의 내용 완전 반영)
+아래 예시는 실제 코드의 동작 방식·값 포맷·단위를 그대로 반영하며, 각 스텝마다
+어떤 값이 생성/전달/업데이트되는지 상세히 보여줍니다.
+
+### Step 0
+시뮬레이터 → 에이전트
+```
+cur_state_0 = [

    * [132.1, 0.00], [129.7, 0.00], [120.1, 0.02],
    * [118.8, 0.01], [115.5, 0.04], [112.2, 0.03] ]

+feat_0 = [1.82, 0.07, 0.03, 12.5, 0.0]
+ +**에이전트 → 시뮬레이터** +
+selected_action_0 = 0.035      # float, U/h
+pump_action_0 = max(min(0.035, max_val), min_val)
+ +**환경 step 후 반환** +
+cgm_1 = 113.7
+reward_1 = -0.5
+done_1 = False
+info_1 = { 'meal': 0, ... }
+ + +### Step 1 +**cur_state_1** +
+[[129.7, 0.00], [120.1, 0.02], [118.8, 0.01], [115.5, 0.04], [112.2, 0.03], [113.7, 0.035]]
+ +**feat_1** (IOB, 시간 등 업데이트): +
+[1.74, 0.11, 0.09, 13.4, 0.0]
+ +**선택 행동 및 환경 step** +
+selected_action_1 = 0.028
+pump_action_1 = ...
+cgm_2 = 117.2, reward_2 = -0.42, done_2 = False, info_2={...}
+```
+
+### Step 2
+cur_state_2: (최신값 slide in), feat_2: ..., selected_action_2, ...
+
+### Step 3
+cur_state_3~`selected_action_3` 과정 동일, 값만 시간 흐름에 따라 갱신
+
+### Step 4
+글로벌 규칙:
+- cur_state/feat: step 진행에 따라 shift, 값 갱신.
+- selected_action: 매 step마다 새로 산출, 펌프변환.
+- 시뮬레이터 출력: 혈당/리워드/종료/부가정보 등 반환.
+
+---
+## 5. feat, cur_state 차원과 값의 유동성/불변성, 단위 (중요 논의 요약)
+- cur_state

    * * 각행 = [혈당, 인슐린] 등, 단위: mg/dL(or mmol/L), U/h
    * * feature_history, n_features는 환경/옵션으로 고정. 단위 및 크기 변하지 않음.

+- feat

    * * 값은 매 스텝 변함(동적).
    * * **차원수(길이)는 환경/feature 옵션에 의해 초기화 시 고정**
    * * 환경타입/옵션 바뀌거나 완전히 새 환경 접목시만 shape이 변함
    * * 한 환경 내에서는 episode·step과 무관하게 feat shape(차원)은 불변

+- selected_action

    * * float, U/h(인슐린) 단위, min/max 지정됨.
    *

+---
+## 6. 주요 코드 근거/참조
+- G2P2C/agents/g2p2c/worker.py (rollout, 행동/상태 루프)
+- G2P2C/utils/statespace.py (state/feat 관리, update)
+- G2P2C/agents/g2p2c/models.py (get_action, 정책)
+- G2P2C/utils/pumpAction.py (pump_action 변환)
+- G2P2C/environments/simglucose/simglucose/envs/simglucose_gym_env.py (환경 내부)
+
+---
+## 결론 및 사용상의 시사점
+- 에이전트와 시뮬레이터는 step별로 주기적으로 state, feat → action → (갱신된)state, feat 순환구조로 상호작용.
+- shape(차원)은 환경 옵션에 따라 결정되며, 값만 시간경과에 따라 변동
+- 실험 형태/환경 변경 시, 해당 입력 데이터 shape만 신경 쓰면 타 환경에도 에이전트 재활용이 용이.
+
+---
+
+이 문서는 실제 구현 및 여러 환경/옵션의 “실제 데이터” 관점에서 agent-simulator interface 흐름을 체계적으로 요약합니다.
*** End Patch
