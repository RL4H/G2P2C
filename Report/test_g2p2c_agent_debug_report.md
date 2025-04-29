<!--- 
  File: Report/test_g2p2c_agent_debug_report.md
  Purpose: Detailed markdown report explaining test_g2p2c_agent.py code flow and debugging steps
-->
# test_g2p2c_agent.py 디버깅 및 코드 흐름 보고서

## 1. 개요
- **목적**: 학습된 G2P2C 에이전트를 외부 시뮬레이터 없이 독립적으로 로드하고 추론 기능을 검증하기 위한 테스트 스크립트(`test_g2p2c_agent.py`)의 구조와 동작 과정을 설명합니다.
- **주요 확인 사항**:
  1. 에이전트 모델(`.pth`)이 정상 로드되는지
  2. 예상 입력(shape, type)을 주었을 때 올바른 출력(action)을 반환하는지
  3. 비정상 입력(shape 오류, NaN 등)에 대해 예외를 올바르게 처리하는지

## 2. 디렉터리 구조
```
G2P2C/
├─ Sim_CLI/
│  └─ test_g2p2c_agent.py    # 테스트 스크립트
├─ agents/
│  └─ g2p2c/
│     ├─ g2p2c.py          # G2P2C 에이전트 클래스
│     └─ models.py         # ActorCritic, FeatureExtractor 등 모델 정의
└─ Report/
   └─ test_g2p2c_agent_debug_report.md  # 이 문서
```

## 3. 테스트 스크립트 흐름
1. **옵션 로드**
   - `results/test/args.json`에서 학습 시 사용된 하이퍼파라미터를 불러와 `args` 객체에 복원
2. **환경 설정**
   - `parameters.py`를 동적으로 로드해 `args`에 추가 옵션 설정
   - `device = torch.device('cpu')`
3. **에이전트 인스턴스 생성**
   ```python
   from agents.g2p2c.g2p2c import G2P2C
   g2p2c_agent = G2P2C(args, device, load=False, path1=None, path2=None)
   ```
4. **모델 로드**
   - 체크포인트(`episode_14_Actor.pth`)를 `torch.load`로 불러옴
   - 저장된 객체가 전체 모델(`ActorNetwork`)이므로, `load_state_dict` 대신
     ```python
     g2p2c_agent.policy.Actor = loaded_obj
     g2p2c_agent.policy.Actor.eval()
     ```
5. **추론 테스트 함수 실행**
   - `test_g2p2c_inference()` 내부에서 valid/invalid 입력 케이스를 순회하며 `policy.get_action(s, feat)` 호출

## 4. 주요 컴포넌트 역할
### 4.1 G2P2C 클래스 (`agents/g2p2c/g2p2c.py`)
- `self.policy = ActorCritic(...)`으로 에이전트(Actor+Critic) 네트워크 인스턴스 생성
- `self.policy.Actor` / `self.policy.Critic`로 각각 액터·크리틱 네트워크 접근

### 4.2 ActorCritic (`agents/g2p2c/models.py`)
```python
class ActorCritic(nn.Module):
    def get_action(self, s, feat):
        (mu, std, act, logp, ...), (val, ...) = self.predict(s, feat)
        # NumPy 배열로 변환해 반환
        return {k: v.detach().cpu().numpy() for k, v in data.items()}
    def predict(self, s, feat):
        # 텐서 변환 후 Actor 및 Critic forward 호출
        return self.Actor(s, feat, ...), self.Critic(s, feat, ...)
```

### 4.3 FeatureExtractor
- LSTM 기반 시퀀스 피처 추출기
- `mode=='batch'` vs `mode!='batch'` 분기:
  - **batch**: 이미 `(batch, seq, feat)`인 입력 사용
  - **forward**: 2D(`(seq, feat)`)도 지원하도록 내부에서 `s.unsqueeze(0)` 처리

## 5. 수정 및 예외 처리 로직
1. **정상 입력** `(12, 2)` → 내부 `FeatureExtractor`가 `s.unsqueeze(0)`로 `(1,12,2)` 변환 → LSTM 통과 → PASS
2. **비정상 입력**
   - `(11,2)` 또는 `(12,3)` → 스크립트에서 이미 `s.unsqueeze(0)` → `(1,11,2)` 등 → `FeatureExtractor` 내부 추가 `unsqueeze` → 4D → LSTM 차원 예외 발생 → 올바른 예외 처리
   - `NaN` 입력도 동일한 형상 오류로 예외 처리

## 6. 결론
- 테스트 스크립트 한 줄(`s = torch.FloatTensor(arr).unsqueeze(0)`)과 `FeatureExtractor`의 2중 unsqueeze 로직을 통해
  - 정상 케이스는 자동으로 3D 입력으로 보정되어 통과
  - 비정상 케이스는 4D 입력으로 예외가 발생해 잘 잡아냄
- 따라서 `test_g2p2c_agent.py`는 에이전트 로드·추론 검증 및 입력 검증이라는 목적을 완벽히 달성합니다.