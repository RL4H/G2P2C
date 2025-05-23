# G2P2C 강화학습 작동 방식 정리

이 문서는 `experiments/run_RL_agent.py`로 시작하는 학습 루프가 내부적으로 어떤 모듈을 호출하여 강화학습을 수행하는지 코드 관점에서 설명한다.

## 1. 실험 초기화 단계

`run_RL_agent.py`의 `main()` 함수에서는 `Options` 클래스로 사용자 인자를 파싱한 뒤, `set_agent_parameters()`를 통해 선택한 알고리즘을 초기화한다. 이후 환경 목록을 불러오고 에이전트의 `run()` 메서드를 실행한다.

```python
171  def main():
172      args = Options().parse()
173      args, agent, device = set_agent_parameters(args)
...
188      if args.sim == 'dmms':
189          setup_dmms_dirs(args)
...
191      patients, env_ids = get_patient_env()
192      agent.run(args, patients, env_ids, args.seed)
```
위와 같이 실험 디렉터리 생성과 시드 고정 이후 `agent.run()`이 실제 학습을 시작한다.【F:experiments/run_RL_agent.py†L171-L192】

## 2. 환경 구성

에이전트는 `utils.core.get_env()`를 통해 환경 객체를 생성한다. `--sim dmms` 옵션이 지정되면 `Sim_CLI/dmms_env.DmmsEnv` 클래스를 사용하여 외부 시뮬레이터 DMMS.R과 통신한다. 주요 로직은 다음과 같다.

```python
42      def reset(self) -> Step:
43          self.close()
44          self.episode_counter += 1
45          self.results_dir = self.io_root / f"episode_{self.episode_counter}"
46          self.results_dir.mkdir(parents=True, exist_ok=True)
47          self._start_process(self.results_dir)
48          resp = httpx.get(f"{self.server_url}/get_state")
...
65          return self.Step(observation=obs, reward=0.0, done=False, info=default_info)

67      def step(self, action: Any) -> Step:
68          payload = {"action": action}
69          resp = httpx.post(f"{self.server_url}/env_step", json=payload)
70          resp.raise_for_status()
71          data = resp.json()
72          obs_val = data.get("cgm")
73          reward = data.get("reward", 0.0)
74          done = data.get("done", False)
...
87          return self.Step(observation=obs, reward=reward, done=done, info=default_info)
```
DMMS.R 실행 파일을 `subprocess.Popen`으로 호출하고 FastAPI 서버(`/env_step`)와 HTTP 통신하여 매 스텝 결과를 받아오는 구조이다.【F:Sim_CLI/dmms_env.py†L42-L88】

## 3. 워커(Worker)와 롤아웃 수집

`agents/g2p2c/worker.py`의 `Worker` 클래스는 환경과 상호작용하며 데이터를 수집한다. `rollout()` 메서드는 한 에피소드의 경험을 `Memory` 버퍼에 저장하고, 필요한 통계 값을 계산한다.

```python
12  class Worker:
...
43      def init_env(self):
47          self.init_state = self.env.reset()
48          self.cur_state, self.feat = self.state_space.update(cgm=self.init_state.CGM, ins=0, meal=0)
...
66      def rollout(self, policy):
75          for n_steps in range(0, rollout_steps):
76              policy_step = policy.get_action(self.cur_state, self.feat)
78              rl_action, pump_action = self.pump.action(agent_action=selected_action,
79                                                        prev_state=self.init_state, prev_info=None)
80              state, _reward, is_done, info = self.env.step(pump_action)
81              reward = composite_reward(self.args, state=state.CGM, reward=_reward)
90              if self.worker_mode == 'training':
92                  self.memory.store(self.cur_state, self.feat, policy_step['action'][0],
93                                    reward, policy_step['state_value'], policy_step['log_prob'], scaled_cgm, self.counter)
95              self.cur_state, self.feat = self.state_space.update(cgm=state.CGM, ins=pump_action,
96                                                                  meal=info['remaining_time'], hour=(self.counter+1),
97                                                                  meal_type=info['meal_type'], carbs=info['future_carb'])
105             criteria = state.CGM <= 40 or state.CGM >= 600 or self.counter > stop_factor
106             if criteria:
108                 final_val = policy.get_final_value(self.cur_state, self.feat)
109                 self.memory.finish_path(final_val)
```
환경 스텝을 진행하며 얻은 `(state, action, reward, value, logprob)` 등을 메모리에 저장하고, 에피소드 종료 시 `finish_path()`로 마지막 상태 값을 기록한다.【F:agents/g2p2c/worker.py†L12-L109】

## 4. 에이전트(G2P2C) 업데이트

`G2P2C` 클래스의 `run()` 메서드는 여러 `Worker`가 수집한 데이터를 사용해 정책과 가치 함수를 학습한다. 학습 루프의 핵심은 `update()` 호출이다.

```python
392      def run(self, args, patients, env_ids, seed):
406          worker_agents = [Worker(args, 'training', patients, env_ids, i+5, i, self.device) for i in range(self.n_training_workers)]
409          # ppo learning
410          for rollout in range(0, 30000):
414              for i in range(self.n_training_workers):
415                  data, actor_bgp_rmse, a_horizonBG_rmse = worker_agents[i].rollout(self.policy)
416                  self.old_states[i] = data['obs']
...
432              self.update(rollout)
433              self.policy.save(rollout)
```
각 워커가 저장한 버퍼를 합쳐 `update()`를 수행하고, 주기적으로 모델을 저장한다.【F:agents/g2p2c/g2p2c.py†L392-L433】

`update()` 내부에서는 일반적인 PPO 방식으로 어드밴티지 계산 후 정책(`train_pi`)과 가치망(`train_vf`)을 최적화한다. 추가로 보조 학습(`train_aux`)과 계획 단계(`train_MCTS_planning`)가 조건부로 실행된다.

```python
356      def update(self, rollout):
358          if self.return_type == 'discount':
359              if self.normalize_reward:
360                  self.reward = self.reward_normaliser(self.reward, self.first_flag)
361              self.adv, self.v_targ = self.compute_gae()
...
366          self.prepare_rollout_buffer()
368          self.model_logs[0], self.model_logs[5] = self.train_pi()
369          self.model_logs[1], self.model_logs[2], self.model_logs[3], self.model_logs[4]  = self.train_vf()
371          if self.aux_mode != 'off' and self.AuxiliaryBuffer.buffer_filled:
372              if (rollout + 1) % self.aux_frequency == 0:
373                  self.aux_model_logs[0], self.aux_model_logs[1], self.aux_model_logs[2], self.aux_model_logs[3] = self.train_aux()
375          if self.use_planning and self.start_planning:
377              self.planning_model_logs[0], self.planning_model_logs[1] = self.train_MCTS_planning()
```
이 부분에서 어드밴티지(`adv`)와 목표 가치(`v_targ`)를 계산하고, 정책·가치 네트워크 및 부가 모듈들을 학습한다.【F:agents/g2p2c/g2p2c.py†L356-L377】

## 5. FastAPI 서버와의 연동

`Sim_CLI/main.py`는 FastAPI 서버로, JavaScript 플러그인이 이 서버에 상태를 보내면 `infer_action()`으로부터 다음 행동을 받아 DMMS.R에 전달한다. 서버는 매 스텝과 에피소드 종료 시 경험을 저장하여 학습에 활용할 수 있다.

---
이와 같이 `run_RL_agent.py` → `G2P2C.run()` → `Worker.rollout()` → `DmmsEnv.step/reset()`의 흐름으로 데이터를 수집하고, `G2P2C.update()`를 통해 파라미터가 갱신되는 구조로 강화학습이 이루어진다.
