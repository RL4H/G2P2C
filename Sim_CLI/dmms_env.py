"""Gym 스타일 DMMS.R 환경 래퍼.

DmmsEnv 클래스는 DMMS.R 시뮬레이터를 서브프로세스로 실행하고
FastAPI 서버와 HTTP 통신하여 관측값과 보상을 주고 받는다.
문서 `docs/RL_Workflow.md` 에 제시된 예시 코드를 참고해 구현한다.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from environments.simglucose.simglucose.simulation.env import Step, Observation


class DmmsEnv:
    """DMMS.R 시뮬레이터와 통신하는 간단한 Gym 호환 환경."""

    def __init__(
        self,
        exe: Path,
        cfg: Path,
        server_url: str = "http://127.0.0.1:5000",
        log_file: str = "log_single.txt",
        io_root: Optional[Path] = None,
    ) -> None:
        self.exe = Path(exe)
        self.cfg = Path(cfg)
        self.server_url = server_url.rstrip("/")
        self.log_file = log_file
        self.io_root = Path(io_root) if io_root else Path("results/dmms_runs")

        self.process: Optional[subprocess.Popen] = None
        self.episode_counter = 0
        self.results_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    def _start_process(self, results_dir: Path) -> None:
        """DMMS.R 프로세스를 시작한다."""
        cmd = [str(self.exe), str(self.cfg), self.log_file, str(results_dir)]
        self.process = subprocess.Popen(cmd)

    # ------------------------------------------------------------------
    def reset(self) -> Step:
        """환경을 초기화하고 첫 관측값을 반환한다."""
        self.close()
        self.episode_counter += 1
        self.results_dir = self.io_root / f"episode_{self.episode_counter}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._start_process(self.results_dir)

        obs_val = None
        # 초기 상태가 준비될 때까지 잠시 대기
        for _ in range(60):
            try:
                resp = httpx.get(f"{self.server_url}/get_state", timeout=1.0)
                if resp.status_code == 200:
                    data = resp.json()
                    obs_val = data.get("cgm")
                    if obs_val is not None:
                        break
            except Exception:
                pass
            time.sleep(1)
        if obs_val is None:
            obs_val = 0.0

        obs = Observation(CGM=obs_val)
        default_info = {
            "sample_time": 5,
            "meal": 0,
            "remaining_time": 0,
            "meal_type": 0,
            "future_carb": 0,
            "day_hour": 0,
            "day_min": 0,
        }
        return Step(observation=obs, reward=0.0, done=False, info=default_info)

    # ------------------------------------------------------------------
    def step(self, action: Any) -> Step:
        """액션을 서버로 전송하고 다음 상태를 받아온다."""
        payload = {"action": action}
        resp = httpx.post(f"{self.server_url}/env_step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs_val = data.get("cgm")
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        info = data.get("info", {}) or {}
        default_info = {
            "sample_time": 5,
            "meal": 0,
            "remaining_time": 0,
            "meal_type": 0,
            "future_carb": 0,
            "day_hour": 0,
            "day_min": 0,
        }
        default_info.update(info)
        obs = Observation(CGM=obs_val)
        return Step(observation=obs, reward=reward, done=done, info=default_info)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """실행 중인 DMMS.R 프로세스를 종료한다."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    # ------------------------------------------------------------------
    def __del__(self) -> None:
        self.close()

