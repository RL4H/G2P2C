import subprocess
import time
from pathlib import Path
from typing import Any
from collections import namedtuple

import gym
from gym import spaces
import httpx


class DmmsEnv(gym.Env):
    """Gym wrapper for interacting with DMMS.R via FastAPI."""

    metadata = {"render.modes": []}

    Step = namedtuple("Step", ["observation", "reward", "done", "info"])
    Observation = namedtuple("Observation", ["CGM"])

    def __init__(self, server_url: str, exe_path: str, cfg_path: str, io_root: str):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.exe_path = str(exe_path)
        self.cfg_path = str(cfg_path)
        self.io_root = Path(io_root)
        self.io_root.mkdir(parents=True, exist_ok=True)
        self.episode_counter = 0
        self.proc: subprocess.Popen | None = None
        self._max_connect_attempts = 20
        self._connect_interval = 0.5

        # older versions of ``gym`` do not accept the ``dtype`` argument in
        # ``spaces.Box``.  Since the exact version used can vary, rely on the
        # default dtype to maximise compatibility.
        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(1,))
        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(12, 2)
        )

    def _start_process(self, results_dir: Path) -> None:
        log_file = results_dir / "dmms.log"
        cmd = [self.exe_path, self.cfg_path, str(log_file), str(results_dir)]
        self.proc = subprocess.Popen(cmd)

    def reset(self) -> Step:
        self.close()
        self.episode_counter += 1
        self.results_dir = self.io_root / f"episode_{self.episode_counter}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._start_process(self.results_dir)
        last_error: Exception | None = None
        for _ in range(self._max_connect_attempts):
            try:
                resp = httpx.get(f"{self.server_url}/get_state")
                resp.raise_for_status()
                break
            except Exception as e:
                last_error = e
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(
                        "DMMS.R process terminated before handshake"
                    ) from e
                time.sleep(self._connect_interval)
        else:
            raise RuntimeError(
                f"Failed to connect to {self.server_url}/get_state after {self._max_connect_attempts} attempts"
            ) from last_error

        data = resp.json()
        obs_val = data.get("cgm")
        info = data.get("info", {})
        default_info = {
            "sample_time": 1,
            "future_carb": 0,
            "remaining_time": 0,
            "day_hour": 0,
            "day_min": 0,
            "meal_type": 0,
            "meal": 0,
        }
        if isinstance(info, dict):
            default_info.update(info)
        obs = self.Observation(CGM=obs_val)
        return self.Step(observation=obs, reward=0.0, done=False, info=default_info)

    def step(self, action: Any) -> Step:
        payload = {"action": action}
        resp = httpx.post(f"{self.server_url}/env_step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs_val = data.get("cgm")
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        info = data.get("info", {})
        default_info = {
            "sample_time": 1,
            "future_carb": 0,
            "remaining_time": 0,
            "day_hour": 0,
            "day_min": 0,
            "meal_type": 0,
            "meal": 0,
        }
        if isinstance(info, dict):
            default_info.update(info)
        obs = self.Observation(CGM=obs_val)
        return self.Step(observation=obs, reward=reward, done=done, info=default_info)

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            self.proc = None
