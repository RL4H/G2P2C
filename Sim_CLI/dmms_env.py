import subprocess
from pathlib import Path
from typing import Any, Tuple

import gym
from gym import spaces
import httpx


class DmmsEnv(gym.Env):
    """Gym wrapper for interacting with DMMS.R via FastAPI."""

    metadata = {"render.modes": []}

    def __init__(self, server_url: str, exe_path: str, cfg_path: str, io_root: str):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.exe_path = str(exe_path)
        self.cfg_path = str(cfg_path)
        self.io_root = Path(io_root)
        self.io_root.mkdir(parents=True, exist_ok=True)
        self.episode_counter = 0
        self.proc: subprocess.Popen | None = None

        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(12, 2), dtype=float)

    def _start_process(self, results_dir: Path) -> None:
        log_file = results_dir / "dmms.log"
        cmd = [self.exe_path, self.cfg_path, str(log_file), str(results_dir)]
        self.proc = subprocess.Popen(cmd)

    def reset(self) -> Any:
        self.close()
        self.episode_counter += 1
        self.results_dir = self.io_root / f"episode_{self.episode_counter}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._start_process(self.results_dir)
        resp = httpx.get(f"{self.server_url}/get_state")
        resp.raise_for_status()
        return resp.json()["obs"]

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        payload = {"action": action}
        resp = httpx.post(f"{self.server_url}/env_step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("obs")
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        info = data.get("info", {})
        return obs, reward, done, info

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
