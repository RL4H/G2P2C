# Guidelines for Contributors

This repository contains code for training and evaluating RL-based insulin dosing agents.
The original environment is **Simglucose**, but the `Sim_CLI` package bridges a trained
agent with the external simulator **DMMS.R** via a FastAPI server and a JavaScript plugin.
The notes below summarise key conventions and design decisions that have emerged in
previous discussions.

## Folder overview

- `agents/` – RL algorithms including the G2P2C implementation.
- `Sim_CLI/` – utilities to connect a trained agent to DMMS.R. Important files are:
  - `main.py`: FastAPI server that loads the agent and exposes API endpoints.
  - `g2p2c_agent_api.py`: helper to load the G2P2C model and run inference.
  - `RL_Agent_Plugin_v1.0.js`: DMMS.R plugin that queries the server for insulin
    actions.
  - `run_dmms_cli.py`: example script showing how to start DMMS.R from Python.
- `results/` – log files written by the server during interaction with DMMS.R.

## Development notes

## RL training with DMMS.R

The current code base primarily performs inference. To train with DMMS.R
consider the following workflow:

1. Modify the JS plugin and the FastAPI server as described above to collect
   step data and handle episode boundaries.
2. Implement a Gym-style environment class that wraps the API communication.
3. Reuse the existing PPO/G2P2C training loops (`experiments/run_RL_agent.py`)
   by replacing the environment with the new DMMS.R wrapper.

These guidelines should help maintainers and contributors keep the code base
consistent while extending the project to support external simulators.
