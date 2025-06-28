#!/bin/bash

# DMMS.R 미세조정 파이프라인 검증 스크립트
# 고정된 1일 시나리오에서 학습 파이프라인 테스트

echo "=== DMMS.R G2P2C Fine-tuning Pipeline Verification ==="
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# 필수 파일 존재 확인
echo "Checking prerequisites..."

CHECKPOINT_DIR="/mnt/c/Users/user/Desktop/G2P2C/results/test/checkpoints"
DMMS_EXE="C:\\Program Files\\The Epsilon Group\\DMMS.R\\simulator\\DMMS.R.exe"
DMMS_CFG="C:\\Users\\user\\Documents\\DMMS.R\\config\\Sim_CLI_1.2.xml"

if [ ! -f "${CHECKPOINT_DIR}/episode_195_Actor.pth" ]; then
    echo "ERROR: Actor checkpoint not found: ${CHECKPOINT_DIR}/episode_195_Actor.pth"
    exit 1
fi

if [ ! -f "${CHECKPOINT_DIR}/episode_195_Critic.pth" ]; then
    echo "ERROR: Critic checkpoint not found: ${CHECKPOINT_DIR}/episode_195_Critic.pth"
    exit 1
fi

echo "✓ Checkpoints found"

# DMMS.R 파일 존재 확인 (Windows 경로이므로 실제 확인은 Python에서)
echo "✓ DMMS.R paths configured"

echo ""
echo "Starting DMMS.R fine-tuning with debug configuration..."
echo "Configuration:"
echo "  - Agent: G2P2C"
echo "  - Mode: DMMS.R debug (single scenario optimization)"
echo "  - Checkpoint: episode_195"
echo "  - Workers: 1"
echo "  - Learning rates: 5e-5"
echo "  - Episode length: 288 steps (1 day)"
echo "  - Rollout length: 64 steps"
echo ""

cd experiments

python run_RL_agent_finetune.py \
  --agent g2p2c \
  --folder_id dmms_pipeline_verification \
  --sim dmms \
  --dmms_exe "C:\\Program Files\\The Epsilon Group\\DMMS.R\\simulator\\DMMS.R.exe" \
  --dmms_cfg "C:\\Users\\user\\Documents\\DMMS.R\\config\\Sim_CLI_1.2.xml" \
  --fine_tune_from_checkpoint 195 \
  --device cpu \
  --debug 1 \
  --dmms_debug_mode \
  --return_type average \
  --action_type exponential \
  --seed 42

echo ""
echo "=== Verification Complete ==="
echo "Check results in: /mnt/c/Users/user/Desktop/G2P2C/results/dmms_pipeline_verification/"
echo "Key files to monitor:"
echo "  - debug.log: Real-time training progress"
echo "  - model_log.csv: Learning metrics (avg_rew, val_loss, exp_var)"
echo "  - dmms_finetune_logs/: DMMS.R interaction logs"