#!/bin/bash

# run_extended_dmms_training.sh
# Complete workflow for extended DMMS training and evaluation

set -e  # Exit on any error

# Default parameters
EPISODES=100
CHECKPOINT=195
PYTHON_CMD="python"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "Command $1 not found. Please install it first."
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Extended DMMS Training and Evaluation Workflow"
    echo ""
    echo "Options:"
    echo "  -e, --episodes NUM      Number of episodes for training (default: 100)"
    echo "  -c, --checkpoint NUM    Starting checkpoint episode (default: 195)"
    echo "  -p, --python CMD        Python command to use (default: python)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --episodes 150 --checkpoint 195"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--episodes)
            EPISODES="$2"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ] || [ ! -d "agents" ] || [ ! -d "experiments" ]; then
    print_error "Please run this script from the G2P2C project root directory."
    exit 1
fi

# Check required commands
check_command $PYTHON_CMD

print_info "=== Extended DMMS Training Workflow ==="
print_info "Episodes to train: $EPISODES"
print_info "Starting checkpoint: episode_$CHECKPOINT"
print_info "Python command: $PYTHON_CMD"
echo ""

# Verify checkpoint exists
CHECKPOINT_PATH="results/test/checkpoints/episode_${CHECKPOINT}_Actor.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    print_error "Checkpoint not found: $CHECKPOINT_PATH"
    print_info "Available checkpoints:"
    ls -la results/test/checkpoints/episode_*_Actor.pth 2>/dev/null || print_warning "No checkpoints found"
    exit 1
fi

print_success "Checkpoint verified: $CHECKPOINT_PATH"

# Step 1: Check DMMS.R environment
print_info "Step 1: Verifying DMMS.R environment..."

if [ ! -f "Sim_CLI/RL_Agent_Plugin_v1.0.js" ]; then
    print_error "DMMS.R plugin not found: Sim_CLI/RL_Agent_Plugin_v1.0.js"
    exit 1
fi

print_success "DMMS.R plugin found"

# Step 2: Run extended training
print_info "Step 2: Starting extended training..."
print_warning "This will take significant time depending on episode count."
print_info "Training will automatically save checkpoints every 10% of episodes."

TRAINING_START=$(date)
print_info "Training started at: $TRAINING_START"

$PYTHON_CMD experiments/run_extended_dmms_training.py \
    --extended_episodes $EPISODES \
    --fine_tune_from_checkpoint $CHECKPOINT \
    --dmms_debug_mode

TRAINING_STATUS=$?

if [ $TRAINING_STATUS -eq 0 ]; then
    print_success "Extended training completed successfully!"
else
    print_error "Extended training failed with exit code: $TRAINING_STATUS"
    exit $TRAINING_STATUS
fi

# Step 3: Find the latest training results
print_info "Step 3: Locating training results..."

LATEST_RESULT=$(ls -td results/extended_dmms_*ep_* 2>/dev/null | head -n1)

if [ -z "$LATEST_RESULT" ]; then
    print_error "Could not find training results directory"
    exit 1
fi

print_success "Training results found: $LATEST_RESULT"

# Step 4: Generate performance evaluation
print_info "Step 4: Generating performance evaluation..."

$PYTHON_CMD experiments/evaluate_dmms_performance.py \
    --experiment_dir "$LATEST_RESULT"

EVALUATION_STATUS=$?

if [ $EVALUATION_STATUS -eq 0 ]; then
    print_success "Performance evaluation completed!"
else
    print_warning "Performance evaluation completed with warnings (exit code: $EVALUATION_STATUS)"
fi

# Step 5: Display summary
print_info "Step 5: Training Summary"
echo ""
print_success "=== EXTENDED TRAINING WORKFLOW COMPLETE ==="
echo ""
print_info "Training Details:"
echo "  • Episodes trained: $EPISODES"
echo "  • Starting checkpoint: episode_$CHECKPOINT"
echo "  • Training started: $TRAINING_START"
echo "  • Training finished: $(date)"
echo ""
print_info "Results Location: $LATEST_RESULT"
echo ""
print_info "Generated Files:"
echo "  • Training logs: $LATEST_RESULT/model_log.csv"
echo "  • Evaluation report: $LATEST_RESULT/evaluation/performance_report.json"
echo "  • Training plots: $LATEST_RESULT/evaluation/training_progress.png"
echo "  • Checkpoints: $LATEST_RESULT/checkpoints/"
echo ""

# Check if training improved
if [ -f "$LATEST_RESULT/evaluation/performance_report.json" ]; then
    print_info "Performance Improvements:"
    $PYTHON_CMD -c "
import json
try:
    with open('$LATEST_RESULT/evaluation/performance_report.json', 'r') as f:
        report = json.load(f)
    if 'learning_progress' in report and 'improvement' in report['learning_progress']:
        improvements = report['learning_progress']['improvement']
        for metric, change in improvements.items():
            direction = 'improved' if change > 0 else 'declined'
            print(f'  • {metric}: {change:+.2f}% ({direction})')
    else:
        print('  • Performance metrics not available in report')
except Exception as e:
    print(f'  • Could not parse performance report: {e}')
"
fi

echo ""
print_info "Next Steps:"
echo "  1. Review training plots in: $LATEST_RESULT/evaluation/"
echo "  2. Check final checkpoint: $LATEST_RESULT/checkpoints/"
echo "  3. Run comparison evaluation if needed:"
echo "     $PYTHON_CMD experiments/evaluate_dmms_performance.py \\"
echo "       --experiment_dir \"$LATEST_RESULT\" \\"
echo "       --run_evaluation"
echo ""
print_success "Extended DMMS training workflow completed successfully!"