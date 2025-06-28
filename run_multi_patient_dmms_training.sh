#!/bin/bash

# Multi-Patient DMMS Training Script for Adult#002 and Adult#008
# This script sequentially trains G2P2C agents on different patient configurations
# without modifying the original run_extended_dmms_training.py

set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Base configuration
EPISODES=30
CHECKPOINT=195
BASE_CONFIG_DIR="/mnt/c/Users/user/Desktop/G2P2C/config"  # WSL path for file checking
DMMS_CONFIG_DIR="C:\\Users\\user\\Desktop\\G2P2C\\config"  # Windows path for Python script
DMMS_EXE="C:\\Program Files\\The Epsilon Group\\DMMS.R\\simulator\\DMMS.R.exe"
PYTHON_SCRIPT="experiments/run_extended_dmms_training.py"

# Patient configurations
declare -a PATIENTS=("adult002" "adult008")
declare -a CONFIG_FILES=("RL_scenario_1_07_13_19_50_adult2.xml" "RL_scenario_1_07_13_19_50_adult8.xml")

# Results directory for aggregated outputs
RESULTS_BASE_DIR="results/multi_patient_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_BASE_DIR"

# Logging
LOG_FILE="$RESULTS_BASE_DIR/multi_patient_training.log"
echo "Multi-Patient DMMS Training Started: $(date)" | tee "$LOG_FILE"
echo "Patients: ${PATIENTS[*]}" | tee -a "$LOG_FILE"
echo "Episodes per patient: $EPISODES" | tee -a "$LOG_FILE"
echo "Starting checkpoint: episode_$CHECKPOINT" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

# =============================================================================
# Functions
# =============================================================================

check_config_file() {
    local config_file="$1"
    local wsl_path="$BASE_CONFIG_DIR/$config_file"
    
    if [ ! -f "$wsl_path" ]; then
        echo "ERROR: Configuration file not found: $wsl_path" | tee -a "$LOG_FILE"
        echo "INFO: Please ensure the file exists at the specified location" | tee -a "$LOG_FILE"
        return 1
    fi
    echo "INFO: Configuration file verified: $wsl_path" | tee -a "$LOG_FILE"
    return 0
}

train_patient() {
    local patient_name="$1"
    local config_file="$2"
    local patient_index="$3"
    
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Training Patient: $patient_name ($patient_index/2)" | tee -a "$LOG_FILE"
    echo "Config File: $config_file" | tee -a "$LOG_FILE"
    echo "Start Time: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # Check if config file exists (using WSL path)
    if ! check_config_file "$config_file"; then
        echo "SKIPPING: Patient $patient_name due to missing config file" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # Full Windows path for Python script
    local full_config_path="$DMMS_CONFIG_DIR\\$config_file"
    
    # Create patient-specific arguments
    local patient_args=(
        --extended_episodes "$EPISODES"
        --fine_tune_from_checkpoint "$CHECKPOINT"
        --dmms_cfg "$full_config_path"
    )
    
    # Training start time
    local start_time=$(date +%s)
    
    # Execute training
    echo "INFO: Starting training for $patient_name..." | tee -a "$LOG_FILE"
    echo "INFO: Command: python $PYTHON_SCRIPT ${patient_args[*]}" | tee -a "$LOG_FILE"
    
    if python "$PYTHON_SCRIPT" "${patient_args[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "SUCCESS: Patient $patient_name training completed in ${duration}s" | tee -a "$LOG_FILE"
        
        # Copy results to aggregated directory
        copy_patient_results "$patient_name"
        
        return 0
    else
        echo "ERROR: Patient $patient_name training failed" | tee -a "$LOG_FILE"
        return 1
    fi
}

copy_patient_results() {
    local patient_name="$1"
    
    # Find the most recent extended_dmms folder
    local latest_result=$(find results -maxdepth 1 -type d -name "extended_dmms_*" | sort -r | head -n 1)
    
    if [ -n "$latest_result" ] && [ -d "$latest_result" ]; then
        local patient_result_dir="$RESULTS_BASE_DIR/$patient_name"
        mkdir -p "$patient_result_dir"
        
        echo "INFO: Copying results from $latest_result to $patient_result_dir" | tee -a "$LOG_FILE"
        cp -r "$latest_result"/* "$patient_result_dir/" 2>/dev/null || true
        
        # Create summary for this patient
        create_patient_summary "$patient_name" "$patient_result_dir"
    else
        echo "WARNING: Could not find results directory for $patient_name" | tee -a "$LOG_FILE"
    fi
}

create_patient_summary() {
    local patient_name="$1"
    local result_dir="$2"
    
    local summary_file="$result_dir/patient_training_summary.txt"
    
    cat > "$summary_file" << EOF
Patient Training Summary
========================
Patient: $patient_name
Training Date: $(date)
Episodes: $EPISODES
Starting Checkpoint: episode_$CHECKPOINT

Configuration:
- Config File: ${CONFIG_FILES[$((${#PATIENTS[@]} - 2))]}
- DMMS Exe: $DMMS_EXE

Results Directory: $result_dir

Training Logs:
See multi_patient_training.log in parent directory

Evaluation:
Run evaluation scripts on the checkpoints in $result_dir/checkpoints/
EOF

    echo "INFO: Patient summary created: $summary_file" | tee -a "$LOG_FILE"
}

generate_final_report() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Multi-Patient Training Completed" | tee -a "$LOG_FILE"
    echo "End Time: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # Create final summary report
    local final_report="$RESULTS_BASE_DIR/FINAL_REPORT.md"
    
    cat > "$final_report" << EOF
# Multi-Patient DMMS Training Report

## Summary
- **Training Date**: $(date)
- **Patients Trained**: ${PATIENTS[*]}
- **Episodes per Patient**: $EPISODES
- **Starting Checkpoint**: episode_$CHECKPOINT

## Patient Results

EOF

    # Add patient-specific sections
    for i in "${!PATIENTS[@]}"; do
        local patient="${PATIENTS[$i]}"
        local config="${CONFIG_FILES[$i]}"
        
        cat >> "$final_report" << EOF
### Patient: $patient
- **Config File**: $config
- **Results Directory**: \`$RESULTS_BASE_DIR/$patient\`
- **Checkpoints**: \`$RESULTS_BASE_DIR/$patient/checkpoints/\`
- **Training Logs**: \`$RESULTS_BASE_DIR/$patient/training/\`

EOF
    done
    
    cat >> "$final_report" << EOF

## Next Steps

1. **Evaluation**: Run performance evaluation on each patient's final checkpoint
2. **Comparison**: Compare TIR, hypoglycemia, and hyperglycemia metrics across patients
3. **Analysis**: Analyze learning curves and convergence patterns

## Commands for Evaluation

\`\`\`bash
# Evaluate patient adult#002
python experiments/evaluate_dmms_performance.py --folder_id $RESULTS_BASE_DIR/adult002

# Evaluate patient adult#008  
python experiments/evaluate_dmms_performance.py --folder_id $RESULTS_BASE_DIR/adult008
\`\`\`

## Files Generated
- Training logs: \`$RESULTS_BASE_DIR/multi_patient_training.log\`
- Patient results: \`$RESULTS_BASE_DIR/adult002/\` and \`$RESULTS_BASE_DIR/adult008/\`
- This report: \`$RESULTS_BASE_DIR/FINAL_REPORT.md\`
EOF

    echo "INFO: Final report generated: $final_report" | tee -a "$LOG_FILE"
    echo "INFO: All results saved to: $RESULTS_BASE_DIR" | tee -a "$LOG_FILE"
}

# =============================================================================
# Main Execution
# =============================================================================

echo "INFO: Starting multi-patient training sequence..." | tee -a "$LOG_FILE"

# Check if required directories and files exist
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT" | tee -a "$LOG_FILE"
    exit 1
fi

# Track successful and failed training sessions
declare -a successful_patients=()
declare -a failed_patients=()

# Train each patient sequentially
for i in "${!PATIENTS[@]}"; do
    patient="${PATIENTS[$i]}"
    config="${CONFIG_FILES[$i]}"
    
    if train_patient "$patient" "$config" "$((i+1))"; then
        successful_patients+=("$patient")
    else
        failed_patients+=("$patient")
    fi
    
    # Add delay between patients to ensure proper cleanup
    if [ $((i+1)) -lt ${#PATIENTS[@]} ]; then
        echo "INFO: Waiting 10 seconds before next patient..." | tee -a "$LOG_FILE"
        sleep 10
    fi
done

# Generate final report
generate_final_report

# Print final status
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "TRAINING SESSION COMPLETE" | tee -a "$LOG_FILE"
echo "Successful: ${successful_patients[*]:-None}" | tee -a "$LOG_FILE"
echo "Failed: ${failed_patients[*]:-None}" | tee -a "$LOG_FILE"
echo "Results Directory: $RESULTS_BASE_DIR" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Exit with appropriate code
if [ ${#failed_patients[@]} -eq 0 ]; then
    echo "SUCCESS: All patients trained successfully!" | tee -a "$LOG_FILE"
    exit 0
else
    echo "WARNING: Some patients failed to train. Check logs for details." | tee -a "$LOG_FILE"
    exit 1
fi