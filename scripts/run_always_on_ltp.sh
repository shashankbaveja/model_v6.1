#!/bin/bash

# Script to run always_on_ltp.py with proper environment setup
# This script should be run from the project root directory

# Set the project directory (adjust this path to your actual project location)
PROJECT_DIR="/Users/shashankbaveja/Main/Projects/KiteConnectAPI/Model_v5"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate conda environment (KiteConnect)
source /opt/anaconda3/bin/activate && conda activate /opt/anaconda3/envs/KiteConnect

# Create logs directory if it doesn't exist
mkdir -p always_on_logs

# Run the always_on_ltp.py script with logging
python src/always_on_ltp.py >> always_on_logs/always_on_ltp.log 2>&1

echo "Always-on LTP monitoring completed at $(date)" >> always_on_logs/always_on_ltp.log 
