#!/bin/bash

# FPL Nightly Update Script
# This script runs the data update process and handles errors

echo "🌙 FPL Nightly Update Started at $(date)"

# Set environment
export PYTHONPATH="/root/fpl-test:$PYTHONPATH"
cd /root/fpl-test

# Create log directory if it doesn't exist
mkdir -p logs

# Log file
LOG_FILE="logs/nightly_update_$(date +%Y%m%d_%H%M%S).log"

# Redirect all output to log file
exec > >(tee -i $LOG_FILE)
exec 2>&1

echo "📊 Starting data update process..."
echo "📁 Working directory: $(pwd)"
echo "📝 Log file: $LOG_FILE"

# Activate virtual environment
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Run the update
echo "🚀 Running updater.py..."
python updater.py

UPDATE_EXIT_CODE=$?

if [ $UPDATE_EXIT_CODE -eq 0 ]; then
    echo "✅ Update completed successfully!"
    echo "🔄 Restarting Streamlit service to load new data..."

    # Restart the Streamlit service to load new data
    sudo systemctl restart fpl-streamlit.service

    if [ $? -eq 0 ]; then
        echo "✅ Streamlit service restarted successfully"
    else
        echo "❌ Failed to restart Streamlit service"
    fi

    echo "🎉 Nightly update completed successfully at $(date)"

else
    echo "❌ Update failed with exit code: $UPDATE_EXIT_CODE"

    # Send notification (if mail is configured)
    if command -v mail &> /dev/null; then
        echo "FPL Update Failed - Exit Code: $UPDATE_EXIT_CODE at $(date)" | mail -s "FPL Update Alert" root@localhost
    fi

    echo "💥 Nightly update failed at $(date)"
    exit 1
fi

echo "📈 Cleaning up old log files..."
# Keep only last 7 days of logs
find logs -name "nightly_update_*.log" -type f -mtime +7 -delete

echo "🏁 Script completed at $(date)"
