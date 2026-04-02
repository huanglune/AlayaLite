#!/bin/bash

# Run a command and log its output to a file.
# Usage: ./run_and_log.sh <description> <command...>
# Example: ./run_and_log.sh build_backend make build
# Default log timezone: America/Los_Angeles
# Override with: RUN_AND_LOG_TZ=<IANA timezone>

if [ "$#" -lt 2 ]; then
    echo "Error: insufficient arguments."
    echo "Usage: ./run_and_log.sh <description> <command...>"
    echo "Example: ./run_and_log.sh build_backend make build"
    exit 1
fi

# Sanitize description for use in filename
RAW_DESC=$1
SAFE_DESC=$(echo "$RAW_DESC" | tr -c 'a-zA-Z0-9\-_' '_')
SAFE_DESC=${SAFE_DESC%_}

shift
COMMAND=("$@")
COMMAND_DISPLAY=$(printf '%q ' "${COMMAND[@]}")
COMMAND_DISPLAY=${COMMAND_DISPLAY% }

# Default to Los Angeles time for log filenames and command execution.
LOG_TZ=${RUN_AND_LOG_TZ:-America/Los_Angeles}
export TZ="$LOG_TZ"
LOG_DATE=$(date +"%Y-%m-%d")
TIMESTAMP=$(date +"%y%m%d-%H%M")
DISPLAY_TIME=$(date +"%Y-%m-%d %H:%M:%S %Z %z")

LOG_DIR="logs/${LOG_DATE}"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/${TIMESTAMP}-${SAFE_DESC}.log"

echo "==================================================" | tee -a "$LOG_FILE"
echo "[TIME]  Start: $DISPLAY_TIME" | tee -a "$LOG_FILE"
echo "[START] Command: $COMMAND_DISPLAY" | tee -a "$LOG_FILE"
echo "[LOG]   File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# Run command, merging stdout and stderr, tee to both screen and log
"${COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"

# Capture the exit code of the actual command (not tee)
EXIT_CODE=${PIPESTATUS[0]}

echo "--------------------------------------------------" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Command succeeded (Exit Code: 0)" | tee -a "$LOG_FILE"
else
    echo "[ERROR] Command failed (Exit Code: $EXIT_CODE)" | tee -a "$LOG_FILE"
fi
echo "==================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
