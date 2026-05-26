#!/usr/bin/env bash
# Train MeloTTS with distributed data parallel (DDP) via torchrun.
#
# Usage:
#   bash melo/scripts/train.sh <config_path> <num_gpus>
#
# Example:
#   bash melo/scripts/train.sh configs/config.json 4

set -euo pipefail

CONFIG=$1
GPUS=$2
MODEL_NAME=$(basename "$(dirname "$CONFIG")")

PORT=10902

# Auto-resume loop: the training process may occasionally crash due to a gloo
# bug on some GPU configurations.  This loop restarts it automatically.
while :
do
    torchrun \
        --nproc_per_node="$GPUS" \
        --master_port="$PORT" \
        -m melo.training.train --c "$CONFIG" --model "$MODEL_NAME"

    # Kill any lingering Python processes related to this config.
    for PID in $(ps -aux | grep "$CONFIG" | grep python | awk '{print $2}')
    do
        echo "Killing PID $PID"
        kill -9 "$PID"
    done
    sleep 30
done