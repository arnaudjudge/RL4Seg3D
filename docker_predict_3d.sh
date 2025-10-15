#!/usr/bin/env bash
# Wrapper for RL4Seg3D container execution
# Usage: ./docker_predict_3d.sh [input_path=<HOST_PATH>] [output_path=<HOST_PATH>] [other args...]

set -e

IMAGE="arnaudjudge/rl4seg3d:latest"

# Default host paths

# Parse arguments
INPUT_PATH=""
OUTPUT_PATH=""
OTHER_ARGS=()

for arg in "$@"; do
    case $arg in
        input_path=*)
            INPUT_PATH="${arg#*=}"
            ;;
        output_path=*)
            OUTPUT_PATH="${arg#*=}"
            ;;
        *)
            OTHER_ARGS+=("$arg")  # keep all other args
            ;;
    esac
done

# Prepend /MOUNT/ to input/output paths
CONTAINER_INPUT="/MOUNT/$INPUT_PATH"
CONTAINER_OUTPUT="/MOUNT/$OUTPUT_PATH"

docker run -it --rm \
  --ipc host \
  --gpus all \
  -v "$(pwd)":/MOUNT \
  --user "$(id -u):$(id -g)" \
  "${IMAGE}" predict_3d \
      input_path="$CONTAINER_INPUT" \
      output_path="$CONTAINER_OUTPUT" \
      "${OTHER_ARGS[@]}"
