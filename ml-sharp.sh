#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/home/user/AI/ml-sharp"
ENV_NAME="sharp"

# Default command parts (editable via CLI)
INPUT_DIR="./input"
OUTPUT_DIR="./output"
CHECKPOINT="sharp_2572gikvuh.pt"
SBS_IMAGE=true
FAST_PREVIEW_RENDER=true
BATCH_SIZE="3"
STEREO_STRENGTH="0.15"
SBS_MIN_OPACITY="0.005"
FOCAL_LENGTH="20"

# New stereo flags (safe defaults preserve prior behavior)
STEREO_MODE="parallel"                 # toe_in (backward-compatible behavior) or parallel
STEREO_CONVERGENCE_DEPTH=""          # only used when STEREO_MODE=parallel; empty => not passed
STEREO_CONVERGENCE_NORM="1.0"           # only used when STEREO_MODE=parallel; empty => not passed

# --- Helpers ---
deactivate_all_conda() {
  # Only possible if conda is initialized in this shell
  if ! command -v conda >/dev/null 2>&1; then
    return 0
  fi

  # If CONDA_SHLVL is set, deactivate repeatedly
  local shlvl="${CONDA_SHLVL:-0}"
  if [[ "$shlvl" =~ ^[0-9]+$ ]] && (( shlvl > 0 )); then
    while (( shlvl > 0 )); do
      # shellcheck disable=SC1091
      conda deactivate || true
      shlvl=$((shlvl - 1))
    done
  fi
}

ensure_conda_shell() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' not found on PATH."
    echo "Tip: run 'source <miniconda>/etc/profile.d/conda.sh' or initialize conda for your shell."
    exit 1
  fi

  # Ensure 'conda activate' works in non-interactive shells
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

build_command() {
  local cmd="sharp predict -i \"$INPUT_DIR\" -o \"$OUTPUT_DIR\" -c \"$CHECKPOINT\""

  if [[ "$SBS_IMAGE" == "true" ]]; then
    cmd+=" --sbs-image"
  fi
  if [[ "$FAST_PREVIEW_RENDER" == "true" ]]; then
    cmd+=" --fast-preview-render"
  fi

  cmd+=" --batch-size $BATCH_SIZE"
  cmd+=" --stereo-strength $STEREO_STRENGTH"
  cmd+=" --sbs-min-opacity $SBS_MIN_OPACITY"
  cmd+=" --focal-length $FOCAL_LENGTH"

  # New flags (safe defaults)
  cmd+=" --stereo-mode $STEREO_MODE"
  if [[ "${STEREO_MODE,,}" == "parallel" ]]; then
    # Depth takes precedence over norm if both are set
    if [[ -n "$STEREO_CONVERGENCE_DEPTH" ]]; then
      cmd+=" --stereo-convergence-depth $STEREO_CONVERGENCE_DEPTH"
    elif [[ -n "$STEREO_CONVERGENCE_NORM" ]]; then
      cmd+=" --stereo-convergence-norm $STEREO_CONVERGENCE_NORM"
    fi
  fi

  echo "$cmd"
}

print_current_settings() {
  echo
  echo "Current settings:"
  echo "  Project dir               : $PROJECT_DIR"
  echo "  Conda env                 : $ENV_NAME"
  echo "  Input dir (-i)            : $INPUT_DIR"
  echo "  Output dir (-o)           : $OUTPUT_DIR"
  echo "  Checkpoint (-c)           : $CHECKPOINT"
  echo "  --sbs-image               : $SBS_IMAGE"
  echo "  --fast-preview-render     : $FAST_PREVIEW_RENDER"
  echo "  --batch-size              : $BATCH_SIZE"
  echo "  --stereo-strength         : $STEREO_STRENGTH"
  echo "  --sbs-min-opacity         : $SBS_MIN_OPACITY"
  echo "  --focal-length            : $FOCAL_LENGTH"
  echo "  --stereo-mode             : $STEREO_MODE"
  echo "  --stereo-convergence-depth: ${STEREO_CONVERGENCE_DEPTH:-<unset>}"
  echo "  --stereo-convergence-norm : ${STEREO_CONVERGENCE_NORM:-<unset>}"
  echo
  echo "Command to run:"
  echo "  $(build_command)"
  echo
}

prompt_nonempty() {
  local label="$1"
  local val=""
  while true; do
    read -r -p "$label" val
    if [[ -n "$val" ]]; then
      echo "$val"
      return 0
    fi
    echo "Value cannot be empty."
  done
}

prompt_number() {
  local label="$1"
  local val=""
  while true; do
    read -r -p "$label" val
    # Accept ints or floats (basic)
    if [[ "$val" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
      echo "$val"
      return 0
    fi
    echo "Please enter a number."
  done
}

prompt_bool() {
  local label="$1"
  local val=""
  while true; do
    read -r -p "$label (y/n): " val
    case "${val,,}" in
      y|yes) echo "true"; return 0 ;;
      n|no)  echo "false"; return 0 ;;
      *) echo "Please enter y or n." ;;
    esac
  done
}

prompt_choice() {
  local label="$1"
  shift
  local choices=("$@")
  local val=""

  while true; do
    {
      echo "$label"
      local i
      for i in "${!choices[@]}"; do
        echo "  $((i+1))) ${choices[$i]}"
      done
      printf "Select [1-%d]: " "${#choices[@]}"
    } >&2

    read -r val
    if [[ "$val" =~ ^[0-9]+$ ]] && (( val >= 1 && val <= ${#choices[@]} )); then
      echo "${choices[$((val-1))]}"
      return 0
    fi
    echo "Invalid choice." >&2
  done
}

change_settings_menu() {
  while true; do
    print_current_settings
    echo "Change settings:"
    echo "  1) input dir (-i)"
    echo "  2) output dir (-o)"
    echo "  3) checkpoint (-c)"
    echo "  4) batch-size"
    echo "  5) stereo-strength"
    echo "  6) sbs-min-opacity"
    echo "  7) focal-length"
    echo "  8) stereo-mode (toe_in/parallel)"
    echo "  9) stereo-convergence-depth (parallel only; empty disables)"
    echo "  10) stereo-convergence-norm (parallel only; empty disables)"
    echo "  11) toggle --sbs-image"
    echo "  12) toggle --fast-preview-render"
    echo "  13) done"
    echo
    read -r -p "Select an option [1-13] (default 13): " choice
    choice="${choice:-13}"

    case "$choice" in
      1) INPUT_DIR="$(prompt_nonempty 'New input dir: ')" ;;
      2) OUTPUT_DIR="$(prompt_nonempty 'New output dir: ')" ;;
      3) CHECKPOINT="$(prompt_nonempty 'New checkpoint path/name: ')" ;;
      4) BATCH_SIZE="$(prompt_number 'New batch-size: ')" ;;
      5) STEREO_STRENGTH="$(prompt_number 'New stereo-strength: ')" ;;
      6) SBS_MIN_OPACITY="$(prompt_number 'New sbs-min-opacity: ')" ;;
      7) FOCAL_LENGTH="$(prompt_number 'New focal-length: ')" ;;
      8) STEREO_MODE="$(prompt_choice 'Select stereo-mode:' toe_in parallel)" ;;
      9) read -r -p "New stereo-convergence-depth (empty to unset): " STEREO_CONVERGENCE_DEPTH ;;
      10) read -r -p "New stereo-convergence-norm (empty to unset): " STEREO_CONVERGENCE_NORM ;;
      11) SBS_IMAGE="$(prompt_bool 'Enable --sbs-image?')" ;;
      12) FAST_PREVIEW_RENDER="$(prompt_bool 'Enable --fast-preview-render?')" ;;
      13) return 0 ;;
      *) echo "Invalid choice." ;;
    esac
  done
}


change_preset_menu() {
  while true; do
    echo
    echo "Stereo presets (sets --stereo-mode parallel and uses --stereo-convergence-norm):"
    echo "  1) Rule of thumb (default starting point)"
    echo "     - stereo-strength: 0.10"
    echo "     - convergence-norm: 1.0  (screen plane at the scene's focus depth)"
    echo
    echo "  2) Natural stereo"
    echo "     - stereo-strength: 0.08"
    echo "     - convergence-norm: 1.0  (comfortable, subtle depth)"
    echo
    echo "  3) Cinematic pop-out"
    echo "     - stereo-strength: 0.12"
    echo "     - convergence-norm: 0.8  (more pop-out; stronger foreground depth)"
    echo
    echo "  4) Background depth emphasis"
    echo "     - stereo-strength: 0.10"
    echo "     - convergence-norm: 2.0  (pushes convergence farther; emphasizes background depth)"
    echo
    echo "  5) Back"
    echo
    read -r -p "Select a preset [1-5]: " preset_choice

    case "$preset_choice" in
      1)
        STEREO_MODE="parallel"
        STEREO_STRENGTH="0.1"
        STEREO_CONVERGENCE_DEPTH=""
        STEREO_CONVERGENCE_NORM="1.0"
        echo "Applied: Rule of thumb."
        return 0
        ;;
      2)
        STEREO_MODE="parallel"
        STEREO_STRENGTH="0.08"
        STEREO_CONVERGENCE_DEPTH=""
        STEREO_CONVERGENCE_NORM="1.0"
        echo "Applied: Natural stereo."
        return 0
        ;;
      3)
        STEREO_MODE="parallel"
        STEREO_STRENGTH="0.12"
        STEREO_CONVERGENCE_DEPTH=""
        STEREO_CONVERGENCE_NORM="0.8"
        echo "Applied: Cinematic pop-out."
        return 0
        ;;
      4)
        STEREO_MODE="parallel"
        STEREO_STRENGTH="0.1"
        STEREO_CONVERGENCE_DEPTH=""
        STEREO_CONVERGENCE_NORM="2.0"
        echo "Applied: Background depth emphasis."
        return 0
        ;;
      5)
        return 0
        ;;
      *)
        echo "Invalid choice."
        ;;
    esac
  done
}

run_menu() {
  while true; do
    print_current_settings
    echo "Main menu:"
    echo "  1) Run (default)"
    echo "  2) Change Preset"
    echo "  3) Change settings"
    echo "  4) Quit"
    echo
    read -r -p "Select an option [1-4] (default 1): " choice
    choice="${choice:-1}"

    case "$choice" in
      1)
        echo "Running in: $PROJECT_DIR"
        local prev_dir
        prev_dir="$(pwd)"
        cd "$PROJECT_DIR"
        local cmd
        cmd="$(build_command)"
        echo "Executing:"
        echo "  $cmd"
        set +e
        # shellcheck disable=SC2086
        eval "$cmd"
        local exit_code=$?
        set -e
        cd "$prev_dir"
        echo
        echo "Process exited with code: $exit_code"
        read -r -p "Press Enter to return to menu..." _
        continue
        ;;
      2)
        change_preset_menu
        ;;
      3)
        change_settings_menu
        ;;
      4)
        echo "Quit."
        return 0
        ;;
      *)
        echo "Invalid choice."
        ;;
    esac
  done
}

# --- Main ---
ensure_conda_shell
deactivate_all_conda
conda activate "$ENV_NAME"

run_menu
