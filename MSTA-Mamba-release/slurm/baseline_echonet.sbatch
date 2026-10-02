#!/bin/bash
#SBATCH --job-name=msta_base_echo
#SBATCH --partition=a100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm /tmp/msta_mpl
export MPLCONFIGDIR=/tmp/msta_mpl
"${PYTHON_BIN:-python}" tools/train_baseline.py --config "${CONFIG_PATH:-configs/baseline_echonet.yaml}"
