#!/bin/bash
#SBATCH --job-name=qwen3vl-54cpu
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --output=./models/vlm/logs/%x-%j.out
#SBATCH --error=./models/vlm/logs/%x-%j.out
set -euo pipefail

ROOT=.
PY=$HOME/envs/fer/bin/python
MANIFEST="$ROOT/stim_frames_54/stimulus_manifest.json"
OUT="$ROOT/analysis/model_rdms_54"
export HF_HOME="$ROOT/.hf"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export PYTHONPATH="$ROOT/models/vlm:${PYTHONPATH:-}"

module load StdEnv/2023 python/3.12
source $HOME/envs/fer/bin/activate
cd "$ROOT/models"
mkdir -p "$ROOT/models/vlm/logs" "$OUT"

"$PY" extract_qwen_rdms.py --model_id Qwen/Qwen3-VL-8B-Instruct \
  --manifest "$MANIFEST" --out-dir "$OUT" --nframes 16 --frame-list \
  --dtype bfloat16 --device cpu --out-name qwen3vl_pretrained_16f_weights

"$PY" extract_qwen_rdms.py --model_id Qwen/Qwen3-VL-8B-Instruct \
  --manifest "$MANIFEST" --out-dir "$OUT" --nframes 16 --frame-list \
  --dtype bfloat16 --device cpu --random --out-name qwen3vl_random_16f_weights

"$PY" build_emotion_rdms_54.py --dir "$OUT" \
  --models qwen3vl_pretrained_16f_weights qwen3vl_random_16f_weights
