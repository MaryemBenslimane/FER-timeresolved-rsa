#!/bin/bash
#SBATCH --job-name=qwen3vl-rdm
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=./models/vlm/logs/%x-%j.out
#SBATCH --error=./models/vlm/logs/%x-%j.out

# Qwen3-VL-8B-Instruct video-mode RDMs, pretrained and random-weight control.
# 18 stimulus videos -> 16 temporal groups each (32 duplicated canonical frames,
# fused in temporal pairs) -> canonical (n_layers, 16, 18, 18) RDMs.
set -uo pipefail

VLM=./models/vlm
OUT=./analysis/vlm_runs
MODEL_ID=Qwen/Qwen3-VL-8B-Instruct

module load StdEnv/2023 python/3.12
source $HOME/envs/fer/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=./.hf
export HF_HUB_OFFLINE=1          # compute nodes have no internet; weights are cached
export TOKENIZERS_PARALLELISM=false

echo "node=$(hostname)"; nvidia-smi -L
mkdir -p "$OUT" "$VLM/logs"
cd "$VLM"

STATUS=0
for VARIANT in pretrained random; do
    EXTRA=()
    [ "$VARIANT" = "random" ] && EXTRA=(--random)
    RUN="$OUT/qwen3vl_$VARIANT"
    echo "================ Qwen3-VL $VARIANT"
    if python vlm_extract_video_mode.py \
            --metadata_csv "$VLM/stimuli_metadata_canon.csv" \
            --output_dir "$RUN" \
            --model_id "$MODEL_ID" \
            --nframes 16 --frame-list \
            --dtype bfloat16 --device cuda --seed 42 \
            "${EXTRA[@]}"; then
        python compute_rdms_from_vlm_activations.py \
            --run-dir "$RUN" \
            --out-name "qwen3vl_${VARIANT}_16f_weights" \
            && echo "OK $VARIANT"
    else
        echo "FAILED $VARIANT" >&2
        STATUS=1
    fi
done

echo "=== done at $(date +%T)"
exit $STATUS
