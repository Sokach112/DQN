#!/usr/bin/env bash
# Run the three multi-seed variants for one seed concurrently on CPU.
# Usage: bash run_multiseed_batch.sh <seed>
# Output: outputs/assignment-2026-10-18-long/multiseed/<variant>-s<seed>/
set -u
cd "$(dirname "$0")/.."
SEED=${1:?usage: run_multiseed_batch.sh <seed>}
OUT=outputs/assignment-2026-10-18-long/multiseed
mkdir -p "$OUT/logs"
COMMON=(--game PongNoFrameskip-v4 --steps 200000 --seed "$SEED" --device cpu
        --torch-threads 1 --buffer-size 50000 --learning-starts 20000
        --end-epsilon 0.01 --target-update-interval 1000 --batch-size 32
        --test-interval 25000 --test-episodes 10 --train-log-interval 10000
        --save-interval 0 --final-test-episodes 10 --eval-seed-offset 100
        --output "$OUT")

run_variant() {
  local tag=$1; shift
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/long_experiments.py \
    --tag "$tag" "${COMMON[@]}" "$@" > "$OUT/logs/${tag}-run.log" 2>&1 &
}

run_variant "fast-eps-s${SEED}" --learning-rate 1e-4 --epsilon-timestep 100000
run_variant "slow-eps-s${SEED}" --learning-rate 1e-4 --epsilon-timestep 250000
run_variant "low-lr-s${SEED}" --learning-rate 5e-5 --epsilon-timestep 100000
wait
echo "batch seed ${SEED} done"
