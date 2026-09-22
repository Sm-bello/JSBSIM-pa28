#!/usr/bin/env bash
# PHI-SPIKE full paper campaign launcher
# Preserves prototype baseline under experiments/prototype_baseline/
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
echo "=== PHI-SPIKE Paper Campaign ==="
echo "Root: $ROOT"
echo "Prototype metrics kept at: experiments/prototype_baseline/"
echo ""

QUICK="${QUICK:-0}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-40}"
SEEDS="${SEEDS:-0,1,2,3,4}"
VARIANTS="${VARIANTS:-vanilla_snn,snn_physics_loss,snn_membrane_conditioning,snn_temporal_physics,full_phi_spike}"

mkdir -p experiments/full_campaign experiments/ablations experiments/robustness experiments/figures logs datasets/full

echo "[1/5] Full experimental campaign (multi-seed, all ablations) …"
if [[ "$QUICK" == "1" ]]; then
  python -m training.train_campaign \
    --quick \
    --epochs 8 \
    --batch-size 16 \
    --device "$DEVICE" \
    --seeds 0,1 \
    --variants "$VARIANTS" \
    --n-clean 6 \
    --sequence-len 40 \
    --save-dir experiments/full_campaign \
    2>&1 | tee logs/campaign_quick.log
else
  python -m training.train_campaign \
    --epochs "$EPOCHS" \
    --batch-size 32 \
    --device "$DEVICE" \
    --seeds "$SEEDS" \
    --variants "$VARIANTS" \
    --n-clean 40 \
    --sequence-len 80 \
    --save-dir experiments/full_campaign \
    2>&1 | tee logs/campaign_full.log
fi

echo ""
echo "[2/5] Publication figures from campaign metrics …"
python -m evaluation.metrics_and_figures \
  --metrics-dir experiments/full_campaign \
  --figure-dir experiments/figures \
  2>&1 | tee logs/figures.log

echo ""
echo "[3/5] Robustness matrix …"
python -m robustness.run_robustness \
  --device "$DEVICE" \
  $( [[ "$QUICK" == "1" ]] && echo --quick ) \
  --out-dir experiments/robustness \
  2>&1 | tee logs/robustness.log

echo ""
echo "[4/5] Edge streaming smoke (batch-size=1) …"
python -m edge_emulation.streaming_infer 2>&1 | tee logs/edge.log || true

echo ""
echo "[5/5] Verification gate …"
python -m verification.run_verification 2>&1 | tee logs/verification.log || true

echo ""
echo "=== Campaign complete ==="
echo "Results:"
echo "  experiments/full_campaign/campaign_summary.json"
echo "  experiments/figures/"
echo "  experiments/robustness/robustness_matrix.json"
echo "  experiments/prototype_baseline/   ← original 30-epoch JSON + checkpoints"
echo ""
echo "Next: inspect summary, then tighten residual / literature claim as needed."
