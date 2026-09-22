# PHI-SPIKE — Piper Cherokee/Warrior (PA-28)

## What this is

This repo trains and evaluates PHI-SPIKE — a physics-informed spiking
neural network for aircraft fault diagnosis — on a JSBSim-simulated
Piper PA-28.

## What we used it for

This is the **generalization test**. The core PHI-SPIKE research
claim — that coupling a physics residual into a spiking neuron's
membrane dynamics beats treating physics as an ordinary loss term —
was first established on a Cessna 172/182. A result on one airframe
could just be a lucky fit to that aircraft's specific dynamics. PA-28
was deliberately chosen as a genuinely different test, not a
same-family variation: different manufacturer, low-wing instead of
high-wing, different control-surface geometry and aerodynamic layout.
The question this repo answers is not "does PHI-SPIKE work" — that's
already been tested elsewhere — it's **"does the physics-conditioning
advantage survive a real airframe change, or was it specific to the
Cessna family?"**

Same 5-variant ablation, same 10-class fault taxonomy, as the primary
repo — see there for the full research framing. What's specific to
this repo is airframe, not methodology.

## Verified trim envelope

**1,600–2,900 ft altitude, 77–89 kt airspeed.** This is noticeably
narrower than the Cessna 172/182 envelope — do not reuse those
aircraft's altitude/airspeed ranges here; JSBSim's trim solver will
fail outside this band. If you extend the dataset generation config,
re-verify the envelope rather than assume it transfers.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.6 scipy==1.16.3 pandas==2.3.3 jsbsim==1.3.1 \
            tqdm==4.67.1 pyarrow==22.0.0 scikit-learn==1.8.0 xgboost==3.2.0
```
Developed and verified against Python 3.11.14.

## Quickstart

```bash
python -m training.train_campaign \
  --epochs 3 --batch-size 16 --device cpu --seeds 0 \
  --variants vanilla_snn --n-clean 6 --sequence-len 40 --aircraft pa28 \
  --save-dir experiments/smoke_test
```

Real campaign:
```bash
python -m training.train_campaign \
  --epochs 40 --batch-size 32 --device cpu --seeds 0,1,2,3,4 \
  --variants vanilla_snn,snn_physics_loss,snn_membrane_conditioning,snn_temporal_physics,full_phi_spike \
  --n-clean 40 --sequence-len 80 --aircraft pa28 \
  --save-dir experiments/full_campaign/pa28
```
Launch with `nohup ... &; disown`, not `tmux`.

## Provenance: this repo inherited fixes, it didn't discover them

Every numerical-stability fix here was found and fixed on the Cessna
182 repo first, then explicitly reconciled onto this codebase — it did
not independently work on the first attempt either. When first
audited, this repo had the JSBSim trim fix but was **missing** the LIF
shape-bug fix, the membrane-potential clamp, and the gradient
value-clip — all three were ported over and then independently
verified here, not assumed to carry over silently:

- **JSBSim trim**: `propulsion/set-running = -1` (not
  `propulsion/engine/set-running`, which silently no-ops) + `do_trim(1)`
  + trim-relative control application.
- **LIF layer shape bug**: removed an auto-transpose heuristic that
  scrambled batch and time dimensions during backpropagation.
- **Membrane potential clamp**: `torch.clamp(..., -50.0, 50.0)` on
  synaptic current and membrane potential every timestep, in both the
  vanilla and physics-conditioned LIF layers.
- **Gradient value-clipping**: `clip_grad_value_(..., 5.0)` before the
  existing norm-based clip — norm-clipping alone was insufficient
  against the exploding-gradient pattern found on this project.

**Verified on this airframe specifically** (not inferred from the
Cessna results): 30/30 successful trims during dataset generation, and
a real training run — `vanilla_snn` and `full_phi_spike`, 3 epochs,
zero non-finite batches in either, both learning (F1 0.30 and 0.33
respectively at that small scale).

**Not yet done**: `snn_physics_loss`, `snn_membrane_conditioning`, and
`snn_temporal_physics` have not been individually run to completion on
this airframe (only the two structurally distinct code paths —
`LIFLayer` and `PhysicsConditionedLIF` — were tested). Full-scale
(`n_clean=40`) has also not been verified here yet. Run the smoke test
below before committing to a real campaign.

## Before you trust any result from this repo

- Run the 3-epoch smoke test first. `[skipped N non-finite batches]`
  should read `0`. If it climbs epoch over epoch, stop.
- Full scale (`n_clean=40`) is memory-heavy — confirmed to OOM-kill a
  <8GB box on the sibling Cessna repo. Check `free -h` partway through
  a real run here too, since it hasn't been separately confirmed safe
  on this airframe.
- Use `nohup ... &; disown` for long runs, not `tmux`.

## Repository structure

```
simulation/       JSBSim aircraft wrapper, trim, telemetry generation
fault_injection/  10-class fault taxonomy, randomized onset/severity injection
physics/          Physics-consistency residual computation
encoding/         Telemetry -> spike encoding
models/
  vanilla_snn/    Baseline LIF network (no physics coupling)
  phi_spike/      Physics-conditioned LIF + the 4 physics-aware variants
datasets/         Dataset assembly, envelope config, windowing
training/         train_campaign.py -- main entry point
baselines/        Non-spiking comparison models
robustness/       Noise/dropout/severity robustness sweeps
edge_emulation/   Streaming inference latency benchmarks
evaluation/       Metrics aggregation, figures
verification/     Pipeline sanity checks
dashboard/        Results visualization server
```

## License

JSBSim is LGPL-licensed — check compatibility before choosing a
license for this repository if redistributing.

## Citation

_(add once published)_
