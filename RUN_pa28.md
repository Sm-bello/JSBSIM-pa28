# Cloud run config — pa28 (Piper Cherokee/Warrior)

Verified this session:
- Trim: reliable within altitude 1500-3000ft, airspeed 75-90kt.
  Outside that band (tried up to 4000ft/95kt) trim fails.
- 60s hold test at 2000ft/85kt: altitude drifted 0.4ft over 60s, pitch/roll
  essentially flat. Clean.
- `generate_clean_trajectories(aircraft='pa28')` verified 10/10 through
  the validity gate after the envelope patch (see CHANGES_trim_fix_pass.md).

## Cloud command
```bash
cd ~/PHI-SPIKE
source .venv/bin/activate
python -m training.train_campaign \
  --epochs 40 --batch-size 32 --device cpu --seeds 0,1,2,3,4 \
  --variants vanilla_snn,snn_physics_loss,snn_membrane_conditioning,snn_temporal_physics,full_phi_spike \
  --n-clean 40 --sequence-len 80 \
  --aircraft pa28 \
  --save-dir experiments/full_campaign/pa28
```

## Before you launch the real campaign
Run a SHORT smoke check first (few minutes, not hours) to catch anything
envelope-specific I haven't seen:
```bash
python -m training.train_campaign --quick --aircraft pa28 \
  --save-dir experiments/smoke_test/pa28
```
Confirm no trim-gate warnings in the log before committing to the full
40-epoch x 5-seed run.

## Still open
- pa28's envelope is narrower than c172x's. If any downstream script
  (robustness sweep, fault-severity scaling) assumes c172x's wider range,
  it needs the same aircraft-aware treatment this patch gave the dataset
  factory. Not yet checked for pa28 specifically.
