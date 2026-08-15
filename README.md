# H0918 Ankles-Locked (Torque-Actuated) Gait RL

Reinforcement-learning training setup for a **torque-motor-actuated**
variant of the SCONE `H0918` planar gait model, with the **ankle and
back joints rigidly locked**. Built on top of
[sconegym](https://github.com/tgeijten/sconegym) and
[depRL](https://github.com/tgeijten/depRL/tree/sconegym).

## Prerequisites

**working SCONE + SconePy + sconegym + depRL install
first** is needed-- this repo does not replace or fork those, it plugs into
them. Follow the  order:

1. **SCONE Studio** (includes SconePy) -- https://scone.software
   A Hyfydy license is required for anything beyond the free OpenSim
   models; see https://hyfydy.com.
2. **sconegym** -- https://github.com/tgeijten/sconegym
   ```
   git clone https://github.com/tgeijten/sconegym
   cd sconegym
   pip install -e .
   ```
3. **depRL** (sconegym-compatible fork) --
   https://github.com/tgeijten/depRL/tree/sconegym
   ```
   git clone -b sconegym https://github.com/tgeijten/depRL
   cd depRL
   pip install -e .
   ```
4. Then, in this repo:
   ```
   pip install -r requirements.txt
   ```

Verify the base installation works *before* using this repo, e.g. by
running sconegym's own `example_environment.py` / `example_deprl.py`.

## Setup

This repo does **not** need to be copied into the sconegym install.
`sconegym_ext/torque_gaitgym.py` imports sconegym as a normal
installed package (`from sconegym.gaitgym import GaitGym`), so as
long as sconegym is `pip install -e`'d and importable, this repo can
live anywhere.

The only requirement: `sconegym_ext` must be **importable** from
wherever you launch training (i.e. run commands from this repo's
root, or `pip install -e .` this repo too / add it to `PYTHONPATH`).

## Training

```bash
python -m deprl.main configs/torque_h0918_ankles_locked.yaml
```

The config's `tonic.header` includes `import sconegym_ext`, which
runs the `gym.envs.registration.register(...)` call in
`sconegym_ext/__init__.py` and makes the environment ID
`sconewalk_h0918_ankles_locked-v1` available -- this is what
`deprl.environments.Gym('sconewalk_h0918_ankles_locked-v1', ...)`
in the config resolves to.

Results (checkpoints, logs) are saved automatically to the SCONE
results folder configured in SCONE Studio (not `working_dir`, which
is ignored for sconegym/Hyfydy runs -- this is depRL's default
behavior).

## Visualizing results

In SCONE Studio: Optimization Results pane -> navigate to any
checkpoint (`.pt`) file -> double-click to run rollouts -> results
appear as `.sto` files in a `run_checkpoint_<step>` subfolder next to
the checkpoint -> double-click any `.sto` to view the motion.

## Reward function 
xBriefly:

| Term | Purpose |
|---|---|
| `velocity` | Tent-shaped reward peaking at `target_vel`; penalizes both under- and over-shooting, including standing still or walking backward |
| `height` | Encourages staying upright / not collapsing |
| `upright` | Penalizes pelvis tilt |
| `grf` | Penalizes excessive ground reaction force |
| `joint_limit` | Penalizes torque generated at a joint's hard mechanical stop |
| `limit_proximity` | Anticipatory: penalizes approaching a hip/knee limit *before* impact, not just after |
| `smooth` | Penalizes jerky step-to-step action changes |
| `leg_symmetry` / `asymmetry` | Penalizes one leg doing all the work while the other drags |
| `impact` | Penalizes sudden jumps in total ground contact force (hard/stomping landings) |

`ankle_r`, `ankle_l`, and `back` are excluded from `joint_limit` /
`limit_proximity` since they are rigidly locked by design (`0..0`
range) -- any force there reflects contact reaction, not policy
behavior.

