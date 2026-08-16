# H0918 Ankles-Locked (Torque-Actuated) Gait RL

Reinforcement-learning training setup for a **torque-motor-actuated**
variant of the SCONE `H0918` planar gait model, with the **ankle and
back joints rigidly locked**. Built on top of
[sconegym](https://github.com/tgeijten/sconegym) and
[depRL](https://github.com/tgeijten/depRL/tree/sconegym).

the policy directly outputs torque (via `joint_motor` actuators) for 4 DOFs: `hip_r`,

## Prerequisites

A **working SCONE + SconePy + sconegym + depRL install first** is
needed -- this repo does not replace or fork those, it plugs into
them. Follow this order:

1. **SCONE Studio** (includes SconePy) -- https://scone.software
   A Hyfydy license is required for anything beyond the free OpenSim
   models; see https://hyfydy.com.
2. **sconegym** -- https://github.com/tgeijten/sconegym
   ```bash
   git clone https://github.com/tgeijten/sconegym
   cd sconegym
   pip install -e .
   ```
3. **depRL** (sconegym-compatible fork) --
   https://github.com/tgeijten/depRL/tree/sconegym
   ```bash
   git clone -b sconegym https://github.com/tgeijten/depRL
   cd depRL
   pip install -e .
   ```
4. Then, in this repo:
   ```bash
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

The `TorqueGaitGym` reward is a weighted sum of the terms below (all
coefficients configurable per-term via `env_args` in the yaml):

| Term | Purpose |
|---|---|
| `velocity` | penalizes both under- and over-shooting, including standing still or walking backward -- not a flat plateau, so overspeeding is never reward-neutral |
| `height` | Encourages staying upright / not collapsing |
| `upright` | Penalizes pelvis tilt (either direction) |
| `grf` | Penalizes excessive ground reaction force |
| `joint_limit` | Penalizes torque generated at a joint's hard mechanical stop (reactive -- after contact) |
| `limit_proximity` | Anticipatory version of the above: penalizes approaching a hip/knee limit *before* impact, based on normalized distance to the DOF's range edge |
| `smooth` | Penalizes jerky step-to-step action changes |
| `asymmetry` / `leg_symmetry` | Compares mean right-leg effort (`|hip_r|+|knee_r|`) vs. mean left-leg effort, accumulated over the **entire episode so far** -- directly penalizes "one leg does all the work, the other drags" |
| `impact` | Penalizes sudden jumps in total ground contact force step-to-step -- a proxy for hard/stomping landings, since per-geometry (heel vs. toe) contact force isn't exposed by the sconepy API used here |
| `self_contact` | Penalizes contact force on any body other than the feet (e.g. knee-on-knee) |

`ankle_r`, `ankle_l`, and `back` are excluded from `joint_limit` /
`limit_proximity` since they are rigidly locked by design (`0..0`
range) -- any force there reflects contact reaction, not policy
behavior.