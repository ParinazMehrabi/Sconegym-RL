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

## Environment Extension

The custom environment:

    TorqueGaitGym

inherits from:

    sconegym.gaitgym.GaitGym

It adds:

-   torque-based actuation
-   torque scaling
-   action rate limiting
-   custom reward terms
-   ankle-locked H0918 support

## Action Space

The policy outputs normalized actions:

    [-1,1]

The actions are converted to torque commands:

    torque = action * action_scale

The current configuration uses:

    action_scale: 120.0

Action mapping:

  Action   Joint
  -------- --------
  0        hip_r
  1        knee_r
  2        hip_l
  3        knee_l

## Reward Function

The reward is a weighted sum of:

### Velocity

Encourages walking close to the target velocity and penalizes excessive
overspeed.

### Height

Encourages maintaining sufficient COM height.

### Upright

Penalizes pelvis orientation errors.

### Ground Reaction Force

Penalizes excessive contact forces.

### Joint Limit

Penalizes torque generation near mechanical limits.

Locked joints such as ankle and back are excluded because they are
rigidly constrained.

### Limit Proximity

Provides an anticipatory penalty before reaching joint limits.

### Smoothness

Penalizes abrupt changes in torque commands.

### Leg Symmetry

Compares accumulated effort between the right and left legs to reduce
one-sided gait behavior.

### Impact

Penalizes sudden increases in total contact force.

## Training Configuration

Current important parameters:

``` yaml
vel_coeff: 6.0
height_coeff: 3.5
grf_coeff: -0.08
joint_limit_coeff: -0.3
limit_proximity_coeff: -3.0
smooth_coeff: -0.1
leg_symmetry_coeff: 1.0
action_scale: 120.0
action_rate_limit: 0.15
```

DEP configuration uses:

-   DEP exploration
-   Tuned MPO
-   AdaptiveEnergyBuffer replay

## Training

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run training:

``` bash
python -m deprl.main configs/torque_h0918_ankles_locked.yaml
```

The environment is registered through:

    sconegym_ext

and the environment ID is:

    sconewalk_h0918_ankles_locked-v1

## Resume Training

Continue from checkpoint:

``` yaml
resume: true
```

Start a new experiment:

``` yaml
resume: false
```

## Project Structure

    h0918-ankles-locked-torque-rl

    ├── configs
    ├── models
    ├── sconegym_ext
    │   ├── gaitgym.py
    │   ├── init_v0.py
    │   └── __init__.py
    ├── requirements.txt
    └── README.md

## References

SCONE Gym: https://github.com/tgeijten/sconegym

depRL: https://github.com/tgeijten/depRL/tree/sconegym

SCONE: https://scone.software
