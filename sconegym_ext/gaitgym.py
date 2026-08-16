import os
# Set non-random initial muscle activations
import sys
from abc import ABC, abstractmethod
from typing import Optional

import gym
import numpy as np

from sconetools import sconepy

def find_model_file(model_file):
    this_dir, this_file = os.path.split(__file__)
    return os.path.join(this_dir, "data", model_file)


DEFAULT_REW_KEYS = {
    "vel_coeff": 10.0,
    "grf_coeff": 0.0,
    "joint_limit_coeff": 0.0,
    "smooth_coeff": 0.0,
    "nmuscle_coeff": 0.0,
    "self_contact_coeff": 0.0,
}


class SconeGym(gym.Env, ABC):
    """
    Main general purpose class that gives you a gym-ready sconepy interface
    It has to be inherited by the environments you actually want to use and
    some methods have to be defined (see end of class). This class would probably
    be a good starting point for new environments.
    New environments also have to be registered in sconegym/__init__.py !
    """

    def __init__(self,
                 model_file,
                 left_leg_idxs,
                 right_leg_idxs,

                 
                 root_body_name = 'pelvis',
                 foot_body_name = 'calcn',
                 clip_actions = False,
                 target_vel = 1.2,
                 leg_switch = True,
                 use_delayed_sensors = False,
                 use_delayed_actuators = False,
                 run = False,
                 obs_type = '2D',
                 init_activations_mean = 0.3,
                 init_activations_std = 0.1,
                 min_com_height = 0.5,
                 min_head_height = 0.9,
                 fall_recovery_time = 0.0,
                 rew_keys = DEFAULT_REW_KEYS,
                 *args, **kwargs):
        # Internal settings
        self.episode = 0
        self.total_reward = 0.0
        self.init_dof_pos_std = 0.05
        self.init_dof_vel_std = 0.1
        self.init_load = 0.5
        self.init_activations_mean = init_activations_mean
        self.init_activations_std = init_activations_std
        self.min_com_height = min_com_height
        self.min_head_height = min_head_height
        self.step_size = 0.01
        self.total_steps = 0
        self.steps = 0
        self.fall_time = -1.0
        self.has_reset = False
        self.store_next = False
        # Reward coefficients from kwargs
        for k, v in rew_keys.items():
            setattr(self, k, float(v))
        self.target_vel = target_vel
        self.use_delayed_sensors = use_delayed_sensors
        self.use_delayed_actuators = use_delayed_actuators
        self.clip_actions = clip_actions
        self.leg_switch = leg_switch
        self.run = run
        self.obs_type = obs_type
        self.left_leg_idxs = left_leg_idxs
        self.right_leg_idxs = right_leg_idxs
        self.root_body_name = root_body_name
        self.left_foot_body_name = foot_body_name + "_l"
        self.right_foot_body_name = foot_body_name + "_r"
        self.fall_recovery_time = fall_recovery_time
        super().__init__(*args, **kwargs)
        sconepy.set_log_level(3)
        self.model = sconepy.load_model(model_file)
        self.init_dof_pos = self.model.dof_position_array().copy()
        self.init_dof_vel = self.model.dof_velocity_array().copy()
        self.set_output_dir("DATE_TIME." + self.model.name())
        self._find_head_body()
        self._setup_action_observation_spaces()

    def step(self, action):
        """
        takes an action and advances environment by 1 step.
        """
        if self.clip_actions:
            action = np.clip(action, 0, 0.5)
        else:
            action = np.clip(action, 0, 1.0)
        if not self.has_reset:
            raise Exception("You have to call reset() once before step()")

        if self.use_delayed_actuators:
            self.model.set_delayed_actuator_inputs(action)
        else:
            self.model.set_actuator_inputs(action)

        self.model.advance_simulation_to(self.time + self.step_size)
        reward = self._get_rew()
        obs = self._get_obs()
        done = self._get_done()
        reward = self._apply_termination_cost(reward, done)
        self.time += self.step_size
        self.total_reward += reward

        if done:
            if self.store_next:
                self.model.write_results(
                    self.output_dir, f"{self.episode:05d}_{self.total_reward:.3f}"
                )
                self.store_next = False
            self.episode += 1
        return obs, reward, done, {}

    def write_now(self):
        if self.store_next:
            self.model.write_results(
                self.output_dir, f"{self.episode:05d}_{self.total_reward:.3f}"
            )
            self.store_next = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """
        Reset and randomize the initial state.
        """
        self.episode_number = np.random.randint(0, 1000000)
        self.model.reset()
        self.has_reset = True
        self.time = 0
        self.total_reward = 0.0
        self.steps = 0
        self.fall_time = -1.0

        # Check if data should be stored (slow)
        self.model.set_store_data(self.store_next)
        # Randomize initial pose
        dof_pos = self.init_dof_pos + np.random.normal(
            0, self.init_dof_pos_std, len(self.init_dof_pos)
        )
        self.model.set_dof_positions(dof_pos)
        dof_vel = self.init_dof_vel + np.random.normal(
            0, self.init_dof_vel_std, len(self.init_dof_vel)
        )
        self.model.set_dof_velocities(dof_vel)
        if self.leg_switch:
            if np.random.uniform() < 0.5:
                self._switch_legs()
        if self.init_activations_std != 0:
            # Randomize initial muscle activations
            muscle_activations = np.clip(
                np.random.normal(
                    self.init_activations_mean,
                    self.init_activations_std,
                    size=len(self.model.muscles()),
                ),
                0.01,
                1.0,
            )
        else:
            # Set non-random initial muscle activations
            muscle_activations = np.ones((len(self.model.muscles()),)) * self.init_activations_mean
        self.prev_acts = muscle_activations
        self.prev_excs = self.model.muscle_excitation_array()
        self.model.init_muscle_activations(muscle_activations)

        # Initialize state and equilibrate muscles
        self.model.init_state_from_dofs()

        if self.init_load > 0:
            self.model.adjust_state_for_load(self.init_load)
        obs = self._get_obs()
        if return_info:
            return obs, (obs, {})
        else:
            return obs

    def store_next_episode(self):
        """
        Primes the environment to store the next episode.
        This also calls reset() to ensure that the data is
        written correctly.
        """
        self.store_next = True
        self.reset()

    def set_output_dir(self, dir_name):
        self.output_dir = sconepy.replace_string_tags(dir_name)

    def manually_load_model(self):
        self.model = sconepy.load_model(self.model_file)
        self.model.set_store_data(True)

    def render(self, *args, **kwargs):
        """
        Not yet supported
        """
        return

    def model_velocity(self):
        return self.model.com_vel().x

    def _setup_action_observation_spaces(self):
        num_act = len(self.model.actuators())
        self.action_space = gym.spaces.Box(
            low=np.zeros(shape=(num_act,)),
            high=np.ones(shape=(num_act,)),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-10000, high=10000, shape=self._get_obs().shape, dtype=np.float32
        )

    def _find_head_body(self):
        head_names = ["torso", "head", "lumbar"]
        self.head_body = None
        for b in self.model.bodies():
            if b.name() in head_names:
                self.head_body = b
        if self.head_body is None:
            raise Exception("Could not find head body")

    def _switch_legs(self):
        """
        Switches leg joint angles. Good for initial
        state randomization.
        """
        pos = self.model.dof_position_array()
        vel = self.model.dof_velocity_array()
        for left, right in zip(self.left_leg_idxs, self.right_leg_idxs):
            pos[left], pos[right] = pos[right], pos[left]
            vel[left], vel[right] = vel[right], vel[left]
        self.model.set_dof_positions(pos)
        self.model.set_dof_velocities(vel)

    def apply_args(self):
        pass

    def _apply_termination_cost(self, reward, done):
        return reward

    # these all need to be defined by environments
    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _get_rew(self):
        pass

    @abstractmethod
    def _get_done(self):
        pass

    @property
    def results_dir(self):
        return sconepy.scone_results_dir()


class GaitGym(SconeGym):
    def __init__(self, model_file, *args, **kwargs):
        self._max_episode_steps = 1000
        super().__init__(model_file, *args, **kwargs)
        self.rwd_dict = None
        self.mass = np.sum([x.mass() for x in self.model.bodies()])

    def _get_obs(self):
        if self.obs_type == '2D':
            return self._get_obs_2d()
        elif self.obs_type == '3D':
            return self._get_obs_3d()
        else:
            raise NotImplementedError

    def _get_obs_3d(self):
        acts = self.model.muscle_activation_array()
        self.prev_acts = self.model.muscle_activation_array().copy()
        self.prev_excs = self.model.muscle_excitation_array()
        dof_values = self.model.dof_position_array()
        dof_vels = self.model.dof_velocity_array()
        # No x or y position in the state
        dof_values[3] = 0.0
        dof_values[5] = 0.0
        return np.concatenate(
            [
                self.model.muscle_fiber_length_array(),
                self.model.muscle_fiber_velocity_array(),
                self.model.muscle_force_array(),
                self.model.muscle_excitation_array(),
                self.head_body.orientation().array(),
                self.head_body.ang_vel().array(),
                self._get_feet_relative_position(),
                dof_values,
                dof_vels,
                acts,
            ],
            dtype=np.float32,
        ).copy()

    def _get_feet_relative_position(self):
        pelvis = (
            [x for x in self.model.bodies() if self.root_body_name in x.name()][0]
            .com_pos()
            .array()
        )
        foot_l = (
            [x for x in self.model.bodies() if self.left_foot_body_name in x.name()][0]
            .com_pos()
            .array()
        )
        foot_r = (
            [x for x in self.model.bodies() if self.right_foot_body_name in x.name()][0]
            .com_pos()
            .array()
        )
        return np.concatenate([foot_l - pelvis, foot_r - pelvis], dtype=np.float32)

    def _get_obs_2d(self):
        acts = self.model.muscle_activation_array()
        self.prev_acts = self.model.muscle_activation_array().copy()
        self.prev_excs = self.model.muscle_excitation_array()
        dof_values = self.model.dof_position_array()
        dof_vels = self.model.dof_velocity_array()
        dof_values[1] = 0.0
        if not self.use_delayed_sensors:
            return np.concatenate(
                [
                    self.model.muscle_fiber_length_array(),
                    self.model.muscle_fiber_velocity_array(),
                    self.model.muscle_force_array(),
                    self.model.muscle_excitation_array(),
                    self.head_body.orientation().array(),
                    self.head_body.ang_vel().array(),
                    self._get_feet_relative_position(),
                    dof_values,
                    dof_vels,
                    acts,
                ],
                dtype=np.float32,
            ).copy()

        else:
            return np.concatenate(
                [
                    self.model.delayed_muscle_fiber_length_array(),
                    self.model.delayed_muscle_fiber_velocity_array(),
                    self.model.delayed_muscle_force_array(),
                    self.model.delayed_vestibular_array(),
                    self.model.muscle_excitation_array(),
                    self.model.muscle_activation_array(),
                ],
                dtype=np.float32,
            ).copy()

    def _get_rew(self):
        """
        Reward function.
        """
        self.total_steps += 1
        self.steps += 1
        return self.custom_reward()

    def custom_reward(self):
        self._update_rwd_dict()
        return np.sum(list(self.rwd_dict.values()))

    def _update_rwd_dict(self):
        self.rwd_dict = {
            "gaussian_vel": self.vel_coeff * self._gaussian_plateau_vel(),
            "grf": self.grf_coeff * self._grf(),
            "smooth": self.smooth_coeff * self._exc_smooth_cost(),
            "number_muscles": self.nmuscle_coeff * self._number_muscle_cost(),
            "constr": self.joint_limit_coeff * self._joint_limit_torques(),
            "self_contact": self.self_contact_coeff * self._get_self_contact(),
        }
        return self.rwd_dict

    def get_rwd_dict(self):
        if not self.rwd_dict:
            self.rwd_dict = self._update_rwd_dict()
        rwd_dict = {k: v for k, v in self.rwd_dict.items()}
        return rwd_dict

    def _number_muscle_cost(self):
        """
        Get number of muscle with activations over 0.15.
        """
        return self._get_active_muscles(0.15)

    def _get_active_muscles(self, threshold):
        """
        Get the number of muscles whose activations is above the threshold.
        """
        return (
            np.sum(
                np.where(self.model.muscle_activation_array() > threshold)[0].shape[0]
            )
            / self.action_space.shape[0]
        )

    def _gaussian_vel(self):
        vel = self.model_velocity()
        return np.exp(-np.square(vel - self.target_vel))

    def _gaussian_plateau_vel(self):
        if self.run:
            return self.model_velocity()
        vel = self.model_velocity()
        if vel < self.target_vel:
            return self._gaussian_vel()
        else:
            return 1.0

    def _exc_smooth_cost(self):
        excs = self.model.muscle_excitation_array()
        delta_excs = excs - self.prev_excs
        return np.mean(np.square(delta_excs))

    def _get_self_contact(self):
        ignore_bodies = ["calcn_r", "calcn_l"]
        contact_force = np.sum(
            [
                np.abs(x.contact_force().array())
                for x in self.model.bodies()
                if x.name() not in ignore_bodies
            ]
        )
        return np.clip(contact_force, -100, 100) / 100

    def _joint_limit_torques(self):
        return np.mean(
            [np.mean(np.abs(x.limit_torque().array())) for x in self.model.joints()]
        )

    def _grf(self):
        grf = self.model.contact_load()
        return max(0, grf - 1.2)

    def _get_done(self) -> bool:
        """
        The episode ends if the center of mass is below min_com_height.
        """
        fall = self.model.com_pos().y < self.min_com_height
        fall = fall or self.head_body.com_pos().y < self.min_head_height
        current_time = self.model.time()
        if fall:
            if self.fall_time < 0:
                self.fall_time = current_time
            if current_time - self.fall_time >= self.fall_recovery_time:
                return True
        else:
            self.fall_time = -1.0

        return False

    @property
    def horizon(self):
        # TODO put this in model kwargs such that it works with deprl
        return 1000


# Tutorial environments to see features
# The Measure one needs to be fixed

# TODO @thomas add right model file
class GaitGymMeasureH0918(GaitGym):
    """
    Shows how to use custom measures from the .scone files in
    python.
    """

    def __init__(self, *args, **kwargs):
        self.delay = False
        super().__init__(find_model_file("H0918_hfd_measure.scone"), *args, **kwargs)

    def custom_reward(self):
        self.rwd_dict = self.create_rwd_dict()
        return self.model.current_measure()
REWARD_KEYS = [
    "vel_coeff", "grf_coeff", "joint_limit_coeff",
    "smooth_coeff", "nmuscle_coeff", "self_contact_coeff",
    "upright_coeff",
]

class TorqueGaitGym(GaitGym):

    ACTUATOR_SIGN = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    MAX_SANE_VELOCITY = 4.0
    BLOWUP_PENALTY = 5.0

    def __init__(
        self,
        model_file,
        *args,
        action_scale=100.0,
        action_rate_limit=0.15,
        step_size=0.01,
        init_load=0.5,
        **kwargs,
    ):
        self.is_torque_actuated = True
        self._max_episode_steps = 1000
        self.action_scale = float(action_scale)
        self.action_rate_limit = float(action_rate_limit)
        self.step_size = float(step_size)
        self.init_load = float(init_load)

        kwargs["init_activations_mean"] = 0.0
        kwargs["init_activations_std"] = 0.0

        rew_keys = dict(kwargs.get("rew_keys", {}))

        rew_keys.setdefault("vel_coeff", 10.0)
        rew_keys.setdefault("height_coeff", 8.0)
        rew_keys.setdefault("upright_coeff", 3.0)
        rew_keys.setdefault("grf_coeff", -0.08)
        rew_keys.setdefault("joint_limit_coeff", -0.05)
        rew_keys.setdefault("smooth_coeff", -0.03)

        rew_keys.setdefault("self_contact_coeff", 0.0)
        rew_keys.setdefault("nmuscle_coeff", 0.0)
        rew_keys.setdefault("asymmetry_coeff", 0.5)
        rew_keys.setdefault("impact_coeff", -0.02)
        rew_keys.setdefault("leg_symmetry_coeff", 0.0)
        rew_keys.setdefault("limit_proximity_coeff", 0.0)
        self._requested_leg_switch = kwargs.get("leg_switch", True)

        kwargs["rew_keys"] = rew_keys
        self.pelvis_tilt_idx = 0
        self.pelvis_tx_idx = 1
        self.pelvis_ty_idx = 2

        self.hip_r_idx = 3
        self.knee_r_idx = 4

        self.hip_l_idx = 5
        self.knee_l_idx = 6

        super().__init__(model_file, *args, **kwargs)

        self.init_dof_pos_std = 0.02
        self.init_dof_vel_std = 0.0
        self.leg_switch = bool(self._requested_leg_switch)

        n_act = len(self.model.actuators())
        self.prev_action = np.zeros(n_act, dtype=np.float32)
        self.current_action = np.zeros(n_act, dtype=np.float32)
        self._blew_up = False
        self._right_effort_sum = 0.0
        self._left_effort_sum = 0.0

        self._printed_index_mapping = False

    def reset(self, *, seed=None, return_info=False, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.episode_number = np.random.randint(0, 1000000)
        self.model.reset()
        self.has_reset = True

        self.time = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.fall_time = -1.0

        self.model.set_store_data(self.store_next)

        dof_pos = np.asarray(self.init_dof_pos, dtype=np.float64).copy()

        if self.init_dof_pos_std > 0.0:
            dof_pos += np.random.normal(0.0, self.init_dof_pos_std, size=len(dof_pos))

        dof_pos[self.pelvis_tilt_idx] = np.clip(dof_pos[self.pelvis_tilt_idx], -0.08, 0.08)
        dof_pos[self.knee_r_idx] = np.clip(dof_pos[self.knee_r_idx], -0.30, -0.10)
        dof_pos[self.knee_l_idx] = np.clip(dof_pos[self.knee_l_idx], -0.30, -0.10)

        self.model.set_dof_positions(dof_pos)
        if self.leg_switch and np.random.uniform() < 0.5:
            dof_pos = self.model.dof_position_array()
            dof_pos[self.hip_r_idx], dof_pos[self.hip_l_idx] = (
                dof_pos[self.hip_l_idx],
                dof_pos[self.hip_r_idx],
            )
            dof_pos[self.knee_r_idx], dof_pos[self.knee_l_idx] = (
                dof_pos[self.knee_l_idx],
                dof_pos[self.knee_r_idx],
            )
            self.model.set_dof_positions(dof_pos)
        dof_vel = np.zeros(len(self.init_dof_vel), dtype=np.float64)
        self.model.set_dof_velocities(dof_vel)
        self.model.init_state_from_dofs()

        if self.init_load > 0.0:
            self.model.adjust_state_for_load(self.init_load)

        n_act = len(self.model.actuators())
        self.prev_action = np.zeros(n_act, dtype=np.float32)
        self.current_action = np.zeros(n_act, dtype=np.float32)
        self._blew_up = False
        self._prev_grf = float(self.model.contact_load())
        self._right_effort_sum = 0.0
        self._left_effort_sum = 0.0
        obs = self._get_obs()

        if return_info:
            return obs, {}
        return obs

    def step(self, action):
        if not self.has_reset:
            raise RuntimeError("You have to call reset() once before step()")

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        action = np.clip(
            action,
            self.prev_action - self.action_rate_limit,
            self.prev_action + self.action_rate_limit,
        )

        self.current_action = action.copy()
        torque = action * self.action_scale * self.ACTUATOR_SIGN
        self.model.set_actuator_inputs(torque)

        self.model.advance_simulation_to(self.time + self.step_size)

        reward = float(self._get_rew())
        obs = self._get_obs()
        done = self._get_done()
        reward = float(self._apply_termination_cost(reward, done))

        self.time += self.step_size
        self.total_reward += reward
        self.prev_action = self.current_action.copy()

        if done:
            if self.store_next:
                self.model.write_results(
                    self.output_dir, f"{self.episode:05d}_{self.total_reward:.3f}"
                )
                self.store_next = False
            self.episode += 1

        return obs, reward, done, {}

    def _setup_action_observation_spaces(self):
        num_act = len(self.model.actuators())

        self.action_space = gym.spaces.Box(
            low=-np.ones(num_act, dtype=np.float32),
            high=np.ones(num_act, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1e4, high=1e4, shape=self._get_obs().shape, dtype=np.float32
        )

    def _get_obs(self):
        dof_values = np.asarray(self.model.dof_position_array(), dtype=np.float32).copy()
        dof_vels = np.asarray(self.model.dof_velocity_array(), dtype=np.float32).copy()
        dof_values[self.pelvis_tx_idx] = 0.0

        head_ori = np.asarray(self.head_body.orientation().array(), dtype=np.float32)
        head_ang_vel = np.asarray(self.head_body.ang_vel().array(), dtype=np.float32)
        feet_rel = np.asarray(self._get_feet_relative_position(), dtype=np.float32)

        current_action = getattr(self, "current_action", None)
        if current_action is None:
            current_action = np.zeros(len(self.model.actuators()), dtype=np.float32)
        else:
            current_action = np.asarray(current_action, dtype=np.float32)

        return np.concatenate(
            [dof_values, dof_vels, head_ori, head_ang_vel, feet_rel, current_action],
            dtype=np.float32,
        ).copy()

    def custom_reward(self):
        self._update_rwd_dict()
        return float(np.sum(list(self.rwd_dict.values())))

    def _update_rwd_dict(self):
        vel = float(self.model_velocity())
        height = float(self.model.com_pos().y)
        if vel <= self.target_vel:
            vel_reward = vel / self.target_vel
        else:
            overspeed = vel - self.target_vel
            vel_reward = 1.0 - (overspeed / self.target_vel)
        vel_reward = float(np.clip(vel_reward, -1.5, 1.0))
        if vel > self.target_vel:
            overshoot = (vel - self.target_vel) / self.target_vel
            vel_reward -= 0.3 * (overshoot ** 2)
            vel_reward = float(np.clip(vel_reward, -1.5, 1.0))

        pelvis_tilt = float(self.model.dof_position_array()[self.pelvis_tilt_idx])
        upright_penalty = pelvis_tilt ** 2
        height_reward = float(np.clip((height - 0.50) / 0.40, 0.0, 1.0))
        smooth_penalty = float(np.mean(np.square(self.current_action - self.prev_action)))

        right_effort = float(
            np.abs(self.current_action[0]) + np.abs(self.current_action[1])
        )
        left_effort = float(
            np.abs(self.current_action[2]) + np.abs(self.current_action[3])
        )
        self._right_effort_sum += right_effort
        self._left_effort_sum += left_effort
        steps_so_far = max(1, self.steps)
        mean_right = self._right_effort_sum / float(steps_so_far)
        mean_left = self._left_effort_sum / float(steps_so_far)
        asymmetry_penalty = abs(mean_right - mean_left)

        current_grf = float(self.model.contact_load())
        prev_grf = getattr(self, "_prev_grf", current_grf)
        impact_jerk = abs(current_grf - prev_grf)
        self._prev_grf = current_grf
        leg_symmetry_term = getattr(self, "leg_symmetry_coeff", 0.0) * asymmetry_penalty

        limit_proximity_term = getattr(self, "limit_proximity_coeff", 0.0) * self._joint_limit_proximity()

        self.rwd_dict = {
            "velocity": self.vel_coeff * vel_reward,
            "height": self.height_coeff * height_reward,
            "upright": -self.upright_coeff * upright_penalty,
            "grf": self.grf_coeff * self._grf(),
            "smooth": self.smooth_coeff * smooth_penalty,
            "joint_limit": self.joint_limit_coeff * self._joint_limit_torques(),
            "self_contact": self.self_contact_coeff * self._get_self_contact(),
            "asymmetry": -getattr(self, "asymmetry_coeff", 0.5) * asymmetry_penalty,
            "leg_symmetry": -leg_symmetry_term,
            "limit_proximity": -limit_proximity_term,
            "impact": -getattr(self, "impact_coeff", 0.02) * impact_jerk,
        }
        return self.rwd_dict

    def _grf(self):
        grf = float(self.model.contact_load())
        return max(0.0, grf - 1.2)

    def _joint_limit_torques(self):
        ignored = {"ankle_r", "ankle_l", "back"}
        joints = [j for j in self.model.joints() if j.name() not in ignored]
        if not joints:
            return 0.0
        return float(np.mean([np.mean(np.abs(j.limit_torque().array())) for j in joints]))

    def _joint_limit_proximity(self):
        ignored = {"ankle_r", "ankle_l", "back"}
        proximities = []
        for j in self.model.joints():
            if j.name() in ignored:
                continue
            try:
                for dof in j.dofs():
                    pos = dof.pos()
                    lo, hi = dof.range()
                    if hi > lo:
                        rel = (pos - lo) / (hi - lo)
                        dist_to_edge = min(rel, 1.0 - rel)
                        proximities.append(max(0.0, 0.1 - dist_to_edge) / 0.1)
            except Exception:
                continue
        if not proximities:
            return 0.0
        return float(np.mean(proximities))

    def _get_self_contact(self):
        ignored = {"calcn_r", "calcn_l"}
        contact_force = np.sum(
            [np.abs(b.contact_force().array()) for b in self.model.bodies() if b.name() not in ignored]
        )
        return float(np.clip(contact_force, 0.0, 100.0) / 100.0)

    def _get_done(self):
        com_height = float(self.model.com_pos().y)
        head_height = float(self.head_body.com_pos().y)

        fall = com_height < self.min_com_height or head_height < self.min_head_height
        current_time = float(self.model.time())

        if fall:
            if self.fall_time < 0.0:
                self.fall_time = current_time
            if current_time - self.fall_time >= self.fall_recovery_time:
                return True
        else:
            self.fall_time = -1.0
        raw_speed = abs(float(self.model_velocity()))
        if raw_speed > self.MAX_SANE_VELOCITY or not np.isfinite(raw_speed):
            self._blew_up = True
            return True

        if self.steps >= self._max_episode_steps:
            return True

        return False

    def _apply_termination_cost(self, reward, done):
        if done and self._blew_up:
            reward -= self.BLOWUP_PENALTY
        return reward

    @property
    def horizon(self):
        return self._max_episode_steps