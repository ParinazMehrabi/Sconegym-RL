"""
Training script for H0918 ankle-locked torque RL.

This file registers the custom SCONE Gym environment
and starts DEP-RL training using the provided configuration.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

import sconegym_ext
import deprl

CONFIG_FILE = (
    "configs/"
    "sconewalk_h0918_ankles_locked.yaml"
)


def main():

    print("=" * 60)
    print("H0918 ankle-locked torque RL training")
    print("=" * 60)

    print("Using config:")
    print(CONFIG_FILE)

    deprl.run(CONFIG_FILE)


if __name__ == "__main__":
    main()