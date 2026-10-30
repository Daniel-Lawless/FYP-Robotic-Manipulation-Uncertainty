"""Palm and finger-pad contact measurements during a grasp, from MuJoCo's own geometry.

Two kinds of measurement are kept apart:

- actual contacts: entries in MuJoCo's contact list for the live simulation, with the
  normal force each one carries;
- geometric clearance: the signed palm-block distance from MuJoCo's collider, computed
  on a copy of the model with a detection margin on the palm. Within that margin the
  collider reports the pair with its true distance (negative means penetration). Up to
  a 10 cm margin, penetration depths match the live simulation's own contacts to within
  0.15 mm; at 50 cm they drift by up to 0.3 mm, while separations agree at any margin.
  So the smallest margin that reports the pair is used: 5 cm, then any wider ones.

mj_geomDistance is not used: for the Panda palm mesh against a box it returns spurious
zeros while the true gap is several centimetres.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robot_uncertainty.sim.primitives import GraspResult
from robot_uncertainty.sim.tabletop import TabletopEnv, grasp_block_from_above

PALM = "gripper0_right_hand_collision"
PADS = ("gripper0_right_finger1_pad_collision", "gripper0_right_finger2_pad_collision")


@dataclass(frozen=True)
class ContactSample:
    """Contact state after one control step."""

    palm_clearance: float  # metres; math.inf if beyond the widest probe margin
    palm_contact: bool  # palm and block in MuJoCo's contact list
    palm_force: float  # total normal force, N
    pad_forces: tuple[float, float]  # total normal force on each pad, N
    pad_points: tuple[np.ndarray, np.ndarray]  # each pad's contact points, block frame


class ContactProbe:
    """Measures palm clearance and pad contacts against one block of a TabletopEnv."""

    def __init__(self, env: TabletopEnv, block: str, margins: tuple[float, ...] = (0.05,)):
        self.env = env
        model = env.sim.model._model
        (block_geom,) = env.blocks[block].contact_geoms
        self.block = model.geom(block_geom).id
        self.palm = model.geom(PALM).id
        self.pads = tuple(model.geom(pad).id for pad in PADS)
        self.probes = []
        for margin in sorted(margins):
            probe_model = copy.deepcopy(model)
            probe_model.geom_margin[self.palm] = margin
            self.probes.append((probe_model, mujoco.MjData(probe_model)))

    def palm_clearance(self) -> float:
        """Signed palm-block distance in metres, or math.inf beyond the widest margin."""
        pair = {self.palm, self.block}
        for probe_model, probe_data in self.probes:
            probe_data.qpos[:] = self.env.sim.data.qpos
            mujoco.mj_kinematics(probe_model, probe_data)
            mujoco.mj_collision(probe_model, probe_data)
            found = [
                c.dist for c in probe_data.contact[: probe_data.ncon] if {c.geom1, c.geom2} == pair
            ]
            if found:
                return min(found)
        return math.inf

    def sample(self) -> ContactSample:
        model, data = self.env.sim.model._model, self.env.sim.data._data
        block_pos = data.geom_xpos[self.block]
        block_rot = data.geom_xmat[self.block].reshape(3, 3)
        palm_force, palm_contact = 0.0, False
        pad_forces, pad_points = [0.0, 0.0], [[], []]
        force = np.zeros(6)
        for i, c in enumerate(data.contact[: data.ncon]):
            pair = {c.geom1, c.geom2}
            if self.block not in pair:
                continue
            other = (pair - {self.block}).pop()
            if other == self.palm:
                mujoco.mj_contactForce(model, data, i, force)
                palm_contact, palm_force = True, palm_force + abs(force[0])
            elif other in self.pads:
                k = self.pads.index(other)
                mujoco.mj_contactForce(model, data, i, force)
                pad_forces[k] += abs(force[0])
                pad_points[k].append((c.pos - block_pos) @ block_rot)
        return ContactSample(
            palm_clearance=self.palm_clearance(),
            palm_contact=palm_contact,
            palm_force=palm_force,
            pad_forces=(pad_forces[0], pad_forces[1]),
            pad_points=tuple(np.reshape(p, (-1, 3)) for p in pad_points),
        )


def record_grasp(
    env: TabletopEnv,
    block: str,
    grasp_height: float | None = None,
    margins: tuple[float, ...] = (0.05,),
) -> tuple[GraspResult, list[ContactSample]]:
    """Run grasp_block_from_above, sampling contacts after every control step.

    Sample i follows control step i + 1; result.phases gives each phase's step count,
    in order, so samples can be split by phase. The env's step is wrapped only for the
    duration of the grasp.
    """
    probe = ContactProbe(env, block, margins)
    samples: list[ContactSample] = []
    step = env.step

    def recording_step(action):
        out = step(action)
        samples.append(probe.sample())
        return out

    env.step = recording_step
    try:
        result = grasp_block_from_above(env, block, grasp_height)
    finally:
        del env.step
    return result, samples
