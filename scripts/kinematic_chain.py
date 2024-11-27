import sys
sys.path.append('../')
from geomotion import rigidbody as rb

import numpy as np
from matplotlib import pyplot as plt

# Set the group as SE2 from rigidbody
G = rb.SE2

class KinematicChain:
    """Simple implementation of a kinematic chain"""

    def __init__(self, links, joint_axes, base_transform=None):
        self.links = links
        self.joint_axes = joint_axes
        self.joint_angles = np.zeros(len(joint_axes))
        self.joint_transforms = [G.identity_element()] * len(joint_axes)
        self.link_positions = [G.identity_element()] * len(links)
        self.base_transform = base_transform if base_transform else G.identity_element()

    def set_base_transform(self, base_transform):
        """Set the base transform of the chain"""
        self.base_transform = base_transform

    def set_configuration(self, joint_angles):
        self.joint_angles = joint_angles
        for i, alpha in enumerate(joint_angles):
            self.joint_transforms[i] = (alpha * self.joint_axes[i]).exp_L

        # Update link positions, starting with the base transform
        self.link_positions[0] = self.base_transform * self.joint_transforms[0] * self.links[0]
        for i in range(1, len(self.link_positions)):
            self.link_positions[i] = (
                self.link_positions[i - 1] * self.joint_transforms[i] * self.links[i]
            )

        return self.link_positions

    def draw(self, ax, color="b"):
        x = [self.base_transform.value[0]]
        y = [self.base_transform.value[1]]

        for l in self.link_positions:
            x.append(l.value[0])
            y.append(l.value[1])
        ax.plot(x, y, marker="o", linestyle="-", color=color)
        ax.set_aspect("equal")