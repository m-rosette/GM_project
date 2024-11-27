import sys
sys.path.append('../')
from geomotion import rigidbody as rb

import numpy as np
import kinematic_chain as kc
from matplotlib import pyplot as plt

# Set the group as SE2 from rigidbody
G = rb.SE2
        
class DiffKinematicChain(kc.KinematicChain):
    """Kinematic chain augmented with Jacobian methods"""

    def __init__(self, links, joint_axes):
        """Initialize a kinematic chain augmented with Jacobian methods"""
        super().__init__(links, joint_axes) 

        # Create placeholders for the last-calculated Jacobian value and the index it was requested for
        self.last_jacobian = None
        self.jacobian_idx = -1  # Starting with an invalid index as this is just a placeholder

    def Jacobian_Ad_inv(self, link_index, output_frame='body'):
        """Calculate the Jacobian by using the Adjoint_inverse to transfer velocities from the joints to the links"""
        # Construct Jacobian matrix J as an ndarray of zeros with as many rows as the group has dimensions,
        # and as many columns as there are joints
        num_rows = G.element_shape[0]
        num_columns = len(self.joint_angles)
        J = np.zeros((num_rows, num_columns))

        link_positions_with_base = [G.identity_element()] + self.link_positions
        selected_link = self.link_positions[link_index - 1]

        for j in range(1, link_index + 1):
            link_position = link_positions_with_base[j - 1]
            joint_axis = self.joint_axes[j - 1]

            g_rel = link_position.inverse * selected_link
            J_joint = g_rel.Ad_inv(joint_axis)

            if output_frame == 'world':
                J_joint = selected_link.TL(J_joint)
            elif output_frame == 'spatial':
                J_joint = selected_link.Ad(J_joint)

            J[:, j - 1] = J_joint.value[:num_rows]

        self.last_jacobian = J.copy()
        self.jacobian_idx = link_index
        return J

    def Jacobian_Ad(self, link_index, output_frame='body'):
        """Calculate the Jacobian by using the Adjoint to transfer velocities from the joints to the origin"""
        num_rows = G.element_shape[0]
        num_columns = len(self.joint_angles)
        J = np.zeros((num_rows, num_columns))

        link_positions_with_base = [G.identity_element()] + self.link_positions
        selected_link = self.link_positions[link_index - 1]

        for j in range(1, link_index + 1):
            link_position = link_positions_with_base[j - 1]
            joint_axis = self.joint_axes[j - 1]

            J_joint = link_position.Ad(joint_axis)

            if output_frame == 'world':
                J_joint = selected_link.TR(J_joint)
            elif output_frame == 'body':
                J_joint = selected_link.Ad_inv(J_joint)

            J[:, j - 1] = J_joint.value[:num_rows]

        self.last_jacobian = J.copy()
        self.jacobian_idx = link_index
        return J