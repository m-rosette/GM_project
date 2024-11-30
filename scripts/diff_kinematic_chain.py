import sys
sys.path.append('../')
from geomotion import (utilityfunctions as ut, rigidbody as rb)

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
        self.dof = len(links)

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
        
    @property
    def redundant(self):
        return self.dof > 3 #can change to 2 if we want a pin connection as grasp
    
    def IK(self, element):
        
        if self.redundant:
            A = self.Jacobian_Ad_inv(self.dof, 'world')
            b = element.value.transpose()
            joint_velocities, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        else:
            J_inv = np.linalg.pinv(self.Jacobian_Ad_inv(self.dof, 'world'))
            joint_velocities = np.matmul(J_inv, element.value.transpose())
            
        return joint_velocities
    
if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links1 = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes1 = [G.Lie_alg_vector([0, 0, 1])] * 3

    # Create a kinematic chain
    kc1 = DiffKinematicChain(links1, joint_axes1)

    # Set the joint angles to pi/4, -pi/2 and 3*pi/4
    kc1.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi])
    
    print(kc1.IK(G.element([0, 1, 0])))
    
    links2 = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0]), G.element([1, 0, 0])]
    joint_axes2 = [G.Lie_alg_vector([0, 0, 1])] * 4
    kc2 = DiffKinematicChain(links2, joint_axes2)
    kc2.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi, -.5 * np.pi])
    
    print(kc2.IK(G.element([0, 1, 0])))