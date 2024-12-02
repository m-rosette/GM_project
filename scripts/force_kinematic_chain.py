# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:04:52 2024

@author: mcrav
"""

from diff_kinematic_chain import DiffKinematicChain
import numpy as np
from geomotion import rigidbody as rb

G = rb.SE2

class ForceKinematicChain(DiffKinematicChain):

    def __init__(self,
                 links,
                 joint_axes,
                 stiffnesses = None,
                 damping = None):

        """Initialize a kinematic chain augmented with Jacobian methods"""

        # Call the constructor for the base KinematicChain class
        super().__init__(links, joint_axes)
        
        self.stiffnesses = np.array(stiffnesses) if stiffnesses else np.zeros(len(joint_axes))
        self.damping = np.array(damping) if damping else np.zeros(len(joint_axes))
    
    def response_forces(self, F_E):
        F_alpha = np.matmul(F_E, self.Jacobian_Ad_inv(self.dof, 'world'))    
        return (F_alpha)
        
    def compute_joint_torques(self, desired_angles, joint_velocities):
        """
        Calculate joint torques using Proportional-Derivative (PD) control.
        
        Args:
            desired_angles (list or array): Desired joint angles.
            joint_velocities (list or array): Current joint velocities.
        
        Returns:
            numpy.ndarray: Computed joint torques.
        """
        # Convert inputs to NumPy arrays for element-wise operations
        desired_angles = np.array(desired_angles)
        current_angles = np.array(self.joint_angles)
        joint_velocities = np.array(joint_velocities)
        
        # Calculate the position error
        position_error = desired_angles - current_angles

        # Proportional (P) control: Torque proportional to the position error
        P_torque = self.stiffnesses * position_error

        # Derivative (D) control: Torque proportional to the negative velocity
        D_torque = -self.damping * joint_velocities

        # Total torque is the sum of P and D components
        return P_torque + D_torque
    
    def equilibrium_end_effector_force(self, desired_angles, joint_velocities):
        """
        Compute the end-effector force required to conform the chain to the desired configuration.
        """
        joint_torques = self.compute_joint_torques(desired_angles, joint_velocities)
        J_transpose = self.Jacobian_Ad_inv(self.dof, 'world').T
        F_end_effector = np.linalg.pinv(J_transpose) @ joint_torques
        return F_end_effector
    
    
if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links1 = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes1 = [G.Lie_alg_vector([0, 0, 1])] * 3

    # Create a kinematic chain
    kc1 = ForceKinematicChain(links1, joint_axes1)

    # Set the joint angles to pi/4, -pi/2 and 3*pi/4
    kc1.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi])
    
    print(kc1.response_forces(np.array([0.0, 1.0, 0.0])))
    
    links2 = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0]), G.element([1, 0, 0])]
    joint_axes2 = [G.Lie_alg_vector([0, 0, 1])] * 4
    kc2 = ForceKinematicChain(links2, joint_axes2)
    kc2.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi, -.5 * np.pi])
    
    print(kc2.response_forces(np.array([0.0, 1.0, 0.0])))

    # JOINT DYNAMICS
    # Calculate required end-effector forces to reach a desired joint configuration 

    # Stiffness and damping for each joint
    stiffnesses = [10, 15, 20]
    damping = [1, 1.5, 2]

    # Create the kinematic chain
    kc = ForceKinematicChain(links1, joint_axes1, stiffnesses, damping)

    # Set the current configuration and desired configuration
    current_angles = [0.25 * np.pi, -0.5 * np.pi, 0.75 * np.pi]
    desired_angles = [0.1 * np.pi, -0.4 * np.pi, 0.6 * np.pi]
    joint_velocities = [0.0, 0.0, 0.0]
    
    kc.set_configuration(current_angles)
    
    # Compute the required end-effector force
    F_end_effector = kc.equilibrium_end_effector_force(desired_angles, joint_velocities)
    print("Required End-Effector Force:", F_end_effector)