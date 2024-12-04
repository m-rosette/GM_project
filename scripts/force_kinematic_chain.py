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
                 damping = None,
                 start_config = None):

        """Initialize a kinematic chain augmented with Jacobian methods"""

        # Call the constructor for the base KinematicChain class
        super().__init__(links, joint_axes)
        
        self.stiffnesses = np.array(stiffnesses) if stiffnesses else np.zeros(len(joint_axes))
        self.damping = np.array(damping) if damping else np.zeros(len(joint_axes))

        self.start_config = start_config
    
    def response_forces(self, F_E):
        F_alpha = np.matmul(F_E, self.Jacobian_Ad_inv(self.dof, 'world'))    
        return (F_alpha)
    
    def response_forces_with_dynamics(self, F_E, desired_config, joint_vel):
        """
        Map the end-effector force F_E to joint torques while considering the system's
        stiffness and damping reaction forces.
        """
        # Inverse dynamics to compute joint space forces from end-effector force
        F_alpha = np.matmul(F_E, self.Jacobian_Ad_inv(self.dof, 'world'))

        # Reaction forces due to deviation from the current joint configuration
        configuration_deviation = -self.stiffnesses * (desired_config - self.start_config)
        damping_reaction = -self.damping * joint_vel

        # Total reaction force: static (stiffness) + dynamic (damping)
        joint_reactions = configuration_deviation + damping_reaction

        # Combine the joint reaction forces with external forces
        return F_alpha + joint_reactions
        
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
        Compute the total force at the end effector considering both the control
        and the system's resistance to configuration changes.
        """
        # Control torques
        joint_torques = self.compute_joint_torques(desired_angles, joint_velocities)
        
        # Reaction forces (stiffness and damping)
        reaction_forces = self.response_forces_with_dynamics(np.zeros_like(joint_torques), desired_angles, joint_velocities)
        
        # Total joint torques (control + reaction)
        total_joint_torques = joint_torques + reaction_forces

        # Compute the Jacobian at the current configuration
        J_inv = np.linalg.pinv(self.Jacobian_Ad_inv(self.dof, 'world'))
        
        # Optional Regularization Step (if needed for ill-conditioned Jacobian)
        epsilon = 1e-6  # Regularization parameter
        J_regularized = np.linalg.pinv(J_inv @ J_inv.T + epsilon * np.eye(J_inv.shape[1])) @ J_inv

        # Compute end-effector force using the regularized Jacobian
        F_end_effector = J_regularized @ total_joint_torques
        return F_end_effector
    
    def calculate_end_effector_forces_over_trajectory(self, trajectory, joint_velocities):
        """
        Calculate the resultant end-effector forces along a kinematic chain trajectory.
        
        Args:
            trajectory (numpy.ndarray): Joint angle configurations over time (shape: dof x n_frames).
            joint_velocities (numpy.ndarray): Joint velocities over time (shape: dof x n_frames).
            
        Returns:
            numpy.ndarray: End-effector forces at each time step (shape: 3 x n_frames for 3D force vectors).
        """
        n_frames = trajectory.shape[1]
        end_effector_forces = np.zeros((3, n_frames))  # Assuming 3D forces

        for i in range(n_frames):
            # Set the chain to the current configuration
            self.set_configuration(trajectory[:, i])
            
            # Compute the end-effector force
            F_end_effector = self.equilibrium_end_effector_force(
                desired_angles=trajectory[:, i],
                joint_velocities=joint_velocities[:, i]
            )
            
            # Store the result
            end_effector_forces[:, i] = F_end_effector

        return end_effector_forces
    

    
    def compute_joint_torques_from_forces(self, end_effector_forces):
        """
        Compute joint torques to recreate specified end-effector forces.
        
        Args:
            end_effector_forces (numpy.ndarray): Desired forces at the end-effector.

        Returns:
            numpy.ndarray: Joint torques to recreate the forces.
        """
        J = self.Jacobian_Ad_inv(self.dof, 'world')  # Jacobian for current configuration
        J_transpose = J.T
        joint_torques = J_transpose @ end_effector_forces
        return joint_torques
    
    def compute_joint_torques_for_trajectory(self, end_effector_trajectory, joint_trajectory):
        """
        Compute joint torques over a trajectory.

        Args:
            end_effector_trajectory (list): Desired end-effector forces/positions as an array of arrays.
            joint_trajectory (list): Corresponding joint configurations.

        Returns:
            numpy.ndarray: Joint torques over the trajectory.
        """
        num_steps = len(end_effector_trajectory)
        joint_torques_trajectory = []

        for i in range(num_steps):
            # Set the current configuration
            self.set_configuration(joint_trajectory[i])
            # Compute joint torques for the given end-effector forces
            torques = self.compute_joint_torques_from_forces(end_effector_trajectory[i])
            joint_torques_trajectory.append(torques)

        return np.array(joint_torques_trajectory)

    def compute_inverse_dynamics_torques(self, end_effector_positions, end_effector_velocities, end_effector_forces):
        """
        Compute the required joint torques for a trajectory of desired end-effector positions and forces.
        
        Args:
            end_effector_positions (list): Desired positions of the end-effector.
            end_effector_velocities (list): Desired velocities of the end-effector.
            end_effector_forces (list): Desired forces at the end-effector.

        Returns:
            numpy.ndarray: Joint torques over the trajectory.
        """
        num_steps = len(end_effector_positions)
        joint_torques = []

        for i in range(num_steps):
            # Calculate the current joint angles using inverse kinematics
            current_angles = self.inverse_kinematics(end_effector_positions[i])
            self.set_configuration(current_angles)
            # Map end-effector forces to joint torques
            torques = self.compute_joint_torques_from_forces(end_effector_forces[i])
            joint_torques.append(torques)

        return np.array(joint_torques)

    
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