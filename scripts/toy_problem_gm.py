import sys
sys.path.append('../')
from geomotion import rigidbody as rb

import numpy as np
from scipy.optimize import minimize
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
        
class DiffKinematicChain(KinematicChain):
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


class TetheredManipulation:
    def __init__(self, controllable_chain, passive_chain, G):
        """Initialize the Tethered Manipulation with controllable and passive chains"""
        self.controllable_chain = controllable_chain
        self.passive_chain = passive_chain
        self.G = G

    def fk_chain1(self, theta1, theta2):
        """Forward kinematics for the controllable chain (Chain 1)"""
        self.controllable_chain.set_configuration([theta1, theta2])
        # Return the end effector position (last link position)
        return self.controllable_chain.link_positions[-1].value[:2]

    def fk_chain2(self, theta3_desired):
        """Forward kinematics for the passive chain (Chain 2)"""
        self.passive_chain.set_configuration([theta3_desired])
        # Return the end effector position (last link position)
        return self.passive_chain.link_positions[-1].value[:2]

    def error_function(self, deltas, theta1, theta2, theta3_desired):
        """Error function to minimize the distance between end effectors"""
        delta_theta1, delta_theta2 = deltas  # Control inputs for Chain 1
        theta1_new = theta1 + delta_theta1
        theta2_new = theta2 + delta_theta2
        
        # Compute the end effector positions
        x1_new = self.fk_chain1(theta1_new, theta2_new)
        x2_desired = self.fk_chain2(theta3_desired)
        
        # Rigid attachment constraint: minimize distance between end effectors
        error = np.linalg.norm(x1_new - x2_desired)
        return error

    def control_law_with_rigid_attachment_optimized(self, theta1, theta2, theta3_desired):
        """Control law using optimization to minimize the rigid attachment error"""
        # Initial guess for deltas
        initial_guess = [0.0, 0.0]
        
        # Optimization to find the deltas
        result = minimize(
            self.error_function, initial_guess, args=(theta1, theta2, theta3_desired),
            method='BFGS'
        )
        
        # Extract the optimized deltas and the final error
        delta_theta1, delta_theta2 = result.x
        return np.array([delta_theta1, delta_theta2]), result.fun
    
    def plot_chains(self, theta1, theta2, theta3, theta3_desired, optimized_theta1=None, optimized_theta2=None):
        """Plot the initial and final configurations of both chains"""
        
        # Plot the initial configuration (before optimization)
        self.controllable_chain.set_configuration([theta1, theta2])
        self.passive_chain.set_configuration([theta3])
        
        # Create the plot
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Plot Chain 1 (controllable chain) in blue
        self.controllable_chain.draw(ax, color='blue') #, label='Controllable Chain (Initial)')

        # Plot Chain 2 (passive chain) in red
        self.passive_chain.draw(ax, color='magenta') #, label='Passive Chain (Desired)')

        # If we have optimized values, plot the optimized controllable chain
        if optimized_theta1 is not None and optimized_theta2 is not None:
            self.controllable_chain.set_configuration([optimized_theta1, optimized_theta2])
            self.controllable_chain.draw(ax, color='green') #, label='Controllable Chain (Optimized)')
            
            self.passive_chain.set_configuration([theta3_desired])
            self.passive_chain.draw(ax, color='red')

        # Add labels and legend
        ax.set_title('Kinematic Chains Configuration')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(['Controllable Chain (Initial)', 'Passive Chain (Initial)', 'Controllable Chain (Final)', 'Passive Chain (Final)'])

        # Show the plot
        plt.axis('equal')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    thetas = [np.pi / 2, -np.pi / 2, np.pi /2]

    # Controllable chain setup
    controllable_links = [G.element([1, 0, 0]), G.element([1, 0, 0])]
    controllable_axes = [G.Lie_alg_vector([0, 0, 1])] * 2
    controllable_chain = KinematicChain(controllable_links, controllable_axes)
    controllable_chain.set_configuration(thetas[:2])  # Example angles

    # Passive chain setup
    passive_links = [G.element([1, 0, 0])]
    passive_axes = [G.Lie_alg_vector([0, 0, 1])] * 1
    passive_chain = KinematicChain(passive_links, passive_axes)

    # Set the base of the passive chain at (1, 0)
    passive_chain.set_base_transform(G.element([1, 0, 0]))

    # Set the configuration of the passive chain
    passive_chain.set_configuration([thetas[2]])  # Example angles

    # Example desired position for the passive chain (end effector)
    theta3_desired = - 4 * np.pi / 3

    # Create the TetheredManipulation instance
    tethered_manipulator = TetheredManipulation(controllable_chain, passive_chain, G)

    # Call the control law with optimization to minimize the error
    deltas, final_error = tethered_manipulator.control_law_with_rigid_attachment_optimized(thetas[0], thetas[1], theta3_desired)
    
    print("Optimized Control Inputs (deltas):", deltas)
    print("Final Error (rigid attachment):", final_error)

    # Plot the optimized configuration of both chains
    tethered_manipulator.plot_chains(thetas[0], thetas[1], thetas[2], theta3_desired, optimized_theta1=thetas[0]+deltas[0], optimized_theta2=thetas[1]+deltas[1])