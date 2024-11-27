import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt


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
        

class TetheredManipulationWithForces:
    def __init__(self, controllable_chain, passive_chain, G, stiffness, damping):
        """Initialize the Tethered Manipulation with controllable and passive chains, stiffness, and damping"""
        self.controllable_chain = controllable_chain
        self.passive_chain = passive_chain
        self.G = G
        self.stiffness = stiffness
        self.damping = damping

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

    def passive_joint_forces(self, theta3, theta3_desired, theta3_dot):
        """Compute the required forces for the passive joint"""
        force = self.stiffness * (theta3_desired - theta3) + self.damping * theta3_dot
        return force

    def error_function(self, deltas, theta1, theta2, theta3, theta3_desired, theta3_dot):
        """Error function to minimize the distance between end effectors and apply forces"""
        delta_theta1, delta_theta2 = deltas  # Control inputs for Chain 1
        theta1_new = theta1 + delta_theta1
        theta2_new = theta2 + delta_theta2
        
        # Compute the end effector positions
        x1_new = self.fk_chain1(theta1_new, theta2_new)
        x2_desired = self.fk_chain2(theta3_desired)
        
        # Rigid attachment constraint: minimize distance between end effectors
        error = np.linalg.norm(x1_new - x2_desired)
        
        return error

    def control_law_with_forces_optimized(self, theta1, theta2, theta3, theta3_dot, theta3_desired):
        """Control law using optimization to minimize the error and account for forces"""
        # Initial guess for deltas
        initial_guess = [0.0, 0.0]
        
        # Optimization to find the deltas
        result = minimize(
            self.error_function, initial_guess, args=(theta1, theta2, theta3, theta3_desired, theta3_dot),
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