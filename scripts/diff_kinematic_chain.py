import sys
sys.path.append('../')
from geomotion import (utilityfunctions as ut, rigidbody as rb)

from scipy.integrate import solve_ivp
import time
import numpy as np
import kinematic_chain as kc
from matplotlib import pyplot as plt
from scipy import integrate

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
    
    def traj_to_v(self, positions, velocities):
        
        n = velocities.shape[1]
        v = np.zeros([3,n])

        for i in range(n):
            self.set_configuration(positions[:,i])
            J = self.Jacobian_Ad_inv(self.dof, 'world')
            alpha_dot = velocities[:,i]
            v[:,i] = np.matmul(J, alpha_dot)
        
        return v

    def v_to_traj(self, times, initial_config, v, timeout_seconds=10):
        """
        Convert end-effector velocities to joint trajectories using integration, with a timeout.
        If timeout occurs, fallback to a manual integration approach.

        Args:
            times (np.array): Array of time points for integration.
            initial_config (np.array): Initial configuration of the kinematic chain.
            v (np.array): Array of end-effector velocities.
            timeout_seconds (int): Maximum time allowed for solving, in seconds.

        Returns:
            np.array: Joint positions at each time step.
        """
        n = times.shape[0]
        dof = self.dof  # Degrees of freedom of the kinematic chain

        # Initialize storage for joint positions
        positions = np.zeros([dof, n])
        positions[:, 0] = np.array(initial_config)

        def dynamics(t, y):
            """
            Differential equation defining joint velocity dynamics.

            Args:
                t (float): Current time.
                y (np.array): Current joint positions.

            Returns:
                np.array: Joint velocities.
            """
            # Check for timeout during the solver's iterations
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError("ODE solver exceeded the time limit.")

            # Set the current configuration for IK
            self.set_configuration(y.flatten().tolist())

            # Interpolate end-effector velocity for each DoF at time t
            v_t = np.array([np.interp(t, times, v_row) for v_row in v])

            # Compute joint velocities using the inverse kinematics function
            alpha_dot = self.IK(G.element(v_t))
            return alpha_dot

        # Record the start time
        start_time = time.time()

        try:
            # Solve the ODE using solve_ivp
            solution = solve_ivp(
                dynamics,
                [times[0], times[-1]],
                initial_config,
                t_eval=times,
                method='RK23',
            )

            # Check if the solver finished within the allowed time
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"ODE solver exceeded the {timeout_seconds}s time limit.")

            # Store the results
            positions = solution.y

            # Ensure final configuration is set
            self.set_configuration(positions[:, -1])

        except TimeoutError as e:
            print(f"Solver timeout: {e}")
            print("Falling back to manual integration.")

            # Manual integration loop
            velocities = np.zeros([dof, n])  # Optional: Store velocities if needed
            positions[:, 0] = np.array(initial_config)
            self.set_configuration(positions[:, 0])

            for i in range(1, n):
                dt = times[i] - times[i - 1]  # Time step

                # Compute joint velocities using IK
                alpha_dot = self.IK(G.element(v[:, i]))
                velocities[:, i] = alpha_dot

                # Integrate each joint's velocity to update positions
                for j in range(dof):
                    positions[j, i] = positions[j, i - 1] + integrate.simpson([0, alpha_dot[j]], x=[0, dt])

                # Update configuration for the next step
                self.set_configuration(positions[:, i])

        return positions   


if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links1 = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes1 = [G.Lie_alg_vector([0, 0, 1])] * 3

    # Create a kinematic chain
    kc1 = DiffKinematicChain(links1, joint_axes1)

    # # Set the joint angles to pi/4, -pi/2 and 3*pi/4
    # kc1.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi])
    
    # links2 = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0]), G.element([1, 0, 0])]
    # joint_axes2 = [G.Lie_alg_vector([0, 0, 1])] * 4
    # kc2 = DiffKinematicChain(links2, joint_axes2)
    # kc2.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi, -.5 * np.pi])


    # End effector vel test
    test_config = [0.25 * np.pi, -0.5 * np.pi, 0.75 * np.pi]
    test_velocity = np.array([1, 0, 0])  # Example velocity in the world frame
    kc1.set_configuration(test_config)
    J = kc1.Jacobian_Ad_inv(kc1.dof, 'world')
    joint_velocities = np.linalg.pinv(J) @ test_velocity  # Solve for joint velocities

    ee_velocity = J @ joint_velocities
    print("Expected end-effector velocity:", test_velocity)
    print("Computed end-effector velocity:", ee_velocity)
    print("Error:", np.linalg.norm(test_velocity - ee_velocity))

    # Joint vel test
    test_joint_vel = [0.1, 0.1, 0.1]
    test_config = [0.25 * np.pi, -0.5 * np.pi, 0.75 * np.pi]
    kc1.set_configuration(test_config)
    J = kc1.Jacobian_Ad_inv(kc1.dof, 'world')

    end_eff_vel = J @ test_joint_vel

    J_inv = np.linalg.pinv(J)
    joint_velocities = np.matmul(J_inv, end_eff_vel)
    print(joint_velocities)


