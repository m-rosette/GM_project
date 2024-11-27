import numpy as np
from matplotlib import pyplot as plt
from traj_gen import TrajectoryGenerator
from diff_kinematic_chain import DiffKinematicChain

import sys
sys.path.append('../')
from geomotion import rigidbody as rb

# Set the group as SE2 from rigidbody
G = rb.SE2

if __name__ == "__main__":
    # Define manipulator parameters
    links = [G.element([1, 0, 0]), G.element([1, 0, 0])]
    joint_axes = [G.Lie_alg_vector([0, 0, 1])] * len(links)

    # Create a kinematic chain object
    kinematic_chain = DiffKinematicChain(links, joint_axes)

    # Define trajectory generation parameters
    T = 2.0  # Duration of the trajectory in seconds
    num_points = 100  # Number of trajectory points
    initial_thetas = [np.pi / 2, -np.pi / 2]  # Initial joint angles
    final_thetas = [0, np.pi]  # Final joint angles

    # Initialize the trajectory generator
    trajectory_generator = TrajectoryGenerator(T, num_points)

    # Generate joint trajectories
    times, positions, velocities, accelerations, joint_names = trajectory_generator.generate_joint_trajectories(
        initial_thetas, final_thetas
    )

    # Initialize storage for end-effector velocities
    end_effector_velocities = []

    # Loop through trajectory points to compute end-effector velocities
    for q, dq in zip(zip(*positions), zip(*velocities)):
        # Update joint angles and link positions in the kinematic chain
        kinematic_chain.set_configuration(q)

        # Calculate the Jacobian for the end effector (last link)
        jacobian = kinematic_chain.Jacobian_Ad_inv(len(links), output_frame='world')

        # Compute end-effector velocity
        v_end_effector = jacobian @ dq
        end_effector_velocities.append(v_end_effector)

    # Convert end-effector velocities to a numpy array for easier manipulation
    end_effector_velocities = np.array(end_effector_velocities)

    # Plot end-effector velocities
    plt.figure(figsize=(8, 6))
    plt.plot(times[0], end_effector_velocities[:, 0], label="End Effector X Velocity")
    plt.plot(times[0], end_effector_velocities[:, 1], label="End Effector Y Velocity")
    plt.plot(times[0], end_effector_velocities[:, 2], label="End Effector Angular Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.title("End Effector Velocities")
    plt.legend()
    plt.grid()
    plt.show()
