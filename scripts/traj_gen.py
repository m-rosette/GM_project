import numpy as np
from matplotlib import pyplot as plt


class TrajectoryGenerator:
    """Class for generating quintic joint trajectories"""

    def __init__(self, duration, num_points=100):
        self.duration = duration
        self.num_points = num_points

    def quintic_trajectory(self, q0, qT):
        """Generate a quintic trajectory for a single joint"""
        t = np.linspace(0, self.duration, self.num_points)
        # Coefficients for quintic trajectory
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = 10 * (qT - q0) / self.duration**3
        a4 = -15 * (qT - q0) / self.duration**4
        a5 = 6 * (qT - q0) / self.duration**5

        # Compute trajectory
        q = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
        dq = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
        ddq = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
        return t, q, dq, ddq

    def generate_joint_trajectories(self, joint_angles, final_angles):
        """
        Generate quintic trajectories for all joints.

        Args:
            joint_angles (list of float): Initial angles of the joints.
            final_angles (list of float): Final angles of the joints.

        Returns:
            tuple: NumPy arrays of times, positions, velocities, accelerations for all joints,
                and a list of joint names.
        """
        num_joints = len(joint_angles)
        num_points = self.num_points
        times = np.zeros((num_joints, num_points))
        positions = np.zeros((num_joints, num_points))
        velocities = np.zeros((num_joints, num_points))
        accelerations = np.zeros((num_joints, num_points))
        joint_names = [f"Joint {i + 1}" for i in range(num_joints)]

        for i, (q0, qT) in enumerate(zip(joint_angles, final_angles)):
            t, q, dq, ddq = self.quintic_trajectory(q0, qT)
            times[i, :] = t
            positions[i, :] = q
            velocities[i, :] = dq
            accelerations[i, :] = ddq

        return times, positions, velocities, accelerations, joint_names

    def plot_joint_trajectories(self, times, positions, velocities, accelerations, joint_names):
        """
        Plot position, velocity, and acceleration profiles for multiple joints on the same plots.

        Args:
            times (list of arrays): List of time arrays for each joint.
            positions (list of arrays): List of position arrays for each joint.
            velocities (list of arrays): List of velocity arrays for each joint.
            accelerations (list of arrays): List of acceleration arrays for each joint.
            joint_names (list of str): List of names for each joint.
        """
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        plt.rcParams.update({'font.size': 12})

        # Plot positions
        for t, q, name in zip(times, positions, joint_names):
            axs[0].plot(t, q, label=f"{name}")
        axs[0].set_ylabel("Position (rad)", fontsize=14)
        axs[0].grid()
        axs[0].legend()

        # Plot velocities
        for t, dq, name in zip(times, velocities, joint_names):
            axs[1].plot(t, dq)#, label=f"{name} Velocity")
        axs[1].set_ylabel("Velocity (rad/s)", fontsize=14)
        axs[1].grid()
        axs[1].legend()

        # Plot accelerations
        for t, ddq, name in zip(times, accelerations, joint_names):
            axs[2].plot(t, ddq)#, label=f"{name} Acceleration")
        axs[2].set_ylabel("Acceleration (rad/s^2)", fontsize=14)
        axs[2].set_xlabel("Time (s)", fontsize=14)
        axs[2].grid()
        axs[2].legend()

        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    # Define parameters for the trajectory
    T = 2.0  # Duration of the trajectory in seconds
    num_points = 100  # Number of points in the trajectory
    initial_thetas = [np.pi / 2, -np.pi / 2, 0]  # Initial joint angles
    final_thetas = [0, np.pi, np.pi/2]  # Final joint angles

    # Initialize the TrajectoryGenerator
    trajectory_generator = TrajectoryGenerator(T, num_points)

    # Generate trajectories for all joints
    times, positions, velocities, accelerations, joint_names = trajectory_generator.generate_joint_trajectories(
        initial_thetas, final_thetas
    )

    # Plot all joint trajectories
    trajectory_generator.plot_joint_trajectories(times, positions, velocities, accelerations, joint_names)