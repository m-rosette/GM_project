import sys
sys.path.append('../')
from geomotion import rigidbody as rb
from kinematic_chain import KinematicChain
from diff_kinematic_chain import DiffKinematicChain
from traj_gen import TrajectoryGenerator
from tethered_manipulation import TetheredManipulation, TetheredManipulationWithForces

import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# Set the group as SE2 from rigidbody
G = rb.SE2


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