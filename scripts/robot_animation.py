# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:30:55 2024

@author: mcrav
"""

from diff_kinematic_chain import DiffKinematicChain
from traj_gen import TrajectoryGenerator
from force_kinematic_chain import ForceKinematicChain

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from geomotion import rigidbody as rb

G = rb.SE2

matplotlib.use('Qt5Agg')

COLOR = {
    "dark_blue": [51 / 255, 34 / 255, 136 / 255],
    "dark_green": [17 / 255, 119 / 255, 51 / 255],
    "teal": [68 / 255, 170 / 255, 153 / 255],
    "light_blue": [136 / 255, 204 / 255, 238 / 255],
    "yellow": [221 / 255, 204 / 255, 119 / 255],
    "salmon": [204 / 255, 102 / 255, 119 / 255],
    "light_purple": [170 / 255, 68 / 255, 153 / 255],
    "dark_purple": [136 / 255, 34 / 255, 85 / 255],
}

class RobotAnimation:
    
    def __init__(self, active_chain, passive_chain, duration, n_frames, start_config_passive, end_config, start_config_active, draw_base_offset):
        
        self.active_chain = active_chain
        self.passive_chain = passive_chain
        self.duration = duration
        self.traj_generator = TrajectoryGenerator(duration=duration,num_points=n_frames)

        self.draw_base_offset = draw_base_offset
        
        self.times, self.passive_trajectory, self.passive_alpha_dot, _, _ = \
            self.traj_generator.generate_joint_trajectories(start_config_passive, end_config)
        
        self.passive_end_effector_forces = self.passive_chain.calculate_end_effector_forces_over_trajectory( \
            self.passive_trajectory, self.passive_alpha_dot)
        
        self.dof = len(start_config_passive)
        self.n_frames = n_frames
        
        self.fig = plt.figure()
        self.ax = self.create_subplots()

        self.times = self.times[0]
        
        self.v_E_passive = self.passive_chain.traj_to_v(self.passive_trajectory, self.passive_alpha_dot)
        self.active_trajectory = self.active_chain.v_to_traj(self.times, start_config_active, self.v_E_passive)
        
    def create_subplots(self):
        
        axes = []
        
        axes.append(self.fig.add_subplot(1,2,1))
        
        for i in range(self.dof):
            idx = (i+1)*2
            axes.append(self.fig.add_subplot(self.dof,2,idx))
            
        return axes
        

    def animate(self, i):
        
        if i%10 == 0:
            print("{} / {}".format(i,self.n_frames))
        
        lines = []
        
        color_order = list(COLOR.keys())
        
        self.ax[0].clear()
        self.passive_chain.set_configuration(self.passive_trajectory[:,i])
        lines.append(self.passive_chain.draw(self.ax[0], color=COLOR["teal"])[0])
        self.active_chain.set_configuration(self.active_trajectory[:,i])
        lines.append(self.active_chain.draw(self.ax[0], color=COLOR["salmon"], offset=self.draw_base_offset)[0])
        
        self.ax[0].set_xlim([-6,6])
        self.ax[0].set_ylim([-6,6])

        # Add force arrows for passive chain
        end_effector_position = self.passive_chain.link_positions[-1].value[:2]
        force_vector = self.passive_end_effector_forces[:, i]
        
        # Colors and components for x and y components
        components = [("x", force_vector[0], 0, "red"), ("y", 0, force_vector[1], "green")]
        
        # Loop through the components and plot arrows
        for comp_name, force_x, force_y, color in components:
            arrow = self.ax[0].quiver(
                end_effector_position[0],  # Base x position
                end_effector_position[1],  # Base y position
                force_x,                   # x-component of force (for 'x' or 'y' direction)
                force_y,                   # y-component of force (for 'x' or 'y' direction)
                color=color,
                scale=50,  # Adjust as needed
                label=f"Force {comp_name}",
            )
            lines.append(arrow)

        # Add a legend to show force component labels
        self.ax[0].legend(loc="upper right") 
    
        # Update the joint plots
        for j in range(self.dof):
            self.ax[j+1].clear()
            lines.append(self.ax[j+1].plot(self.times, self.passive_trajectory[j,:],
                              color=COLOR[color_order[j]])[0])
            lines.append(self.ax[j+1].plot(self.times[i], self.passive_trajectory[j,i],
                              marker = 'o', color=COLOR[color_order[j]])[0])
                        
            
        return lines
            
if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links1 = [G.element([1, 0, 0]), G.element([1, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes1 = [G.Lie_alg_vector([0, 0, 1])] * 3

    # Stiffness and damping for each joint
    stiffnesses = [5, 5, 5]
    damping = [0.1, 0.1, 0.1]

    # Create the controllable kinematic chain
    # kc1 = DiffKinematicChain(links1, joint_axes1)
    # start_config_active = [np.pi/6, np.pi/6, np.pi/6]
    start_config_active = [np.pi/4, np.pi/2, np.pi/4]
    kc1 = ForceKinematicChain(links1, joint_axes1)
    kc1.set_configuration(start_config_active)
    active_base_offset = np.array([2.05, 0])
    # kc1.set_base_transform(active_base_offset)
    
    # Create another kinematic chain
    # kc2 = DiffKinematicChain(links1, joint_axes1)
    # start_config_passive = [np.pi/6, np.pi/6, np.pi/6]
    # desired_config_passive = [np.pi/2, np.pi/4, np.pi/4]
    start_config_passive = [3*np.pi/4, -np.pi/2, -np.pi/4]
    desired_config_passive = [np.pi/2, -0.05, -np.pi/2]
    kc2 = ForceKinematicChain(links1, joint_axes1, stiffnesses, damping, start_config_passive)
    kc2.set_configuration(start_config_passive)

    num_frames = 30

    robo_animator = RobotAnimation(kc1, kc2, 5, num_frames, start_config_passive, desired_config_passive, start_config_active, draw_base_offset=active_base_offset)
    
    ani = FuncAnimation(robo_animator.fig, robo_animator.animate, frames=num_frames, interval=20, blit=True)
    # plt.show()
    ani.save("test_IK.gif",fps=20)