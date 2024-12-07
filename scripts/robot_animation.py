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
        
        axes.append(self.fig.add_subplot(1, 2, 1))
        
        for i in range(self.dof):
            idx = (i+1)*2
            axes.append(self.fig.add_subplot(self.dof, 2, idx))
            
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
        
        self.ax[0].set_xlim([-3,9])
        self.ax[0].set_ylim([-2,6])

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
            
            # Adding joint labels on the right side of the subplot
            self.ax[j+1].text(1.05, 0.5, f'Joint {j+1}', transform=self.ax[j+1].transAxes,
                             fontsize=12, verticalalignment='center', rotation=90, ha='left')

            # Set x-axis labels only at the bottom subplot
            if j == self.dof - 1:
                self.ax[j+1].tick_params(axis='x', labelbottom=True)
            else:
                self.ax[j+1].tick_params(axis='x', labelbottom=False)

        # Add x-axis label to the bottom-right subplot
        self.ax[-1].set_xlabel('Time (s)')
        self.ax[0].set_xlabel('X')
        self.ax[0].set_ylabel('Y')

        plt.tight_layout()  

        return lines

            
if __name__ == "__main__":
    # System parameters
    num_links = 5
    link_len = 1
    stiffness = 5
    damping = 0.1
    num_frames = 50
    duration = 5

    # # Three link system
    # active_base_offset = np.array([2.05, 0]) # used for three link system
    # start_config_active = [np.pi/4, np.pi/2, np.pi/4]
    # start_config_passive = [3*np.pi/4, -np.pi/2, -np.pi/4]
    # desired_config_passive = [np.pi/4, -0.05, -0.05]

    # Five link system
    active_base_offset = np.array([4.05, 0])
    start_config_active = [np.pi/4, np.pi/4, np.pi/4, np.pi/4, 0.05]
    start_config_passive = [3*np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4, -0.05]
    desired_config_passive = [np.pi/2, -np.pi/8, -np.pi/8, -np.pi/4, -np.pi/4]

    # Create a list of links, all extending in the x direction with different lengths
    links1 = [G.element([link_len, 0, 0])] * num_links

    # Create a list of three joint axes, all in the rotational direction
    joint_axes1 = [G.Lie_alg_vector([0, 0, 1])] * num_links

    # Stiffness and damping for each joint
    stiffnesses = [stiffness] * num_links
    damping = [damping] * num_links

    # Create the controllable kinematic chain
    kc1 = ForceKinematicChain(links1, joint_axes1)
    kc1.set_configuration(start_config_active)
    
    # Create another kinematic chain
    kc2 = ForceKinematicChain(links1, joint_axes1, stiffnesses, damping, start_config_passive)
    kc2.set_configuration(start_config_passive)

    robo_animator = RobotAnimation(kc1, kc2, duration, num_frames, start_config_passive, desired_config_passive, start_config_active, draw_base_offset=active_base_offset)
    
    ani = FuncAnimation(robo_animator.fig, robo_animator.animate, frames=num_frames, interval=20, blit=True)
    # plt.show()
    ani.save("test_IK.gif",fps=20)