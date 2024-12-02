# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:30:55 2024

@author: mcrav
"""

from diff_kinematic_chain import DiffKinematicChain
from traj_gen import TrajectoryGenerator

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
    
    def __init__(self, active_chain, passive_chain, duration, n_frames, start_config, end_config):
        
        self.active_chain = active_chain
        self.passive_chain = passive_chain
        self.duration = duration
        self.traj_generator = TrajectoryGenerator(duration=duration,num_points=n_frames)
        
        self.times, self.passive_trajectory, self.passive_alpha_dot, _, _ = \
        self.traj_generator.generate_joint_trajectories(start_config, end_config)
        
        self.dof = len(start_config)
        self.n_frames = n_frames
        
        self.fig = plt.figure()
        self.ax = self.create_subplots()

        self.times = self.times[0]
        
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
        
        self.ax[0].set_xlim([-3,3])
        self.ax[0].set_ylim([-3,3])
        
        for j in range(self.dof):
            self.ax[j+1].clear()
            lines.append(self.ax[j+1].plot(self.times, self.passive_trajectory[j,:],
                              color=COLOR[color_order[j]])[0])
            lines.append(self.ax[j+1].plot(self.times[i], self.passive_trajectory[j,i],
                              marker = 'o', color=COLOR[color_order[j]])[0])
                        
            
        return lines
            
if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links1 = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes1 = [G.Lie_alg_vector([0, 0, 1])] * 3

    # Create a kinematic chain
    kc1 = DiffKinematicChain(links1, joint_axes1)
    
    # Create another kinematic chain
    T_controlled = G.element([3, 0, np.pi])
    kc2 = DiffKinematicChain(links1, joint_axes1)
    kc2.set_base_transform(T_controlled)

    robo_animator = RobotAnimation(kc1, kc2, 10, 200, [0.0,0.0,0.0], [.25 * np.pi, -.5 * np.pi, .75 * np.pi])
    
    ani = FuncAnimation(robo_animator.fig, robo_animator.animate, frames=200, interval=20, blit=True)
    plt.show()
    ani.save("test_base_transform.gif",fps=20)