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
                 damping = None):

        """Initialize a kinematic chain augmented with Jacobian methods"""

        # Call the constructor for the base KinematicChain class
        super().__init__(links, joint_axes)
        
        self.stiffnesses = stiffnesses
        self.damping = damping
    
    
    def response_forces(self, F_E):
        
        F_alpha = np.matmul(F_E, self.Jacobian_Ad_inv(self.dof, 'world'))    
        
        return(F_alpha)
    
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