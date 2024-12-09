# Tethered Deformable Object Manipulation
## ROB 541 Geometric Mechanics - Final Project
#### Marcus Rosette & Miranda Cravetz

This repository serves as the codebase for determining the joint commands required by a 2D active (controllable) kinematic chain to conform a 2D passive kinematic chain to a desired configuration.


##### The process involves the following steps:
1. Define the desired trajectory of the passive chain.
2. Map the desired joint velocities of the passive chain to the resulting end-effector velocities.
3. Compute the resultant forces at the end-effector of the passive chain. 
4. Perform the following simultaneously:
    1. Map the passive chain's end-effector velocity to the active chain's joint-space velocity.
    2. Numerically integrate the joint-space velocities to compute the active chain's joint-space configurations.

This pipeline was tested on three active/passive chain configurations: two-link chains (insufficient), three-link chains (sufficient), and five-link chains (redundant).

##### Results
*Insufficient* 
![](https://github.com/m-rosette/GM_project/main/docs/insufficient_two_link.gif)

*Sufficient*
![](https://github.com/m-rosette/GM_project/main/docs/sufficient_three_link.gif)

*Redundant*
![](https://github.com/m-rosette/GM_project/main/docs/redundant_five_link.gif)