# Simulation Plotting

This script focuses on simulating and plotting the dynamic characteristics of the robot arm—specifically tracking the **joint velocities** and **joint torques** over time.

The simulation executes a specific sequence of movements where the robot navigates to its default working configuration and a designated pickup and place position. This sequence evaluates two different trajectory generation methods: joint-space and Cartesian-space trajectory types.

During these trajectory sequences, the script logs the velocity and force/torque applied at each joint at every simulation step. These values are then plotted to visualize the dynamic demands placed on the robot's motors throughout the pickup operation.