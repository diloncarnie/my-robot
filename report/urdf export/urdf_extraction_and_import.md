# URDF Extraction and Import

This section outlines how the robot arm's Unified Robot Description Format (URDF) model is extracted from Onshape and imported into the Python Robotics Toolbox.

### 1. Extraction from Onshape (`urdf-exporter.py`)
The URDF and corresponding 3D mesh files are exported directly from the Onshape CAD assembly using the `onshape_robotics_toolkit` library. 
- The script parses the CAD document into a kinematic graph, preserving the mathematical relationships and physical properties (mass, inertia, etc.) of the joints and links.
- It then uses a URDF serializer to generate the `.urdf` XML file and downloads the required visual and collision meshes (STL files) into an output directory.

### 2. Import into Robotics Toolbox (`CustomRobotClass.py`)
Once exported, the URDF model is loaded into the Robotics Toolbox environment.
- The class inherits from the standard `ERobot` class provided by the `roboticstoolbox` library.
- It reads the exported `robot-arm.urdf` file to automatically build the robot's complete kinematic model, enabling the toolbox to perform subsequent calculations.
- Finally, the class initializes the robot model and registers several predefined joint configurations (such as "default", "working", and "singular") for access during analysis and simulation.

### 3. Import into CoppeliaSim
The extracted URDF file is also imported directly into the CoppeliaSim environment using its built-in URDF Importer module. 
- This process accurately translates the URDF's visual meshes, collision geometries, and physical properties (such as link masses and inertia tensors) into native CoppeliaSim objects.
- It establishes the correct hierarchical kinematic tree, converting URDF joints into CoppeliaSim joint objects capable of modeling real-world physical constraints and friction.
- By utilizing CoppeliaSim's physics engine (e.g., Bullet or Newton), this configuration allows for accurate, real-time dynamic simulation. It provides a robust testing ground for executing trajectories and analyzing dynamic responses, such as the required joint torques and resulting joint velocities during movement.