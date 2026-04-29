# 1. Kinematic Modeling with the CustomRobot Class

The `rms.py` script relies on the `CustomRobot` class to provide the fundamental kinematic model of the manipulator. This class inherits from the Robotics Toolbox `ERobot` class and is instantiated by parsing the robot's URDF (Unified Robot Description Format) file.

When the URDF is loaded, the Robotics Toolbox automatically extracts the geometric and inertial properties of the robot's links and joints. Crucially, it translates this information into an **Elementary Transform Sequence (ETS)**. The ETS is a string of fundamental translations and rotations (e.g., translation along X, rotation about Z) that precisely describe the spatial relationship from the robot's base to its end-effector. 

This automatically generated ETS kinematic model is the mathematical backbone for all subsequent spatial calculations in the toolbox:

*   **Forward Kinematics ($f(q)$):** The toolbox uses the ETS parameters to calculate the exact position and orientation ($SE(3)$ pose) of the end-effector for any given set of joint angles ($q$). It does this by sequentially multiplying the transformation matrices defined by the ETS from the base frame up to the tip frame.

---

# 2. Inverse Kinematics

The `rms.py` script leverages the Robotics Toolbox for Python to perform Inverse Kinematics (IK), which is the mathematical process of calculating the specific joint angles required to achieve a desired position and orientation (pose) of the robot's end-effector in 3D space.

**Function Used:**
The script uses the `ikine_LM()` function to solve for the target joint configurations.

**Mathematical Theory and Internal Workings:**
Under the hood, `ikine_LM()` employs the **Levenberg-Marquardt (LM) algorithm**. Because calculating the exact joint angles algebraically is often too complex for a multi-jointed robot, the LM algorithm finds the solution iteratively by "guessing and checking." 

The algorithm heavily relies on the ETS kinematic model provided by the `CustomRobot` class to perform this iterative optimization:

1.  **Forward Kinematics for Error Checking:** Starting from an initial guess (provided as `q0=current_q`), the algorithm uses the ETS model to compute the Forward Kinematics. This tells the algorithm exactly where the end-effector is currently located based on the guessed joint angles. It then calculates the spatial error between this guessed pose and the actual desired target pose.
2.  **The Jacobian Matrix for Adjustments:** To determine how to adjust the joints to reduce this error, the algorithm uses the manipulator's geometric Jacobian matrix, which is derived directly from the ETS model. The Jacobian acts as a guide, mapping how small velocity changes in the joints will affect the velocity (and thus position) of the end-effector.
3.  **Iterative Step Updates:** The LM algorithm uses the Jacobian and the calculated spatial error to compute a step update, moving the joint angles closer to the target. It specifically uses a "damped" approach (adding a stabilization factor). This mathematically prevents the calculations from failing if the robot approaches a **singularity** (a physical configuration where the robot temporarily loses a degree of freedom).

By continuously looping through these steps—using the ETS model to check the error and the ETS-derived Jacobian to adjust the angles—the algorithm eventually converges on the correct joint configuration that places the end-effector at the desired target pose.

---

# 3. Trajectory Planning

The script implements two distinct methods for trajectory generation within the `move_to_pose` function, controlled by the `straight_line` boolean parameter. Trajectory planning ensures the robot transitions smoothly between spatial waypoints without demanding infinite acceleration from its motors.

**Joint-Space Trajectory (Non-Straight Line)**
When `straight_line` is `False`, the script plans the motion entirely within the robot's joint space.

*   **Function Used:** `rtb.tools.trajectory.jtraj(current_q, sol.q, times)`
*   **Mathematical Theory:** `jtraj` generates a smooth, time-parameterized path directly between the initial joint angles and the final target joint angles (which were previously calculated by the IK solver). Under the hood, this function utilizes **quintic (fifth-order) polynomials**. A quintic polynomial is mathematically ideal because it allows the algorithm to specify boundary conditions for position, velocity, and acceleration at both the start and end of the motion (typically setting start and end velocities and accelerations to zero). This results in a mathematically smooth motion profile that minimizes "jerk" (the derivative of acceleration), ensuring stable movement. However, because the interpolation is mapped purely across joint angles, the resulting physical path of the end-effector in Cartesian space will generally be an arc or curve, not a straight line.

**Cartesian-Space Trajectory (Straight Line)**
When `straight_line` is `True`, the script strictly enforces a linear path for the end-effector in 3D operational space.

*   **Functions Used:** `rtb.tools.trajectory.ctraj(current_t, target_t, len(times))` followed by the `ikine_LM()` solver.
*   **Mathematical Theory:** `ctraj` performs interpolation directly on the $SE(3)$ transformation matrices representing the start and end poses. 
    1.  **Translational Interpolation:** The Cartesian position vector (x, y, z) is linearly interpolated over time between the starting and ending coordinates, ensuring the strict straight-line geometry.
    2.  **Rotational Interpolation:** The 3D orientation is interpolated using **Spherical Linear Interpolation (SLERP)** on quaternions. SLERP ensures that the rotation transitions smoothly along the shortest possible path on a 4D hypersphere, avoiding the mathematical singularities (like gimbal lock) and distortion issues that arise from linearly interpolating Euler angles.
    3.  Once this mathematically perfect Cartesian path is generated (resulting in an array of intermediate $SE(3)$ matrices), it must be translated into joint commands. The script feeds this entire Cartesian trajectory array back into the `ikine_LM()` solver, which iteratively solves the inverse kinematics for every single interpolated point along that straight line.