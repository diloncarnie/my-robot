# Static Analysis

The static analysis calculates the holding torques required at each joint to maintain the robot in various predefined configurations against the force of gravity (gravity compensation). 

To compute these values, the script utilizes the **jacob0()`** function from the robotics toolbox to find the base-frame Jacobian matrix for each link's center of mass. The required joint torque is then determined by multiplying the transpose of this Jacobian matrix by the gravitational force vector acting on the link.
