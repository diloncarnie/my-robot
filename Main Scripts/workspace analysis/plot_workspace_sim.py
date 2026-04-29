import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from CustomRobotClass import CustomRobot


SCENE_PATH = os.path.abspath('rms-workspace.ttt')
JOINTS = 5

def draw_workspace_in_sim(samples=50000, sing_threshold=0.015):
    print("Connecting to CoppeliaSim...")
    try:
        client = RemoteAPIClient()
        sim = client.require('sim')
        simConvex = client.require('simConvex')
    except Exception as e:
        print(f"Failed to connect to CoppeliaSim. Make sure it is open! Error: {e}")
        return
        
    print("Loading scene...")
    sim.loadScene(SCENE_PATH)
    sim.startSimulation()
    # Give the simulation a moment to load and start
    time.sleep(1.0)
    
    robot = CustomRobot()
    print(f"Robot loaded: {robot.name}")
    
    try:
        qlim = robot.qlim
        if qlim is None or np.all(qlim == 0):
            print(f"Using default joint limits (-pi to pi) for {JOINTS} joints")
            qlim = np.array([[-np.pi] * JOINTS, [np.pi] * JOINTS])
        else:
            print("Using joint limits from URDF model")
    except AttributeError:
        print(f"Using default joint limits (-pi to pi) for {JOINTS} joints")
        qlim = np.array([[-np.pi] * JOINTS, [np.pi] * JOINTS])
    
    print(f"Generating {samples} random joint configurations...")
    q_random = np.random.uniform(qlim[0], qlim[1], (samples, JOINTS))
    
    print("Calculating kinematics and Jacobians... This may take a moment.")
    points = []
    points_interior = []
    
    for i, q in enumerate(q_random):
        # Calculate forward kinematics
        pose = robot.fkine(q).A
        px, py, pz = pose[0, 3], pose[1, 3], pose[2, 3]
        
        # 1. Find Jacobian matrix (in base frame)
        J = robot.jacob0(q)
        
        # 2. Determine singularity by checking the singular values
        s = np.linalg.svd(J, compute_uv=False)
        min_sv = np.min(s)
        
        # If the minimum singular value is below the threshold, it's a boundary/singularity
        if min_sv < sing_threshold:
            points.extend([px, py, pz])
        else:
            # Optionally keep some interior points for visualization
            if np.random.random() < 0.05: # Keep 5% of interior points
                points_interior.extend([px, py, pz])
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{samples} configurations...")
            
    print(f"Found {len(points)//3} singular configurations (Boundaries).")
    print(f"Drawing workspace points in CoppeliaSim...")
    
    usePointCloud = True
    extractConvexHull = True

    if usePointCloud:
        # Draw Boundary Points (Singularities) in Magenta
        if len(points) > 0:
            ptcld = sim.createPointCloud(0.02, 10, 0, 5) # smaller point size for boundary
            sim.insertPointsIntoPointCloud(ptcld, 0, points, [1, 0, 1] * (len(points)//3))
            
        # Draw Interior Points in Cyan (at lower density)
        if len(points_interior) > 0:
            ptcld_int = sim.createPointCloud(0.02, 10, 0, 5)
            sim.insertPointsIntoPointCloud(ptcld_int, 0, points_interior, [0, 1, 1] * (len(points_interior)//3))
        
    if extractConvexHull and len(points) > 0:
        # Using the singular (boundary) points to define the hull
        vertices, indices = simConvex.qhull(points)
        shape = sim.createShape(3, 0, vertices, indices)
        sim.alignShapeBB(shape, [0, 0, 0, 0, 0, 0, 1])
        sim.setShapeColor(shape, None, 0, [1, 0, 1]) # Magenta hull
        sim.setShapeColor(shape, None, 4, [0.3]) # Semi-transparent

    print("\nDone! Please check your CoppeliaSim window.")
    print("Magenta = Workspace Boundary (Singularities)")
    print("Cyan = Interior Workspace (Non-singular)")
    print("Press Ctrl+C in the terminal to stop the simulation and exit.")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        sim.stopSimulation()

if __name__ == '__main__':
    # Increase samples for better boundary resolution if needed
    draw_workspace_in_sim(samples=50000, sing_threshold=0.015)
