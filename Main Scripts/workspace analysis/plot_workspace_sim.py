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
        print(f"Connection failed: {e}")
        return
        
    print("Loading scene...")
    sim.loadScene(SCENE_PATH)
    sim.startSimulation()
    time.sleep(1.0)
    
    robot = CustomRobot()
    print(f"Robot: {robot.name}")
    
    try:
        qlim = robot.qlim
        if qlim is None or np.all(qlim == 0):
            qlim = np.array([[-np.pi] * JOINTS, [np.pi] * JOINTS])
    except AttributeError:
        qlim = np.array([[-np.pi] * JOINTS, [np.pi] * JOINTS])
    
    print(f"Generating {samples} configurations...")
    q_random = np.random.uniform(qlim[0], qlim[1], (samples, JOINTS))
    
    points = []
    points_interior = []
    
    for i, q in enumerate(q_random):
        pose = robot.fkine(q).A
        px, py, pz = pose[0, 3], pose[1, 3], pose[2, 3]
        
        J = robot.jacob0(q)
        s = np.linalg.svd(J, compute_uv=False)
        min_sv = np.min(s)
        
        if min_sv < sing_threshold:
            points.extend([px, py, pz])
        else:
            if np.random.random() < 0.05:
                points_interior.extend([px, py, pz])
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{samples}")
            
    print(f"Drawing points in CoppeliaSim...")
    
    if len(points) > 0:
        ptcld = sim.createPointCloud(0.02, 10, 0, 5)
        sim.insertPointsIntoPointCloud(ptcld, 0, points, [1, 0, 1] * (len(points)//3))
        
    if len(points_interior) > 0:
        ptcld_int = sim.createPointCloud(0.02, 10, 0, 5)
        sim.insertPointsIntoPointCloud(ptcld_int, 0, points_interior, [0, 1, 1] * (len(points_interior)//3))
    
    if len(points) > 0:
        vertices, indices = simConvex.qhull(points)
        shape = sim.createShape(3, 0, vertices, indices)
        sim.alignShapeBB(shape, [0, 0, 0, 0, 0, 0, 1])
        sim.setShapeColor(shape, None, 0, [1, 0, 1])
        sim.setShapeColor(shape, None, 4, [0.3])

    print("\nDone. Magenta = Boundary, Cyan = Interior.")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        sim.stopSimulation()

if __name__ == '__main__':
    draw_workspace_in_sim(samples=50000, sing_threshold=0.015)
