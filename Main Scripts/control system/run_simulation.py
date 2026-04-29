from typing import List
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
from spatialmath import SE3
import roboticstoolbox as rtb
from roboticstoolbox.robot.ERobot import ERobot
import easyocr
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from CustomRobotClass import CustomRobot

SCENE_PATH = os.path.abspath('main-simulation.ttt')
JOINT_BASE_PATH = '/RobotArm/joint'
GRIPPER_JOINT_PATH = '/left_drive_joint'
SIMULATED_OBJECTS = ['/Envelope1', '/Envelope2','/Envelope3','/Envelope4']
JOINTS = 5
DEFAULT_MAX_VEL_DEG = 360.0
DEFAULT_MAX_ACCEL_DEG = 360.0
DEFAULT_MAX_JERK_DEG = 720.0
reader = easyocr.Reader(['en'], gpu=True)

def pose_to_se3(pose: List[float]) -> SE3:
    x, y, z, qx, qy, qz, qw = pose

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)

    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)

    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    t = np.array(
        [
            [r00, r01, r02, x],
            [r10, r11, r12, y],
            [r20, r21, r22, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return SE3(t, check=False)

def move_to_pose(
    sim,
    robot: ERobot,
    joints: List[int],
    waypoint_pose: List[float],
    current_q: np.ndarray,
    max_vel: np.ndarray,
    max_accel: np.ndarray,
    max_jerk: np.ndarray,
    duration_s: float = 2.0,
    straight_line: bool = False,
) -> np.ndarray:
    target_t = pose_to_se3(waypoint_pose)
    
    if duration_s <= 0.0:
        raise ValueError('duration_s must be > 0')
  
    dt = sim.getSimulationTimeStep()
    times = np.arange(0.0, duration_s + 0.5 * dt, dt, dtype=float)

    if straight_line:
        current_t = robot.fkine(current_q)
        cartesian_traj = rtb.tools.trajectory.ctraj(current_t, target_t, len(times))
        sol = robot.ikine_LM(cartesian_traj, q0=current_q)
        
        if not np.all(sol.success):
            print('Warning: IK failed at one or more points along the Cartesian trajectory.')
              
        traj_q = sol.q
        final_q = traj_q[-1]
    else:
        sol = robot.ikine_LM(target_t, q0=current_q)
        
        if not sol.success:
            print(f'IK failed: {sol.reason}')
            return current_q
            
        traj = rtb.tools.trajectory.jtraj(current_q, sol.q, times)
        traj_q = traj.q
        final_q = sol.q

    for q in traj_q:
        for i, joint_handle in enumerate(joints):
            motion_params = [
                float(max_vel[i]),
                float(max_accel[i]),
                float(max_jerk[i]),
            ]
            sim.setJointTargetPosition(joint_handle, float(q[i]), motion_params)
        sim.step()

    return np.array(final_q, dtype=float)

def set_gripper(sim, joint_handle, open_state: bool, delay: int = 30, object_name: str = None, make_dynamic: bool = False):
    angle_deg = 0.0 if open_state else -35.0
    angle_rad = np.deg2rad(angle_deg)
    motion_params = [np.deg2rad(360), np.deg2rad(360), np.deg2rad(720)]

    if object_name and open_state:
        try:
            obj_handle = sim.getObject(object_name)
            sim.setObjectParent(obj_handle, -1, True)
            sim.setObjectFloatParam(obj_handle, 3005, 0.0001)
            
            if make_dynamic:
                sim.setObjectInt32Param(obj_handle, 3003, 0)
                sim.setObjectInt32Param(obj_handle, 3004, 1)
                print(f"Gripper opening: {object_name} released.")
            else:
                sim.setObjectInt32Param(obj_handle, 3003, 1)
                print(f"Gripper opening: {object_name} released.")
        except Exception as e:
            print(f"Warning: Drop failed for '{object_name}': {e}")

    sim.setJointTargetPosition(joint_handle, float(angle_rad), motion_params)
    for _ in range(delay):
        sim.step()

    if object_name and not open_state:
        try:
            obj_handle = sim.getObject(object_name)
            gripper_handle = sim.getObject('/RobotGripper')
            sim.setObjectParent(obj_handle, gripper_handle, True)
            sim.setObjectFloatParam(obj_handle, 3005, 0.0001)
            sim.setObjectInt32Param(obj_handle, 3003, 1)
            sim.setObjectInt32Param(obj_handle, 3004, 0)
            print(f"Gripper closed: {object_name} attached.")
        except Exception as e:
            print(f"Warning: Pick failed for '{object_name}': {e}")

def get_vision_sensor_snapshot(sim, sensor_handle):
    img, res = sim.getVisionSensorImg(sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape([res[1], res[0], 3])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def load_json_mapping(file_path):
    try:
        with open(file_path, 'r') as f:
            raw_mapping = json.load(f)
            height_mapping = {int(k): v for k, v in raw_mapping.items()}
    except FileNotFoundError:
        print("Error: height_mapping.json not found.")
        height_mapping = None
    return height_mapping
      
def run_ocr(snapshot):
    results = reader.readtext(snapshot)
    results.sort(key=lambda x: x[0][0][1])
    height_mapping = load_json_mapping('height_mapping.json')
    
    if not results or not height_mapping:
        return None, None

    bbox, text, _ = results[0]
    words = text.split()
    
    last_name = words[1]
    sorting_letter = last_name[0].upper()
    
    y_coord = int(bbox[0][1])
    closest_y = min(height_mapping.keys(), key=lambda k: abs(k - y_coord))
    assigned_index = height_mapping[closest_y]

    print(f"TARGET: {text} | SORT: {sorting_letter} | INDEX: {assigned_index}")

    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))
    cv2.rectangle(snapshot, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(snapshot, f"Sort: {sorting_letter} (Idx: {assigned_index})", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return sorting_letter, assigned_index
       
def reset_objects(sim):
    print("Resetting objects...")
    for name in SIMULATED_OBJECTS:
        try:
            handle = sim.getObject(name)
            sim.setObjectParent(handle, -1, True)
            sim.setObjectInt32Param(handle, 3003, 1)
            sim.setObjectInt32Param(handle, 3004, 1)
        except:
            pass

def main(robot: ERobot) -> None:
    print('Starting simulation')
    client = RemoteAPIClient()
    sim = client.require('sim')

    sim.loadScene(SCENE_PATH)
    sim.setStepping(True)
    sim.startSimulation()
    sim.step()

    vision_sensor = sim.getObject('/visionSensor')
    
    joints: List[int] = []
    for i in range(JOINTS):
        joints.append(sim.getObject(JOINT_BASE_PATH, {'index': i}))

    gripper_joint = sim.getObject(GRIPPER_JOINT_PATH)

    max_vel = np.deg2rad(np.full(JOINTS, DEFAULT_MAX_VEL_DEG, dtype=float))
    max_accel = np.deg2rad(np.full(JOINTS, DEFAULT_MAX_ACCEL_DEG, dtype=float))
    max_jerk = np.deg2rad(np.full(JOINTS, DEFAULT_MAX_JERK_DEG, dtype=float))

    current_q = np.deg2rad([0, 0, 0, 0, 0])
    
    default_position = "/default"
    w = sim.getObjectPose(sim.getObject(default_position), sim.handle_world)
    current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0)

    try:
        while True:
            sim.setStepping(False)
            snapshot = get_vision_sensor_snapshot(sim, vision_sensor)
            sorting_letter, assigned_index = run_ocr(snapshot)
            sim.setStepping(True)
            
            if sorting_letter is None or assigned_index is None:
                print("No detected letters. Stopping.")
                break
            
            envelope_name = f"/Envelope{assigned_index + 1}"
            index_wp = f"/index{assigned_index}"
            index_buffer_wp = f"/index{assigned_index}buffer"
            pigeonhole_wp = f"/pigeonhole{sorting_letter}"
            pigeonhole_buffer_wp = f"/pigeonhole{sorting_letter}buffer"

            w = sim.getObjectPose(sim.getObject(index_wp), sim.handle_world)
            current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0)
            set_gripper(sim, gripper_joint, open_state=False, object_name=envelope_name)
            
            w = sim.getObjectPose(sim.getObject(index_buffer_wp), sim.handle_world)
            current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=3.5, straight_line=True)
            
            w = sim.getObjectPose(sim.getObject(default_position), sim.handle_world)
            current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0)
            
            w = sim.getObjectPose(sim.getObject(pigeonhole_wp), sim.handle_world)
            current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0)
            
            w = sim.getObjectPose(sim.getObject(pigeonhole_buffer_wp), sim.handle_world)
            current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0, straight_line=True)
            set_gripper(sim, gripper_joint, open_state=True, object_name=envelope_name, make_dynamic=True)

            w = sim.getObjectPose(sim.getObject(default_position), sim.handle_world)
            current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0)
        
    finally:
        print("Cleaning up.")
        reset_objects(sim)
        sim.stopSimulation()

if __name__ == '__main__':
    robot = CustomRobot()
    main(robot)
