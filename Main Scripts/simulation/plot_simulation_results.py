from typing import List

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import  numpy as np
import cv2
import matplotlib.pyplot as plt
from spatialmath import SE3
import roboticstoolbox as rtb
from roboticstoolbox.robot.ERobot import ERobot
import easyocr
import json
import os
from CustomRobotClass import CustomRobot



SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation-test.ttt')
JOINT_BASE_PATH = '/RobotArm/joint'
ROBOT_TIP = '/tip'
GRIPPER_JOINT_PATH = '/left_drive_joint'  # Adjusted path for root-level import
SIMULATED_OBJECTS = ['/Envelope1', '/Envelope2','/Envelope3','/Envelope4']
JOINTS = 5
DEFAULT_MAX_VEL_DEG = 360.0
DEFAULT_MAX_ACCEL_DEG = 360.0
DEFAULT_MAX_JERK_DEG = 720.0
reader = easyocr.Reader(['en'], gpu=True)


def pose_to_se3(pose: List[float]) -> SE3:
    x, y, z, qx, qy, qz, qw = pose

    # CoppeliaSim pose quaternion order: (qx, qy, qz, qw)
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
    metrics: dict = None,
) -> np.ndarray:
    target_t = pose_to_se3(waypoint_pose)
    
    if duration_s <= 0.0:
        raise ValueError('duration_s must be > 0')
  
    # Dynamically fetch the simulation time step (dt) to guarantee perfect synchronization
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

 
    tip_handle = sim.getObject('/tip')

    for step_idx, q in enumerate(traj_q):
        for i, joint_handle in enumerate(joints):
            motion_params = [
                float(max_vel[i]),
                float(max_accel[i]),
                float(max_jerk[i]),
            ]
            sim.setJointTargetPosition(joint_handle, float(q[i]), motion_params)
        sim.step()
        
        if metrics is not None:
            t = sim.getSimulationTime()
            metrics['times'].append(t)
            
            lin_vel, _ = sim.getObjectVelocity(tip_handle)
            import math
            ee_vel_mag = math.sqrt(lin_vel[0]**2 + lin_vel[1]**2 + lin_vel[2]**2)
            metrics['ee_vels'].append(ee_vel_mag)
            
            for i, joint_handle in enumerate(joints):
                jv = sim.getJointVelocity(joint_handle)
                jt = sim.getJointForce(joint_handle)
                metrics['joint_vels'][i].append(jv)
                metrics['joint_torques'][i].append(jt)

    return np.array(final_q, dtype=float)


def set_gripper(sim, joint_handle, open_state: bool, delay: int = 30, object_name: str = None, make_dynamic: bool = False):
    """
    Controls the gripper.
    open_state: True for 0 deg (open), False for -35 deg (closed).
    object_name: Optional name of an object to pick up or drop.
    make_dynamic: If True, toggles object between static (carried) and dynamic (dropped).
    """
    angle_deg = 0.0 if open_state else -35.0
    angle_rad = np.deg2rad(angle_deg)
    
    # Fast motion parameters for the gripper servo
    motion_params = [np.deg2rad(360), np.deg2rad(360), np.deg2rad(720)]

    # IMMEDIATE ACTION for Opening (Drop)
    if object_name and open_state:
        try:
            obj_handle = sim.getObject(object_name)
            sim.setObjectParent(obj_handle, -1, True)
            
            # Set mass (3005 = sim.shapefloatparam_mass)
            sim.setObjectFloatParam(obj_handle, 3005, 0.0001)
            
            if make_dynamic:
                # 3003 = static, 3004 = respondable
                sim.setObjectInt32Param(obj_handle, 3003, 0) # 0 = dynamic
                sim.setObjectInt32Param(obj_handle, 3004, 1) # 1 = respondable
                print(f"Gripper opening: {object_name} released (Dynamic & Respondable, Mass: 0.001).")
            else:
                sim.setObjectInt32Param(obj_handle, 3003, 1) # 1 = static
                print(f"Gripper opening: {object_name} released (Static, Mass: 0.001).")
        except Exception as e:
            print(f"Warning: Immediate drop failed for '{object_name}': {e}")

    # START MOTION
    sim.setJointTargetPosition(joint_handle, float(angle_rad), motion_params)
    # Step simulation a few times to ensure the command is processed
    for _ in range(delay):
        sim.step()

    # DELAYED ACTION for Closing (Pick up)
    if object_name and not open_state:
        try:
            obj_handle = sim.getObject(object_name)
            gripper_handle = sim.getObject('/RobotGripper')
            sim.setObjectParent(obj_handle, gripper_handle, True)
            
            # Set mass and force to static/non-respondable while carried
            sim.setObjectFloatParam(obj_handle, 3005, 0.0001)
            sim.setObjectInt32Param(obj_handle, 3003, 1) # 1 = static
            sim.setObjectInt32Param(obj_handle, 3004, 0) # 0 = non-respondable
            print(f"Gripper closed: {object_name} attached to RobotGripper (Static, Non-respondable, Mass: 0.001).")
        except Exception as e:
            print(f"Warning: Delayed pick failed for '{object_name}': {e}")


def get_vision_sensor_snapshot(sim, sensor_handle):
    """Retrieves a single frame from the vision sensor and returns it as an OpenCV image."""
    img, res = sim.getVisionSensorImg(sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape([res[1], res[0], 3])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def load_json_mapping(file_path):
    try:
        with open(file_path, 'r') as f:
            raw_mapping = json.load(f)
            # Convert string keys back to integers for distance comparison
            height_mapping = {int(k): v for k, v in raw_mapping.items()}
    except FileNotFoundError:
        print("Error: height_mapping.json not found. Please run extract-height.py first.")
        height_mapping = None
    return height_mapping
      
def run_ocr(snapshot):
    # Run EasyOCR
    results = reader.readtext(snapshot)
    # We sort so the item with the LOWEST Y-coordinate (top of screen) is first
    # x[0] is the bbox, x[0][0][1] is the Y-value of the top-left corner
    results.sort(key=lambda x: x[0][0][1])
    # Load the height mapping from JSON
    height_mapping = load_json_mapping('height_mapping.json')
    
    if not results or not height_mapping:
        return None, None

    # Get the topmost result
    bbox, text, _ = results[0]
    words = text.split()
    
    last_name = words[1]
    sorting_letter = last_name[0].upper()
    
    # Extract the top-left y-coordinate
    y_coord = int(bbox[0][1])
    # Find the closest height in the mapping (OCR coordinates can vary slightly)
    closest_y = min(height_mapping.keys(), key=lambda k: abs(k - y_coord))
    assigned_index = height_mapping[closest_y]

    print(f"TARGET: {text} | SORT BY LAST NAME: {sorting_letter} | ASSIGNED INDEX: {assigned_index}")

    # Drawing the Green Box and Label
    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))
    cv2.rectangle(snapshot, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(snapshot, f"Sort: {sorting_letter} (Idx: {assigned_index})", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return sorting_letter, assigned_index
       



def main(robot: ERobot) -> None:
    print('Program started')
    client = RemoteAPIClient()
    sim = client.require('sim')

    sim.loadScene(SCENE_PATH)
    sim.setStepping(True)
    sim.startSimulation()
    sim.step()

    joints: List[int] = []
    for i in range(JOINTS):
        joints.append(sim.getObject(JOINT_BASE_PATH, {'index': i}))

 
    # Motion constraints for sim.setJointTargetPosition(..., motionParams).
    max_vel = np.deg2rad(np.full(JOINTS, DEFAULT_MAX_VEL_DEG, dtype=float))
    max_accel = np.deg2rad(np.full(JOINTS, DEFAULT_MAX_ACCEL_DEG, dtype=float))
    max_jerk = np.deg2rad(np.full(JOINTS, DEFAULT_MAX_JERK_DEG, dtype=float))

    # Keep joint state in Python (do not fetch from CoppeliaSim between waypoints).
    current_q = np.deg2rad([0, 0, 0, 0, 0])
    
    
    # Start movement to dynamic trajectory waypoints
    print("Starting Movement")
    
    trajectory_metrics = {
        'times': [],
        'ee_vels': [],
        'joint_vels': [[] for _ in range(JOINTS)],
        'joint_torques': [[] for _ in range(JOINTS)]
    }

    default_position = "/default"
    w = sim.getObjectPose(sim.getObject(default_position), sim.handle_world) #Go to default position
    current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0, metrics=trajectory_metrics)
    
    w = sim.getObjectPose(sim.getObject(default_position), sim.handle_world) #Go to default position
    current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0, metrics=trajectory_metrics)

    w = sim.getObjectPose(sim.getObject("/index0"), sim.handle_world) #Go to pickup position for selected Envelope
    current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0, metrics=trajectory_metrics)
    
    
    w = sim.getObjectPose(sim.getObject("/index0buffer"), sim.handle_world) #Go to buffer position for selected Envelope
    current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=3.5, straight_line=True, metrics=trajectory_metrics)
    
    w = sim.getObjectPose(sim.getObject(default_position), sim.handle_world) #Go to default position
    current_q = move_to_pose(sim, robot, joints, w, current_q, max_vel, max_accel, max_jerk, duration_s=2.0, metrics=trajectory_metrics)
    
    print("Script interrupted or finished. Cleaning up simulation.")
    sim.stopSimulation()

    # Plot final trajectory metrics
    fig1 = plt.figure('End-Effector Velocity', figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(trajectory_metrics['times'], trajectory_metrics['ee_vels'], label='EE Vel Magnitude')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('End-Effector Velocity')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    fig1.tight_layout()
    
    fig2 = plt.figure('Joint Velocities', figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    for i in range(JOINTS):
        ax2.plot(trajectory_metrics['times'], trajectory_metrics['joint_vels'][i], label=f'J{i+1}')
    ax2.set_ylabel('Joint Velocities (rad/s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Joint Velocities')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    fig2.tight_layout()
    
    fig3 = plt.figure('Joint Torques', figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    for i in range(JOINTS):
        ax3.plot(trajectory_metrics['times'], trajectory_metrics['joint_torques'][i], label=f'J{i+1}')
    ax3.set_ylabel('Joint Torques (Nm)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Joint Torques')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    fig3.tight_layout()
    
    plt.show()


if __name__ == '__main__':
    robot = CustomRobot()
    main(robot)
