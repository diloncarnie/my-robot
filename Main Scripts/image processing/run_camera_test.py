import os
import sys
import cv2
import json
import numpy as np
import easyocr
import time

CoppeliaSim_PYTHON_PATH = r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python"
sys.path.append(CoppeliaSim_PYTHON_PATH)
sys.path.append(os.path.join(CoppeliaSim_PYTHON_PATH, 'zmqRemoteApi'))
from coppeliasim_zmqremoteapi_client import RemoteAPIClient 

reader = easyocr.Reader(['en'], gpu=False)
SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rms-camera.ttt')

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
    cv2.putText(snapshot, f"Sorting Letter: {sorting_letter}  Index: {assigned_index}", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.imwrite("ocr_detection_output.jpg", snapshot)
    return sorting_letter, assigned_index

def main():
    print('Starting camera test')
    client = RemoteAPIClient()
    sim = client.require('sim')

    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)

    sim.loadScene(SCENE_PATH)
    sim.setStepping(True)
    sim.startSimulation()
    sim.step()

    vision_sensor = sim.getObject('/visionSensor')
    snapshot = get_vision_sensor_snapshot(sim, vision_sensor)
    sorting_letter, assigned_index = run_ocr(snapshot)
    
    print(f"RESULT: {sorting_letter} | INDEX: {assigned_index}")

    sim.stopSimulation()
    print("Done.")

if __name__ == '__main__':
    main()
