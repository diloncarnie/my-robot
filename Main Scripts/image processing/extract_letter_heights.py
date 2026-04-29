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
SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rms.ttt')

def get_vision_sensor_snapshot(sim, sensor_handle):
    img, res = sim.getVisionSensorImg(sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape([res[1], res[0], 3])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

def main():
    print('Starting OCR extraction')
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
    
    results = reader.readtext(snapshot)
    results.sort(key=lambda x: x[0][0][1])

    height_mapping = {}
    current_index = 0

    for (bbox, text, prob) in results:
        if prob > 0.3:
            y_coord = int(bbox[0][1])
            print(f"TARGET: {text} | Y: {y_coord} | IDX: {current_index}")
            height_mapping[y_coord] = current_index
            current_index += 1

    with open('height_mapping.json', 'w') as f:
        json.dump(height_mapping, f, indent=4)

    sim.stopSimulation()
    print("Done.")

if __name__ == '__main__':
    main()
