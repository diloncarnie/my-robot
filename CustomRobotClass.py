import  numpy as np
from roboticstoolbox.robot.ERobot import ERobot
import os


class CustomRobot(ERobot):
    def __init__(self):
        urdf_path = os.path.abspath("URDF/robot-arm.urdf")
        links, name, urdf_string, urdf_filepath = self.URDF_read(urdf_path)
        super().__init__(
            links,
            name=name.upper(),
            manufacturer="Heriot-Watt University Dubai",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
        self.addconfiguration("default configuration", np.zeros(5))
        self.addconfiguration("working configuration", [0, 0, 0.79, 0.79, 0])
        self.addconfiguration("singular configuration", [0, 0.79, 0, 0, 0])
        