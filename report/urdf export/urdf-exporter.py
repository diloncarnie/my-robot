from onshape_robotics_toolkit.connect import Client
from onshape_robotics_toolkit.formats.urdf import URDFSerializer
from onshape_robotics_toolkit.graph import KinematicGraph
from onshape_robotics_toolkit.parse import CAD
from onshape_robotics_toolkit.robot import Robot
from onshape_robotics_toolkit.utilities import setup_default_logging

DOCUMENT_URL = "https://cad.onshape.com/documents/6cad1129a03f26c16c97901b/w/498ad2e526330b7e761f06df/e/1aa9615a2348221a2004f3bd"

setup_default_logging(file_path="robot-arm.log", console_level="INFO")

client = Client(env=".env")
cad = CAD.from_url(DOCUMENT_URL, client=client, max_depth=0)
graph = KinematicGraph.from_cad(cad, use_user_defined_root=True)
robot = Robot.from_graph(kinematic_graph=graph, client=client, name="robot-arm")

URDFSerializer().save(robot, "output_gripper/robot-gripper.urdf", download_assets=True, mesh_dir="output_gripper/meshes")