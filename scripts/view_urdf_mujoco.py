import mujoco
import mujoco.viewer
import argparse
import os

parser = argparse.ArgumentParser()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
xml_path = os.path.join(project_root, "urdf", "ur10.urdf")
parser.add_argument("--xml-path", default=xml_path)
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(args.xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)