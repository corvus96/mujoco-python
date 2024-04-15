
import mujoco 
import numpy as np
import time
from mujoco import viewer
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Now you can access your environment variables
pc_dir = os.getenv("PC_FOLDER")

def control_inv():
     # simulate time
    simu_time = 5
    path = pc_dir + "/mujoco-python/diff_car.xml"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    print(data.xpos)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
      show_right_ui=False,) as viewer:

      # Reset the free camera.
      mujoco.mjv_defaultFreeCamera(model, viewer.cam)

      # Close the viewer automatically after simu_time wall-seconds.
      start = time.time()
      while viewer.is_running() and time.time() - start < simu_time:
        step_start = time.time()

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)