import mujoco 
import numpy as np
import time
from mujoco import viewer
from kinematics.inverse_kinematics import GaussNewton, GradientDescent, LM, IKinematics
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Now you can access your environment variables
pc_dir = os.getenv("PC_FOLDER")


def main():

    # simulate time
    simu_time = 5
    path = pc_dir + "/mujoco-python/scene.xml"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    key_name = "home"
    key_id = model.key(key_name).id

    # Initialize the goal position
    #goal  = [0.49, 0.13, 0.59]
    # Adjust zero matrix for rotational and traslational jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    # Get the end-effector id
    body_id = model.body("wrist_3_link").id
    # select a step size
    step_size = 0.5
    # tolerance accepted for the optimization algorithm 
    tol = 0.01
    # Guess an initial angle position for each link
    #Init position.
    pi = np.pi
    init_pos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0]

    def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
        as a function of time t and frequency f."""
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        z = 0.5
        return np.array([x, y, z])


    in_k = IKinematics(model, data, jacp, jacr, step_size, LM(), tol)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
            show_right_ui=False,) as viewer:
      # Reset the simulation.
      mujoco.mj_resetDataKeyframe(model, data, key_id)

      # Reset the free camera.
      mujoco.mjv_defaultFreeCamera(model, viewer.cam)
      # Enable site frame visualization.
      viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

      # Close the viewer automatically after simu_time wall-seconds.
      start = time.time()
      while viewer.is_running() and time.time() - start < simu_time:
        step_start = time.time()
        goal = circle(data.time, 0.1, 0.5, 0.0, 0.5)
        # mj_step can be replaced with code that also evaluates
        in_k.calculate(goal, init_pos, body_id)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()