import math
from pickle import TRUE
import mujoco 
import numpy as np

class Car:
    def __init__(self, model, data, wheel_radius, robot_base):
        self.model = model
        self.data = data
        self.wheel_radius = wheel_radius
        self.robot_base = robot_base
        self.prev_error = [0.0, 0.0] # index 0 = velocity error and index 1 = omega_error 

    def check_limits(self):
        pass

    def update_state(self):
        # Get current state variables
        omega_r = self.data.sensordata[0]
        omega_l = self.data.sensordata[1]
        theta = self.data.qpos[6]
        # Update state
        x_prime = (self.wheel_radius / 2.0) * np.cos(theta) * (omega_r + omega_l)
        y_prime = (self.wheel_radius / 2.0) * np.sin(theta) * (omega_r + omega_l)
        omega =  (omega_r - omega_l) * (self.wheel_radius / self.robot_base)
        return x_prime, y_prime, theta, omega

    def pd_controller(self, e, e_prime, kp, kd):
        return kp * e + kd * e_prime

    def compute_pose(self,goal, body_id, kp, kd, threshold=0.2):
        linear_ctrl = 0.5
        condition = goal - self.data.body(body_id).xpos
        if np.linalg.norm(condition) >= threshold:
            # position control
            x = self.data.body(body_id).xpos[0]
            y = self.data.body(body_id).xpos[1]
            x_prime, y_prime, theta, omega = self.update_state()
            theta_d = math.atan2(goal[1] - y, goal[0] - x)
            e_theta = theta_d - theta
            omega_d = ((x - goal[0]) * y_prime - (y - goal[1]) * x_prime) / ((x - goal[0])**2 + (y - goal[1])**2)
            e_omega = omega_d - omega
            w_ctrl = self.pd_controller(e_theta, e_omega, kp, kd)
            self.data.ctrl[0] = linear_ctrl - w_ctrl
            self.data.ctrl[1] = linear_ctrl + w_ctrl
            mujoco.mj_step(self.model, self.data)
            #orientation control
           




