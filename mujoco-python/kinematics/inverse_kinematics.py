from abc import abstractmethod
import mujoco
import numpy as np

class Optimizer:
    def compute(self, data, jac, error, step_size, alpha, damping):
        pass

class GradientDescent(Optimizer):
    def compute(self, data, jac, error, step_size, alpha, damping):
        grad = alpha * jac.T @ error
        data.qpos += step_size * grad
        return data.qpos

class GaussNewton(Optimizer):
    def compute(self, data, jac, error, step_size, alpha, damping):
        jac_product = jac.T @ jac
        if np.isclose(np.linalg.det(jac_product), 0):
            # Use the pseudo-inverse because the matrix is singular
            jac_inv = np.linalg.pinv(jac_product) @ jac.T
        else:
            jac_inv = np.linalg.inv(jac_product) @ jac.T
        delta_q = jac_inv @ error
        data.qpos += step_size * delta_q
        return data.qpos


class  LM(Optimizer):
    def compute(self, data, jac, error, step_size, alpha, damping):
        n = jac.shape[1]
        I = np.identity(n)
        jac_product = jac.T @ jac + damping * I
        if np.isclose(np.linalg.det(jac_product), 0):
            # Use the pseudo-inverse because the matrix is singular
            jac_inv = np.linalg.pinv(jac_product) @ jac.T
        else:
            jac_inv = np.linalg.inv(jac_product) @ jac.T
        delta_q = jac_inv @ error
        data.qpos += step_size * delta_q
        return data.qpos

class IKinematics:
    def __init__(self, model, data, jacp, jacr, step_size, optim : Optimizer, tol=0.01, damping=0.15, alpha=0.5) -> None:
        self.model = model
        self.data = data
        self._optim = optim
        self.jacp = jacp
        self.jacr = jacr
        self.alpha = alpha
        self.damping = damping
        self.step_size = step_size
        self.tol = tol

    def optim(self) -> Optimizer:
        return self._optim

    def check_limits(self, q):
        return [max(self.model.jnt_range[i][0], min(self.model.jnt_range[i][1], q[i])) for i in range(len(self.model.jnt_range))]

    def calculate(self, goal, init_pos, body_id):
        self.data.qpos = init_pos
        # Calculate the current end-effector pose with Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        current_p = self.data.body(body_id).xpos
        error = goal - current_p
        if np.linalg.norm(error) >= self.tol:
            # Calculate the jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            # Apply the correspond minimization method to optimize the error
            q = self._optim.compute(self.data, self.jacp, error, self.step_size, self.alpha, self.damping)
            q = self.check_limits(q)
            self.data.qpos = q
            # apply FK to change the angles
            self.data.ctrl = q 
            mujoco.mj_step(self.model, self.data)
            # calculate the error for the new position of end-effector
            error = goal - self.data.body(body_id).xpos
        
        

