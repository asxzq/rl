
import numpy as np

class Landing_Earth_NoMass_NoThrustMagnitudeConstraint(object):

    def __init__(self):
        # 对象初始化
        self.g = np.array([0,  0, -9.8])
        self.reset_state()

    def reset_state(self, state0=None):
        # 状态初始化
        if state0 is None:
            # self.r0 = np.array([500, 1600, 200.0])
            # self.v0 = np.array([-5, 26.7, 0.0])
            self.r0 = np.array([50.0, 5.0, 100.0])
            self.v0 = np.array([-3.0, -1.0, -15.0])
            self.energy0 = 0.0
            self.fuel0 = 0.0
            self.state0 = np.hstack((self.r0, self.v0))

            self.rf = np.array([0.0, 0.0, 0.0])
            self.vf = np.array([0.0, 0.0, 0.0])
        else:
            self.state0 = state0

            self.rf = np.array([0.0, 0.0, 0.0])
            self.vf = np.array([0.0, 0.0, 0.0])

    def fun_dynamics(self, state, t, u):
        # 动力学方程

        r = state[0:3]
        v = state[3:6]
        r_dot = v
        v_dot = np.copy(u) + self.g
        state_dot = np.hstack((r_dot, v_dot))
        return state_dot

    def fun_state_update(self, state, delta_t, u):
        # 动力学一个步长更新

        t = 0
        k1 = self.fun_dynamics(state, t, u)
        k2 = self.fun_dynamics(state + delta_t * k1 / 2, t + delta_t / 2, u)
        k3 = self.fun_dynamics(state + delta_t * k2 / 2, t + delta_t / 2, u)
        k4 = self.fun_dynamics(state + delta_t * k3, t + delta_t, u)
        state_next = state + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return state_next














































