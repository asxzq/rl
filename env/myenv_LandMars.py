
import numpy as np
import math
from numba_funcs import landing3d_diff
import matplotlib.pyplot as plt
import random

class Memory:
    def __init__(self):
        self.memory = []

    def push(self, state):
        self.memory.append(np.copy(state))

    def getxyz(self):
        m = np.array(self.memory, dtype=float)
        return m[:, 0], m[:, 1], m[:, 2]
    def clear(self):
        self.memory = []

def EulerAndQuaternionTransform(intput_data):

    if type(intput_data) == list:
        data_len = len(intput_data)
    else:
        data_len = intput_data.shape[0]
    angle_is_not_rad = False
    if data_len == 3:
        r = 0
        p = 0
        y = 0
        if angle_is_not_rad:  # 180 ->pi
            r = math.radians(intput_data[0])
            p = math.radians(intput_data[1])
            y = math.radians(intput_data[2])
        else:
            r = intput_data[0]
            p = intput_data[1]
            y = intput_data[2]

        sinp = math.sin(p / 2)
        siny = math.sin(y / 2)
        sinr = math.sin(r / 2)

        cosp = math.cos(p / 2)
        cosy = math.cos(y / 2)
        cosr = math.cos(r / 2)

        w = cosr * cosp * cosy + sinr * sinp * siny
        x = sinr * cosp * cosy - cosr * sinp * siny
        y = cosr * sinp * cosy + sinr * cosp * siny
        z = cosr * cosp * siny - sinr * sinp * cosy
        return np.array([w, x, y, z], dtype=np.float32)

    elif data_len == 4:

        w = intput_data[0]
        x = intput_data[1]
        y = intput_data[2]
        z = intput_data[3]

        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        if angle_is_not_rad:  # pi -> 180
            r = math.degrees(r)
            p = math.degrees(p)
            y = math.degrees(y)
        return np.array([r, p, y], dtype=np.float32)


class LandMars(object):
    def __init__(self):
        self.env_name = "LandMars"
        self.state_dim = 14
        self.action_dim = 3
        self.delta_t = 0.1
        self.memory = Memory()
        self.max_step = 450
        self.action_max = 1
        self.if_discrete = False
        self.continuous = True
        self.target_return = 3000
        self.thrust_min = 0.3

    def reset(self):

        self.state = np.array([np.random.uniform(-30, 30), np.random.uniform(400, 500), np.random.uniform(-30, 30),
                               0.0, 0.0, 0.0,
                               1.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 6100.0], dtype=float)


        self.state_init = np.copy(self.state)

        self.memory.clear()

        self.params = np.array([math.pi / 36.0, 120000.0, 600, 9.8], dtype=float)

        self.inertia_tensor = np.diag(np.array([10000.0, 3000.0, 10000.0], dtype=float))


        self.thrust_center = np.array([0.0, -0.5, 0.0], dtype=float)

        self.buf = np.zeros_like(self.state)


        self.max_landv = 2.0
        self.max_vx = 30 / 4.5 * 2
        self.max_vz = 30 / 4.5 * 2
        self.max_vy = math.sqrt(2 * self.params[3] * self.state_init[1])

        self.prev_shaping = None
        self.total_reward = 0.0
        self.init_distance_xz = np.linalg.norm(np.array([self.state_init[0], self.state_init[2]], dtype=float))

        self.num_step = -1
        state_out_init, _, _, _ = self.step(action=np.zeros((self.action_dim,), dtype=float))

        return state_out_init



    def truestate_outstate(self):
        state_ = np.copy(self.state)

        state_[0] = self.state[0] / abs(self.state_init[0])
        state_[1] = self.state[1] / abs(self.state_init[1])
        state_[2] = self.state[2] / abs(self.state_init[2])
        state_[3] = self.state[3] / self.max_vx
        state_[4] = self.state[4] / self.max_vy
        state_[5] = self.state[5] / self.max_vy

        state_[13] = self.state[13] / self.state_init[13]
        return state_

    def step(self, action):

        control = np.copy(action)
        control[0] = np.clip(action[0], -1, 1) / 2 + 0.5
        # print(control[0])
        assert control[0] >= 0 and control[0] <= 1.0

        landing3d_diff(self.state, control.astype(float), self.params, self.inertia_tensor, self.thrust_center, self.buf)

        self.state += self.delta_t * self.buf
        # print(self.state[6:10])
        self.state[6:10] /= np.linalg.norm(self.state[6:10])

        state_out = self.truestate_outstate()

        Euler = EulerAndQuaternionTransform(state_out[6:10])



        reward = math.exp(-np.linalg.norm(state_out[:3])
                          - np.linalg.norm(state_out[3:6])
                          - np.linalg.norm(Euler) / 5
                          - np.linalg.norm(state_out[10:13]) / 10
                          ) - 0.001 * abs(self.buf[13]) * self.delta_t - 0.05 + math.exp(-np.linalg.norm(state_out[3:6])) * (state_out[1] < 0.25)

        '''
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        '''


        self.total_reward += reward
        done = False

        self.num_step += 1
        if (state_out[1] > 1.2) and (not done):
            reward = -100
            done = True

        if (abs(state_out[0]) > 3 * abs(self.state_init[0]) or abs(state_out[2]) > 3 * abs(self.state_init[2])) and \
        (not done):
            reward = -100
            done = True

        if self.num_step >= self.max_step and (not done):
            done = True
            reward = -100

        if self.state[1] < 0 and (not done):
            done = True
            if np.linalg.norm(self.state[:3]) < 4:
                reward = self.total_reward / 2 + 5 * math.exp(np.linalg.norm(self.state[:3]) / 4)
            else:
                reward = - self.total_reward / 2

        self.memory.push(self.state)
        return state_out.astype(np.float32), reward, done, {}

    def render(self):
        x, y, z = self.memory.getxyz()
        # fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(z, x, y, cmap='Blues')
        ax1.scatter3D(0.0, 0.0, 0.0, cmap='Reds')
        ax1.scatter3D(50, 50, 500, cmap='Greys')
        ax1.scatter3D(-50, -50, 0.0, cmap='Greys')
        plt.show()


if __name__ == "__main__":
    a = LandMars()
    a.reset()
    s = a.state
    print(s[:6])
    ep_r = 0
    for i in range(400):
        if i < 90:
            s_, r, done, _ = a.step(np.array([-1.0, 0, 0], dtype=np.float32))
        else:
            s_, r, done, _ = a.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        # print(s_[:6])
        ep_r += r
        s = s_
        if done:
            break
    print(a.state[0:6], a.state[13], a.num_step)
    print(ep_r)
    a.render()





