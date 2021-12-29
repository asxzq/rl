import numpy as np
import math
from numba_funcs import landing3d_diff
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

import random

class Memory:
    def __init__(self):
        self.memory = []
        self.theta = []

    def push(self, state, theta):
        self.memory.append(np.copy(state))
        self.theta.append(np.copy(theta))

    def getxyz(self):
        m = np.array(self.memory, dtype=float)
        return m[:, 0], m[:, 1], m[:, 2]

    def getvxyz(self):
        m = np.array(self.memory, dtype=float)
        return m[:, 3], m[:, 4], m[:, 5]

    def gettheta(self):
        return np.array(self.theta, dtype=float)

    def clear(self):
        self.memory = []
        self.theta = []

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
        self.delta_t = 0.1
        self.memory = Memory()
        self.max_step = 300
        self.max_step = 1000
        self.action_max = 1
        self.if_discrete = False
        self.continuous = True
        self.target_return = 3000
        self.thrust_min = 0.3

    def reset(self):

        self.state = np.array([np.random.uniform(-8, 8), np.random.uniform(80, 100), np.random.uniform(-8, 8),
                               0.0, -2.0, 0.0,
                               1.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 6100.0], dtype=float)


        self.state_init = np.copy(self.state)

        self.memory.clear()

        self.params = np.array([math.pi / 36.0, 120000.0, 600, 9.8], dtype=float)

        self.inertia_tensor = np.diag(np.array([10000.0, 3000.0, 10000.0], dtype=float))


        self.thrust_center = np.array([0.0, 0.0, 0.0], dtype=float)

        self.buf = np.zeros_like(self.state)


        self.max_landv = 2.0
        self.max_vx = 0.8
        self.max_vz = 0.8
        self.max_vy = math.sqrt(2 * self.params[3] * 60)
        
        self.total_reward = 0.0
        self.init_distance_xz = np.linalg.norm(np.array([self.state_init[0], self.state_init[2]], dtype=float))

        self.num_step = -1
        state_out_init, _, _, _ = self.step(action=np.zeros((self.action_dim,), dtype=float))

        return state_out_init



    def truestate_outstate(self):
        state_ = np.copy(self.state)
        # norm
        state_[0] = self.state[0] / 8
        state_[1] = self.state[1] / 100
        state_[2] = self.state[2] / 8
        state_[3] = self.state[3] / (self.max_vx + 1e-4)
        state_[4] = self.state[4] / (self.max_vy + 1e-4)
        state_[5] = self.state[5] / self.max_vy

        state_[13] = self.state[13] / self.state_init[13]
        return state_

    def step(self, action):
        # action to real control
        control = np.copy(action)
        control[0] = np.clip(action[0], -1, 1) / 2 + 0.5
        assert (control[0] >= 0.0) and (control[0] <= 1.0)

        # calculate diff
        landing3d_diff(self.state, control.astype(float), self.params, self.inertia_tensor, self.thrust_center, self.buf)

        # update state
        self.state += self.delta_t * self.buf
        self.state[6:10] /= np.linalg.norm(self.state[6:10])
        assert (self.state[6] >= -1) and (self.state[6] <= 1)

        # state to state_out
        state_out = self.truestate_outstate()

        # calculate state_value
        # calculate theta between [0, 1, 0] and rocket attitude
        qw, qx, qy, qz = self.state[6:10]
        Rm = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ])
        attitude = Rm @ np.array([[0.0], [1.0], [0.0]])
        theta = math.acos(attitude[1] / np.linalg.norm(attitude))
        # calculate value_array which is used to calculate value
        (x, y, z, vx, vy, vz) = state_out[:6]
        cal_coefficient = np.array([1, 1, 1, 1, 1, 1, 1])
        value_array = np.array([
            x, y, z, vx, vy, vz, 0.0
        ]) * cal_coefficient
        # calculate value--reward
        value = 80 * math.exp(-np.linalg.norm(value_array))
        reward = np.clip(value - self.total_reward, 0.0, 1000.0)

        # game over or not
        done = False
        
        if np.abs(theta) > np.pi / 2:
            done = True
            reward = -50
        

        self.num_step += 1
        if self.num_step > self.max_step:
            done = True
            reward = -1 * self.total_reward
        
        if self.state[1] < 0 and (not done):
            done = True
            if np.linalg.norm(self.state[:3]) < 1:
                reward = self.total_reward * 0.25
            else:
                reward = -self.total_reward / 2

        # update total_reward
        self.total_reward += reward
        self.memory.push(self.state, theta)

        return state_out.astype(np.float32), reward, done, {}

    def render(self):
        #import cv2
        x, y, z = self.memory.getxyz()
        #fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(z, x, y, cmap='Blues')
        ax1.scatter3D(0.0, 0.0, 0.0, cmap='Reds')
        ax1.scatter3D(50, 50, 500, cmap='Greys')
        ax1.scatter3D(-50, -50, 0.0, cmap='Greys')
        plt.show()






    def save_points(self):

        save_path = "step%03d" % self.num_step + "_reward%3f" % self.total_reward + ".txt"
        print(save_path)
        np.savetxt(save_path, np.array(self.memory.memory)[:, :3])


if __name__ == "__main__":
    a = LandMars()
    a.reset()
    s = a.state
    print(s[:6])
    ep_r = 0
    for i in range(400):
        if i < 117:
            s_, r, done, _ = a.step(np.array([-0.5, 0.00, 0], dtype=np.float32))
        else:
            s_, r, done, _ = a.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        # print(s_[:6])
        print(r)
        ep_r += r
        s = s_
        if done:
            break
    print(a.state[0:6], a.state[13], a.num_step)
    print(ep_r)
    #a.save_points()
    a.render()





