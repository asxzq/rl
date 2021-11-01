

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
e = math.exp(1)

class LandMars(object):
    def __init__(self):
        self.name = "LandMars"
        self.state_dim = 14
        self.action_dim = 3
        self.delta_t = 0.005
        self.memory = Memory()

    def reset(self):
        # 当地坐标系：x北极，y轴竖直向上，z轴东
        # 本体系：y纵轴向上，z轴在竖直平面内垂直x轴向下
        self.state = np.array([0.3 * (random.random() - 0.5), 1.0, 0.3 * (random.random() - 0.5), 0.0, 0.0, 0.0, \
                               1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=float)
        # 力的单位是N
        self.params = np.array([math.pi / 18.0, 40, 300, 9.8], dtype=float)
        # 转动惯例
        self.inertia_tensor = np.eye(3, dtype=float)
        # 推力作用点
        self.thrust_center = np.array([0.0, 0.0, 0.0])
        # 更新
        self.buf = np.zeros_like(self.state)
        self.init_distance = np.linalg.norm(self.state[:3])
        self.init_distance_xz = np.linalg.norm(np.array([self.state[0], self.state[2]], dtype=float))
        self.max_landv = 2.0
        self.memory.clear()
        return self.state

    def step(self, control):
        #print(control)
        landing3d_diff(self.state, control, self.params, self.inertia_tensor, self.thrust_center, self.buf)
        self.state += self.delta_t * self.buf
        xz = np.array([self.state[0], self.state[2]], dtype=float)
        vxz = np.array([self.state[3], self.state[5]], dtype=float)
        # step dm distance
        reward = -0.5 \
                 - self.buf[13] * self.delta_t \
                 + math.exp(1 - np.linalg.norm(xz) / self.init_distance_xz) - 1.0 \
                 - np.linalg.norm(self.state[3:6]) / self.max_landv \
                 - np.dot(xz, vxz) / (np.linalg.norm(xz) * np.linalg.norm(vxz)) \
                 - self.state[1] * self.state[4] / abs(self.state[4]) / abs(self.state[1])
                 #- np.dot(self.state[3:6], self.state[0:3]) * 1.5
        #print(reward, control)
        # 结束条件是，y<0
        # 质量衰减不能太多、不能向上飞，落点不能偏离太多,落地速度不能太大，落地角速度不能太大，落地姿态也不能太大
        if self.state[1] < 0 or self.state[13] < 1 or self.state[1] > 1.2:
            done = True
            #print(self.state[13],np.linalg.norm(self.state[:3]))
            reward = 5 * (math.exp(1 - np.linalg.norm(xz) / self.init_distance_xz) - 1.0)

            if np.linalg.norm(self.state[:3]) > 0.05 * self.init_distance:
                reward -= 20
        else:
            done = False
        self.memory.push(self.state)
        return self.state, reward, done, 1

    def render(self):
        x,y,z = self.memory.getxyz()
        #fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(z, x, y, cmap='Blues')  # 绘制散点图
        ax1.scatter3D(0.0, 0.0, 0.0, cmap='Reds')
        ax1.scatter3D(1.0, 1.0, 1.0, cmap='Greys')
        plt.show()



if __name__ == "__main__":
    a = LandMars()
    a.reset()
    s = a.state
    print(s[:6])
    for i in range(10000):
        s_, r, done = a.step(np.array([1.0, 0.0, 0.0],dtype=float))
        # print(s_[:6])
        s = s_
    a.render()





