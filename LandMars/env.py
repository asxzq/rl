

import numpy as np
import math
from numba_funcs import landing3d_diff
import matplotlib.pyplot as plt

class Memory:
    def __init__(self):
        self.memory = []

    def push(self, state):
        self.memory.append(np.copy(state))

    def getxyz(self):
        m = np.array(self.memory, dtype=float)
        return m[:, 0], m[:, 1], m[:, 2]

class LandMars(object):
    def __init__(self):
        self.name = "LandMars"
        self.state_dim = 14
        self.action_dim = 3
        self.delta_t = 0.01
        self.memory = Memory()

    def reset(self):
        # 当地坐标系：x北极，y轴竖直向上，z轴东
        # 本体系：y纵轴向上，z轴在竖直平面内垂直x轴向下
        self.state = np.array([1.0, 1.0, 1.0, -2.0, 0.0, 2.0, \
                               1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=float)
        # 力的单位是N
        self.params = np.array([math.pi / 18.0, 40, 300, 9.8], dtype=float)
        # 转动惯例
        self.inertia_tensor = np.eye(3, dtype=float)
        # 推力作用点
        self.thrust_center = np.array([0.0, -1.0, 0.0])
        # 更新
        self.buf = np.zeros_like(self.state)


    def step(self, control):
        landing3d_diff(self.state, control, self.params, self.inertia_tensor, self.thrust_center, self.buf)
        self.state += self.delta_t * self.buf
        reward = -1.0 - self.buf[13] * self.delta_t
        # 结束条件是，y<0
        # 质量衰减不能太多、不能向上飞，落点不能偏离太多,落地速度不能太大，落地角速度不能太大，落地姿态也不能太大
        if self.state[1] < 0:
            done = True
        else:
            done = False
        self.memory.push(self.state)
        return self.state, reward, done

    def render(self):
        x,y,z = self.memory.getxyz()
        #fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(x, y, z, cmap='Blues')  # 绘制散点图
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
    print(a.memory.memory)
    a.render()





