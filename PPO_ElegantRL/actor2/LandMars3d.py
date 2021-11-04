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
        p = math.asin(2 * (w * y - z * x))
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
        self.delta_t = 0.005
        self.memory = Memory()
        self.max_step = 400
        self.action_max = 1
        self.if_discrete = False
        self.continuous = True
        self.target_return = 180
        self.thrust_min = 0.3

    def reset(self):
        # 当地坐标系：x北极，y轴竖直向上，z轴东
        # 本体系：y纵轴向上，z轴在竖直平面内垂直x轴向�?
        self.state = np.array([0.3 * (random.random() - 0.5), 1.0, 0.3 * (random.random() - 0.5), 0.0, 0.0, 0.0, \
                               1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        self.init_y = 1.0
        # 力的单位是N
        self.params = np.array([math.pi / 18.0, 20, 300, 9.8], dtype=float)
        # 转动惯例
        self.inertia_tensor = np.eye(3, dtype=float)
        # 推力作用�?
        self.thrust_center = np.array([0.0, 0.0, 0.0])
        # 更新
        self.buf = np.zeros_like(self.state)
        self.init_distance_xz = np.linalg.norm(np.array([self.state[0], self.state[2]], dtype=float))
        self.max_landv = 2.0
        self.memory.clear()

        self.prev_shaping = None

        self.num_step = -1
        state_out_init, _, _, _ = self.step(action=np.zeros((self.action_dim,), dtype=np.float32))
        return state_out_init


    def reset_test(self):
        # 当地坐标系：x北极，y轴竖直向上，z轴东
        # 本体系：y纵轴向上，z轴在竖直平面内垂直x轴向�?
        self.state = np.array([0., 1.0, 0., 0.0, 0.0, 0.0, \
                               1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        self.init_y = 1.0
        # 力的单位是N
        self.params = np.array([math.pi / 18.0, 20, 300, 9.8], dtype=float)
        # 转动惯例
        self.inertia_tensor = np.eye(3, dtype=float)
        # 推力作用�?
        self.thrust_center = np.array([0.0, 0.0, 0.0])
        # 更新
        self.buf = np.zeros_like(self.state)
        self.init_distance_xz = np.linalg.norm(np.array([self.state[0], self.state[2]], dtype=float))
        self.max_landv = 2.0
        self.memory.clear()
        self.max_a = self.params[1] / self.state[13] - self.params[3]
        self.prev_shaping = None

        self.num_step = -1
        state_out_init, _, _, _ = self.step(action=np.zeros((self.action_dim,), dtype=np.float32))
        return state_out_init

    def truestate_outstate(self):
        state_ = np.copy(self.state)
        state_[1] = self.state[1]
        # 归一�?
        state_[4] = self.state[4] / self.params[3]
        return state_

    def step(self, action):

        control = np.copy(action)

        if action[0] > 0:
            control[0] = np.clip(action[0], 0.0, 1.0) * (1 - self.thrust_min) + self.thrust_min
            assert control[0] >= self.thrust_min and control[0] <= 1.0
        else:
            control[0] = 0.0
        #print(control)
        landing3d_diff(self.state, control, self.params, self.inertia_tensor, self.thrust_center, self.buf)

        self.state += self.delta_t * self.buf

        #print("state     :", self.state[:6])
        '''
        reward = -0.5 \
                 - self.buf[13] * self.delta_t \
                 + math.exp(1 - np.linalg.norm(self.state[:3]) / self.init_distance)  - 1.0 \
                 - np.linalg.norm(self.state[3:6]) / self.max_landv \
                 - np.dot(xz, vxz) / (np.linalg.norm(xz) * np.linalg.norm(vxz)) * 2
        '''
        # 输出的state,经过换算,适合训练
        state_out = self.truestate_outstate()
        #print("state_out :", state_out[:6])
        Euler = EulerAndQuaternionTransform(state_out[6:10])

        reward = 0
        # 位置、速度、姿�?
        shaping = -100 * np.sqrt(state_out[0] * state_out[0] + state_out[1] * state_out[1] + state_out[2] * state_out[2]) \
                -  10 * np.sqrt(state_out[3] * state_out[3] + state_out[4] * state_out[4] + state_out[5] * state_out[5]) \
                - 50 * (abs(Euler[0]) + abs(Euler[1]) + abs(Euler[2]))  \
                - 50 * np.sqrt(state_out[9] * state_out[9] + state_out[10] * state_out[10] + state_out[11] * state_out[11]) \
                + 10 * state_out[13]

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # 结束条件是，y<0
        # 质量衰减不能太多、不能向上飞，落点不能偏离太�?落地速度不能太大，落地角速度不能太大，落地姿态也不能太大
        done = False
        distance_xz = np.sqrt(self.state[0] ** 2 + self.state[2] ** 2)
        #if distance_xz > 2 * self.init_distance_xz:
            #reward = -100
            #done = True
        if self.state[1] < 0 and (not done):
            done = True

            if distance_xz <= self.init_distance_xz * 0.4:
                reward = 100
            elif distance_xz <= self.init_distance_xz * 0.8:
                reward = 30
            else:
                reward = -50

            if abs(self.state[4]) > self.max_landv:
                reward -= 100

        self.num_step += 1
        if (self.state[1] > 1.2 or self.num_step >= self.max_step) and (not done):
            reward = -100
            done = True

        self.memory.push(self.state)
        return state_out.astype(np.float32), reward, done, {}






    def render(self):
        x, y, z = self.memory.getxyz()
        #fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(z, x, y, cmap='Blues')  # 绘制散点�?
        ax1.scatter3D(0.0, 0.0, 0.0, cmap='Reds')
        ax1.scatter3D(0.5, 0.5, 1.0, cmap='Greys')
        ax1.scatter3D(-0.5, -0.5, 1.0, cmap='Greys')
        plt.show()



if __name__ == "__main__":
    a = LandMars()
    a.reset_test()
    s = a.state
    print(s[:6])
    ep_r = 0
    for i in range(200):
        if i < 95:
            s_, r, done, _ = a.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        else:
            s_, r, done, _ = a.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        # print(s_[:6])
        ep_r += r
        print(i,":",r)
        s = s_
        if done:
            break
    print(a.state[0:6], a.num_step)
    print(ep_r)
    a.render()





