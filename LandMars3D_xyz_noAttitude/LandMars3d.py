import numpy as np
import math
from numba_funcs import landing3d_diff
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sympy as sym
import optimal_a
from scipy.spatial.transform import Rotation as R

import random

class Memory:
    def __init__(self):
        self.memory = []
        self.theta = []
        self.memory_a = []
        self.control = []

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
        self.memory_a = []
        self.control = []


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
        self.delta_t = 0.04
        self.t = 0
        self.memory = Memory()
        self.max_step = 300
        self.max_step = 3000
        self.action_max = 1
        self.if_discrete = False
        self.continuous = True
        self.target_return = 3000
        self.thrust_min = 0.3

    def reset(self):

        self.state = np.array([np.random.uniform(-30, 30), np.random.uniform(400, 500), np.random.uniform(-30, 30),
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
        self.max_vx = 4
        self.max_vz = 4
        self.max_vy = math.sqrt(2 * self.params[3] * 300)
        
        self.total_reward = 0.0
        self.init_distance_xz = np.linalg.norm(np.array([self.state_init[0], self.state_init[2]], dtype=float))

        self.num_step = -1
        self.t = self.num_step * self.delta_t
        state_out_init, _, _, _ = self.step(action=np.zeros((self.action_dim,), dtype=float),
                                            noise=np.zeros((self.action_dim,), dtype=float))

        return state_out_init

    def truestate2outstate(self):
        state_ = np.copy(self.state)
        # norm
        state_[0] = self.state[0] / 30
        state_[1] = self.state[1] / 500
        state_[2] = self.state[2] / 30
        state_[3] = self.state[3] / (self.max_vx + 1e-4)
        state_[4] = self.state[4] / (self.max_vy + 1e-4)
        state_[5] = self.state[5] / self.max_vy

        state_[13] = self.state[13] / self.state_init[13]
        return state_

    def step(self, action, noise):
        # action to real control
        control = np.copy(action)
        control[0] = np.clip(action[0], -1, 1) / 2 + 0.5
        assert (control[0] >= 0.0) and (control[0] <= 1.0)

        # calculate diff
        landing3d_diff(self.state, control.astype(float), self.params, self.inertia_tensor, self.thrust_center, self.buf)

        self.buf[3:6] += noise
        # update state
        self.state += self.delta_t * self.buf
        self.state[6:10] /= np.linalg.norm(self.state[6:10])
        assert (self.state[6] >= -1) and (self.state[6] <= 1)

        # state to state_out
        state_out = self.truestate2outstate()

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
        self.t = self.num_step * self.delta_t

        if self.num_step > self.max_step:
            done = True
            reward = -1 * self.total_reward
        
        if self.state[1] < 0 and (not done):
            done = True
            if np.linalg.norm(self.state[:3]) < 4:
                reward = self.total_reward * 0.25
            else:
                reward = -self.total_reward / 2

        # update total_reward
        self.total_reward += reward

        # memory
        self.memory.push(self.state, theta)
        self.buf[4] += self.params[3]
        self.memory.memory_a.append(np.copy(self.buf))
        self.memory.control.append(control)

        # if done cal->optimal
        if done:
            self.tf = self.fun_optimaltime_determination()
            # print(self.tf)
            self.fun_get_episode()
            len_ = len(self.memory.memory_a) - 1
            self.tf_rl = len_ * self.delta_t

        return state_out.astype(np.float32), reward, done, {}

    def fun_optimaltime_determination(self):

        # 确定最优的转移时间 t_optimal
        state0 = self.state_init[:6]
        r0 = state0[0:3]
        v0 = state0[3:6]
        self.g = np.array([0, -self.params[3], 0], dtype=float)
        g = self.g
        tf = sym.symbols('tf')
        H_tf = np.linalg.norm(g) ** 2 / 2 * tf ** 4 - 2 * np.linalg.norm(v0) ** 2 * tf ** 2 - 12 * np.dot(v0, r0) * tf - 18 * np.linalg.norm(r0) ** 2
        t_optimal = sym.solve(H_tf, tf)
        return np.float(t_optimal[1])

    def fun_get_episode(self):

        self.rf = self.state[:3]
        self.vf = self.state[3:6]
        self.state0 = np.copy(self.state_init[:6])
        # print(self.state0)
        # print(self.rf, self.vf)

    def fun_ODE(self, state, t):
        # 动力学方程
        u = self.fun_controller1(state, t)
        # u2 = self.fun_controller2(state, t)
        # print(u, u2)
        state_dot = self.fun_dynamics(state, t, u)

        return state_dot

    def fun_controller1(self, state, t):

        # 最优控制规律，根据总的飞行时间计算
        r0 = self.state0[0:3]
        v0 = self.state0[3:6]
        rf = self.rf
        vf = self.vf
        g = self.g

        u1 = 6 * (self.tf * v0 + self.tf * vf + 2 * r0 - 2 * rf) / self.tf ** 3 * t
        u2 = 2 * (2 * self.tf * v0 + self.tf * vf + 3 * r0 - 3 * rf) / self.tf ** 2 + g

        u = u1 - u2
        return u

    def fun_controller2(self, state, t):

        # 最优控制规律，根据总的飞行时间计算
        r = state[0:3]
        v = state[3:6]
        rf = self.rf
        vf = self.vf
        g = self.g
        t_go = (self.tf-t)

        u2 = 2 * (2 * t_go * v + t_go * vf + 3 * r - 3 * rf) / t_go ** 2 + g

        u = - u2
        return u

    def fun_dynamics(self, state, t, u):
        # 动力学方程
        r = state[0:3]
        v = state[3:6]
        r_dot = v
        v_dot = np.copy(u) + self.g
        state_dot = np.hstack((r_dot, v_dot))
        return state_dot

    def render(self):
        # import cv2
        x, y, z = self.memory.getxyz()
        plt.figure(1)
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(z, x, y, cmap='Blues')
        ax1.scatter3D(0.0, 0.0, 0.0, cmap='Reds')
        ax1.scatter3D(50, 50, 500, cmap='Greys')
        ax1.scatter3D(-50, -50, 0.0, cmap='Greys')

        plt.figure(2)
        len_ = len(self.memory.memory_a) - 1
        t = np.linspace(0, len_ * self.delta_t, len_)
        a = np.linalg.norm(np.array(self.memory.memory_a, dtype=float)[1:, 3:6], axis=1)
        plt.plot(t, a, 'b', label='|a|')
        plt.figure(3)
        plt.plot(np.array(self.memory.control, dtype=float)[1:, 0], 'b', label='|a|')

        plt.show()

    def render_(self):

        import matplotlib.animation as animation
        import time

        x, y, z = self.memory.getxyz()
        vx, vy, vz = self.memory.getvxyz()
        theta = self.memory.gettheta()
        plt.ion();  # 开启interactive mode 成功的关键函数
        azim = 90 - math.atan(- x[0] / z[0]) / math.pi * 180
        elev = 30

        for i in range(len(x)):
            plt.clf()  # 清除之前画的图

            title = "x:%3f  " % z[i] + "y:%3f  " % x[i] + "z:%3f" % y[i] + "\n" + \
                "vx:%3f  " % vz[i] + "vy:%3f  " % vx[i] + "z:%3f" % vy[i] + "\n" + \
                "attitude:%3f rad" % theta[i] + "time:%1f s" % (i * self.delta_t)
            # print(title)
            fig = plt.gcf()  # 获取当前图
            ax = fig.gca(projection='3d')  # 获取当前轴
            ax.view_init(elev, azim)  # 设定角度
            ax.scatter3D(z[:i], x[:i], y[:i], cmap='blue')
            ax.scatter3D(0.0, 0.0, 0.0, cmap='Reds')
            ax.scatter3D(50, 50, 500, cmap='Greys')
            ax.scatter3D(-50, -50, 0.0, cmap='Greys')
            plt.title("%s" % title)
            if i==0 :
                plt.pause(10)
            plt.pause(0.01)  # 暂停一段时间，不然画的太快会卡住显示不出来

            elev, azim = ax.elev, ax.azim  # 将当前绘制的三维图角度记录下来，用于下一次绘制（放在ioff()函数前后都可以，但放在其他地方就不行）

            plt.ioff()  # 关闭画图窗口Z

        # 加这个的目的是绘制完后不让窗口关闭
        plt.show()

    def render_legend(self):
        t_tra = np.linspace(0, self.tf, 1000)
        state_tra = odeint(self.fun_ODE, self.state0, t_tra, args=())
        control_tra = np.array([self.fun_controller1(self.state0, t) for t in t_tra])
        x, y, z = self.memory.getxyz()
        len_ = len(self.memory.memory_a) - 1

        plt.figure(1)
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(state_tra[:, 2], state_tra[:, 0], state_tra[:, 1], 'red', label='op')  # 绘制空间曲线
        ax1.plot3D(z, x, y, 'blue', label='rl')
        plt.legend()
        ax1.scatter3D(0, 0, 0, cmap='Reds')  # 绘制散点图
        plt.title('total time_op:' + '%.2f' % self.tf + '\n' + 'total time_rl:' + str(self.delta_t * len_))
        plt.grid()



        t = np.linspace(0, self.tf_rl, len_)
        a = np.linalg.norm(np.array(self.memory.memory_a, dtype=float)[1:, 3:6], axis=1)

        plt.figure(2)
        plt.plot(t, a, 'blue', label='rl:|u|')
        plt.plot(t_tra, np.linalg.norm(control_tra, axis=1), 'red', label='op:|u|')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('u')
        plt.title('total time_op:' + '%.2f' % self.tf + '\n' + 'total time_rl:' + str(self.delta_t * len_))
        plt.grid()

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





