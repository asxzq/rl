
import numpy as np
from scipy.integrate import odeint
import sympy as sym
import sys
# sys.path.append("c:/Users/Lynn/PycharmProjects/20210310_Landing_Problem_Rearrange/2_Energy_Optimal/1_Problem")
# from Problem import Landing_Earth_NoMass_NoThrustMagnitudeConstraint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Landing_Earth_NoMass_NoThrustMagnitudeConstraint:
    def __init__(self):
        self.g = np.array([0, 9.8, 0], dtype=float)
        self.state0 = self.reset_state()
        self.rf = np.array([0, 0, 0], dtype=float)
        self.vf = np.array([0, 0, 0], dtype=float)

    def reset_state(self):
        return np.array([np.random.uniform(-30, 30), np.random.uniform(400, 500), np.random.uniform(-30, 30),
                               0.0, 0.0, 0.0,])


    def fun_dynamics(self, state, t, u):


        state_out = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        state_out[0] = state[3]
        state_out[1] = state[4]
        state_out[2] = state[5]
        state_out[3] = u[0]
        state_out[4] = u[1]
        state_out[5] = u[2]
        dydt = [state[3], state[4], state[5], u[0], u[1], u[2]]
        return dydt

class Simulation(object):

    def __init__(self):

        self.pro = Landing_Earth_NoMass_NoThrustMagnitudeConstraint()
        self.tf = self.fun_optimaltime_determination()
        print(self.tf)
        self.fun_get_episode()

    def fun_optimaltime_determination(self):

        # 确定最优的转移时间 t_optimal
        state0 = self.pro.state0
        r0 = state0[0:3]
        v0 = state0[3:6]
        g = self.pro.g

        tf = sym.symbols('tf')
        H_tf = np.linalg.norm(g) ** 2 / 2 * tf ** 4 - 2 * np.linalg.norm(v0)**2 * tf**2 - 12 * np.dot(v0, r0) *tf - 18 * np.linalg.norm(r0)**2
        t_optimal = sym.solve(H_tf, tf)
        print(t_optimal)

        return np.float(t_optimal[1])


    def fun_get_episode(self):

        self.pro.reset_state()
        self.state0 = np.copy(self.pro.state0)
        t_tra = np.linspace(0, self.tf, 1000)
        state_tra = odeint(self.fun_ODE, self.state0, t_tra, args=())
        control_tra = np.array([self.fun_controller1(self.state0, t) for t in t_tra])
        #
        #for i in range(len(t_tra)):
            #print(state_tra[i])
        plt.figure(1)
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(state_tra[:, 0], state_tra[:, 1], state_tra[:, 2], 'blue', label='r')  # 绘制空间曲线
        plt.legend()
        ax1.scatter3D(0, 0, 0, cmap='Reds')  # 绘制散点图
        plt.title('total time:' + str(self.tf))
        plt.grid()

        plt.figure(2)
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(state_tra[:, 3], state_tra[:, 4], state_tra[:, 5], 'blue', label='v')  # 绘制空间曲线
        plt.legend()
        plt.title('total time:' + str(self.tf))
        plt.grid()

        plt.figure(3)
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(control_tra[:, 0], control_tra[:, 1], control_tra[:, 2], 'blue', label='u')  # 绘制空间曲线
        plt.legend()
        plt.title('total time:' + str(self.tf))
        plt.grid()

        plt.figure(4)
        plt.plot(t_tra, control_tra[:, 0], 'b', label='ux')
        plt.plot(t_tra, control_tra[:, 1], 'r', label='uy')
        plt.plot(t_tra, control_tra[:, 2], 'm:', label='uz')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('u')
        plt.title('total time:' + str(self.tf))
        plt.grid()

        plt.figure(5)
        plt.plot(t_tra, np.linalg.norm(control_tra, axis=1), 'b', label='|u|')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('u')
        plt.title('total time:' + str(self.tf))
        plt.grid()

        # plt.figure(6)
        # plt.plot(t_tra, state_tra[:, -1], 'b', label='energy(t)')
        # plt.legend(loc='best')
        # plt.xlabel('t')
        # plt.ylabel('energy')
        # plt.title('total time:' + str(self.tf))
        # plt.grid()

        plt.show()

    def fun_ODE(self, state, t):
        # 动力学方程
        u = self.fun_controller2(state, t)
        # u2 = self.fun_controller2(state, t)
        # print(u, u2)
        state_dot = self.pro.fun_dynamics(state, t, u)

        return state_dot

    def fun_controller1(self, state, t):

        # 最优控制规律，根据总的飞行时间计算
        r0 = self.state0[0:3]
        v0 = self.state0[3:6]
        rf = self.pro.rf
        vf = self.pro.vf
        g = self.pro.g

        u1 = 6 * (self.tf * v0 + self.tf * vf + 2 * r0 - 2 * rf) / self.tf ** 3 * t
        u2 = 2 * (2 * self.tf * v0 + self.tf * vf + 3 * r0 - 3 * rf) / self.tf ** 2 + g

        u = u1 - u2
        return u

    def fun_controller2(self, state, t):

        # 最优控制规律，根据总的飞行时间计算
        r = state[0:3]
        v = state[3:6]
        rf = self.pro.rf
        vf = self.pro.vf
        g = self.pro.g
        t_go = (self.tf-t)

        u2 = 2 * (2 * t_go * v + t_go * vf + 3 * r - 3 * rf) / t_go ** 2 + g

        u = - u2
        return u



if __name__ == '__main__':

    simu = Simulation()


