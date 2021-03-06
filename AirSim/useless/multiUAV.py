import numpy as np
import matplotlib.pyplot as plt
import math


def norm_sigma(point, zeta=0.2):
    if type(point) == np.ndarray:
        dim = point.ndim
        if dim == 3:
            norm2 = np.linalg.norm(point, axis=2)
            return (np.sqrt(1 + norm2 * norm2 * zeta) - 1) / zeta
        else:
            norm2 = np.linalg.norm(point, axis=1).T
            return (np.sqrt(1 + norm2 * norm2 * zeta) - 1) / zeta
    else:
        return (np.sqrt(1 + point * point * zeta) - 1) / zeta


def delta_norm_sigma(point, zeta=0.2):
    if type(point) == np.ndarray:
        dim = point.ndim
        if dim == 3:
            norm2 = np.linalg.norm(point, axis=2)
            out = np.copy(point)
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    out[i, j] /= math.sqrt(1 + norm2[i, j] * norm2[i, j] * zeta)
            return out
        else:
            norm2 = np.linalg.norm(point, axis=1)
            return point / np.sqrt(1 + norm2 * norm2 * zeta)
    else:
        return point / np.sqrt(1 + point * point * zeta)


'''
# 势能函数 a
# 无人机间的斥力势能
def Psi_a(z, k1=1, d1=2, zeta=0.3):
    NormSigma_d1 = norm_sigma(d1, zeta)
    # print(NormSigma_d1)
    z_ = np.clip(z, 0, NormSigma_d1)
    return k1 * np.power((z_ - NormSigma_d1), 2)

# 无人机间的斥力
def DeltaPsi_a(z, k1=1, d1=2, zeta=0.3):
    NormSigma_d1 = norm_sigma(d1, zeta)
    # print(NormSigma_d1)
    z_ = np.clip(z, 0, NormSigma_d1)
    return 2 * k1 * (z_ - NormSigma_d1)


# 势能函数 b
# 无人机间的引力势能
def Psi_b(z, k2=0.2, d2=10, r=12, zeta=0.3):
    NormSigma_d2 = norm_sigma(d2, zeta)
    NormSigma_r = norm_sigma(r, zeta)
    print(NormSigma_d2, NormSigma_r)
    z_ = np.clip(z, NormSigma_d2, NormSigma_r)
    return k2 * np.power((z_ - NormSigma_d2), 2)

# 无人机间的引力
def DeltaPsi_b(z, k2=0.2, d2=10, r=12, zeta=0.3):
    NormSigma_d2 = norm_sigma(d2, zeta)
    NormSigma_r = norm_sigma(r, zeta)
    z_ = np.clip(z, NormSigma_d2, NormSigma_r)
    return 2 * k2 * (z_ - NormSigma_d2)
'''


class Data:
    def __init__(self):
        self.q_mat = np.array([
            [-15, 10, 4],
            [8, -15, 3],
            [5, 4, 2],
            [-5, -8, 1]
        ], dtype=float)


class MultiUAV:
    # def __init__(self, num=4, alpha=np.pi/2, theta=0, scale=3, k1=1, k2=0.8, k3=0.3, zeta=0.2, d1=2, d2=10, r=12):
    def __init__(self, num=4, alpha=np.pi/2, theta=np.pi/2, scale=3, k1=0.6, k2=0.8, k3=0.9,
                 zeta=0.2, d1=2, d2=10, d3=5, r=100):
        self.num_uav = num
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.zeta = zeta
        self.d1 = d1
        self.d2 = d2
        # 无人机感知障碍的半径为d3
        self.d3 = d3
        self.r = r

        self.c = np.zeros((self.num_uav, 3), dtype=float)
        self.make_c(num, alpha, theta, scale)

        self.A = np.zeros((self.num_uav, self.num_uav), dtype=float)
        self.L = np.zeros((self.num_uav, self.num_uav), dtype=float)
        self.H = np.zeros((self.num_uav, self.num_uav), dtype=float)
        self.make_constant()

        self.t = 0
        self.delta_t = 0.02

        self.q_mat = np.zeros((self.num_uav, 3), dtype=float)
        self.p_mat = np.zeros((self.num_uav, 3), dtype=float)
        self.q_d = np.array([0, 0, 0], dtype=float)
        self.p_d = np.array([0, 0, 0], dtype=float)

        # 障碍物集合，坐标 半径
        self.o_mat = np.array([
            [2, 80, 20, 3],
            [-3, 50, 20, 4],
        ], dtype=float)
        # 障碍物相对无人机的位置
        self.x_ik = np.zeros((self.num_uav, self.o_mat.shape[0], 3), dtype=float)

        # 无人机相对虚领航者的位置 / 速度
        self.x = np.zeros_like(self.q_mat, dtype=float)
        self.v = np.zeros_like(self.p_mat, dtype=float)
        self.make_x__v()

        # 无人机间相对位置
        self.x_ij = np.zeros((self.num_uav + 1, self.num_uav + 1, 3), dtype=float)
        self.NormSigma_x_ij = np.zeros((self.num_uav + 1, self.num_uav + 1), dtype=float)
        self.Delta_NormSigma_x_ij = np.zeros((self.num_uav + 1, self.num_uav + 1), dtype=float)

        self.memory_x = []
        self.memory_q = []
        self.memory_q_true = []
        self.memory_v = []

    def make_constant(self):
        ones = np.ones((self.num_uav, self.num_uav), dtype=float)
        eyes = np.eye(self.num_uav)
        self.A = ones - eyes
        self.L = self.num_uav * eyes - ones
        self.H = self.L + eyes

    def test_make_constant(self):
        self.A = np.array([
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=float)
        self.L = np.array([
            [2, -1, 0, -1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 1]
        ], dtype=float)
        self.H = np.array([
            [2, -1, 0, -1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 1]
        ], dtype=float)

    def test_init(self):
        # 位置q 和速度p
        self.q_mat = np.array([
            [-15, 3, 24],
            [8, -12, 23],
            [5, 2, 17],
            [-5, -8, 15]
        ], dtype=float)
        self.q_d = self.q_n_1()
        self.p_mat = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=float)
        self.p_d = self.p_n_1()

    def load(self, data):
        pass

    def make_x__v(self):
        for i in range(self.num_uav):
            self.x[i] = self.q_mat[i] - self.q_d
        for i in range(self.num_uav):
            self.v[i] = self.p_mat[i] - self.p_d

    def make_x_ij(self):
        num = self.x.shape[0]
        for i in range(num + 1):
            self.x_ij[i, i] = np.zeros((3,), dtype=float)
            for j in range(i + 1, num + 1):
                if j < num:
                    self.x_ij[i, j] = self.x[j] - self.x[i]
                else:
                    if i < num:
                        self.x_ij[i, j] = -self.x[i]
                    else:
                        self.x_ij[i, j] = self.x[0] - self.x[0]
                self.x_ij[j, i] = -1.0 * self.x_ij[i, j]

    def step(self):

        dx, dv = self.calculate_u()

        self.x += dx * self.delta_t
        self.v += dv * self.delta_t
        self.t += self.delta_t
        # self.q_d = self.q_n_1()
        # self.p_d = self.p_n_1()

        for i in range(self.num_uav):
            self.q_mat[i] = self.x[i] + self.q_d
            self.p_mat[i] = self.v[i] + self.p_d

        self.memory_x.append(np.copy(self.x))
        self.memory_v.append(np.copy(self.v))
        self.memory_q.append(np.copy(self.q_mat))

    def calculate_u(self):

        self.make_x__v()
        self.make_x_ij()
        self.NormSigma_x_ij = norm_sigma(self.x_ij, self.zeta)
        self.Delta_NormSigma_x_ij = delta_norm_sigma(self.x_ij, self.zeta)

        f_a = self.f_a()
        f_b = self.f_b()
        f_c = self.f_c()
        f = f_a + f_b + f_c
        dx = np.copy(self.v)
        dv = -self.k1 * self.H @ self.x - self.k2 * self.v + f + self.k1 * self.H @ self.c

        return dx, dv

    def f_a(self):
        f_a = np.zeros_like(self.x, dtype=float)
        delta_psi_a = self.delta_psi_a()
        for i in range(f_a.shape[0]):
            for j in range(self.NormSigma_x_ij.shape[1]):
                f_a[i] += delta_psi_a[i, j] * self.Delta_NormSigma_x_ij[i, j]
        return f_a

    # 必须内部定义，不然报错
    def delta_psi_a(self):
        norm_sigma_d1 = norm_sigma(self.d1)
        z_ = np.clip(self.NormSigma_x_ij, 0, norm_sigma_d1)
        return 2 * self.k1 * (z_ - norm_sigma_d1)

    def f_b(self):
        f_b = np.zeros_like(self.x, dtype=float)
        delta_psi_b = self.delta_psi_b()
        for i in range(f_b.shape[0]):
            for j in range(self.NormSigma_x_ij.shape[1]):
                f_b[i] += delta_psi_b[i, j] * self.Delta_NormSigma_x_ij[i, j]
        return f_b

    # 无人机间的引力
    def delta_psi_b(self):
        norm_sigma_d2 = norm_sigma(self.d2)
        norm_sigma_r = norm_sigma(self.r)
        z_ = np.clip(self.NormSigma_x_ij, norm_sigma_d2, norm_sigma_r)
        return 2 * self.k2 * (z_ - norm_sigma_d2)

    def f_c(self):
        f_c = np.zeros_like(self.x, dtype=float)
        for i in range(self.num_uav):
            for k in range(self.o_mat.shape[0]):
                tmp = self.o_mat[k, :3] - self.q_mat[i]
                self.x_ik[i, k] = tmp * (1 - self.o_mat[k, 3] / np.linalg.norm(tmp))
        delta_psi_c = self.delta_psi_c()
        delta_norm_sigma_xo_ik = delta_norm_sigma(self.x_ik, zeta=self.zeta)
        predict = self.predict_crash()

        for i in range(self.num_uav):
            for k in range(self.o_mat.shape[0]):
                f_c[i] += predict[i, k] * delta_psi_c[i, k] * delta_norm_sigma_xo_ik[i, k]
        return f_c

    def delta_psi_c(self):
        norm_sigma_d3 = norm_sigma(self.d3)
        z_ = norm_sigma(self.x_ik, zeta=self.zeta)
        return 2 * self.k3 * (z_ - norm_sigma_d3) * (z_ < norm_sigma_d3)

    def predict_crash(self):
        predict = np.zeros((self.num_uav, self.o_mat.shape[0]), dtype=float)
        norm_p = np.linalg.norm(self.p_mat, axis=1) + 1e-6
        norm_ik = np.linalg.norm(self.x_ik, axis=2) + 1e-6
        for k in range(self.o_mat.shape[0]):
            norm_ik[:, k] += self.o_mat[k, 3]
        for i in range(self.num_uav):
            for k in range(self.o_mat.shape[0]):
                cos = np.dot(self.p_mat[i], self.x_ik[i, k]) / norm_p[i] / norm_ik[i, k]
                sin = np.cross(self.p_mat[i], self.x_ik[i, k]) / norm_p[i] / norm_ik[i, k]
                if cos < 0:
                    predict[i, k] = 0
                else:
                    if np.linalg.norm(sin) * norm_ik[i, k] <= self.o_mat[k, 3]:
                        predict[i, k] = 1
                    else:
                        predict[i, k] = 0
        return predict

    def q_n_1(self):
        return np.array([
            # -10 * math.cos(self.t), 10 * math.sin(self.t), 10
            # -1 * math.cos(self.t), 3 * self.t, 1 * math.sin(self.t)
            -0 * math.cos(self.t), 2 * self.t, 20
        ], dtype=float)

    def p_n_1(self):
        return np.array([
            # 1 * math.sin(self.t), 3, 1 * math.cos(self.t)
            0 * math.sin(self.t), 2, 0
        ], dtype=float)

    def make_c(self, num, alpha, theta, scale):
        rotation1 = self.rotation(axis=2, theta=alpha)
        rotation2 = self.rotation(axis=1, theta=-theta)
        rotation = np.matmul(rotation2, rotation1)
        c_init = np.zeros((num, 3))
        for i in range(num):
            c_init[i, 0] = math.cos(i * np.pi / num * 2)
            c_init[i, 1] = math.sin(i * np.pi / num * 2)
        self.c = scale * np.matmul(c_init, rotation)

    @staticmethod
    # axis = 0, 1, 2
    def rotation(axis, theta):
        ro = np.zeros((3, 3), dtype=float)
        ro[axis, axis] = 1
        ro[(axis + 1) % 3, (axis + 1) % 3] = math.cos(theta)
        ro[(axis + 2) % 3, (axis + 1) % 3] = math.sin(-theta)
        ro[(axis + 1) % 3, (axis + 2) % 3] = math.sin(theta)
        ro[(axis + 2) % 3, (axis + 2) % 3] = math.cos(theta)
        return ro

    def render(self):
        points_x = np.array(self.memory_x)
        points_q = np.array(self.memory_q)
        points_q_true = np.array(self.memory_q_true)
        points_v = np.array(self.memory_v)
        n_points = len(self.memory_x)
        t = np.linspace(0, self.t, n_points)
        color = ['r', 'b', 'y', 'c', 'g', 'm', 'peru', 'gray', 'pink']
        label = ['UAV1', 'UAV2', 'UAV3', 'UAV4', 'UAV5', 'UAV6', 'UAV7', 'UAV8', 'UAV9']

        plt.figure(1)
        plt.title('x_y_z')
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(points_x[0, :, 0], points_x[0, :, 1], points_x[0, :, 2], marker="v", s=20)
        for i in range(self.num_uav):
            ax1.plot3D(points_x[:, i, 0], points_x[:, i, 1], points_x[:, i, 2], color=color[i], label=label[i])
        ax1.scatter3D(points_x[n_points - 1, :, 0], points_x[n_points - 1, :, 1], points_x[n_points - 1, :, 2],
                      marker="*", s=100)
        plt.legend()

        plt.figure(2)
        ax1 = plt.axes(projection='3d')
        ax1.set_ylim(-30, 30)
        ax1.set_zlim(0, 100)
        ax1.scatter3D(points_q[0, :, 0], points_q[0, :, 1], points_q[0, :, 2], marker="v", s=100)
        for i in range(self.num_uav):
            ax1.plot3D(points_q[:, i, 0], points_q[:, i, 1], points_q[:, i, 2], color=color[i], label=label[i])

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        for k in range(self.o_mat.shape[0]):
            x = self.o_mat[k, 3] * np.outer(np.cos(u), np.sin(v)) + self.o_mat[k, 0]
            y = self.o_mat[k, 3] * np.outer(np.sin(u), np.sin(v)) + self.o_mat[k, 1]
            z = self.o_mat[k, 3] * np.outer(np.ones(np.size(u)), np.cos(v)) + self.o_mat[k, 2]
            # Plot the surface
            # ax.plot_surface(x, y, z, color='b')
            ax1.scatter3D(x, y, z, color=color[8 - k], marker=".", s=12)

        ax1.scatter3D(points_q[n_points - 1, :, 0], points_q[n_points - 1, :, 1], points_q[n_points - 1, :, 2],
                      marker="*", s=100)
        plt.legend()

        fig, ax = plt.subplots(1, 3)
        ax[0].set_title("x")
        ax[1].set_title("y")
        ax[2].set_title("z")
        for j in range(3):
            for i in range(self.num_uav):
                ax[j].plot(t, points_x[:, i, j], color=color[i], label=label[i])
            ax[j].legend()

        fig, ax = plt.subplots(1, 3)
        ax[0].set_title("vx")
        ax[1].set_title("vy")
        ax[2].set_title("vz")
        for j in range(3):
            for i in range(self.num_uav):
                ax[j].plot(t, points_v[:, i, j], color=color[i], label=label[i])
            ax[j].legend()


        plt.figure(5)
        ax1 = plt.axes(projection='3d')
        ax1.set_ylim(-30, 30)
        ax1.set_zlim(0, 100)
        ax1.scatter3D(points_q_true[0, :, 0], points_q_true[0, :, 1], points_q_true[0, :, 2], marker="v", s=100)
        for i in range(self.num_uav):
            ax1.plot3D(points_q_true[:, i, 0], points_q_true[:, i, 1], points_q_true[:, i, 2], color=color[i], label=label[i])
        ax1.scatter3D(points_q_true[n_points - 1, :, 0], points_q_true[n_points - 1, :, 1], points_q_true[n_points - 1, :, 2],
                      marker="*", s=100)
        ax1.scatter3D(points_q[n_points - 1, :, 0], points_q[n_points - 1, :, 1], points_q[n_points - 1, :, 2],
                      marker="*", s=100)

        # plt.figure(5)

        plt.show()


if __name__ == "__main__":
    m = MultiUAV()
    m.test_init()
    m.test_make_constant()
    for _ in range(3000):
        m.step()

    m.make_c(m.num_uav, alpha=-np.pi/2, theta=0, scale=4)

    for _ in range(2000):
        m.step()
    print(m.x)
    print(m.q_d)
    m.render()





