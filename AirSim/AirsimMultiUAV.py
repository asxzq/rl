import sys
import airsim
import multiUAV
import numpy as np
import math
import time
from scipy import linalg
sys.path.append("F:\\code\\ReinforceLearning\\easy-rl\\mycodes\\rl\\AirSim")

def move_by_acceleration_horizontal(orientation, ax_cmd, ay_cmd, z_cmd, duration=1, vehicle_name=''):
    # 读取自身yaw角度
    # state = client.simGetGroundTruthKinematics()
    angles = airsim.to_eularian_angles(orientation)
    yaw_my = angles[2]
    g = 9.8  # 重力加速度
    sin_yaw = math.sin(yaw_my)
    cos_yaw = math.cos(yaw_my)
    A_psi = np.array([[sin_yaw, cos_yaw], [-cos_yaw, sin_yaw]])
    A_psi_inverse = np.linalg.inv(A_psi)
    angle_h_cmd = 1/g * np.dot(A_psi_inverse, np.array([[-ax_cmd], [-ay_cmd]]))
    a_x_cmd = math.atan(angle_h_cmd[0, 0])
    a_y_cmd = -math.atan(angle_h_cmd[1, 0])
    client.moveByRollPitchYawZAsync(a_x_cmd, a_y_cmd, 0, z_cmd, duration, vehicle_name=vehicle_name)
    return


def q_n_1(t):
    return np.array([2 * t, 0, -60], dtype=float)


def p_n_1(t):
    return np.array([2, 0, 0], dtype=float)


if __name__ == "__main__":

    airsim_on = True

    uav_num = 9
    clients = []
    for i in range(uav_num):
        clients.append(airsim.MultirotorClient())
    uav_name = []
    for i in range(uav_num):
        uav_name.append("UAV" + str(i+1))
    futures = []
    for i in range(uav_num):
        clients[i].enableApiControl(True, uav_name[i])  # 获取控制权
        clients[i].armDisarm(True, uav_name[i])  # 解锁（螺旋桨开始转动）
        future = clients[i].takeoffAsync(vehicle_name=uav_name[i])
        futures.append(future)

    for i in range(uav_num):
        futures[i].join()

    futures = []
    for i in range(uav_num):
        future = clients[i].moveToZAsync(-60, 10, vehicle_name=uav_name[i])
        futures.append(future)

    for i in range(uav_num):
        futures[i].join()

    m = multiUAV.MultiUAV(num=uav_num, alpha=0, scale=4)
    print(m.c)
    m.o_mat = np.array([
        [80, 5, 60, 3],
        [100, -3, 60, 4],
    ], dtype=float)

    for step in range(1500):
        futures = []
        m.q_d = q_n_1(m.t)
        m.q_d[1] *= -1
        m.q_d[2] *= -1
        m.p_d = p_n_1(m.t)
        m.p_d[1] *= -1
        m.p_d[2] *= -1
        if step == 0:
            for i in range(uav_num):
                state = clients[i].simGetGroundTruthKinematics(vehicle_name=uav_name[i])
                m.q_mat[i] = np.array([state.position.x_val, -state.position.y_val, -state.position.z_val])
                m.p_mat[i] = np.array([state.linear_velocity.x_val, -state.linear_velocity.y_val, -state.linear_velocity.z_val])
            print(m.q_mat)
        true_q = np.zeros_like(m.q_mat, dtype=float)

        m.step()

        for i in range(uav_num):
            state = clients[i].simGetGroundTruthKinematics(vehicle_name=uav_name[i])
            true_q[i] = np.array([state.position.x_val, -state.position.y_val, -state.position.z_val])
        m.memory_q_true.append(true_q)
        for i in range(uav_num):
            clients[i].moveByVelocityAsync(m.p_mat[i, 0], -m.p_mat[i, 1], -m.p_mat[i, 2], 1.0, vehicle_name=uav_name[i])
            # clients[i].moveByVelocityAsync(0, 10, 10, 1.0, vehicle_name=uav_name[i])

        # time.sleep(m.delta_t * 0.95)

    # 循环结束
    for i in range(uav_num):
        clients[i].simPause(False)
    for i in range(uav_num):
        if i != uav_num - 1:  # 降落
            clients[i].landAsync(vehicle_name=uav_name[i])
        else:
            clients[i].landAsync(vehicle_name=uav_name[i])
    for i in range(uav_num):
        clients[i].armDisarm(False, vehicle_name=uav_name[i])  # 上锁
        clients[i].enableApiControl(False, vehicle_name=uav_name[i])  # 释放控制权

    m.render()
