import sys
import airsim
import multiUAV1
import numpy as np
import math
import time
from scipy import linalg


def q_n_1(t):
    return np.array([2 * t, 0, -60], dtype=float)


def p_n_1(t):
    return np.array([2, 0, -0.5], dtype=float)


if __name__ == "__main__":

    airsim_on = True

    num_uav = 4

    name_uavc = 'UAVc'
    clientc = airsim.MultirotorClient()
    clients = []
    for i in range(num_uav):
        clients.append(airsim.MultirotorClient())

    uav_name = []
    for i in range(num_uav):
        uav_name.append("UAV" + str(i+1))

    origins = np.array([
        [50, -6, 0],
        [53, 0, 0],
        [56, 0, 0],
        [50, -3, 0]
    ], dtype=float)
    originc = np.array([50, 0, 0], dtype=float)
    
    rotation_integrator_controller_T = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])

    futures = []
    for i in range(num_uav):
        clients[i].enableApiControl(True, uav_name[i])  # 获取控制权
        clients[i].armDisarm(True, uav_name[i])  # 解锁（螺旋桨开始转动）
        future = clients[i].takeoffAsync(vehicle_name=uav_name[i])
        futures.append(future)
    clientc.enableApiControl(True, name_uavc)
    clientc.armDisarm(True, name_uavc)
    future = clientc.takeoffAsync(vehicle_name=name_uavc)
    futures.append(future)
    for i in range(num_uav):
        futures[i].join()

    print(origins)
    print(origins @ rotation_integrator_controller_T)

    futures = []
    for i in range(num_uav):
        future = clients[i].moveToZAsync(-20, 10, vehicle_name=uav_name[i])
        futures.append(future)
    future = clientc.moveToZAsync(-20, 10, vehicle_name=name_uavc)
    futures.append(future)
    for i in range(num_uav):
        futures[i].join()

    m = multiUAV1.MultiUAVController(num=num_uav, alpha=0, scale=4)
    m.o_mat = np.array([
        [80, 5, 60, 3],
        [100, -3, 60, 4],
    ], dtype=float)

    UAVs_q = np.zeros((num_uav, 3), dtype=float)
    UAVs_p = np.zeros((num_uav, 3), dtype=float)
    UAVc_q = np.zeros((3, ), dtype=float)
    UAVc_p = np.zeros((3, ), dtype=float)
    UAVc_p_ = np.zeros((3, ), dtype=float)

    clientc.moveByVelocityAsync(-2, 0, 0, 1000, vehicle_name=name_uavc)
    for step in range(2000):
        a0 = time.time()
        state = clientc.simGetGroundTruthKinematics(vehicle_name=name_uavc)
        UAVc_q = np.array([state.position.x_val, -state.position.y_val, -state.position.z_val])
        UAVc_p = np.array([state.linear_velocity.x_val, -state.linear_velocity.y_val, -state.linear_velocity.z_val])
        UAVc_q += originc @ rotation_integrator_controller_T
        for i in range(num_uav):
            state = clients[i].simGetGroundTruthKinematics(vehicle_name=uav_name[i])
            UAVs_q[i] = np.array([state.position.x_val, -state.position.y_val, -state.position.z_val])
            UAVs_p[i] = np.array([state.linear_velocity.x_val, -state.linear_velocity.y_val, -state.linear_velocity.z_val])
        UAVs_q += origins @ rotation_integrator_controller_T
        a = m.calculate_u(UAVs_q, UAVs_p, UAVc_q, UAVc_p, 0)
        UAVs_p += a
        for i in range(num_uav):
            clients[i].moveByVelocityAsync(UAVs_p[i, 0], -UAVs_p[i, 1], -UAVs_p[i, 2], 2.0, vehicle_name=uav_name[i])
        
        time.sleep(0.02)
        a1 = time.time()
        print(a1 - a0)

    # 循环结束
    for i in range(num_uav):
        clients[i].simPause(False)
    for i in range(num_uav):
        if i != num_uav - 1:  # 降落
            clients[i].landAsync(vehicle_name=uav_name[i])
        else:
            clients[i].landAsync(vehicle_name=uav_name[i])
    for i in range(num_uav):
        clients[i].armDisarm(False, vehicle_name=uav_name[i])  # 上锁
        clients[i].enableApiControl(False, vehicle_name=uav_name[i])  # 释放控制权

    # m.render()
