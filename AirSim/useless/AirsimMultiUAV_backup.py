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

    origin_UAVs = np.array([
        [50, -6, -30],
        [53, 0, -30],
        [56, 0, -30],
        [50, -3, -30]
    ], dtype=float)
    origin_c = np.array([50, 0, -30], dtype=float)
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

    print(origin_UAVs)
    print(origin_UAVs @ rotation_integrator_controller_T)

    futures = []
    for i in range(num_uav):
        future = clients[i].moveToZAsync(-20, 10, vehicle_name=uav_name[i])
        futures.append(future)
    future = clientc.moveToZAsync(-70, 10, vehicle_name=name_uavc)
    futures.append(future)
    for i in range(num_uav):
        futures[i].join()

    m = multiUAV1.MultiUAVController(num=num_uav, alpha=0, scale=4)
    m.o_mat = np.array([
        [80, 5, 60, 3],
        [100, -3, 60, 4],
    ], dtype=float)

    UAVs_q_in_game_UAVs_origin = np.zeros((num_uav, 3), dtype=float)
    UAVs_q_in_game_Airsim_origin = np.zeros((num_uav, 3), dtype=float)
    UAVs_q_in_controller = np.zeros((num_uav, 3), dtype=float)

    UAVs_p_in_game = np.zeros((num_uav, 3), dtype=float)
    UAVs_p_in_controller = np.zeros((num_uav, 3), dtype=float)

    # 领航者相对领航者出生点的位置
    UAVc_q_in_game_UAVc_origin = np.zeros((3,), dtype=float)
    # 领航者相对于游戏原点的位置
    UAVc_q_in_game_Airsim_origin = np.zeros((3,), dtype=float)
    # 领航者在控制器中的位置
    UAVc_q_in_controller = np.zeros((3,), dtype=float)

    UAVc_p_in_game = np.zeros((3,), dtype=float)
    UAVc_p_in_controller = np.zeros((3,), dtype=float)
    # UAVc_p_ = np.zeros((3,), dtype=float)

    clientc.moveByVelocityAsync(-2, 0, 0, 2000, vehicle_name=name_uavc)
    for step in range(2500):

        if step == 400:
            print(m.c)
            m.make_c(num_uav, alpha=np.pi / 2, theta=0, scale=4)
            print(m.c)

        state = clientc.simGetGroundTruthKinematics(vehicle_name=name_uavc)
        UAVc_q_in_game_UAVc_origin = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
        UAVc_q_in_game_Airsim_origin = UAVc_q_in_game_UAVc_origin + origin_c
        UAVc_q_in_controller = UAVc_q_in_game_Airsim_origin @ rotation_integrator_controller_T
        UAVc_p_in_game = np.array(
            [state.linear_velocity.x_val, state.linear_velocity.y_val, state.linear_velocity.z_val])
        UAVc_p_in_controller = UAVc_p_in_game @ rotation_integrator_controller_T
        for i in range(num_uav):
            state = clients[i].simGetGroundTruthKinematics(vehicle_name=uav_name[i])
            UAVs_q_in_game_UAVs_origin[i] = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
            UAVs_q_in_game_Airsim_origin = UAVs_q_in_game_UAVs_origin + origin_UAVs
            UAVs_p_in_game[i] = np.array([state.linear_velocity.x_val, state.linear_velocity.y_val, state.linear_velocity.z_val])
        UAVs_q_in_controller = UAVs_q_in_game_Airsim_origin @ rotation_integrator_controller_T
        UAVs_p_in_controller = UAVs_p_in_game @ rotation_integrator_controller_T

        a = m.calculate_u(UAVs_q_in_controller, UAVs_p_in_controller, UAVc_q_in_controller, UAVc_p_in_controller, 0)
        UAVs_p_in_controller += a
        UAVs_p_in_game = UAVs_p_in_controller @ rotation_integrator_controller_T
        for i in range(num_uav):
            clients[i].moveByVelocityAsync(UAVs_p_in_game[i, 0], UAVs_p_in_game[i, 1], UAVs_p_in_game[i, 2], 2.0,
                                           vehicle_name=uav_name[i])

        time.sleep(0.02)


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