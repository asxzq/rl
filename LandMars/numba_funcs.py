
# °


import numpy as np
from numba import jit
import sys, os, math, random, time


# 参数：
# state: 13维 (
#       x, y, z (m), 
#       vx, vy, vz (m/s), 
#       qw, qx, qy, qz (无量纲), 
#       avx, avy, avz (rad/s), 
#       mass (kg)
#       )
# control: 3维 (
#       throttle (0~1无量纲),
#       gimbalx (-1~1无量纲), 
#       gimbalz (-1~1无量纲)
#       )
# params: 4维 (
#       max_gimbal (rad), 
#       max_thrust (N), 
#       isp (s), 
#       g (m/s^2)
#       )
# inertia_tensor: 3x3矩阵 （本体系下的）
# thrust_center 3维向量(m)
# buf 13维向量，用于存放函数输出结果

@jit(nopython=True, cache=True)
def landing3d_diff(state, control, params, inertia_tensor, thrust_center, buf):
    x, y, z, vx, vy, vz, qw, qx, qy, qz, avx, avy, avz, mass = state
    throttle, gimbalx, gimbalz = control
    max_gimbal, max_thrust, isp, g = params
    
    # 旋转矩阵
    local2world_r = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ])
    
    # 推力和力臂
    thrust_dir_local = np.array([-gimbalz * max_gimbal, 1.0, gimbalx * max_gimbal]) # tan小角近似
    thrust_dir_local /= np.linalg.norm(thrust_dir_local)
    thrust_dir = local2world_r @ thrust_dir_local
    thrust_mag = max_thrust * throttle
    arm = local2world_r @ thrust_center
    thrust = thrust_mag * thrust_dir
    
    # 加速度
    acc = thrust / mass
    
    # 角加速度
    torque = np.cross(arm, thrust)
    torque_local = local2world_r.T @ torque
    avel_local = local2world_r.T @ state[10:13]
    angular_acc_local = np.linalg.inv(inertia_tensor) @ (torque_local - np.cross(avel_local, inertia_tensor @ avel_local))
    angular_acc = local2world_r @ angular_acc_local
    
    # 质量变化率
    dm = - thrust_mag / (isp * 9.81)
    
    # 四元数与角速度的关系
    avxl, avyl, avzl = avel_local
    dq = 0.5 * np.array([
        [    0, -avxl, -avyl, -avzl],
        [ avxl,     0,  avzl, -avyl],
        [ avyl, -avzl,     0,  avxl],
        [ avzl,  avyl, -avxl,     0],
    ]) @ state[6:10]
    
    # 输出
    # xyz
    buf[0:3] = state[3:6]
    # xyz vel
    buf[3:6] = [
        acc[0],                                    # dvx
        acc[1] - g,                                # dvy
        acc[2]                                     # dvz
    ]
    # quaternion
    buf[6:10] = dq
    # angular vel
    buf[10:13] = angular_acc
    # mass
    buf[13] = dm


