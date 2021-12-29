from agent import AgentPPO
import torch
import gym
from LandMars3d import LandMars
import numpy as np
import matplotlib.pyplot as plt
import math
import random

env = LandMars()
# state_dim = env.observation_space.shape[0]
state_dim = env.state_dim 

# action_dim = env.action_space.shape[0]
action_dim = env.action_dim

agent = AgentPPO()
agent.init(512, state_dim, action_dim)
agent.act.load_state_dict(torch.load("AgentPPO_LandMars_0/actor.pth"))
agent.act.eval()


def make_noise(t, flag):
    if flag == 1:
        return np.random.uniform(-1, 1) * np.array([np.random.uniform(-1, 1),
                                                        np.random.uniform(-1, 1),
                                                        np.random.uniform(-1, 1)] / np.sqrt(3), dtype=float)
    elif flag == 0:
        return 0.25 * np.array([0, 0, 0], dtype=float)


print("模型加载成功")
tf_op_noise_off = []
tf_rl_noise_off = []
tf_xz_noise_off = []

tf_op_noise_on = []
tf_rl_noise_on = []
tf_xz_noise_on = []

for i in range(1000):
    s = env.reset()
    ep_r = 0
    step = 0
    while True:
        action, noise = agent.select_action(s)
        noise = make_noise(env.t, flag=0)
        s_, reward, done, info = env.step(action, noise)
        # 计算r
        s = s_  # 更新环境
        ep_r += reward
        step += 1

        if done:
            print('Episode:', i, ' Reward:', ep_r, env.state[0:6], ' step', step)
            tf_op_noise_off.append(env.tf)
            tf_rl_noise_off.append(env.tf_rl)
            tf_xz_noise_off.append(np.array([env.state[0], env.state[2]], dtype=float))
            break
    env.render()
# env.close()
tf_xz_noise_off = np.array(tf_xz_noise_off)

#
# plt.figure(1)
# plt.plot(tf_op_noise_off, 'red', label='op')
# plt.plot(tf_rl_noise_off, 'blue', label='rl')
#
# plt.figure(2)
# plt.scatter(tf_xz_noise_off[:, 0], tf_xz_noise_off[:, 1])
#
#
# for i in range(1000):
#     s = env.reset()
#     ep_r = 0
#     step = 0
#     while True:
#         action, noise = agent.select_action(s)
#         noise = make_noise(env.t, flag=1)
#         s_, reward, done, info = env.step(action, noise)
#         # 计算r
#         s = s_  # 更新环境
#         ep_r += reward
#         step += 1
#
#         if done:
#             print('Episode:', i, ' Reward:', ep_r, env.state[0:6], ' step', step)
#             print('s', env.state[0], env.state[2])
#             tf_op_noise_on.append(env.tf)
#             tf_rl_noise_on.append(env.tf_rl)
#             tf_xz_noise_on.append(np.array([env.state[0], env.state[2]], dtype=float))
#             break
#     # env.render_legend()
# # env.close()
#
# tf_xz_noise_on = np.array(tf_xz_noise_on)
#
#
# plt.figure(3)
# plt.plot(tf_op_noise_on, 'red', label='op')
# plt.plot(tf_rl_noise_on, 'blue', label='rl')
#
# plt.figure(4)
# plt.scatter(tf_xz_noise_on[:, 0], tf_xz_noise_on[:, 1])
#
# plt.show()
