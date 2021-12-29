from agent import AgentPPO
import torch
import gym
from LandMars3d_500 import LandMars_500
from LandMars3d_250 import LandMars_250
from LandMars3d_100 import LandMars_100
import numpy as np
import matplotlib.pyplot as plt


env = []
env.append(LandMars_500())
env.append(LandMars_250())
env.append(LandMars_100())
env_single = LandMars_500()
state_dim = env[0].state_dim
action_dim = env[1].action_dim


agent = []
agent.append(AgentPPO())
agent.append(AgentPPO())
agent.append(AgentPPO())
for i in range(len(agent)):
    agent[i].init(512, state_dim, action_dim)
agent[0].act.load_state_dict(torch.load("Land_500/AgentPPO_LandMars_0/actor.pth"))
agent[1].act.load_state_dict(torch.load("Land_250/AgentPPO_LandMars_0/actor.pth"))
agent[2].act.load_state_dict(torch.load("Land_100/AgentPPO_LandMars_0/actor.pth"))
for i in range(len(agent)):
    agent[i].act.eval()

stage = 0

tf_xz_mul = []
tf_xz_single = []
print("模型加载成功")
for i in range(1000):
    ep_r = 0
    step = 0
    stage = 0
    s = env[stage].reset()
    s_single = env_single.reset(flag = 1, state = np.copy(env[stage].state))
    while True:
        #env.render()
        action, noise = agent[stage].select_action(s)
        #print(action)
        s_, reward, done, info = env[stage].step(action)
        # 计算r
        s = s_  # 更新环境
        ep_r += reward
        step += 1
        if env[stage].state[1] < env[stage].max_y * 0.25 and stage < len(env) - 1:
            # print('Episode:', i, 'stage:', stage, env[stage].state[0:6])
            stage += 1
            s = env[stage].reset(flag=1, state=env[stage - 1].state)
            # print('Episode:', i, 'stage:', stage, env[stage].state[0:6])
        if done:
            print('Episode:', i, ' Reward:', ep_r, env[stage].state[0:6],' step', step)
            tf_xz_mul.append(np.array([env[stage].state[0], env[stage].state[2]], dtype=float))
            break

    # control = np.array(env[0].memory.control, dtype=float)[1:, 0]
    # for k in range(1, len(env)):
    #     control = np.hstack((control, np.array(env[k].memory.control, dtype=float)[1:, 0]))
    # plt.figure(1)
    # plt.plot(control, 'b', label='|a|')


    while True:
        #env.render()
        action, noise = agent[0].select_action(s_single)
        #print(action)
        s_, reward, done, info = env_single.step(action)
        # 计算r
        s_single = s_  # 更新环境
        ep_r += reward
        step += 1
        if done:
            print('Episode:', i, ' Reward:', ep_r, env_single.state[0:6],' step', step)
            tf_xz_single.append(np.array([env_single.state[0], env_single.state[2]], dtype=float))
            break

    # control1 = np.array(env_single.memory.control, dtype=float)[1:, 0]
    # plt.figure(2)
    # plt.plot(control1, 'b', label='|a|')
    # plt.show()
# env.close()
tf_xz_mul = np.array(tf_xz_mul)
tf_xz_single = np.array(tf_xz_single)

plt.figure(1)
plt.scatter(tf_xz_mul[:, 0], tf_xz_mul[:, 1], c='red',label='mul')
plt.scatter(tf_xz_single[:, 0], tf_xz_single[:, 1], c='blue', label='single')
plt.legend()
plt.show()
