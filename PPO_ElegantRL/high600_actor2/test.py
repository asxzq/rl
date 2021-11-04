from agent import AgentPPO
import torch
import gym
from LandMars3d import LandMars
import numpy as np


env = LandMars()
#state_dim = env.observation_space.shape[0]  # 状态空间，state
state_dim = env.state_dim  # 状态空间，state

#action_dim = env.action_space.shape[0]
action_dim = env.action_dim

agent = AgentPPO()
agent.init(512, state_dim, action_dim)
agent.act.load_state_dict(torch.load("AgentPPO_LandMars_0/actor.pth"))

print("模型加载成功")
for i in range(30):
    s = env.reset()
    print(env.state[:6])
    ep_r = 0
    step = 0
    while True:
        #env.render()
        action, noise = agent.select_action(s)
        #print(action)
        s_, reward, done, info = env.step(action)
        # 计算r
        s = s_  # 更新环境
        ep_r += reward
        step += 1
        if done:
            print('Episode:', i, ' Reward:', ep_r, ' step', step)
            break
    print(env.state[:6])
    env.render()