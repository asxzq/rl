from net import AgentPPO
import torch
import gym
import numpy as np


env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0]  # 状态空间，state
action_dim = env.action_space.shape[0]

agent = AgentPPO()
agent.init(512, state_dim, action_dim)
agent.act.load_state_dict(torch.load("actor.pth"))

print("模型加载成功")
for i in range(30):
    s = env.reset()
    ep_r = 0
    step = 0
    while True:
        env.render()
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
