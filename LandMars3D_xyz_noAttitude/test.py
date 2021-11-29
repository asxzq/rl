from agent import AgentPPO
import torch
import gym
from LandMars3d import LandMars
import numpy as np
import matplotlib.pyplot as plt

env = LandMars()
#state_dim = env.observation_space.shape[0]  
state_dim = env.state_dim 

#action_dim = env.action_space.shape[0]
action_dim = env.action_dim

agent = AgentPPO()
agent.init(512, state_dim, action_dim)
agent.act.load_state_dict(torch.load("AgentPPO_LandMars_0/actor.pth"))
agent.act.eval()


tf_op = []
tf_rl = []



print("模型加载成功")
for i in range(300):
    s = env.reset()
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
            print('Episode:', i, ' Reward:', ep_r, env.state[0:6],' step', step)
            tf_op.append(env.tf)
            tf_rl.append(env.tf_rl)
            break
    #env.render_legend()
# env.close()

plt.figure(1)
plt.plot(tf_op, 'red', label='op')
plt.plot(tf_rl, 'blue', label='rl')
plt.show()