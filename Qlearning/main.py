import gym
import torch
import numpy as np
from Env import CliffWalkingWapper
from Qlearning import Qlearning

import time
ONTRAIN = False

# 参数
seed = 1
train_episode = 400
eval_episode = 40
# 创建环境
env = gym.make("CliffWalking-v0")
env = env = CliffWalkingWapper(env)
env.seed(seed) # 设置随机种子

QL = Qlearning()

def train():
    print('开始训练！')
    for i_ep in range(train_episode):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset()  # 重置环境,即开始新的回合
        while True:
            # env.render()
            action = QL.choose_action(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
            QL.learn(state, action, reward, next_state, done)  # Q学习算法更新
            state = next_state  # 更新状态
            ep_reward += reward
            if done:
                break
        print("回合数：{}/{}，奖励{:.1f}".format(i_ep + 1, train_episode, ep_reward))
    print('完成训练！')



def eval():
    print('开始测试！')
    for i_ep in range(eval_episode):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset()  # 重置环境,即开始新的回合
        while True:
            env.render()
            action = QL.predict_action(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
            #print(state, action, next_state)
            state = next_state  # 更新状态
            ep_reward += reward
            time.sleep(0.3)
            if done:
                break
        print("回合数：{}/{}，奖励{:.1f}".format(i_ep + 1, eval_episode, ep_reward))
    print('完成测试！')


if __name__ == "__main__":
    if ONTRAIN:
        train()
        QL.save()
    #QL.show_Qtable()
    else:
        QL.load()
    #QL.show_Qtable()
        eval()
        env.close()