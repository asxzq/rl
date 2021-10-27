import torch
import numpy as np
from collections import defaultdict
import math


class Sarsa(object):
    def __init__(self):
        self.epsilon_start = 0.9  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01   # e-greedy策略中的终止epsilon
        self.epsilon_decay = 300  # e-greedy策略中epsilon的衰减率
        self.sample_num = 0       # choose_action 访问次数
        self.action_dim = 4       # 动作数量
        self.alpha = 0.1          # 学习率
        self.gamma = 0.9          # reward 衰减率
        # deflautdict 保证字典索引为空时，返回一个 全零的数组
        self.Q_table = defaultdict(lambda: np.zeros(self.action_dim))
        print(self.Q_table)

    def choose_action(self,state):
        self.sample_num += 1
        epsilon = self.epsilon_end +(self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_num / self.epsilon_decay)
        # epsilon贪心
        if np.random.uniform(0, 1) > epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.randint(self.action_dim)
        return action

    def predict_action(self,state):
        #不使用策略
        action = np.argmax(self.Q_table[str(state)])
        return action

    def learn(self,state, action, reward, next_state, next_action, done):
        Q_ = self.Q_table[str(state)][action]
        if done:
            Q_target = reward
        else:
            # Sarsa算法 取的是正常下一步取得的动作，而Qlearning算法 取的是当前Q-table或者pi策略下最大值的动作
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][next_action]
        self.Q_table[str(state)][action] += self.alpha * (Q_target - Q_)

    def show_Qtable(self):
        for key in self.Q_table.keys():
            print(f"{key}:{self.Q_table[key]}")

    def sort_Qtable(self):
        #print(self.Q_table)
        keys = sorted(self.Q_table.keys())
        for key in keys:
            print(f"{key}:{self.Q_table[key]}")

    def save(self):
        import dill
        torch.save(obj=self.Q_table, f="Sarsa_model.pkl", pickle_module=dill)
        print("保存模型成功！")

    def load(self):
        import dill
        self.Q_table = torch.load(f= 'Sarsa_model.pkl', pickle_module=dill)
        print("加载模型成功！")