# DQN本质上只用了一个网络
# 为了解决q_target计算不稳定问题，引入另外的target_net，得到NatureDQN
# 此代码实际为 NatureDQN
#  Q-learning，
#  Double Q-learning（解决Q-learning值函数过估计问题），
#  DQN（解决Q-learning大状态空间、动作空间问题），
#  Double DQN（解决DQN值函数过估计问题），
#  Dueling-DQN（状态价值和动作价值分离）
#  NoisyNet-DQN（解决e-贪心策略在网络输出的概率上采样、使之偏离实际过程的问题
#             采用在网络中加入噪声来解决。噪声在每回合内不变，则可以保证该回合内的所有动作有相同的趋势
#             这种方式 叫做   依赖状态的探索(state-dependent exploration)）
import torch
import numpy as np
class NoisyNetDQN:
    def __init__(self):
        self.lr  = 0.00001
