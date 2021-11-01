"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from share_adam import SharedAdam
import numpy as np
import gym
import math, os
os.environ["OMP_NUM_THREADS"] = "1"

from  env import LandMars

UPDATE_GLOBAL_ITER = 6
GAMMA = 0.9
TRAIN_MAX_EP = 3000
TEST_MAX_EP = 30
MAX_EP_STEP = 200
ONTRAIN = False
'''
env = LandMars()
N_S = env.state_dim
N_A = env.action_dim
'''
env = gym.make("LunarLanderContinuous-v2")
    #env.unwrapped
N_S = env.observation_space.shape[0]  # 状态空间，state
N_A = env.action_space.shape[0]


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.a1 = nn.Linear(s_dim, 200)
        self.a2 = nn.Linear(200, 100)
        self.mu = nn.Linear(100, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.c2 = nn.Linear(100, 50)
        self.v = nn.Linear(50, 1)
        set_init([self.a1, self.a2, self.mu, self.c1, self.c2, self.v])

        action_std = 1.0
        self.distribution = torch.distributions.MultivariateNormal
        self.action_var = torch.full((a_dim,), action_std * action_std)

        self.max_r = -10000

    def forward(self, x):
        a1 = F.relu(self.a1(x))
        a2 = F.relu(self.a2(a1))
        mu = torch.tanh(self.mu(a2))

        c1 = F.relu(self.c1(x))
        c2 = F.relu(self.c2(c1))
        values = self.v(c2)
        return mu, values

    def choose_action(self, s):
        self.training = False
        mu, _ = self.forward(s)
        m = self.distribution(mu[0].data, torch.diag(self.action_var))
        return m.sample().numpy()

    def choose_action_(self, s):
        self.training = False
        mu, _ = self.forward(s)
        m = self.distribution(mu[0].data, torch.diag(self.action_var))
        return mu[0].detach().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        m = self.distribution(mu[0].data, torch.diag(self.action_var))
        log_prob = m.log_prob(a)
        #entropy = 0.5 + 0.5 * math.log(2 * math.pi) + m.entropy()  # exploration
        exp_v = log_prob * td.detach() # + 0.005 * m.entropy()
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss

    def save(self, path='net_continuous.pth'):
        state = {
            'a1_state_dict': self.a1.state_dict(),
            'a2_state_dict': self.a2.state_dict(),
            'mu_state_dict': self.mu.state_dict(),
            #'sigma_state_dict': self.sigma.state_dict(),
            'c1_state_dict': self.c1.state_dict(),
            'c2_state_dict': self.c2.state_dict(),
            'v_state_dict': self.v.state_dict()
        }
        torch.save(state, path)

    def load(self, path='net_continuous.pth'):
        checkpoint = torch.load(path)
        self.a1.load_state_dict(checkpoint['a1_state_dict'])
        self.a2.load_state_dict(checkpoint['a2_state_dict'])
        self.mu.load_state_dict(checkpoint['mu_state_dict'])
        #self.sigma.load_state_dict(checkpoint['sigma_state_dict'])
        self.c1.load_state_dict(checkpoint['c1_state_dict'])
        self.c2.load_state_dict(checkpoint['c2_state_dict'])
        self.v.load_state_dict(checkpoint['v_state_dict'])



class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        #self.env = LandMars()
        self.env = gym.make("LunarLanderContinuous-v2")

    def run(self):
        total_step = 1
        while self.g_ep.value < TRAIN_MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

            if ep_r > self.lnet.max_r:
                self.lnet.max_r = ep_r
                if ep_r > self.gnet.max_r:
                    self.gnet.max_r = ep_r
                    self.gnet.save()
        print("while done")
        self.res_queue.put(None)
        print("worker done")


def train():
    gnet = Net(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(4)]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    # while里面很耗时
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    [w.join() for w in workers]

    gnet.save(path= 'last.pth')

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()


def test():
    gnet = Net(N_S, N_A)  # global network
    #gnet.load(path= 'last.pth')
    gnet.load()
    print('开始测试!')
    for i in range(TEST_MAX_EP):
        ep_r = 0  # reward per episode
        s = env.reset()
        print(s[:6])
        step = 0
        while True:
            env.render()
            action = gnet.choose_action_(v_wrap(s[None, :]))
            s_, r, done, _ = env.step(action)
            s = s_
            step += 1
            # 计算r
            # r = (r + 8.1) / 8.1
            ep_r += r
            if done:
                print(s[:6])
                print('Episode:', i, ' Reward:', ep_r, ' Step', step)
                break
        #env.render()

if __name__ == "__main__":
    if ONTRAIN:
        train()
    else:
        test()