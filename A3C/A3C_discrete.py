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
import gym
import os

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
TRAIN_MAX_EP = 3000
TEST_MAX_EP = 30
ONTRAIN = False

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical
        self.max_r = -10000


    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    def save(self, path='net_discrete.pth'):
        state = {
            'pi1_state_dict': self.pi1.state_dict(),
            'pi2_state_dict': self.pi2.state_dict(),
            'v1_state_dict': self.v1.state_dict(),
            'v2_state_dict': self.v2.state_dict()
        }
        torch.save(state, path)

    def load(self, path='net_discrete.pth'):
        checkpoint = torch.load(path)
        self.pi1.load_state_dict(checkpoint['pi1_state_dict'])
        self.pi2.load_state_dict(checkpoint['pi2_state_dict'])
        self.v1.load_state_dict(checkpoint['v1_state_dict'])
        self.v2.load_state_dict(checkpoint['v2_state_dict'])


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = gym.make('CartPole-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < TRAIN_MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            ep_step = 0
            while True:
                # if self.name == 'w00':
                    # self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, reward, done, _ = self.env.step(a)

                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2

                ep_r += r
                ep_step += 1
                if abs(x) > env.x_threshold or abs(theta) > env.theta_threshold_radians or ep_step > 1000:
                    done = True
                else:
                    done = False

                # 存储记忆
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                # 更新网络
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break

                s = s_
                total_step += 1

            # 保存max_r对应的网络参数
            if ep_r > self.lnet.max_r:
                self.lnet.max_r = ep_r
                if ep_r > self.gnet.max_r:
                    self.gnet.max_r = ep_r
                    self.gnet.save()
        self.res_queue.put(None)



def train():
    gnet = Net(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    # 进程之间共享的内存
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    # 并行计算
    print(mp.cpu_count())
    # 根据CPU数量创建多个进程
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    # 多个进程启动
    [w.start() for w in workers]
    res = []  # record episode reward to plot

    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    # join([timeout])：是用来阻塞当前上下文，直至该进程运行结束，一个进程可以被join()多次
    [w.join() for w in workers]

    gnet.save()
    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

def test():
    gnet = Net(N_S, N_A)  # global network
    gnet.load()
    print('开始测试!')
    for i in range(TEST_MAX_EP):
        ep_r = 0  # reward per episode
        s = env.reset()
        step = 0
        while True:
            env.render()
            action = gnet.choose_action(v_wrap(s[None, :]))
            s_, reward, done, _ = env.step(action)
            s = s_
            step += 1
            # 计算r
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            ep_r += r
            if abs(x) > env.x_threshold or abs(theta) > env.theta_threshold_radians:
                print('Episode:', i, ' Reward:', ep_r, ' step', step)
                break


if __name__ == "__main__":
    if ONTRAIN:
        train()
    else:
        test()