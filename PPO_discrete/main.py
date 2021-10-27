import gym
from PPO import PPO
import numpy as np
import gym
#设定参数
train_episode = 600
eval_episode = 40
ONTRAIN = False
batch_size = 8
learn_frequent = 20  # 更新


#模型参数
hidden_dim = 256


def make_env():
    env = gym.make('CartPole-v0')
    env.unwrapped
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim,hidden_dim,action_dim,batch_size)
    return env, agent

def train():
    print("开始训练")
    max_r = 0
    for i in range(train_episode):
        s = env.reset()
        ep_r = 0
        step = 0
        if i == 200:
            agent.update_lr(1e-5)
        if i == 350:
            agent.update_lr(1e-6)
        while True:
            #env.render()
            a, p, v = agent.choose_action(s)
            #print(a)
            s_,reward,done,info = env.step(a)
            # 计算r
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.3
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            if abs(x)>env.x_threshold or abs(theta) > env.theta_threshold_radians or step == 5000:
                done = True
            else:
                done = False
            agent.memory(s, a, p, v, reward, done)
            s = s_ # 更新环境
            ep_r += r
            step += 1
            if step % learn_frequent == 0:
                agent.learn()
            if done:
                print('Episode:', i, ' Reward:',  ep_r, ' step', step)
                break
        if max_r < ep_r:
            max_r = ep_r
            agent.save()
            print("模型加载成功,reward",max_r)


def eval():
    agent.load()
    print("模型加载成功")
    for i in range(eval_episode):
        s = env.reset()
        ep_r = 0
        step = 0
        while True:
            env.render()
            action, probs, value = agent.choose_action(s)
            s_, reward, done, info = env.step(action)
            # 计算r
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.3
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            s = s_  # 更新环境
            ep_r += r
            step += 1
            if abs(x)>env.x_threshold or abs(theta)> env.theta_threshold_radians:
                print('Episode:', i, ' Reward:', ep_r, ' step', step)
                break


if __name__=="__main__":
    env,agent = make_env()
    if ONTRAIN:
        train()
    else:
        eval()