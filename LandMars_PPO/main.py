from PPO import PPO
import numpy as np
import math
#from env import LandMars
import gym
# 设定参数
train_episode = 300
eval_episode = 40

ONTRAIN = 0
TRAIN_LOAD_OLDNET = 0
TEST_LOAD_MAXNET = 0
batch_size = 16
learn_frequent = 100  # 更新

# 模型参数
hidden_dim = 64


def make_env():
    env = gym.make("LunarLanderContinuous-v2")
    #env.unwrapped
    state_dim = env.observation_space.shape[0]  # 状态空间，state
    action_dim = env.action_space.shape[0]

    print('state_dim:',state_dim,'action_dim:',action_dim)
    agent = PPO(state_dim, hidden_dim, action_dim, batch_size)
    return env, agent


def train():
    print("开始训练")
    max_r = -10000
    rewards = []
    steps = []
    total_step = 0
    for i in range(train_episode):
        s = env.reset()
        ep_r = 0
        ep_step = 0
        agent.action_std = 0.5 * math.exp(-2 * i / train_episode)
        while True:
            #env.render()
            a, p, v = agent.choose_action(s)
            # print(a)
            s_, reward, done, info = env.step(a)

            ep_r += reward
            ep_step += 1
            if ep_step > 300:
                done = True
            agent.memory(s, a, p, v, reward - 1.0, done)
            s = s_  # 更新环境
            if (len(agent.memory_states) + 1) % learn_frequent == 0 or done:
                agent.learn()
            if done:
                print('Episode:', i, ' Reward:', ep_r, ' step', ep_step)
                break
        if (i+1) % 20 == 0:
            agent.save()
            print("模型保存成功")
        if ep_r > max_r:
            max_r = ep_r
            agent.save(actor_path='actor_max.pt', critic_path='critic_max.pt')
            print("模型保存成功,reward", max_r)
        total_step += ep_step
        rewards.append(ep_r)
        steps.append(total_step)
    import matplotlib.pyplot as plt
    plt.plot(steps, rewards)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()


def eval():
    if TEST_LOAD_MAXNET:
        agent.load(actor_path='actor_max.pt', critic_path='critic_max.pt')
    else:
        agent.load()
    print("模型加载成功")
    for i in range(eval_episode):
        s = env.reset()
        ep_r = 0
        step = 0
        while True:
            env.render()

            action, probs, value = agent.choose_action_(s)
            s_, reward, done, info = env.step(action)
            # 计算r
            s = s_  # 更新环境
            ep_r += reward
            step += 1
            if done:
                print('Episode:', i, ' Reward:', ep_r, ' step', step)
                break



if __name__ == "__main__":
    env, agent = make_env()
    if ONTRAIN:
        if TRAIN_LOAD_OLDNET:
            agent.load(actor_path='oldnet\\actor.pt', critic_path='oldnet\\critic.pt')
        train()
    else:
        eval()
