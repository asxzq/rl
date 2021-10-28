# A2C在线学习
import gym
from A2C import A2C
train_episode = 5000
eval_episode = 40
ONTRAIN = True
def env_agent(seed):
    env = gym.make('CartPole-v0')
    env.seed(seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = A2C(n_states, n_actions)
    return env, agent

def train(env, agent):
    print('开始训练!')
    max_r = 0
    for i in range(train_episode):
        s = env.reset()
        ep_r = 0
        step = 0
        while True:
            a, p, v = agent.choose_action(s)
            # print(a)
            s_, reward, done, info = env.step(a)
            # 计算r
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.3
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            if abs(x) > env.x_threshold or abs(theta) > env.theta_threshold_radians or step == 5000:
                done = True
            else:
                done = False
            s = s_  # 更新环境
            ep_r += r
            step += 1
            agent.learn(s,a,p,reward,v,s_,done)
            if done:
                print('Episode:', i, ' Reward:', ep_r, ' step', step)
                break
        if max_r < ep_r:
            max_r = ep_r
            agent.save()
    print('完成训练！')



def eval(env, agent):
    print('开始测试!')
    agent.load()
    for i in range(eval_episode):
        ep_r= 0  # reward per episode
        s = env.reset()
        step = 0
        while True:
            env.render()
            a, p, v = agent.choose_action(s)
            # print(a)
            s_, reward, done, info = env.step(a)
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

    print('完成测试！')



if __name__ == "__main__":
    if ONTRAIN:
        # 训练
        env, agent = env_agent(seed=1)
        train(env, agent)
    else:
        # 测试
        env, agent = env_agent(seed=10)
        eval(env, agent)
