import gym
from PG import PG
#设定参数
train_episode = 300
eval_episode = 40
ONTRAIN = False
batch_size = 8

#模型参数
hidden_dim = 36


def make_env():
    env = gym.make('CartPole-v0')
    #env.unwrapped
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PG(state_dim,hidden_dim,batch_size)
    return env, agent

def train():
    for i in range(train_episode):
        s = env.reset()
        ep_r = 0
        step = 0
        while True:
            #env.render()
            a = agent.choose_action(s)
            s_,reward,done,info = env.step(a)
            # 计算r
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.3
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            if done:
                reward = 0.0
            agent.memory(s,r,a)

            s = s_ # 更新环境
            ep_r += r
            step += 1
            if done:
                print('Episode:', i, ' Reward:',  ep_r, ' step', step)
                break
        # 保证每次训练的数据，模型参数theta不变，在epsilon外训练
        # 比较浪费数据
        if len(agent.memory_state) >= batch_size:
            agent.learn()
            agent.memory_clear()

    agent.save()


def eval():
    agent.load()
    for i in range(eval_episode):
        s = env.reset()
        ep_r = 0
        step = 0
        while True:
            env.render()
            a = agent.choose_action_(s)
            s_, reward, done, info = env.step(a)
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