import gym
import torch
from DuelingDQN import DuelingDQN
train_episode = 300
eval_episode = 40
ONTRAIN = False
def env_agent_config(seed):
    env = gym.make( 'CartPole-v0' )
    env.seed(seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DuelingDQN(n_states, n_actions)
    return env, agent

def train(env, agent):
    print('开始训练!')
    max_r = 0
    for i in range(train_episode):
        state = env.reset()
        ep_r = 0
        step = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.3
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            step += 1
            if abs(x) > env.x_threshold or abs(theta) > env.theta_threshold_radians or step > 3000:
                done = True
            else:
                done = False
            agent.memory.push(state, action, r, next_state, done)
            state = next_state
            agent.update()
            ep_r += r
            if done:
                print('Episode:', i, ' Reward:', ep_r, ' step', step)
                break
        if (i + 1) % 4 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if i == 150:
            agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=0.00003) # 优化器
        if max_r < ep_r:
            max_r = ep_r
            agent.save()
    print('完成训练！')



def eval(env, agent):
    print('开始测试!')
    agent.load()
    for i in range(eval_episode):
        ep_r= 0  # reward per episode
        state = env.reset()
        step = 0
        while True:
            env.render()
            action = agent.predict(state)
            s_, reward, done, _ = env.step(action)
            state = s_
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
        env, agent = env_agent_config(seed=1)
        train(env, agent)
    else:
        # 测试
        env, agent = env_agent_config(seed=10)
        eval(env, agent)
