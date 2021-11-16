import gym
from DDPG import DDPG
from DDPG import OUNoise
import os

class Arguments:
    def __init__(self):
        # agent
        self.action_dim = 0
        self.hidden_dim = 128
        self.state_dim = 0
        # train
        self.break_step = 1000000  # break training after 'total_step > break_step'
        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.gamma = 0.9
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        self.repeat_times = 50
        self.batch_size = 32  # num of transitions sampled from replay buffer.
        self.target_step = 1600  # repeatedly update network to keep critic's loss small
        self.soft_update_tau = 0.01

        '''Arguments for evaluate'''
        self.eval_steps = 4800  # evaluate the agent per eval_gap seconds
        self.eval_times = 4  # number of times that get episode return in first

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)

def eval():
    args = Arguments()
    env = gym.make('Pendulum-v0')
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    agent = DDPG(args)
    agent.load()

    for i in range(10):
        state = env.reset()
        ep_r = 0
        while True:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_r += reward
            if done:
                break
        print("ep_r: ", ep_r)


def train():
    args = Arguments()
    env = gym.make('Pendulum-v0')
    env.unwrapped

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.target_step = max_step * 8
    print(args.target_step)
    agent = DDPG(args)
    ou_noise = OUNoise(env.action_space)  # 动作噪声
    print("begin train")
    total_step = 0
    eval_max_reward = -100000
    eval_times_done = 0
    state_temp = env.reset()
    ou_noise.reset()
    while total_step < args.break_step:
        state = state_temp
        for i in range(args.target_step):
            action = agent.choose_action(state)
            action = ou_noise.get_action(action)
            # env.render()
            next_state, reward, done, _ = env.step(action)
            agent.memory.memorystore(state, action, reward, next_state, done)
            if done:
                state = env.reset()
                ou_noise.reset()
            else:
                state = next_state

        state_temp = state
        total_step += args.target_step

        agent.update()

        if total_step >= args.eval_steps * eval_times_done:
            eval_times_done += 1
            eval_total_reward = 0
            for i in range(args.eval_times):
                state = env.reset()
                while True:
                    #env.render()
                    action = agent.choose_action(state)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    eval_total_reward += reward
                    if done:
                        break
            #env.close()
            eval_avg_reward = eval_total_reward / args.eval_times
            if eval_max_reward < eval_avg_reward:
                eval_max_reward = eval_avg_reward
                print(total_step, ":  ", eval_max_reward)
                agent.save(path="max.pt")
            else:
                print(total_step, ":  ", eval_max_reward, "||", eval_avg_reward)

    agent.save()

if __name__ == "__main__":
    train()
    #eval()