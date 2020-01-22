import gym
from DQN import *
import os
import keyboard as kb

def make_video(env, TrainNet, iterations):
    env = gym.wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    print("start")
    while steps <= iterations:
        print("step")
        action = TrainNet.get_action(observation, 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))

def show_off(env, TrainNet):
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        env.render()
        action = TrainNet.get_action(observation, 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))

def play_game(env, TrainNet, TargetNet, epsilon, copy_step, iter_per_episode):
    rewards = 0
    iter = 0
    done = False
    
    observations = env.reset()
    while iter <= iter_per_episode:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        TrainNet.train(TargetNet)
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards


def play():
    env = gym.make("Acrobot-v1")
    env.reset()
    while not kb.is_pressed('q'):
        env.render()
        actions = 0
        actions = 1 if kb.is_pressed('a') else actions
        actions = 2 if kb.is_pressed('d') else actions
        print(env.action_space.sample())
        env.step(actions)
    env.close()

def main():
    env = gym.make('Acrobot-v1')
    gamma = 0.99
    copy_step = 25
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n
    hidden_units = [64, 64]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    iter_per_episode = 300

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size)
    N = 50
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.08
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step, iter_per_episode)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        if n % 5 == 0:
            print("Progress:", int(n/N*100), "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards)
    print("avg reward for last 100 episodes:", avg_rewards)
    make_video(env, TrainNet, 300)
    env.close()

play()