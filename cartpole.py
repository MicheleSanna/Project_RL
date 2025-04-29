import gymnasium
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from dqn import BaseNetwork, ReplayMemory, Transition
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dqn import DQNTrainer

env = gymnasium.make("CartPole-v1")

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(device)
# Get number of actions from gym action space
a = gymnasium.spaces.discrete.Discrete(2)
print(a)
print(env.action_space)
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

agent = DQNTrainer(device, n_observations, n_actions, batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=0.999, tau=0.001, lr=0.0005)


def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def training_loop(num_episodes = 1000, memory = ReplayMemory(10000)):
    steps_done = 0
    episode_durations = []
    for i_episode in range(num_episodes):
        print(f"Episode {i_episode+1}/{num_episodes}")
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = agent.select_action(steps_done, state, env)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
 
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model(memory)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            agent.update_target_net()

            if done:
                episode_durations.append(t + 1)
                if truncated:
                    print("TRUNCATED")
                break

    print('Complete')
    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()

training_loop()