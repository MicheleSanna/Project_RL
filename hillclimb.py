import gym
import numpy as np
from esarsa import ESARSA, EXPECTED_SARSA_TDControl
from matplotlib import pyplot as plt

def unflatten_taxi(state):
    taxi_row = state[0] // (5 * 5 * 4)
    taxi_col = (state[0] // (5 * 4)) % 5
    passenger_location = (state[0] // 4) % 5
    destination = state[0] % 4

    return np.array([taxi_row, taxi_col, passenger_location, destination]), state[1], state[2], state[3]

def unflatten_cliffwalking(state):
    """Convert 1D state index to (row, col)."""
    row = state[0] // 12
    col = state[0] % 12
    return np.array([row, col]), state[1], state[2], state[3]


env = gym.make("Taxi-v3")


e_sarsa = ESARSA((5,5,5,4), 6)
#state = env.step(1)
n_episodes = 20000
episode_reward = np.zeros(n_episodes)

for i in range(n_episodes):
    
    done = False
    s = env.reset()
    state, reward, _, _ = unflatten_taxi([s[0], 0, None, None])
    action = e_sarsa.get_action_epsilon_greedy(state, 0.1)
    new_state = state
    new_action = action
    
    while not done:
        
        s = env.step(new_action)
        new_state, reward, done, _ = unflatten_taxi(s)
        new_action = e_sarsa.get_action_epsilon_greedy(new_state, 0.1)
        
        e_sarsa.single_step_update(state, action, reward, new_state, done, 0.1)

        state = new_state
        action = new_action
        episode_reward[i] += reward

    print(f"Step: {i}, reward: {episode_reward[i]}")

plt.plot(episode_reward)
plt.xlabel('Episode (x100)')
plt.ylabel('Reward')
plt.title('Episode Reward Over Time')
plt.show()  
    
    
