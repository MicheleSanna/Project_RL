import logging
import torch
import gym
import numpy as np
import random
from tools.helper import get_config
from tools.helper import init_logger
from game.games import NoLimitHoldem
import time
from state_constructor import StateConstructor
from esarsa import ESARSA
from matplotlib import pyplot as plt
from dqn import DQN, ReplayMemory, Transition, QNetworkAgent

def get_action(pot, action_id, big_blind=100):
    match action_id:
        case 0:
            return [0, 0]
        case 1:
            return [1, 0]
        case 2:
            return [2, big_blind]
        case 3:
            return [2, pot*0.25]
        case 4:
            return [2, pot*0.5]
        case 5:
            return [2, pot*1]
        case 6:
            return [2, pot*2]
        case 7:
            return [2, pot*4]
        
def training_loop(env, agent, device, num_episodes = 20000, memory = ReplayMemory(10000)):
    episode_reward=np.zeros(int(n_episodes/100))
    steps_done = 0
    reward = [0,0]
    best_performance = 0
    for i in range(num_episodes):
        feet = int(i*0.01)
        episode_reward[feet] += reward[0] * 0.1
        if (i%100 == 0):
            print(f"Step: {i}, reward of last 100: {episode_reward[feet-1]}, time elapsed: {time.time()-start}")
        if (i%5000 == 0 and i != 0):
            print("REWARD_NOW: ", episode_reward[feet])
            if sum(episode_reward[feet-50:feet-1]) > best_performance:
                best_performance = sum(episode_reward[feet-50:feet-1])
                agent.save_model()
                print("SAVED!")

        done = False
        # Initialize the environment and get its state
        env.reset()
        state_dict = env.state_dict()
        state = state_constructor.construct_state_continuous(state_dict, 0)
        action = agent.select_action(steps_done, state)
        last_action = action
        while not done:
            steps_done += 1
            
            #observation, reward, terminated, truncated, _ = env.step(action.item())
            _, reward, done, a = env.step(get_action(state_dict['main_pot'], action.item()))
            state_dict = env.state_dict()
            
            if state_dict['current_player'] == 0:
                r = torch.tensor([reward[0]], device=device)
                #if terminated:
                #    next_state = None
                next_state = state_constructor.construct_state_continuous(state_dict, 0)

                # Store the transition in memory
                memory.push(state, last_action, next_state, r)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                agent.optimize_model(memory)
                agent.update_target_net()

                action = agent.select_action(steps_done, state)
                last_action = action
            else:
                action = torch.tensor([[random.choices([0, 1, 2, 3, 4, 5, 6, 7], [0.1, 2, 1, 0.5, 0.1, 0.05, 0.02, 0.01])[0]]], device=device, dtype=torch.long)

    print('Complete')
    return episode_reward

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

if __name__ == '__main__':
    n_observations = 8
    n_actions = 8
    args = NoLimitHoldem.ARGS_CLS(n_seats=2,
                                stack_randomization_range=(0, 9700),
                                starting_stack_sizes_list=[300] * 2)
    env = NoLimitHoldem(is_evaluating=False, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())
    state_constructor = StateConstructor(equity_bins=[0.4, 0.6, 1], 
                                         pot_bins=[0, 0.05, 0.1, 0.2, 0.4, 0.8, 9999], 
                                         raise_bins=[0, 0.1, 0.5, 1, 3, 9999], 
                                         big_blind=100, 
                                         initial_stack=10000, 
                                         n_workers=4,
                                         device=device)
    n_episodes = 200000
    agent = QNetworkAgent(device, n_observations, n_actions, 64, 0.99, 0.9, 0.05, 200, 0.001, 0.0005)
    start = time.time()
    print(device)
    episode_reward = training_loop(env, agent, device, num_episodes = n_episodes, memory = ReplayMemory(10000))
    
    """for i in range(n_episodes):
        feet = int(i*0.01)
        episode_reward[feet] += reward[0] * 0.1
        if (i%100 == 0):
                print(f"Step: {i}, reward of last 100: {episode_reward[feet-1]}, time elapsed: {time.time()-start}")
        done = False
        env.reset()
        state = state_constructor.construct_state_bin(env.state_dict(), 0)
        new_state = state
        new_action = action
        state_dict = env.state_dict()
        while not done:
            
            _, reward, done, a = env.step(get_action(state_dict['main_pot'], new_action))
            
            state_dict = env.state_dict()
            
            if state_dict['current_player'] == 0:
                new_state = state_constructor.construct_state_bin(state_dict, 0)
                new_action = e_sarsa.get_action_epsilon_greedy(new_state, 0.1)
                e_sarsa.single_step_update(state, action, reward[0], new_state, done, 0.1)
            else:
                action_space = env.get_legal_actions()
                new_action = random.choices([0, 1, 2, 3, 4, 5, 6, 7], [0.1, 2, 1, 0.5, 0.1, 0.05, 0.02, 0.01])[0]

            state = new_state
            action = new_action"""
 
    print("Time elapsed: ", time.time()-start)
    plt.plot(episode_reward)
    plt.xlabel('Episode (x100)')
    plt.ylabel('Reward')
    plt.title('Episode Reward Over Time')
    plt.show()            