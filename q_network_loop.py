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
from players.nn_player import NNPlayer
from players.random_player import RandomPlayer
from plot import plot


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
        

def training_loop(env, agent, device, adversary, num_episodes = 20000, memory = ReplayMemory(10000), version_name = '1.0'):
    episode_reward=np.zeros(n_episodes)
    reward = [0,0]
    best_performance = 0
    print("Lesgo")
    for i in range(num_episodes):
        if (i%100 == 0 and i != 0):
            print(f"Step: {i}, reward of last 100: {sum(episode_reward[i-100:i])}, time elapsed: {time.time()-start}")
        if (i%5000 == 0 and i != 0):
            print(f"REWARD_NOW: {episode_reward[i-1]}, AVG_REWARD: {sum(episode_reward[i-5000:i])}")
            if sum(episode_reward[i-5000:i]) > best_performance:
                best_performance = sum(episode_reward[i-5000:i])
                agent.save_model(policy_path=f"policy_{version_name}.pth", target_path=f"target_{version_name}.pth")
                print("SAVED!")

        done = False
        # Initialize the environment and get its state
        env.reset()
        state_dict = env.state_dict()
        state = state_constructor.construct_state_continuous(state_dict, 0, False)
        action = agent.select_action(i, state)
        last_action = action
        while not done:
            _, reward, done, a = env.step(get_action(state_dict['main_pot'], action.item()))
            state_dict = env.state_dict()

            if state_dict['current_player'] == 0 or done:
                r = torch.tensor([reward[0]], device=device)
                next_state = state_constructor.construct_state_continuous(state_dict, 0, done)

                # Store the transition in memory
                memory.push(state, last_action, next_state, r)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                agent.optimize_model(memory)
                agent.update_target_net()

                action = agent.select_action(i, state) if not done else None
                last_action = action
            else:
                action = adversary.select_action(state)
        episode_reward[i] = reward[0] 

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
    n_episodes = 100000
    agent = QNetworkAgent(device=device, 
                          n_observations=n_observations, 
                          n_actions=n_actions,
                          replay_memory_size=10000,
                          batch_size=256, 
                          gamma=1, 
                          eps_start=0.9, 
                          eps_end=0.01, 
                          eps_decay=n_episodes-(n_episodes/3), 
                          tau=0.001, 
                          lr=0.0005)
    #agent.load_model(policy_path="policy_1.0.pth", target_path="target_1.0.pth")
    start = time.time()
    print(device)
    #adv = NNPlayer(policy_net=agent.policy_net, policy_net_name="policy_1.0.pth", device=device)
    episode_reward = training_loop(env, 
                                   agent, 
                                   device, 
                                   RandomPlayer(device),
                                   num_episodes = n_episodes, 
                                   memory = ReplayMemory(10000),
                                   version_name="1.0")
    
    reward_averages = np.zeros(n_episodes//100)
    for i in range(0, 100000, 100):
        reward_averages[(i//100)] = sum(episode_reward[i:i+100])*0.1
    print("Time elapsed: ", time.time()-start)
    plot(reward_averages, 25)