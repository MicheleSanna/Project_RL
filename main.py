import logging

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
    

class StaticRandomPlayer:
    """Mandatory class with the player methods"""

    def __init__(self, name='Fausto'):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info


        action = random.choice(action_space)
        return action

if __name__ == '__main__':
    stack = 500
    args = NoLimitHoldem.ARGS_CLS(n_seats=2,
                                stack_randomization_range=(0, 9700),
                                starting_stack_sizes_list=[300] * 2)
    env = NoLimitHoldem(is_evaluating=False, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())
    player = StaticRandomPlayer()
    state_constructor = StateConstructor(equity_bins=[0.25, 0.5, 0.75, 1], 
                                         pot_bins=[0.05, 0.1, 0.2, 0.4, 0.8, 9999], 
                                         raise_bins=[0, 0.1, 0.5, 1, 3, 9999], 
                                         big_blind=100, 
                                         initial_stack=10000, 
                                         n_workers=4)
    e_sarsa = ESARSA((6, 7, 6, 4), 8)
    n_episodes = 200000
    episode_reward=np.zeros(int(n_episodes/100))
    state_0 = env.reset()
    start = time.time()
    step = 0
    reward = [0]


    
    for i in range(n_episodes):
        feet = int(i*0.01)
        episode_reward[feet] += reward[0] * 0.1
        if (i%100 == 0):
                print(f"Step: {i}, reward of last 100: {episode_reward[feet-1]}, time elapsed: {time.time()-start}")
        done = False
        env.reset()
        state = state_constructor.construct_state_bin(env.state_dict(), 0)
        action = e_sarsa.get_action_epsilon_greedy(state, 0.1)
        new_state = state
        new_action = action
        state_dict = env.state_dict()
        while not done:
            
            _, reward, done, a = env.step(get_action(state_dict['main_pot'], new_action))
            
            state_dict = env.state_dict()
            #print("--------------------------------------")
            #print(state_dict)
            if state_dict['current_player'] == 0:
                new_state = state_constructor.construct_state_bin(state_dict, 0)
                new_action = e_sarsa.get_action_epsilon_greedy(new_state, 0.1)
                e_sarsa.single_step_update(state, action, reward[0], new_state, done, 0.1)
                #print("SARSA PLAYER ACTION: ", new_action)
                state = new_state
                action = new_action
            else:
                new_action = random.choices([0, 1, 2, 3, 4, 5, 6, 7], [0.1, 2, 1, 0.5, 0.1, 0.05, 0.02, 0.01])[0]
                #print("RANDOM PLAYER ACTION: ", new_action)

            
 
    print("Time elapsed: ", time.time()-start)
    e_sarsa.save("QValues.npy")
    plt.plot(episode_reward)
    plt.xlabel('Episode (x100)')
    plt.ylabel('Reward')
    plt.title('Episode Reward Over Time')
    plt.show()            




