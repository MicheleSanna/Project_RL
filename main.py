import logging

import gym
import numpy as np
import random
from tools.helper import get_config
from tools.helper import init_logger
from game.games import NoLimitHoldem
import time

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


        action = random.choice(list(action_space))
        return action

stack = 500
args = NoLimitHoldem.ARGS_CLS(n_seats=3,
                            stack_randomization_range=(0, 0),
                            starting_stack_sizes_list=[800] * 3)
env = NoLimitHoldem(is_evaluating=False, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())
player = StaticRandomPlayer()


"""state_0 = env.reset()
print(env.get_legal_actions())
env.render()
print(state_0)
state, reward, done, _ = env.step([2, 85])
print(reward)
state, reward, done, _ = env.step([2, 85])
print(reward)
state, reward, done, _ = env.step([0, 85])
print(reward)
state, reward, done, _ = env.step([1, 85])
state, reward, done, _ = env.step([1, 85])
env.render()
print(env.state_dict())
print(reward)"""

state_0 = env.reset()

start = time.time()
for i in range(100000):
    #env.render()
    if (i%2000 == 0):
        print(i/2000)
    action_space = env.get_legal_actions()
    action = player.action(action_space, state_0, {})
    state, reward, done, _ = env.step([action, random.randint(0, 100)])
    #print(env.state_dict())
    
    #state_dict = env.state_dict()
    #board = state_dict['board_2d']
    #hand = state_dict['seats'][i%3]['hand']
    #current_bet = state_dict['seats'][i%3]['current_bet']
    #fold = state_dict['seats'][i%3]['folded_this_episode']
    if done:
        env.reset()

print("Time elapsed: ", time.time()-start)




