import logging

import gym
import numpy as np
import random
from tools.helper import get_config
from tools.helper import init_logger
from game.games import NoLimitHoldem
import time
from utils import StateConstructor

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


        action = random.choice(action_space[1:])
        return action

if __name__ == '__main__':
    stack = 500
    args = NoLimitHoldem.ARGS_CLS(n_seats=3,
                                stack_randomization_range=(0, 0),
                                starting_stack_sizes_list=[10000] * 3)
    env = NoLimitHoldem(is_evaluating=False, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())
    player = StaticRandomPlayer()


    state_constructor = StateConstructor(equity_bins=[0.25, 0.5, 0.75, 1], pot_bins=[100, 900, 1000], players_in_game=[0, 1, 2, 3, 4], raise_bins=[0, 50, 100, 500], stack_bins=[100, 900, 1000], n_workers=4)

    state_0 = env.reset()

    start = time.time()
    for i in range(300000):
        if (i%1000 == 0):
            print(i/1000)
        action_space = env.get_legal_actions()
        action = player.action(action_space, state_0, {})
        state, reward, done, _ = env.step([action, random.randint(100, 1000)])
        
        if env.state_dict()['current_player'] == 0:
            state_constructor.construct_state(env.state_dict(), 0)
        #board = state_dict['board_2d']
        #hand = state_dict['seats'][i%3]['hand']
        #current_bet = state_dict['seats'][i%3]['current_bet']
        #fold = state_dict['seats'][i%3]['folded_this_episode']
        if done:
            env.reset()

    print("Time elapsed: ", time.time()-start)




