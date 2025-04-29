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
from dqn import BaseNetwork, ReplayMemory, Transition, DQNTrainer
from players.nn_player import NNPlayer, print_state
from players.random_player import RandomPlayer
from players.dqn_trainer_player import DQNTrainerPlayer
from plot import plot_rewards, plot_sum
from performance_tracker import PerformanceTracker


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



def training_loop(env, hero, opponent, num_episodes, version_name):
    performance_tracker = PerformanceTracker(num_episodes, 2, version_name)
    print("Lesgo")
    for i in range(num_episodes):
        # Initialize the environment and get its state
        env.reset()
        state_dict = env.state_dict()
        #First move always of the hero
        state = hero.state_constructor.construct_state_continuous(state_dict, 0, False)
        action = hero.select_action(i, state)
        
        done = False
        while not done:
            _, reward, done, a = env.step(get_action(state_dict['main_pot'], action.item()))
            state_dict = env.state_dict()

            if state_dict['current_player'] == 0 or done:
                action, _ = hero.play(i, state_dict, done, performance_tracker.flops, reward)
            else:
                action, last_phase = opponent.play(i, state_dict, done, performance_tracker.flops, reward)

        performance_tracker.update(i, reward, last_phase, hero)

    return performance_tracker.get_stats()

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
    
    state_constructor_player = StateConstructor(equity_bins=[0.4, 0.6, 1], 
                                                pot_bins=[0, 0.05, 0.1, 0.2, 0.4, 0.8, 9999], 
                                                raise_bins=[0, 0.1, 0.5, 1, 3, 9999], 
                                                big_blind=100, 
                                                initial_stack=10000, 
                                                n_workers=4,
                                                device=device)
    
    state_constructor_adv = StateConstructor(equity_bins=[0.4, 0.6, 1], 
                                                pot_bins=[0, 0.05, 0.1, 0.2, 0.4, 0.8, 9999], 
                                                raise_bins=[0, 0.1, 0.5, 1, 3, 9999], 
                                                big_blind=100, 
                                                initial_stack=10000, 
                                                n_workers=4,
                                                device=device)
    
    n_episodes = 100000
    dqn_trainer = DQNTrainer(device=device, 
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
    #dqn_trainer.load_model(policy_path="policy_4.0.pth", target_path="target_4.0.pth")
    hero = DQNTrainerPlayer(state_constructor=state_constructor_player,
                            trainer=dqn_trainer, 
                            memory=ReplayMemory(10000),
                            device=device)
    
    start = time.time()
    print(device)
    #adv = NNPlayer(policy_net=agent.policy_net, policy_net_name="policy_2.0.pth", device=device)
    episode_reward, flops, empty_hands = training_loop(env, 
                                                        hero = hero,
                                                        opponent = RandomPlayer(device=device),
                                                        #opponent = NNPlayer(state_constructor=state_constructor_adv, policy_net=BaseNetwork(n_observations, n_actions).to(device), policy_net_name="policy_4.0.pth", device=device),
                                                        num_episodes = n_episodes, 
                                                        version_name="test")
    
    reward_averages = np.zeros(n_episodes//100)
    for i in range(0, 100000, 100):
        reward_averages[(i//100)] = sum(episode_reward[i:i+100])*0.1
    print("Time elapsed: ", time.time()-start)
    plot_rewards(reward_averages, 25)
    plot_sum(flops[:, 0], flops[:, 1], 2500, title='Average flops per 100 episodes', y_label='flops')
    plot_sum(empty_hands, empty_hands, 2500, title='Empty hands', y_label='hands')