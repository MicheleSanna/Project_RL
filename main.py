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
from matplotlib import pyplot as plt
from trainers.dqn import BaseNetwork, ReplayMemory, Transition, DQNTrainer
from players.nn_player import NNPlayer, print_state
from players.random_player import RandomPlayer
from players.dqn_trainer_player import DQNTrainerPlayer
from players.nfsp_trainer_player import NFSPTrainerPlayer
from plot import plot_rewards, plot_sum
from performance_tracker import PerformanceTracker
from trainers.nfsp import NFSPTrainer
from training_loop import training_loop


if __name__ == '__main__':

    device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
    )
    
    n_episodes = 100000
    n_observations = 14
    n_actions = 10

    args = NoLimitHoldem.ARGS_CLS(n_seats=2,
                                stack_randomization_range=(0, 9900),
                                starting_stack_sizes_list=[100,100],
                                use_simplified_headsup_obs=True,
                                randomize_positions=True)
    env = NoLimitHoldem(is_evaluating=False, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())

    state_constructor_player = StateConstructor(big_blind=100, 
                                                initial_stack=10000, 
                                                n_workers=4,
                                                device=device)
    state_constructor_adv = StateConstructor(big_blind=100, 
                                             initial_stack=10000, 
                                             n_workers=4,
                                             device=device)


    dqn_trainer = DQNTrainer(device=device, 
                            n_observations=n_observations, 
                            n_actions=n_actions,
                            batch_size=256, 
                            gamma=1, 
                            eps_start=0.12, 
                            eps_end=0.0001, 
                            eps_decay=n_episodes-(n_episodes/3), 
                            tau=400, 
                            lr=0.0005)
    
    dqn_trainer_adv = DQNTrainer(device=device, 
                            n_observations=n_observations, 
                            n_actions=n_actions,
                            batch_size=256, 
                            gamma=1, 
                            eps_start=0.12, 
                            eps_end=0.0001, 
                            eps_decay=n_episodes-(n_episodes/3), 
                            tau=400, 
                            lr=0.0005)
    
    nfsp_trainer = NFSPTrainer(device=device, 
                            n_observations=n_observations, 
                            n_actions=n_actions,
                            batch_size=256, 
                            gamma=1, 
                            eps_start=0.12, 
                            eps_end=0.0001, 
                            eps_decay=n_episodes-(n_episodes/3), 
                            tau=400, 
                            lr=0.0005,
                            eta=0.2,
                            batch_size_behaviour=256)
    
    nfsp_trainer_adv = NFSPTrainer(device=device, 
                            n_observations=n_observations, 
                            n_actions=n_actions,
                            batch_size=256, 
                            gamma=1, 
                            eps_start=0.12, 
                            eps_end=0.0001, 
                            eps_decay=n_episodes-(n_episodes/3), 
                            tau=400, 
                            lr=0.0005,
                            eta=0.2,
                            batch_size_behaviour=256)
    #dqn_trainer.load_model(policy_path="policy_4.0.pth", target_path="target_4.0.pth")
    hero_dqn = DQNTrainerPlayer(state_constructor=state_constructor_player,
                            trainer=dqn_trainer, 
                            replay_memory_size=10000,
                            device=device)
    
    opponent_dqn = DQNTrainerPlayer(state_constructor=state_constructor_adv,
                            trainer=dqn_trainer_adv, 
                            replay_memory_size=10000,
                            device=device)

    hero_nfsp = NFSPTrainerPlayer(state_constructor=state_constructor_player,
                            trainer=nfsp_trainer,
                            replay_memory_size=10000,
                            reservoir_memory_size=200000,
                            device=device)

    opponent_nfsp = NFSPTrainerPlayer(state_constructor=state_constructor_adv,
                            trainer=nfsp_trainer_adv,
                            replay_memory_size=10000,
                            reservoir_memory_size=200000,
                            device=device)
    
    #opponent_nn_policy = NNPlayer(state_constructor=state_constructor_adv, policy_net=BaseNetwork(n_observations, n_actions).to(device), policy_net_name="policy_4.0.pth", device=device)



    start = time.time()
    print(device)
    #adv = NNPlayer(policy_net=agent.policy_net, policy_net_name="policy_2.0.pth", device=device)
    episode_reward, flops, empty_hands = training_loop(env, 
                                                        hero = hero_dqn,
                                                        opponent = RandomPlayer(device=device),
                                                        num_episodes = n_episodes, 
                                                        version_name="nfsp")
    
    reward_averages = np.zeros(n_episodes//100)
    for i in range(0, n_episodes, 100):
        reward_averages[(i//100)] = sum(episode_reward[i:i+100])*0.1
    print("Time elapsed: ", time.time()-start)
    print("Avg mbb per hand: ", (sum(episode_reward)/len(episode_reward))*200)
    plot_rewards(reward_averages, 100)
    plot_sum(flops[:, 0], flops[:, 1], 2500, title='Average flops per 100 episodes', y_label='flops')
    plot_sum(empty_hands, empty_hands, 2500, title='Empty hands', y_label='hands')









# This is a test script to run the environment and print the state dictionary

    