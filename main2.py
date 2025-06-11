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
from players.nn_player import NNPlayer
from players.random_player import RandomPlayer
from players.dqn_trainer_player import DQNTrainerPlayer
from players.nfsp_trainer_player import NFSPTrainerPlayer
from plot import plot_rewards, plot_sum
from performance_tracker import PerformanceTracker
from trainers.nfsp import NFSPTrainer
from training_loop import training_loop

class StateConstructorPlain():
    def __init__(self):
        pass
        
    def construct_state_continuous(self, state, current_player, done):
        return state

def get_action(action_id, big_blind=100):
    match action_id:
        case 0:
            return [0, 0]
        case 1:
            return [1, 0]
        case 2:
            return [2, big_blind]
        case 3:
            return [2, 1.5*big_blind]
        case 4:
            return [2, 2*big_blind]
        case 5:
            return [2, 4*big_blind]
        case 6:
            return [2, 8*big_blind]
        case 7:
            return [2, 16*big_blind]
        case 8:
            return [2, 32*big_blind]
        case 9:
            return [2, 256*big_blind] #All in action
        

def training_loop2(env, hero, opponent, num_episodes, version_name):
    performance_tracker = PerformanceTracker(num_episodes, 2, version_name)
    print("Lesgo")
    for i in range(num_episodes):
        if i%2 == 0:
            hero_seat = 0
            opponent_seat = 1
        else:
            hero_seat = 1
            opponent_seat = 0
        # Initialize the environment and get its state
        state , reward, done, _ = env.reset()
        j = 0
        while not done:
            if j%2 == hero_seat:
                action, last_phase = hero.play(i, torch.from_numpy(state).to(device=device).unsqueeze(0), done, performance_tracker.flops, reward, hero_seat)
            else:
                action, last_phase = opponent.play(i, torch.from_numpy(state).to(device=device).unsqueeze(0), done, performance_tracker.flops, reward, opponent_seat)

            _, reward, done, _ = env.step(get_action(action.item()))
            j += 1


        hero.play(i, torch.from_numpy(state).to(device=device).unsqueeze(0), done, performance_tracker.flops, reward, hero_seat)
        opponent.play(i, torch.from_numpy(state).to(device=device).unsqueeze(0), done, performance_tracker.flops, reward, opponent_seat)
        performance_tracker.update(i, reward, last_phase, hero, hero_seat)

    return performance_tracker.get_stats()

if __name__ == '__main__':

    device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
    )
    
    n_episodes = 500000
    n_observations = 109
    n_actions = 8

    args = NoLimitHoldem.ARGS_CLS(n_seats=2,
                                stack_randomization_range=(0, 9900),
                                starting_stack_sizes_list=[100,100],
                                use_simplified_headsup_obs=True,
                                randomize_positions=True)
    env = NoLimitHoldem(is_evaluating=False, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())

    dqn_trainer = DQNTrainer(device=device, 
                            n_observations=n_observations, 
                            n_actions=n_actions,
                            batch_size=256, 
                            gamma=1, 
                            eps_start=0.1, 
                            eps_end=0.001, 
                            eps_decay=n_episodes-(n_episodes/3), 
                            tau=0.001, 
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
    hero_dqn = DQNTrainerPlayer(state_constructor=StateConstructorPlain(),
                            trainer=dqn_trainer, 
                            replay_memory_size=10000,
                            device=device)

    hero_nfsp = NFSPTrainerPlayer(state_constructor=StateConstructorPlain(),
                            trainer=nfsp_trainer,
                            replay_memory_size=10000,
                            reservoir_memory_size=180000,
                            device=device)
    
    opponent_nfsp = NFSPTrainerPlayer(state_constructor=StateConstructorPlain(),
                            trainer=nfsp_trainer_adv,
                            replay_memory_size=10000,
                            reservoir_memory_size=180000,
                            device=device)
    
    #opponent_nn_policy = NNPlayer(state_constructor=state_constructor_adv, policy_net=BaseNetwork(n_observations, n_actions).to(device), policy_net_name="policy_4.0.pth", device=device)



    start = time.time()
    print(device)
    #adv = NNPlayer(policy_net=agent.policy_net, policy_net_name="policy_2.0.pth", device=device)
    episode_reward, flops, empty_hands = training_loop2(env, 
                                                        hero = hero_dqn,
                                                        opponent = RandomPlayer(device=device),
                                                        num_episodes = n_episodes, 
                                                        version_name="test")
    
    reward_averages = np.zeros(n_episodes//100)
    for i in range(0, n_episodes, 100):
        reward_averages[(i//100)] = sum(episode_reward[i:i+100])*0.1
    print("Time elapsed: ", time.time()-start)
    plot_rewards(reward_averages, 25)
    plot_sum(flops[:, 0], flops[:, 1], 2500, title='Average flops per 100 episodes', y_label='flops')
    plot_sum(empty_hands, empty_hands, 2500, title='Empty hands', y_label='hands')
