import torch
import torch.nn as nn
import torch.nn.functional as F
from memories import Transition, ReplayMemory

class DQNTrainerPlayer():
    def __init__(self, state_constructor, trainer, replay_memory_size, device):
        self.state_constructor = state_constructor
        self.trainer = trainer
        self.memory = ReplayMemory(replay_memory_size)
        self.device = device
        self.last_action = None
        self.last_state = None


    def select_action(self, i, state_dict, player_seat):
        state = self.state_constructor.construct_state_continuous(state_dict, player_seat, False)
        action = self.trainer.select_action(i, state)
        self.last_action = action
        self.last_state = state
        return action
       
        
    def play(self, i, state_dict, done, flops, reward, player_seat):
        r = torch.tensor([reward[player_seat]], device=self.device)
        next_state = self.state_constructor.construct_state_continuous(state_dict, player_seat, done)
        # Store the transition in memory
        if self.last_state is not None:
            self.memory.push(self.last_state, self.last_action, next_state, r)
        self.last_state = next_state

        # Perform one step of the optimization (on the policy network)
        self.trainer.optimize_model(self.memory)
        self.trainer.update_target_net()

        if done:
            action = None
            self.last_action = None
            self.last_state = None
        else:
            action = self.trainer.select_action(i, self.last_state) 
            flops[i, 0] = 1 if action.item() == 0 else 0
            self.last_action = action

        return action, None