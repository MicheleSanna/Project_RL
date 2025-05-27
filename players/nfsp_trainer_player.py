import torch
import torch.nn as nn
import torch.nn.functional as F
from memories import ReplayMemory, ReservoirMemory

class NFSPTrainerPlayer():
    def __init__(self, state_constructor, trainer, replay_memory_size, reservoir_memory_size, device):
        self.device = device
        
        self.state_constructor = state_constructor
        self.trainer = trainer
        self.replay_memory = ReplayMemory(replay_memory_size)
        self.average_policy_memory = ReservoirMemory(reservoir_memory_size)
        self.last_action = None
        self.last_state = None
        self.total_steps = 0


    def select_action(self, i, state_dict, player_seat):
        state = self.state_constructor.construct_state_continuous(state_dict, player_seat, False)
        action, best_response = self.trainer.select_action(i, state)
        self.last_action = action
        self.last_state = state
        if best_response:
            self.average_policy_memory.push(state, action)
        return action
    

    def play(self, i, state_dict, done, flops, reward, player_seat, update_mode='hard'):
        self.total_steps += 1
        
        r = torch.tensor([reward[player_seat]], device=self.device)
        next_state = self.state_constructor.construct_state_continuous(state_dict, player_seat, done)
        # Store the transition in memory
        if self.last_state is not None:
            self.replay_memory.push(self.last_state, self.last_action, next_state, r)
        self.last_state = next_state

        # Perform one step of the optimization (on the policy network)
        self.trainer.optimize_behaviour_net(self.average_policy_memory)
        self.trainer.optimize_model(self.replay_memory)
        if update_mode == 'hard' and self.total_steps % self.trainer.tau == 0:
            self.trainer.hard_update_target_net()
        else:
            self.trainer.soft_update_target_net()
            
        if done:
            action = None
            self.last_action = None
            self.last_state = None
        else:
            action, best_response = self.trainer.select_action(i, next_state) 
            if best_response:
                self.average_policy_memory.push(next_state, action)
            flops[i, 0] = 1 if action.item() == 0 else 0

        self.last_action = action
        return action, None
    

    def print_state(state, adv=False):
        if state is not None:
            game_phase = None
            for i in range(4):
                if state[0][i] == 1:
                    game_phase = i
                    break 
            print(f"{"ADV" if adv else "AGENT"} Game phase: {game_phase} | Equity: {state[0][4]} | Pot: {state[0][5] * 10000} | Stack: {state[0][6] * 10000} | Norm call: {state[0][7] * 10000}")
        else:
            print(f"{"ADV" if adv else "AGENT"} Episode end")