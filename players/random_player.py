import torch
import random


class RandomPlayer():
    def __init__(self, device):
        self.device = device

    def select_action(self, i, state_dict, player_seat):
        return torch.tensor([[random.choices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0])[0]]], device=self.device, dtype=torch.long)
    
    def play(self, i, state_dict, done, flops, reward, player_seat):
        return torch.tensor([[random.choices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0])[0]]], device=self.device, dtype=torch.long), 0