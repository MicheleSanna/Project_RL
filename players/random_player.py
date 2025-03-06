import torch
import random


class RandomPlayer():
    def __init__(self, device):
        self.device = device

    def select_action(self, state):
        return torch.tensor([[random.choices([0, 1, 2, 3, 4, 5, 6, 7], [0.1, 2, 1, 0.5, 0.1, 0.05, 0.02, 0.01])[0]]], device=self.device, dtype=torch.long)