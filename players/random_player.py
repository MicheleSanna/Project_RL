import torch
import random


class RandomPlayer():
    def __init__(self, device, type='random'):
        self.device = device
        self.prob_dist = self.get_prob_dist(type)

    def select_action(self, i, state_dict, player_seat):
        return torch.tensor([[random.choices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], self.prob_dist)[0]]], device=self.device, dtype=torch.long)
    
    def play(self, i, state_dict, done, flops, reward, player_seat):
        return torch.tensor([[random.choices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], self.prob_dist)[0]]], device=self.device, dtype=torch.long), 0
    
    @staticmethod
    def get_prob_dist(type):
        if type == 'random':
            return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif type == 'no-fold':
            return [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif type == 'better':
            return [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        elif type == 'all-in':
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        elif type == 'aggressive':
            return [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0, 0]
        elif type == 'folder':
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif type == 'checker':
            return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            raise ValueError("Unknown player type")