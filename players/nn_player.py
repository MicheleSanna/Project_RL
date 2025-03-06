import torch
import torch.nn as nn


class NNPlayer():
    def __init__(self, policy_net, policy_net_name, device):
        self.policy_net = policy_net
        self.device = device
        self.policy_net.load_state_dict(torch.load(policy_net_name))

    def select_action(self, state):
        return self.policy_net(state).max(1).indices.view(1, 1)