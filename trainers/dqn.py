import gymnasium
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from memories import Transition, ReplayMemory
from network import BaseNetwork

# This is the DQN trainer class that will be used to train the DQN agent.
class DQNTrainer():
    def __init__(self, device, n_observations, n_actions, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr):
        self.device = device
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.action_space = gymnasium.spaces.discrete.Discrete(n_actions)
        
        self.policy_net = BaseNetwork(n_observations, n_actions).to(device)
        self.target_net = BaseNetwork(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)


    def select_action(self, steps_done, state, mode='boltzmann'):
        if mode == 'boltzmann':
            with torch.no_grad():
                q_values = self.policy_net(state)
                probabilities = F.softmax(q_values, dim=1)
                action = torch.multinomial(probabilities, 1)
                return action.view(1, 1)
        else:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * steps_done / self.eps_decay)

            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[self.action_space.sample()]], device=self.device, dtype=torch.long) #before it was env.action_space"""
        

    def optimize_model(self, memory):
        if len(memory) < self.batch_size:
            return
        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))# Transpose the batch, converts batch-array of Transitions to Transition of batch-arrays

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            #SINGLE DQN: next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            next_state_actions = self.policy_net(non_final_next_states).max(1).indices
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def soft_update_target_net(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    
    def hard_update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def save_model(self, policy_path='policy_net1.pth', target_path='target_net1.pth'):
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.target_net.state_dict(), target_path)
        print("SAVED!")

    def load_model(self, policy_path='policy_net.pth', target_path='target_net.pth'):
        self.policy_net.load_state_dict(torch.load(policy_path))
        self.target_net.load_state_dict(torch.load(target_path))
        print("LOADED!")

