import gymnasium
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from memories import Transition, BehaviourTuple, ReplayMemory, ReservoirMemory
from network import BaseNetwork

class NFSPTrainer():
    def __init__(self, device, n_observations, n_actions, batch_size, batch_size_behaviour, gamma, eps_start, eps_end, eps_decay, tau, lr, eta):
        self.device = device
        #Initiaize hyperparameters
        self.n_observations = n_observations
        self.n_actions = n_actions 
        self.batch_size = batch_size #Batch size for training
        self.batch_size_behaviour = batch_size_behaviour #Batch size for average behaviour net training
        self.gamma = gamma #Discount factor 
        self.eps_start = eps_start #Initial epsilon value
        self.eps_end = eps_end #Minimum epsilon value
        self.eps_decay = eps_decay #Epsilon decay rate
        self.tau = tau #Target network soft update frequency
        self.lr = lr #Learning rate
        self.eta = eta #Anticipatory parameter
        self.action_space = gymnasium.spaces.discrete.Discrete(n_actions)
        
        #Initialize networks
        self.policy_net = BaseNetwork(n_observations, n_actions).to(device)
        self.target_net = BaseNetwork(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.behaviour_net = BaseNetwork(n_observations, n_actions).to(device)
        self.criterion_policy = nn.SmoothL1Loss()
        self.criterion_behaviour = nn.CrossEntropyLoss()
        self.optimizer_policy = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.optimizer_behaviour = optim.AdamW(self.behaviour_net.parameters(), lr=lr, amsgrad=True)

    
    def select_action(self, steps_done, state, mode='epsilon-greedy'):
        eta_sample = random.random()
        if eta_sample < self.eta:
            if mode == 'boltzmann':
                with torch.no_grad():
                    q_values = self.policy_net(state)
                    probabilities = F.softmax(q_values, dim=1)
                    action = torch.multinomial(probabilities, 1)
                    return action.view(1, 1), True #We return a true value if we're using epsilon greedy
            else:
                eps_sample = random.random()
                eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                    math.exp(-1. * steps_done / self.eps_decay)

                if eps_sample > eps_threshold:
                    with torch.no_grad():
                        # t.max(1) will return the largest column value of each row.
                        # second column on max result is index of where max element was
                        return self.policy_net(state).max(1).indices.view(1, 1), True #We return a true value if we're using epsilon greedy
                else:
                    return torch.tensor([[self.action_space.sample()]], device=self.device, dtype=torch.long), False #Is it really correct to sample a random action and return True anyway? It's not a best response
        else:
            with torch.no_grad():
                q_values = self.behaviour_net(state)
                probabilities = F.softmax(q_values, dim=1)
                action = torch.multinomial(probabilities, 1)
                return action.view(1, 1), False
            

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
        
        loss = self.criterion_policy(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer_policy.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer_policy.step()


    def optimize_behaviour_net(self, memory):
        if len(memory) < self.batch_size:
            return
        tuples = memory.sample(self.batch_size)
        batch = BehaviourTuple(*zip(*tuples))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)

        
        self.optimizer_behaviour.zero_grad()
        # Compute the predicted average behaviour
        outputs = self.behaviour_net(state_batch)
        loss = self.criterion_behaviour(outputs, action_batch.squeeze(1))
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.behaviour_net.parameters(), 100)
        self.optimizer_behaviour.step()

    
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

    
    def save_model(self, policy_path='policy_net1.pth', target_path='target_net1.pth', behaviour_path='behaviour_net1.pth'):
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.target_net.state_dict(), target_path)
        torch.save(self.behaviour_net.state_dict(), behaviour_path)
        print("SAVED!")

    
    def load_model(self, policy_path='policy_net.pth', target_path='target_net.pth', behaviour_path='behaviour_net.pth'):
        self.policy_net.load_state_dict(torch.load(policy_path))
        self.target_net.load_state_dict(torch.load(target_path))
        self.behaviour_net.load_state_dict(torch.load(behaviour_path))
        print("LOADED!")

        