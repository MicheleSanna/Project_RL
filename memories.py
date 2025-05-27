from collections import namedtuple, deque
from itertools import count
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
BehaviourTuple = namedtuple('BehaviourTuple',('state', 'action'))


class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        #Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class ReservoirMemory():
    def __init__(self, capacity):
        #Initialize the memory with a fixed capacity.
        self.capacity = capacity
        self.memory = []
        self.n_seen = 0  # Total number of items seen so far

    def push(self, *args):
        self.n_seen += 1
        if len(self.memory) < self.capacity:
            self.memory.append(BehaviourTuple(*args))
        else:
            # Replace an existing item with decreasing probability
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.memory[idx] = BehaviourTuple(*args)

    def sample(self, k):
        return random.sample(self.memory, min(k, len(self.memory)))

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]