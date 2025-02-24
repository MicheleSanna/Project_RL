import numpy as np

class ESARSA():
    def __init__(self, space_size, action_size, n_axis, gamma=1, lr_v=0):
        self.gamma = gamma
        self.lr_v = lr_v
        self.QValues = np.zeros((*space_size, action_size))
        #self.e = np.zeros((space_size, action_size))
        self.action_size = action_size
        self.space_size = space_size
        #self.last_state = None
        #self.last_action = None
        #self.last_reward = None


    def single_step_update(self, s, a, r, new_s, done, eps):
        if done:
            deltaQ = r - self.QValues[(*s, a)]
        else:
            deltaQ = r + self.gamma * np.dot(self.QValues[(*new_s,)], self.policy(new_s, eps)) - self.QValues[(*s, a)]
        
        self.QValues[(*s, a)] += self.lr_v * deltaQ

    
    def get_action_epsilon_greedy(self, s, eps):
        if np.random.rand() < eps:
            prob_actions = np.ones(self.action_size) / self.action_size
        else:
            best_value = np.max(self.QValues[(*s,)])
            best_actions = (self.QValues[(*s,)] == best_value)
            prob_actions = best_actions / np.sum(best_actions)
        
        return np.random.choice(self.action_size, p=prob_actions)
    

    def greedy_policy(self):
        return np.argmax(self.QValues, axis=2)
    

    def policy(self, s, eps):
        policy = np.ones(self.action_size) / self.action_size * eps
        best_value = np.max(self.QValues[(*s,)])
        best_actions = (self.QValues[(*s,)] == best_value)
        policy += best_actions / np.sum(best_actions) * (1 - eps)
        return policy
