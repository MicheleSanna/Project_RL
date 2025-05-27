import torch
import torch.nn as nn
import torch.nn.functional as F

class NNPlayer():
    def __init__(self, state_constructor, policy_net, policy_net_name, device):
        self.policy_net = policy_net
        self.device = device
        self.state_constructor = state_constructor
        self.policy_net.load_state_dict(torch.load(policy_net_name))

    def select_action(self, i, state_dict, player_seat):
        state = self.state_constructor.construct_state_continuous(state_dict, player_seat, False)
        #print_state(state, adv=True)
        with torch.no_grad():
            q_values = self.policy_net(state)
            probabilities = F.softmax(q_values, dim=1)
            action = torch.multinomial(probabilities, 1)
            return action.view(1, 1)
        
    def play(self, i, state_dict, done, flops, reward, player_seat):
        if not done:
            last_phase = state_dict['current_round']
            action = self.select_action(i, state_dict, player_seat) if not done else None
            flops[i, 1] = 1 if action.item() == 0 else 0
        else:
            action = None

        return action, last_phase
        

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