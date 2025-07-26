import torch
import torch.nn.functional as F

class NNPlayer():
    def __init__(self, state_constructor, policy_net, policy_net_name, mode, device, is_hero=False):
        self.policy_net = policy_net.to(device)
        self.mode = mode
        self.device = device
        self.state_constructor = state_constructor
        self.policy_net.load_state_dict(torch.load(policy_net_name))
        self.is_hero = 0 if is_hero else 1
        print("LOADED: ", get_param_norm(self.policy_net))

    def select_action(self, i, state_dict, player_seat, done):
        state = self.state_constructor.construct_state_continuous(state_dict, player_seat, done)
        #print_state(state, adv=True)
        if self.mode == 'boltzmann' and not done:
            with torch.no_grad():
                q_values = self.policy_net(state)
                probabilities = F.softmax(q_values, dim=1)
                action = torch.multinomial(probabilities, 1)
                return action.view(1, 1)
        elif not done:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                #print("Logits: ", self.policy_net(state))
                #print("Action: ", self.policy_net(state).max(1).indices.view(1, 1))
                return self.policy_net(state).max(1).indices.view(1, 1)


        
    def play(self, i, state_dict, done, flops, reward, player_seat):
        action = self.select_action(i, state_dict, player_seat, done)
        if not done:
            last_phase = state_dict['current_round']
            #print("Action: ", action.item())
            flops[i, self.is_hero] = 1 if action.item() == 0 else 0
        else:
            action = None
            last_phase = None

        return action, last_phase
        

def print_state(state, adv=False):
    if state is not None:
        game_phase = None
        for i in range(4):
            if state[0][i] == 1:
                game_phase = i
                break 
        print(f"{'ADV' if adv else 'AGENT'} Game phase: {game_phase} | Equity: {state[0][4]} | Pot: {state[0][5] * 10000} | Stack: {state[0][6] * 10000} | Norm call: {state[0][7] * 10000}")
    else:
        print(f"{'ADV' if adv else 'AGENT'} Episode end")

def get_param_norm(net, p=2):
    """Returns the norm of the parameters of the policy_net."""
    total_norm = 0.0
    for param in net.parameters():
        if param.requires_grad:
            total_norm += param.data.norm(p).item() ** p
    return total_norm ** (1. / p)