import numpy as np
from tools.montecarlo_numpy2 import Evaluation
#import cppimport
import torch
import time
from multiprocessing import Pool
from multiprocessing import Process, Queue

#calculator = cppimport.imp("tools.montecarlo_cpp.pymontecarlo")
E = Evaluation()

class StateConstructor():
    def __init__(self, big_blind, initial_stack, equity_bins = [0.4, 0.6, 1], pot_bins = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 9999], raise_bins = [0, 0.1, 0.5, 1, 3, 9999], n_workers=4, device='cpu'):
        self.device = device
        self.equity_bins = equity_bins
        self.pot_bins = pot_bins
        self.raise_bins = raise_bins
        self.Evaluator = Evaluation()
        self.big_blind_inv = 1/big_blind
        self.initial_stack_inv = 1/initial_stack
        self.pool = Pool(n_workers)
        self.equity=None
        self.last_game_phase=None
        self.adv_bet_history = np.zeros(4, dtype=np.float32)


    def construct_state_continuous(self, state_dict, current_player, done):
        if done:
            self.adv_bet_history[0] = 0
            self.adv_bet_history[1] = 0
            self.adv_bet_history[2] = 0
            self.adv_bet_history[3] = 0
            return None
        stack = state_dict['seats'][current_player]['stack']
        adv_stack = state_dict['seats'][current_player-1]['stack']
        game_phase = state_dict['current_round']
        player_bet = state_dict['seats'][current_player]['current_bet']
        adv_bet = state_dict['seats'][current_player - 1]['current_bet']
        pot = state_dict['main_pot'] + player_bet + adv_bet
        
        if game_phase != self.last_game_phase:
            match game_phase:
                case 0:
                    tablecards = []
                    iterations = 500
                case 1:
                    tablecards = state_dict['board_2d'][:3]
                    iterations = 2000
                case 2:
                    tablecards = state_dict['board_2d'][:4]
                    iterations = 1500
                case 3:
                    tablecards = state_dict['board_2d'][:5]
                    iterations = 1000
            self.calculate_equity(tablecards, state_dict['seats'][current_player]['hand'], iterations)
        
        norm_call = ((adv_bet - player_bet) * self.initial_stack_inv )
        self.adv_bet_history[game_phase] += norm_call
        state_array = torch.tensor([0, 0, 0, 0, self.equity, pot * self.initial_stack_inv, stack * self.initial_stack_inv, adv_stack * self.initial_stack_inv, norm_call, self.adv_bet_history[0], self.adv_bet_history[1], self.adv_bet_history[2], self.adv_bet_history[3]], dtype=torch.float32, device=self.device)
        
        state_array[game_phase] = 1
        self.last_game_phase = game_phase
        
        #print(f"Player {current_player} state: {state_array.unsqueeze(0)}")

        return state_array.unsqueeze(0)
    
    
    def construct_state_bin(self, state_dict, current_player):
        stack = state_dict['seats'][current_player]['stack']
        game_phase = state_dict['current_round']
        player_bet = state_dict['seats'][current_player]['current_bet']
        adv_bet = state_dict['seats'][current_player - 1]['current_bet']
        match game_phase:
            case 0:
                tablecards = []
                #self.last_game_phase = game_phase
                #self.equity = 0
                iterations = 500
            case 1:
                tablecards = state_dict['board_2d'][:3]
                iterations = 2000
            case 2:
                tablecards = state_dict['board_2d'][:4]
                iterations = 1500
            case 3:
                tablecards = state_dict['board_2d'][:5]
                iterations = 1000

        pot = state_dict['main_pot'] 
        pot_idx = self.get_bin_index((pot / stack) if stack != 0 else 999  , self.pot_bins)

        
        norm_call = ((adv_bet - player_bet) / (pot + adv_bet + player_bet) ) if adv_bet != 0 else 0
        #print(f"Call: {norm_call} ({state_dict['seats'][current_player - 1]['current_bet'] - state_dict['seats'][current_player]['current_bet']}), Pot: {pot}")
        call_idx = self.get_bin_index(norm_call, self.raise_bins)

        if game_phase != self.last_game_phase:
            self.calculate_equity(tablecards, state_dict['seats'][current_player]['hand'], iterations)
        
        equity_idx = self.get_bin_index(self.equity, self.equity_bins)
        self.last_game_phase = game_phase
        return np.array([equity_idx, pot_idx, call_idx, game_phase])
    
    
    def calculate_equity(self, tablecards, hand, iterations, nplayers=2):
        args_list = [{"card1": hand[0], "card2": hand[1], "tablecards": tablecards, "iterations": iterations, "player_amount": nplayers}] * 4
        self.equity = sum(self.pool.map(run_evaluation_wrapper, args_list))*0.25
    
    @staticmethod
    def get_bin_index(value, bins):
        for i in range(len(bins)):
            if value <= bins[i] and(i == 0 or value > bins[i-1]):
                return i
        return len(bins) - 1
    
    @staticmethod
    def _run_evaluation_wrapper(args):
        return E.run_evaluation(**args)
        

            
"""def _runner(my_cards, cards_on_table, players, iterations=5000):
     #Montecarlo test
     if len(cards_on_table) < 3:
         cards_on_table = {'null'}
     equity = calculator.montecarlo(my_cards, cards_on_table, players, iterations) 
     print("EQ: ", equity)"""

def run_evaluation_wrapper(args):
    return E.run_evaluation(**args)

"""def run_evaluation_process(card1, card2, tablecards, iterations, player_amount):
    result = E.run_evaluation(card1, card2, tablecards, iterations, player_amount)
    queue.put(result)"""

""""queue = Queue()
if __name__ == "__main__":
    my_cards = {'8H', '8D'}
    cards_on_table = {'QH', '7H', '9H', 'JH', 'TH'}
    expected_results = 95.6
    players = 2
    p = Pool(4)
    
    start = time.time() 
    args_list = [{"card1": [6, 0], "card2": [6, 1], "tablecards": [[11, 0], [5, 0]], "iterations": 2500, "player_amount": 3}] * 4
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(sum(p.map(run_evaluation_wrapper, args_list))*0.25)
    print(f"Time elapsed: {time.time() - start}")


    start = time.time() 
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(E.run_evaluation(card1=[6, 0], card2=[6, 1], tablecards=[[11, 0], [5, 0]], iterations=10000, player_amount=3))
    print(f"Time elapsed: {time.time() - start}")

    equity = calculator.montecarlo(my_cards, cards_on_table, players, 10000) 
    print(f"Time elapsed: {time.time() - start}, Equity: {equity}")
    start = time.time()
    equity = calculator.montecarlo(my_cards, cards_on_table, players, 10000) 
    print(f"Time elapsed: {time.time() - start}, Equity: {equity}")
    start = time.time()
    equity = calculator.montecarlo(my_cards, cards_on_table, players, 10000) 
    print(f"Time elapsed: {time.time() - start}, Equity: {equity}")
    start = time.time()
    equity = calculator.montecarlo(my_cards, cards_on_table, players, 10000) 
    print(f"Time elapsed: {time.time() - start}, Equity: {equity}")"""
