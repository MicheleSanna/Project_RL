import numpy as np
from tools.montecarlo_numpy2 import Evaluation
import cppimport
import time
from multiprocessing import Pool
from multiprocessing import Process, Queue

calculator = cppimport.imp("tools.montecarlo_cpp.pymontecarlo")
E = Evaluation()

class StateConstructor():
    def __init__(self, equity_bins, pot_bins, players_in_game, raise_bins, stack_bins, n_workers=4):
        self.equity_bins = equity_bins
        self.pot_bins = pot_bins
        self.players_in_game = players_in_game
        self.raise_bins = raise_bins
        self.stack_bins = stack_bins
        self.Evaluator = Evaluation()
        self.pool = Pool(n_workers)
        self.equity=None
        self.last_game_phase=None


    def construct_state(self, state_dict, current_player):
        game_phase = state_dict['current_round']
        match game_phase:
            case 0:
                tablecards = []
                self.last_game_phase = game_phase
                self.equity = 0.33
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
        pot_idx = self.get_bin_index(pot, self.pot_bins)

        max_bet = max([state_dict['seats'][0]['current_bet'], state_dict['seats'][1]['current_bet'], state_dict['seats'][2]['current_bet']])
        call = state_dict['seats'][current_player]['current_bet'] - max_bet
        call_idx = self.get_bin_index(call, self.raise_bins)

        stack = state_dict['seats'][current_player]['stack']
        stack_idx = self.get_bin_index(stack, self.stack_bins)

        if game_phase != self.last_game_phase:
            self.calculate_equity(tablecards, state_dict['seats'][current_player]['hand'], iterations)
        
        equity_idx = self.get_bin_index(self.equity, self.equity_bins)

        if state_dict['seats'][current_player]['folded_this_episode']:
            players_in_game = 0
        elif state_dict['seats'][current_player-1]['folded_this_episode'] and state_dict['seats'][(current_player + 1) % 3]['folded_this_episode']:
            players_in_game = 1
        elif state_dict['seats'][current_player-1]['folded_this_episode']:
            players_in_game = 2
        elif state_dict['seats'][(current_player + 1) % 3]['folded_this_episode']:
            players_in_game = 3
        else:
            players_in_game = 4
        self.last_game_phase = game_phase

        return np.array([equity_idx, pot_idx, call_idx, stack_idx, game_phase, players_in_game])
    
    def calculate_equity(self, tablecards, hand, iterations, nplayers=3):
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
        

            
def _runner(my_cards, cards_on_table, players, iterations=5000):
     """Montecarlo test"""
     if len(cards_on_table) < 3:
         cards_on_table = {'null'}
     equity = calculator.montecarlo(my_cards, cards_on_table, players, iterations) 
     print("EQ: ", equity)

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
