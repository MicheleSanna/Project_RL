import torch
from performance_tracker import PerformanceTracker


def get_action(action_id, big_blind=100):
    match action_id:
        case 0:
            return [0, 0]
        case 1:
            return [1, 0]
        case 2:
            return [2, big_blind]
        case 3:
            return [2, 1.5*big_blind]
        case 4:
            return [2, 2*big_blind]
        case 5:
            return [2, 4*big_blind]
        case 6:
            return [2, 8*big_blind]
        case 7:
            return [2, 16*big_blind]
        case 8:
            return [2, 32*big_blind]
        case 9:
            return [2, 256*big_blind] #All in action


def training_loop(env, hero, opponent, num_episodes, version_name):
    performance_tracker = PerformanceTracker(num_episodes, 2, version_name)
    print("Lesgo")
    for i in range(num_episodes):
        if i%2 == 0:
            hero_seat = 0
            opponent_seat = 1
        else:
            hero_seat = 1
            opponent_seat = 0
        # Initialize the environment and get its state
        _, reward, done, _ = env.reset()
        state_dict = env.state_dict()
        
        while not done:
            if state_dict['current_player'] == hero_seat:
                action, last_phase = hero.play(i, state_dict, done, performance_tracker.flops, reward, hero_seat)
            else:
                action, last_phase = opponent.play(i, state_dict, done, performance_tracker.flops, reward, opponent_seat)

            _, reward, done, _ = env.step(get_action(action.item()))
            state_dict = env.state_dict()


        hero.play(i, state_dict, done, performance_tracker.flops, reward, hero_seat)
        opponent.play(i, state_dict, done, performance_tracker.flops, reward, opponent_seat)
        performance_tracker.update(i, reward, last_phase, hero, hero_seat)

    return performance_tracker.get_stats()