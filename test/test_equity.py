from state_constructor import StateConstructor
import torch
from game.games import NoLimitHoldem


if __name__ == '__main__':
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
        )
    args = NoLimitHoldem.ARGS_CLS(n_seats=2,
                                stack_randomization_range=(0, 9900),
                                starting_stack_sizes_list=[100,100],
                                use_simplified_headsup_obs=True,
                                randomize_positions=True)
    env = NoLimitHoldem(is_evaluating=False, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())
    state_constructor_player = StateConstructor(big_blind=100, 
                                                    initial_stack=10000, 
                                                    n_workers=4,
                                                    device=device)
    env.reset()
    env.step([1,0])
    env.step([1,0])
    env.step([1,0])
    env.step([1,0])
    env.step([1,0])
    env.render()
    state_dict = env.state_dict()
    state = state_constructor_player.construct_state_continuous(state_dict, 0, False)
    print(state)
    env.step([1,0])
    env.step([1,0])
    env.render()
    state_dict = env.state_dict()
    state = state_constructor_player.construct_state_continuous(state_dict, 0, False)
    print(state)