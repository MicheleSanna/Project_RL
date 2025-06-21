import numpy as np
import time 

class PerformanceTracker():
    def __init__(self, n_episodes, n_players, version_name):
        self.episode_reward=np.zeros(n_episodes)
        self.flops = np.zeros((n_episodes, n_players))
        self.empty_hands=np.zeros(n_episodes)
        self.best_performance = 0
        self.savepath_policy = f"dqn_run_short/policy_{version_name}"
        self.savepath_target = f"dqn_run_short/target_{version_name}"
        self.savepath_behaviour = f"dqn_run_short/behaviour_{version_name}"
        self.start = time.time()

    def update(self, i, reward, last_phase, hero, hero_seat):
        self.episode_reward[i] = reward[hero_seat] 
        self.empty_hands[i] = 1 if last_phase == 0 else 0

        if (i%100 == 0 and i != 0):
            print(f"Step: {i}, avg mbb/hand of last 100: {sum(self.episode_reward[i-100:i])/0.5}, time elapsed: {time.time()-self.start}")
            if (i%10000 == 0):
                print(f"REWARD_NOW: {self.episode_reward[i-1]}, AVG_REWARD: {sum(self.episode_reward[i-10000:i])/50}, Flops player1: {sum(self.flops[i-10000:i, 0])}, Flops player2: {sum(self.flops[i-10000:i, 1])}")        
        if (i%1000 == 0):
            hero.trainer.save_model(policy_path=self.savepath_policy + f"_{i//1000}k.pth", target_path=self.savepath_target + f"_{i//1000}k.pth")
            #hero.trainer.save_model(policy_path=self.savepath_policy + f"_{i//1000}k.pth", target_path=self.savepath_target + f"_{i//1000}k.pth", behaviour_path=self.savepath_behaviour + f"_{i//1000}k.pth")
            #hero.trainer.save_model(policy_path="prova_policy" + f"_{i//1000}k.pth", target_path="prova_target" + f"_{i//1000}k.pth")
      

    def get_stats(self):
        return self.episode_reward, self.flops, self.empty_hands