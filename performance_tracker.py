import numpy as np
import time 

class PerformanceTracker():
    def __init__(self, n_episodes, n_players, version_name):
        self.episode_reward=np.zeros(n_episodes)
        self.flops = np.zeros((n_episodes, n_players))
        self.empty_hands=np.zeros(n_episodes)
        self.best_performance = 0
        self.savepath_policy = f"policy_{version_name}.pth"
        self.savepath_target = f"target_{version_name}.pth"
        self.start = time.time()

    def update(self, i, reward, last_phase, hero):
        self.episode_reward[i] = reward[0] 
        self.empty_hands[i] = 1 if last_phase == 0 else 0

        if (i%100 == 0 and i != 0):
            print(f"Step: {i}, reward of last 100: {sum(self.episode_reward[i-100:i])}, time elapsed: {time.time()-self.start}")
        if (i%5000 == 0 and i != 0):
            print(f"REWARD_NOW: {self.episode_reward[i-1]}, AVG_REWARD: {sum(self.episode_reward[i-5000:i])}, Flops player1: {sum(self.flops[i-5000:i, 0])}, Flops player2: {sum(self.flops[i-5000:i, 1])}")
            if sum(self.episode_reward[i-5000:i]) > best_performance:
                best_performance = sum(self.episode_reward[i-5000:i])
                hero.save_model(policy_path=self.savepath_policy, target_path=self.savepath_target)
                print("SAVED!")

    def get_stats(self):
        print('Complete')
        return self.episode_reward, self.flops, self.empty_hands