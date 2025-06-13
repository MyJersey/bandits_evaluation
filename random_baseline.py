import numpy as np

class NaiveStrategy:
    def __init__(self, K):
        self.K = K

    def run_replay(self, data, max_pulls, verbose=False):
        tot_reward, cum_rewards, pull_history = 0, [], []
        for i, (actual_arm, reward) in enumerate(data):
            chosen_arm = np.random.choice(self.K)
            if chosen_arm == actual_arm:
                tot_reward += reward
                cum_rewards.append(tot_reward)
                pull_history.append(chosen_arm)
                if len(cum_rewards) >= max_pulls:
                    break
        return cum_rewards, pull_history
