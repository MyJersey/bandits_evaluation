import numpy as np


class UCB1:
    """
    UCB1 Multi-Armed Bandit
    """
    def __init__(self, K):
        """
        Initialize the bandit with K arms.

        Args:
            K (int): Number of arms.
        """
        self.K = K
        self.t = 0
        self.avg_rewards = np.zeros(K)
        self.times_pulled = np.zeros(K)
        
    def select_arm(self):
        """
        Select the arm to pull according to the UCB1 strategy.

        Returns:
            int: The index of the selected arm.
        """
        self.t += 1

        
        for arm in range(self.K):
            if self.times_pulled[arm] == 0:
                return arm


        scores = self.avg_rewards + np.sqrt(2 * np.log(self.t) / self.times_pulled)
        return np.argmax(scores)

    def update(self, arm, reward):
        """
        Update the statistics for the selected arm with the new reward.

        Args:
            arm (int): Index of the pulled arm.
            reward (float): Observed reward.
        """
        self.times_pulled[arm] += 1
  
        n = self.times_pulled[arm]
        self.avg_rewards[arm] += (reward - self.avg_rewards[arm]) / n

    def run_replay(self, data, max_pulls, verbose=False):
        """
        Run offline replay evaluation with the provided data.
    
        For each round, select an arm using UCB1, and only update statistics
        if the selected arm matches the arm in the dataset.
        
        Args:
            data (list of (arm, reward)): Each element is a tuple of arm index and observed reward.
    
        Returns:
            (cum_rewards, pull_history)
        """
        cum_reward = 0
        cum_rewards = []
        pull_history = []
        for actual_arm, reward in data:
            arm = self.select_arm()
            if arm == actual_arm:
                self.update(arm, reward)
                cum_reward += reward
                cum_rewards.append(cum_reward)
                pull_history.append(arm)
                if len(cum_rewards) >= max_pulls:
                    break
        return cum_rewards, pull_history