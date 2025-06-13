import numpy as np


class EpsilonGreedy:
    """
    Epsilon-Greedy Multi-Armed Bandit
    """
    def __init__(self, epsilon, K):
        """
        Args:
            epsilon (float): Exploration probability (0 <= epsilon <= 1)
            K (int): Number of arms
        """
        if epsilon < 0 or epsilon > 1:
            raise Exception ('Epsilon should be in [0,1]')
        self.epsilon = epsilon
        self.K = K
        self.avg_rewards = np.zeros(self.K)
        self.tot_rewards = np.zeros(self.K)
        self.times_pulled = np.zeros(self.K)
        
    def get_tot_reward(self):
        """
        Returns the total reward accumulated.
        """
        return sum(self.tot_rewards)
    
    def get_number_pulls(self):
        """
        Returns the total number of times any arm has been pulled.
        """
        return sum(self.times_pulled)
    
    def __str__(self):
        s = f'=== MAB state ===\n'
        s += f'Epsilon Greedy: epsilon={self.epsilon}, arms={self.K}\n'
        s += f'Trained with {self.get_number_pulls()} actions; Total reward = {self.get_tot_reward()}\n'
        s += f'Avg. reward = {self.avg_rewards}\n'
        s += f'# pulls = {self.times_pulled}\n'
        s += '================='
        return s
        
    def select_arm(self):
        """
        Select the best arm according to the strategy.

        Returns:
            arm (int): The index of the selected arm
            explore (int): 1 if exploring, 0 if exploiting
        """
        p = np.random.rand()
        if p > self.epsilon:
            best_arm_idx = np.argmax(self.avg_rewards)
            return (best_arm_idx, 0)
        else:
            return (np.random.choice(range(0,self.K)), 1)

    def update(self, arm, reward):
        """
        Update the model parameters.

        Args:
            arm (int): Selected arm 
            reward (float): Observed reward
        """
        self.times_pulled[arm] += 1
        self.tot_rewards[arm] += reward
        self.avg_rewards[arm] = self.tot_rewards[arm] / self.times_pulled[arm]
        
    def reset(self):
        """
        Reset the state of the bandit.
        """
        self.avg_rewards = np.zeros(self.K)
        self.tot_rewards = np.zeros(self.K)
        self.times_pulled = np.zeros(self.K)
        
    def run_replay(self, data, max_pulls, verbose=False):
        """
        Runs a Replay offline evaluation with the given data.

        Args:
            data (list): List of (arm, reward) pairs
            verbose (bool): Whether to print details
        Returns:
            (cum_rewards, pull_history)
        """
        replay_num_matches = 0
        replay_tot_rewards = 0
        replay_cum_rewards = []
        pull_history = []
        for i, action in enumerate(data):
            chosen_arm, has_explored = self.select_arm()
            actual_arm, reward = action
            if chosen_arm == actual_arm:
                if verbose:
                    print(f'ROUND {replay_num_matches} ({i}) ---')
                    print(f'arm={actual_arm}, chosen_arm={chosen_arm}, match={chosen_arm == actual_arm}, has_explored={has_explored}, reward={reward}')
                    print(f'avg_rewards = {self.avg_rewards}')
                replay_num_matches += 1
                replay_tot_rewards += reward
                replay_cum_rewards.append(replay_tot_rewards)
                pull_history.append(chosen_arm)
                self.update(chosen_arm, reward)
                if replay_num_matches >= max_pulls:
                    break
        return replay_cum_rewards, pull_history