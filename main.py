from epsilon_greedy import EpsilonGreedy
from ucb1 import UCB1
from random_baseline import NaiveStrategy
from utils import load_data

import matplotlib.pyplot as plt
import pandas as pd

# Load data
df, data, K = load_data()

# Initialize models
eg = EpsilonGreedy(0.1, K)
ucb = UCB1(K)
rand = NaiveStrategy(K)

# Run replay evaluation
eg_rewards, _ = eg.run_replay(data, 1000)
ucb_rewards, _ = ucb.run_replay(data, 1000)
rand_rewards, _ = rand.run_replay(data, 1000)

# Print summary
summary = pd.DataFrame({
    "Algorithm": ["EpsilonGreedy", "UCB1", "Random"],
    "CumulativeReward@1000": [eg_rewards[-1], ucb_rewards[-1], rand_rewards[-1]]
})
print(summary)

# Plot
plt.figure(figsize=(8,5))
plt.plot(eg_rewards, label="Epsilon-Greedy")
plt.plot(ucb_rewards, label="UCB1")
plt.plot(rand_rewards, label="Random Baseline")
plt.xlabel("Step (matching round)")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.show()
