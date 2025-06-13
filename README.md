# Bandits Evaluation

This project evaluates and compares three multi-armed bandit algorithms on a news recommendation dataset.

## Dataset Format

It should consist of two columns:
- **Article Type** (int: 0–5): The identifier of the news article (used as bandit arm index)
- **Rating** (float: 0–10): The reward received after selecting an article

All rewards are assumed to come from the same user.

## Algorithms

- `EpsilonGreedy`: Balances exploration and exploitation using an epsilon parameter.
- `UCB1`: Upper Confidence Bound approach.
- `NaiveStrategy`: Random baseline that selects arms uniformly.

## How to Run

```bash
python main.py
```

This will:
- Simulate 1000 arm pulls using each strategy
- Plot cumulative reward over time
- Print final reward comparison

## Project Origin

This exercise is part of the **Data Science in Production** course at IT University of Copenhagen.

Maintainer: Jersey
