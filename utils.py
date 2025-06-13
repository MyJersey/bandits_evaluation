import pandas as pd

def load_data(path="bandits.csv"):
    df = pd.read_csv(path)
    data = list(df.itertuples(index=False, name=None))
    arms = df.iloc[:, 0].unique()
    K = len(arms)
    return df, data, K
