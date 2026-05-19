import pandas as pd

splits = {'train': 'data/train.parquet', 'eval': 'data/eval.parquet'}
df = pd.read_parquet("hf://datasets/junwatu/indonesian-recipes/" + splits["train"])

df.to_csv("indonesian_recipes.csv", index=False)