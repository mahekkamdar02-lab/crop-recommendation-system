"""Create a SYNTHETIC dataset so you can test the pipeline before downloading Kaggle data.
The numbers are random - NOT real agronomy. Never report results from this file."""
import os
import numpy as np
import pandas as pd

CROPS = ["rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram",
         "lentil", "pomegranate", "banana", "mango", "grapes", "watermelon", "muskmelon", "apple",
         "orange", "papaya", "coconut", "cotton", "jute", "coffee"]
rng = np.random.default_rng(0)
rows = []
for i, crop in enumerate(CROPS):
    c = rng.uniform([0, 5, 5, 10, 30, 4.5, 30], [140, 145, 205, 40, 95, 8.5, 280])
    s = np.array([12, 12, 12, 2.5, 6, 0.4, 25])
    X = rng.normal(c, s, size=(100, 7))
    for x in X:
        rows.append([*x, crop, int(rng.integers(1, 7))])
df = pd.DataFrame(rows, columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label", "region"])
df[["N", "P", "K"]] = df[["N", "P", "K"]].clip(lower=0)
df["ph"] = df["ph"].clip(3.5, 9.5)
os.makedirs("data", exist_ok=True)
df.to_csv("data/synthetic_demo.csv", index=False)
print("Wrote data/synthetic_demo.csv  (SYNTHETIC - for pipeline testing only)")
