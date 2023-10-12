import pandas as pd
import os
import urllib.request
from pathlib import Path
#from tqdm import tqdm

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data.csv")

for url, path in zip(df["url"].tolist(), df["name"].tolist()):
    print("downloading from: ", url)
    urllib.request.urlretrieve(url, Path("data") / path)