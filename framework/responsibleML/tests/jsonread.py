import pandas as pd
import numpy as np
import json
import os


with open('sample.json', 'r') as f:
    data = json.load(f)
print(data)
df = pd.DataFrame(data)

#print(df.head())

