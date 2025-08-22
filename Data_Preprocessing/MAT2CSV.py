import pandas as pd
from scripy.io import loadmat

data = loadmat()
x = data[]
df = pd.DataFrame(x, columns=[""])
df.to_csv(r"", index=False)