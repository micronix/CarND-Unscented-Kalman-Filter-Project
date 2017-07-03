import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./build/output.txt', delimiter='\t')

print(df['NIS'])

plt.scatter(df['time_stamp'], df['NIS'])
plt.show()
