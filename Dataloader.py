import pandas as pd
import numpy as np
from sklearn import model_selection
import torch.nn.functional as F

"""
# initialize list of lists
data = [['ABFFGGJS',1, 4, np.nan, np.nan, np.nan],
        ['GGFJJSFGHUON',2, 1,np.nan, np.nan, np.nan],
        ['SFGRGRWGV',5,3,4,8, np.nan],
        ["SRGRSDGFS",4,3,2,np.nan, np.nan],
        ['SGDSDGRG',3, 4, 9, 10, np.nan],
        ['SRGSFGSDGSD', 1,5,2,np.nan, np.nan],
        ['SGDDSGDSG', 2, 6, 1, 3, 5],
        ['GSFDFGDFHDFH', 5, np.nan, np.nan, np.nan, np.nan],
        ['DFHGDFGERHBTHERH', 8, 3, np.nan, np.nan, np.nan],
        ['EHBTETHBTEFHBDEFH', 6, 7, 2, np.nan, np.nan],
        ['EDTHBDTHBDFHBDFHB', 4, 5, np.nan, np.nan, np.nan],
        ['ETHFBFDG', 1, np.nan, np.nan, np.nan, np.nan],
        ['GDFG', 2, 9, np.nan, np.nan, np.nan],
        ['TEDFHETGHERGER', 4, 1, np.nan, np.nan, np.nan],
        ['REGERFESDGFG', 8, 3, np.nan, np.nan, np.nan],
        ['GERFERHETGER', 8, 10, np.nan, np.nan, np.nan],
        ['EDTGHETH', 6, 9, np.nan, np.nan, np.nan],
        ['ETHERFEF', 2, 8, np.nan, np.nan, np.nan],
        ['ERGREGRGETHERHER', 4, 6, 1, 3, 10, 8],
        ['ERHGREGERFER', 6, 4, np.nan, 3, np.nan, np.nan]
]

column_names = ['Sequence', 'Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6']
df = pd.DataFrame(data, columns=column_names)

df.to_csv("test.csv")
"""
df=pd.read_csv('test.csv')

import torch

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")

# Create the pandas DataFrame
y=df.drop(columns=['Sequence'])
y.drop(y.columns[0], axis=1, inplace=True)
#print(y)
X=df['Sequence']
import torch


# Categories
categories =np.array(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'], dtype= float)
num_rows = df.shape[0]
# One-hot encoding
encoded_data = []
#print(y)
for i in range(num_rows):
    row=y.loc[i]
    row = row.values

    encoded_row = [1 if category in row else 0 for category in categories]
    #print(encoded_row)
    encoded_data.append(encoded_row)

print(encoded_data)