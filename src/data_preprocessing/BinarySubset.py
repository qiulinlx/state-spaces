import pandas as pd
import numpy as np

df=pd.read_csv('ClusteredSeq1.csv')
sequence= df['Sequence']

Y = df.drop(df.columns[0], axis=1)

# Categories
categories =np.array(np.arange(0,500), dtype= float) #Get a list of all the clusters
num_rows = df.shape[0] #no. samples in dataset

#One-hot encoding
y = []

for i in range(num_rows):
    '''Encoding each protein sequence's GO annotation as a binary vector of length equal to the number of clusters'''
    row=Y.loc[i]
    row = row.values

    encoded_row = [1 if category in row else 0 for category in categories]
    y.append(encoded_row)


df=pd.DataFrame(y)

max =df[0]


joined_column = pd.concat([sequence, max] , axis=1)

joined_column .to_csv('Binaryset.csv')