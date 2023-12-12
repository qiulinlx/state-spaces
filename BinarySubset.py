import pandas as pd
import numpy as np

df=pd.read_csv('ClusteredSeq1.csv')
sequence= df['Sequence']
# # Remove rows where column 3 (City) has no value (empty string or NaN)
# df1 = df[df["0"].notna() & (df["0"] != '')]
# df1.to_csv('ClusteredSeq1.csv')
# #df=pd.read_csv('subsetdata.csv')

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

# Remove the first column
# # Initialize variables to keep track of the column with the most ones
# max_ones_count = 0
# max_ones_column = None
# # Iterate through the columns
# for col in df.columns:
#     # Count the number of ones in the current column
#     ones_count = df[col].sum()

#     # Check if the current column has more ones than the previous maximum
#     if ones_count > max_ones_count:
#         max_ones_count = ones_count
#         max_ones_column = col

# # Check if a column with the most ones was found
# if max_ones_column is not None:
#     print(f"Column '{max_ones_column}' has the most ones with {max_ones_count} ones.")
# else:
#     print("No column with ones found.")

max =df[0]


joined_column = pd.concat([sequence, max] , axis=1)

joined_column .to_csv('Binaryset.csv')