import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm

import pandas as pd
import csv

tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )

model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()

df=pd.read_csv('Binaryset.csv')
X= df['Sequence']
# Original list of strings
X = X.tolist()

# Add spaces between letters in each element
modified_list = [' '.join(list(sequences)) for sequences in X]
modified_list = ['"' + word + '"' for word in modified_list]

ids = tokenizer.batch_encode_plus(modified_list, add_special_tokens=True, pad_to_max_length=True)

input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]

features = [] 
for seq_num in range(len(embedding)):
    seq_len = (attention_mask[seq_num] == 1).sum()
    seq_emd = embedding[seq_num][1:seq_len-1]
    features.append(seq_emd)

# Specify the CSV file path
csv_file = 'ProtBertdata.csv'

# Write the list of NumPy arrays to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(features)