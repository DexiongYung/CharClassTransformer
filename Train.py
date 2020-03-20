import torch

import torch.nn as nn
import string
import pandas as pd
from Model.Transformer import Transformer
from Dataset.Dataset import Dataset
from torch.utils.data import DataLoader
from Utilities.Convert import *

SOS = 'SOS'
PAD = 'PAD'
EOS = 'EOS'
DECODER_CHARS = ['f', 'm', 'l', 's', SOS, PAD]
NUM_DECODER_CHARS = len(DECODER_CHARS)
ENCODER_CHARS = [c for c in string.printable] + [SOS, PAD, EOS]
NUM_ENCODER_CHARS = len(ENCODER_CHARS)
FORMATS = ['first middle last', 'last, first middle']
NUM_FORMATS = len(FORMATS)

def train(src: list, trg: list):
    optimizer.zero_grad()

    if len(src) + 1 != len(trg):
        raise Exception('src and trg should be the same length')

    src = indexTensor([src], len(src), ENCODER_CHARS)
    trg = targetTensor([trg], len(trg), DECODER_CHARS)
    loss = 0

    for i in range(len(src)):
        prob = transformer.forward(src, trg[0:i+1])
        loss += criterion(prob[i], trg[i + 1])
    
    loss.backward()
    optimizer.step()

def enumerate_train():
    for name, first, middle, last, format in dl:
        sample = torch.distributions.Categorical(torch.ones(NUM_FORMATS) * (1/NUM_FORMATS)).sample()
        first_len = len(first[0])
        middle_len = len(middle[0])
        last_len = len(last[0])

        src, trg = [], []
        if sample == 0:
            src = [c for c in f"{first[0]} {middle[0]} {last[0]}"]
            trg = [DECODER_CHARS[4]] + first_len * [DECODER_CHARS[0]] + [DECODER_CHARS[3]] + middle_len * [DECODER_CHARS[1]] + [DECODER_CHARS[3]] + last_len * [DECODER_CHARS[2]]
        elif sample == 1:
            src = [c for c in f"{last[0]}, {first[0]} {middle[0]}"]
            trg = [DECODER_CHARS[4]] + last_len * [DECODER_CHARS[2]] + [DECODER_CHARS[3]] * 2 + [DECODER_CHARS[0]] * first_len + [DECODER_CHARS[3]] + [DECODER_CHARS[1]] * middle_len
        
        train(src, trg)
    torch.save({'weights': transformer.state_dict()}, os.path.join("Checkpoints/transformer.path.tar"))


transformer = Transformer(NUM_ENCODER_CHARS, NUM_DECODER_CHARS, ENCODER_CHARS.index('PAD'), DECODER_CHARS.index('PAD'))
criterion = nn.CrossEntropyLoss()
lr = 0.005
optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
df = pd.read_csv('Data/labelled.csv')
ds = Dataset(df)
dl = DataLoader(ds, batch_size = 1, shuffle=True)

enumerate_train()