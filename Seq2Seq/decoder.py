import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import nltk 
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Decoder(nn.Module):
  def __init__(self,input_size,vocab_size,hidden_size,num_layers,output):
    super(Decoder,self).__init__()
    self.input_size  = input_size
    self.vocab_size  = vocab_size
    self.hiddern_size = hidden_size
    self.hidden_size = hidden_size
    self.num_layers  = num_layers
    self.output      = output
    self.embd   = nn.Embedding(self.input_size,self.vocab_size)
    self.lstm   = nn.LSTM(self.vocab_size,self.hidden_size,self.num_layers)
    self.linear = nn.Linear(self.hidden_size,self.output)
  def forward(self,input,hidden,cell):
    x             = self.embd(input)
    x,(hidden,cell) = self.lstm(x,(hidden,cell))
    pred          = self.linear(x)*100
    pred          = f.relu(pred)
    pred          = torch.round(pred)
    return pred,hidden,cell