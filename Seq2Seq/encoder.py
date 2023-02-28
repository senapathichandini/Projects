import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import nltk 
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Encoder(nn.Module):
  def __init__(self,input_size,vocab_size,hidden_size,num_layers):
    super(Encoder,self).__init__()
    self.input_size = input_size
    self.vocab_size = vocab_size
    self.hidden_size= hidden_size
    self.num_layers = num_layers
    self.embd       = nn.Embedding(self.input_size,self.vocab_size)
    self.lstm       = nn.LSTM(self.vocab_size,self.hidden_size,self.num_layers)
  def forward(self,input):
    input         = self.embd(input)
    x,(hidden,cell )= self.lstm(input)
    return hidden,cell