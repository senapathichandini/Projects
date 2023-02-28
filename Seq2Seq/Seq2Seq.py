import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import nltk 
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Seq2Seq(nn.Module):
  def __init__(self,encoder,decoder):
    super(Seq2Seq,self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
  def forward(self,input,target):
    outputs = []
    output = []
    hidden,cell = self.encoder(input)
    x = torch.tensor([target[0]]).long()
    for i in range(len(target)):
      x,hidden,cell = self.decoder(x,hidden,cell)
      x = torch.tensor(x[0]).long().to(self.device)
      outputs.append(x[0].item()) 
    return outputs