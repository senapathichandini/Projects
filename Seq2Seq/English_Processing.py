import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import nltk 
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class English_Preprocessing:
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
  def english_processing(self,data):
    sentences = []
    for sent in data:
      sent = re.sub(r'[^\w\s]' ,'',str(sent))
      sentences.append(sent)
    return sentences
  def english_vectorization(self,data):
    tokens =[] 
    word  = data[0].split(" ")
    tokens.append(word[0])
    for i in range(len(word)):
      if word[i] not in tokens:
        tokens.append(word[i])
    for i in range(0,len(data)):
      words = data[i].split(" ")
      for j in range(len(words)):
        if words[j] not in tokens:
          tokens.append(words[j])
    english_dict = {}
    for idx,word in enumerate(tokens):
      english_dict.update({word:idx})
    return english_dict,tokens
  def vector_to_english(self,tokens):
    eng_vectors = {}
    for idx,word in enumerate(tokens):
      eng_vectors.update({idx:word})
    return eng_vectors
  def english_traslation(self,dicts,input):
    output = []
    for i in range(len(input)):
      word = dicts[input[i]]
      output.append(word)
    return output
  def english_to_tokens(self,eng_dict,input):
    tokens = []
    input  = re.sub(r'[^\w\s]', " ",input)
    input  = input.split(" ")
    #tokens.append(len(tokens)+1)
    for i in range(len(input)):
      word = eng_dict[input[i]]
      tokens.append(word)
    #tokens.append(len(tokens)+2)
    tokens = torch.tensor(tokens,dtype = torch.long).to(self.device)
    return tokens