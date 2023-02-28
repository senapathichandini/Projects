import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import nltk 
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from encoder import Encoder
from decoder import Decoder
from Seq2Seq import Seq2Seq
from English_Processing import English_Preprocessing
from Hindi_Processing import Hindi_Preprocessing

torch.set_grad_enabled(True) 
class Model:
  def __init__(self,input_size,output_size,vocab_size,hidden_size,num_layers):
    self.input_size = input_size
    self.output_size = output_size
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_layers   = num_layers
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.encoder = Encoder(self.input_size,self.vocab_size,self.hidden_size,self.num_layers).to(self.device)
    self.decoder = Decoder(self.input_size,self.vocab_size,self.hidden_size,self.num_layers,self.output_size).to(self.device)
    self.model   = Seq2Seq(self.encoder,self.decoder).to(self.device)
    self.optim   = optim.Adam(self.model.parameters() ,lr = 3e-4)
    self.loss    = []
    self.episode = []
    print('started')
    self.english = English_Preprocessing()
    self.eng_processing = self.english.english_processing(english)
    self.eng_dict,self.tokens = self.english.english_vectorization(self.eng_processing)
    self.hindi = Hindi_Preprocessing()
    self.hid_processing = self.hindi.hindi_preprocessing(hindi)
    self.hid_dict,self.hid_tokens = self.hindi.hindi_vectorization(self.hid_processing)
    self.hindi_translation = self.hindi.hindi_translation(self.hid_tokens)
    self.episodes = int(len(self.hid_tokens)*0.80)
    print('ended')
  def save(self):
    torch.save(self.model.state_dict(),"/content/drive/MyDrive/Datasets/seq2seq.pth")
    torch.save(self.encoder.state_dict(),"/content/drive/MyDrive/Datasets/encoder.pth")
    torch.save(self.decoder.state_dict(),"/content/drive/MyDrive/Datasets/decoder.pth")
  def ploting(self):
    plt.plot(self.episode,self.loss)
    plt.xlabel("episodes")
    plt.ylabel("loss")
    plt.savefig("/content/drive/MyDrive/Datasets/seq2seqloss.png")
    plt.close()
  def train(self):
    self.model.train()
    for i in range(self.episodes):
      input = self.english.english_to_tokens(self.eng_dict,self.eng_processing[i])
      target= self.hindi.hindi_to_tokens(self.hid_dict,self.hid_processing[i])
      prediction = self.model(input,target)
      translation = self.hindi.translation(prediction,self.hindi_translation)
      prediction = torch.tensor(prediction).float().to(self.device)      
      loss = (target-prediction)**2
      loss = loss.mean()
      loss.requires_grad = True
      self.save()
      self.optim.zero_grad()
      loss.backward()
      self.optim.step()
      print("=======================================================================")
      print("epochs:",i,"/",self.episodes,"loss:-",loss.item())
      print("*********************************************")
      print("english:",self.eng_processing[i])
      print("*********************************************")
      print("translated",translation)
      self.loss.append(loss.item())
      self.episode.append(i)
      self.ploting()
