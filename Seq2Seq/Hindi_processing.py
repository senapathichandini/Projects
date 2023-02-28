import torch 
import torch.nn as nn

class Hindi_Preprocessing:
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  def hindi_preprocessing(self,data):
    sentence = []
    for sent in data:
      sent = sent.replace("."," ")
      sent = sent.replace(","," ")
      sent = sent.replace("।"," ")
      sent = sent.replace("-"," ")
      sent = sent.replace("("," ")
      sent = sent.replace(")"," ")
      sent = sent.replace("["," ")
      sent = sent.replace("]"," ")
      sent = sent.replace("'"," ")
      sent = sent.replace("?"," ")
      sent = sent.replace("/"," ")
      sent = sent.replace(":"," ")
      sentence.append(sent)
    return sentence
  def hindi_vectorization(self,data):
    tokens =[]
    word = data[0].split(" ")
    tokens.append(word[0])
    for i in range(len(word)):
      if word[i] not in tokens:
        tokens.append(word[i])
    for sent in data:
      words = sent.split(" ")
      for i in range(0,len(words)):
        if words[i] not in tokens:
          tokens.append(words[i])
    hindi_dict = {}
    count = 0
    for i in range(len(tokens)):
      hindi_dict.update({tokens[i]:count})
      count +=1
    return hindi_dict,tokens
  def hindi_translation(self,tokens):
    hindi_vector = {}
    for idx,word in enumerate(tokens):
      hindi_vector.update({idx:word})
    return hindi_vector
  def hindi_to_tokens(self,hindi_dict,input):
    input = input.replace("."," ")
    input = input.replace(","," ")
    input = input.replace("।"," ")
    input = input.replace("-"," ")
    input = input.replace("("," ")
    input = input.replace(")"," ")
    input = input.replace("["," ")
    input = input.replace("]"," ")
    input = input.replace("'"," ")
    input = input.replace("?"," ")
    input = input.replace("/"," ")
    input = input.replace(":"," ")
    words = input.split(" ")
    tokens = []
    #tokens.append(len(tokens)+1)
    for i in range(len(words)):
      word = hindi_dict[words[i]]
      tokens.append(word)
    #tokens.append(len(tokens)+2)
    tokens = torch.tensor(tokens,dtype = torch.long).to(self.device)
    return tokens
  def translation(self,input,vect):
    output = []
    for i in range(len(input)):
      word  = vect[input[i]]
      output.append(word)
    return output