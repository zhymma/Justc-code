import math  
import torch
def get_positional_encoding(d_model, max_sequence_length):  
   # Create positional encoding  
   positional_encoding = torch.zeros(max_sequence_length, d_model)  
   row = torch.arange(0, max_sequence_length).reshape(-1,1)
   col = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
   positional_encoding[:, 0::2] = torch.sin(row / col)  
   positional_encoding[:, 1::2] = torch.cos(row / col)

   return positional_encoding.unsqueeze(0)
positional_encoding = get_positional_encoding(768, 768) 
print(positional_encoding.shape)