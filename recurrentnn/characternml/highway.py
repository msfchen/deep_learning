#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn

class Highway(nn.Module):
    """ Highway Networks have a skip-connection controlled by a dynamic gate """

    def __init__(self, embed_dim: int): 
        """ Init Highway Instance.
        @param embed_dim (int): word embedding dimension
        """    
        super(Highway, self).__init__()
        
        self.conv_out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x_conv_out):
        """ Take x_conv_out, compute the x_highway.
        @param x_conv_out (matrix): batch_size x embed_dim

        @returns scores (matrix): batch_size x embed_dim
        """    
        x_proj = torch.relu(self.conv_out_proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))

        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out

        return x_highway

### END YOUR CODE 

