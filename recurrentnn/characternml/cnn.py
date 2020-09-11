#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn

class CNN(nn.Module):
    """ 1-dimensional convolutions.
        The convolutional layer has two hyperparameters:
            the kernel size k (also called window size), and 
            the number of filters f (also called number of output features or number of output channels).
    """
    def __init__(self, char_embed_dim: int, 
                       word_embed_dim: int, 
                       max_word_length: int=21, 
                       kernel_size: int=5):
        """ Init CNN Instance.
        @param char_embed_dim (int): character embedding dimension  # e_char
        @param word_embed_dim (int): the size of the final word embedding   # e_word (set filter number to be equal to e_word)
        @param max_word_length (int): max word length   # m_word
        @param kernel_size (int): window size
        """    
        super(CNN, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=word_embed_dim, 
            kernel_size=kernel_size,
            bias=True)

        # MaxPool simply takes the maximum across the second dimension
        self.maxpool = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x):
        """ Take x_reshaped, compute the x_vonv_out.
        @param x (tensor): b_size, e_char, m_word
        
        @returns x_conv_out (tensor): b_size, e_word
        """    
        x_conv = self.conv1d(x)     # => b_size, e_word, max_word_length - kernel_size + 1
        x_conv_out = self.maxpool(torch.relu(x_conv)).squeeze()     # => b_size, e_word
        
        return x_conv_out

### END YOUR CODE

