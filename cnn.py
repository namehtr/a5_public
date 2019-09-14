#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """ Init CNN Model.
            @param embed_size (int): Embedding size (dimensionality)
            @param dropout_rate (float): Dropout probability
    """
    def __init__(self, input_channels, output_channels, m_word,kernel_size=5):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.maxp = nn.MaxPool1d(m_word-kernel_size+1)
    def forward(self,x_reshaped):
        x_conv = self.conv(x_reshaped)
        return self.maxp(x_conv)


### END YOUR CODE

