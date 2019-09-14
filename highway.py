#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
### YOUR CODE HERE for part 1h

class Highway(nn.Module):
    """ Init Highway Model.
            @param embed_size (int): Embedding size (dimensionality)
            @param dropout_rate (float): Dropout probability
    """
    def __init__(self, embed_size, dropout_rate=0.3):
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate
        self.proj = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.gate = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self,conv_out : torch.Tensor):
        x_proj = F.relu(self.proj(conv_out))
        x_gate = torch.sigmoid(self.gate(conv_out))
        x_highway = x_proj* x_gate+conv_out*(1-x_gate)
        return self.dropout(x_highway)

### END YOUR CODE

