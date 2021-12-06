###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Adapted: 2021-11-11
###############################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import random
import numpy as np


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """

    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.W_ix = nn.Parameter(torch.Tensor(self.embed_dim, self.hidden_dim))
        self.W_ih = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.W_fx = nn.Parameter(torch.Tensor(self.embed_dim, self.hidden_dim))
        self.W_fh = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.W_gx = nn.Parameter(torch.Tensor(self.embed_dim, self.hidden_dim))
        self.W_gh = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.W_ox = nn.Parameter(torch.Tensor(self.embed_dim, self.hidden_dim))
        self.W_oh = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(self.hidden_dim))
        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        bound = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-bound, bound)
        self.b_f.data = torch.add(self.b_f.data, 1)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, hidden dimension].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        input_len, batch_size, hidden_dim = embeds.size()
        h_t = torch.zeros(batch_size, self.hidden_dim).to(embeds.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(embeds.device)

        output = []
        for t in range(input_len):
            x_t = embeds[t, :, :]
            i_t = torch.sigmoid(x_t @ self.W_ix + h_t @ self.W_ih + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_fx + h_t @ self.W_fh + self.b_f)
            g_t = torch.tanh(x_t @ self.W_gx + h_t @ self.W_gh + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_ox + h_t @ self.W_oh + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            output.append(h_t.unsqueeze(0))

        output = torch.cat(output, dim=0)
        # print("output shape 1 = ", output.shape)
        # # output = output.transpose(0, 1).contiguous()
        # print("output shape = ", output.shape)
        assert output.shape == (input_len, batch_size, self.hidden_dim), "Output shape is wrong !!!"
        return output
        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """

    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.args = args
        self.embedding = nn.Embedding(num_embeddings=args.vocabulary_size, embedding_dim=args.embedding_size)
        # self.LSTM = nn.LSTM(hidden_size=args.lstm_hidden_dim, input_size=args.embedding_size)
        self.LSTM = LSTM(lstm_hidden_dim=args.lstm_hidden_dim, embedding_size=args.embedding_size)
        self.classifier = nn.Linear(args.lstm_hidden_dim, args.vocabulary_size)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # x shape = [input length, batch size]
        embed = self.embedding(x)
        output = self.LSTM(embed)
        # LSTM output shape: [input length, batch size, hidden dimension]

        # logits = self.classifier(output[-1])
        logits = self.classifier(output)

        return logits
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=0.):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        print("starting sampling ...")
        start_letters_ix = torch.randint(self.args.vocabulary_size, (batch_size,))  # shape [batch_size]
        curr_sample_ix = start_letters_ix.unsqueeze(0)  # shape = [1, batch_size]
        output_ix = curr_sample_ix  # shape =  [1, batch_size]
        for i in range(sample_length):
            curr_sample_ix = curr_sample_ix.to(self.args.device)
            next_sample = self.forward(curr_sample_ix)  # next_sample shape = [1, batch_size, vocab_size]
            if temperature == 0:
                next_sample_ix = torch.argmax(next_sample, dim=2)
            else:
                next_sample = F.softmax(next_sample / temperature, dim=2)
                next_sample_ix = torch.multinomial(next_sample[0], 1)
                next_sample_ix = next_sample_ix.squeeze()
                next_sample_ix = next_sample_ix.unsqueeze(0)
            output_ix = torch.cat((output_ix, next_sample_ix.cpu()))
            curr_sample_ix = next_sample_ix

        # output_ix shape: [sample_length, batch_size]

        new_output_ix = torch.transpose(output_ix, 0, 1)
        new_output_ix = new_output_ix.tolist()
        for batch_id in range(len(new_output_ix)):
            for char_id in range(len(new_output_ix[batch_id])):
                x = new_output_ix[batch_id][char_id]
                new_output_ix[batch_id][char_id] = self.args._ix_to_char[x]

        for batch_id in range(len(new_output_ix)):
            new_output_ix[batch_id] = ''.join(new_output_ix[batch_id])

        print("Sampling completed ... ")

        return new_output_ix

        #######################
        # END OF YOUR CODE    #
        #######################
