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

from datetime import datetime
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
# from argparse import Namespace

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    print("Training is starting ...")
    set_seed(args.seed)
    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    data_loader = DataLoader(dataset, args.batch_size,
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)
    # Create model
    args.vocabulary_size = dataset.vocabulary_size
    # args._char_to_ix = dataset._char_to_ix
    # print("char_to_ix", args._char_to_ix)
    print("Vocabulary size = ", args.vocabulary_size)
    # model_args = Namespace(vocabulary_size= dataset.vocabulary_size, embedding_size= args.embedding_size,
    #                        lstm_hidden_dim=args.lstm_hiddend_dim)
    model = TextGenerationModel(args)
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # Training loop
    all_losses = []
    print(len(data_loader))
    for epoch in range(args.num_epochs):
        print("Epoch = ", epoch)
        epoch_loss = 0
        for i, sentence in enumerate(data_loader, 0):
            if i%40 == 0:
                print("Epoch = ", epoch, " i = ", i, '/', len(data_loader))
            inputs, targets = sentence
            # print("Shape of inputs and targets: ", inputs.shape, targets.shape)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            loss = 0
            predictions = model(inputs)
            # print("shape of predictions and targets: ",predictions.shape, targets.shape)
            # loss = criterion(predictions, targets)
            for ch_idx in range(args.input_seq_length):
                # print(predictions[ch_idx].shape, targets[ch_idx].shape)
                l = criterion(predictions[ch_idx], targets[ch_idx])
                loss += l
            loss /= args.input_seq_length
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            epoch_loss += loss
        all_losses.append(epoch_loss)
        print("epoch: ", epoch, " loss = ", epoch_loss)
    print("Training completed")
    return model, all_losses
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    print(args)
    print("==" * 10)
    model, all_losses = train(args)
    print("plotting ...")
    fig, axs = plt.subplots(1)
    x = [e + 1 for e in range(len(all_losses))]
    axs[0].plot(x, all_losses, label='loss')
    axs[0].set_title("Training loss")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig('./rnn_loss_debugging.png')
    print("plotting completed!")
