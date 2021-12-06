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


def write_to_file(sample_sentences, filename):
    with open(filename, 'w') as f:
        for sentence in sample_sentences:
            f.write("%s\n" % sentence)


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
    args._ix_to_char = dataset._ix_to_char
    print("Vocabulary size = ", args.vocabulary_size)
    # model_args = Namespace(vocabulary_size= dataset.vocabulary_size, embedding_size= args.embedding_size,
    #                        lstm_hidden_dim=args.lstm_hiddend_dim)
    model = TextGenerationModel(args)
    model = model.to(args.device)
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # Training loop
    all_losses = []
    print(len(data_loader))
    epochs_sampling = [1, 5, args.num_epochs]

    training_filename = args.txt_file.split("/")[1].split(".")[0]
    sample_filename = "samples/{}_samples.txt".format(training_filename)
    myfile = open(sample_filename, "w")
    for epoch in range(args.num_epochs):
        print("Epoch = ", epoch)
        epoch_loss = 0
        for i, sentence in enumerate(data_loader, 0):
            inputs, targets = sentence
            # print("Shape of inputs and targets: ", inputs.shape, targets.shape)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            loss = 0
            predictions = model(inputs)
            # print("shape of predictions and targets: ",predictions.shape, targets.shape)
            # loss = criterion(predictions, targets)
            print(predictions.shape, targets.shape)
            for ch_idx in range(args.input_seq_length):
                # print(predictions[ch_idx].shape, targets[ch_idx].shape)
                print("shapes of predictions and targets: ", predictions[ch_idx].shape, targets[ch_idx].shape)
                l = criterion(predictions[ch_idx], targets[ch_idx])
                loss += l
            loss /= args.input_seq_length
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            epoch_loss += loss
        epoch_loss /= len(data_loader)
        all_losses.append(epoch_loss.item())
        print("epoch: ", epoch + 1, " loss = ", epoch_loss.item())
        if epoch + 1 in epochs_sampling:
            print("epoch = ", epoch + 1, "sampling now")
            lens = [15, 30, 45]
            for sample_len in lens:
                sample_sentences = model.sample(batch_size=5, sample_length=sample_len, temperature=0)
                # training_filename = args.txt_file.split("/")[1].split(".")[0]
                # sample_filename = "samples/{}_epoch_{}_temperature_{}_sample_length_{}.txt".format(training_filename,
                #                                                                                    epoch + 1, 0,
                #                                                                                    sample_len)
                sample_title = "epoch_{}_temperature_{}_sample_length_{}".format(epoch + 1, 0, sample_len)
                # write_to_file(sample_sentences, sample_filename)
                myfile.write(sample_title)
                for sentence in sample_sentences:
                    myfile.write("%s\n" % sentence)
                myfile.write("\n")
                print("samples ", sample_title, " written to ", sample_filename)
    temps = [0.5, 1, 2]
    for t in temps:
        sample_sentences = model.sample(batch_size=5, sample_length=30, temperature=t)
        # training_filename = args.txt_file.split("/")[1].split(".")[0]
        # sample_filename = "samples/{}_epoch_{}_temperature_{}_sample_length_{}.txt".format(training_filename,
        #                                                                                    args.num_epochs, t, 30)
        # write_to_file(sample_sentences, sample_filename)
        sample_title = "epoch_{}_temperature_{}_sample_length_{}".format(args.num_epochs, t, 30)
        myfile.write(sample_title)
        for sentence in sample_sentences:
            myfile.write("%s\n" % sentence)
        myfile.write("\n")
        print("samples ", sample_title, " written to ", sample_filename)
    myfile.close()
    pretrained_model_name = "text_gen_model.ckpt"
    torch.save(model.state_dict(), pretrained_model_name)
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
    x = [e + 1 for e in range(len(all_losses))]
    plt.plot(x, all_losses, label='loss')
    plt.set_title("Training loss")
    plt.legend()
    plt.savefig('./rnn_loss_debugging.png')
    print("plotting completed!")
