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
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform
from cifar10_utils import get_train_validation_set, get_test_set


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


def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'debug':  # Use this model for debugging
        cnn_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, num_classes)
        )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
        cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    return cnn_model


def accuracy(predictions, targets):
    correct = 0
    for idx in range(len(targets)):
        pred_class = np.argmax(predictions[idx], axis=0)
        if pred_class == targets[idx]:
            correct += 1
    acc = correct / len(targets)

    return acc


def train_model(model_name, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    num_workers = min(2, os.cpu_count())
    pin_memory = True if torch.cuda.is_available() else False
    train_dataset, val_dataset = get_train_validation_set(data_dir)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
    validation_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                       num_workers=num_workers, pin_memory=pin_memory)

    model = get_model(model_name)
    model = model.to(device)
    # Initialize the optimizers and learning rate scheduler. 
    # We provide a recommend setup, which you are allowed to change if interested.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)

    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    best_model = deepcopy(model)
    best_val_acc = 0
    loss_module = nn.CrossEntropyLoss()
    val_accuracies = []
    train_accuracies = []
    val_losses, train_losses = [], []
    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        train_running_loss = 0.0
        train_predictions = np.empty((0, 10), int)
        train_targets = np.empty((0), int)
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_module(outputs, labels)
            loss.backward()
            optimizer.step()

            train_predictions = np.append(train_predictions, outputs.cpu().detach().numpy(), axis=0)
            train_targets = np.append(train_targets, labels.cpu().detach().numpy(), axis=0)
            train_running_loss += loss.item()
        train_losses.append(train_running_loss / len(train_dataloader))
        train_epoch_acc = accuracy(predictions=train_predictions, targets=train_targets)
        train_accuracies.append(train_epoch_acc)

        predictions = np.empty((0, 10), int)
        targets = np.empty((0), int)
        val_running_loss = 0
        for i, val_data in enumerate(validation_dataloader, 0):
            val_inputs, val_labels = val_data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = loss_module.forward(val_outputs, val_labels)
            val_running_loss += val_loss.item()

            predictions = np.append(predictions, val_outputs.cpu().detach().numpy(), axis=0)
            targets = np.append(targets, val_labels.cpu().detach().numpy(), axis=0)
        val_losses.append(val_running_loss / len(validation_dataloader))
        val_epoch_acc = accuracy(predictions=predictions, targets=targets)
        epoch_end = time.time()
        print("epochs: ", epoch, "val_epoch_acc = ", val_epoch_acc, "train_epoch_acc = ", train_epoch_acc,
              "val_loss=", val_loss.item(), "train_loss=", loss.item(), "time/epoch = ", epoch_end-epoch_start)
        val_accuracies.append(val_epoch_acc)
        if val_epoch_acc > best_val_acc:
            best_model = deepcopy(model)
            best_val_acc = val_epoch_acc
        scheduler.step()
    torch.save(best_model.state_dict(), checkpoint_name)
    # pass

    # Load best model and return it.
    model = get_model(model_name)
    model.load_state_dict(torch.load(checkpoint_name))
    print("Model saved to ", checkpoint_name)
    # pass

    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = np.empty((0, 10), int)
    targets = np.empty((0), int)
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions = np.append(predictions, outputs.cpu().detach().numpy(), axis=0)
        targets = np.append(targets, labels.cpu().detach().numpy(), axis=0)

    avg_accuracy = accuracy(predictions=predictions, targets=targets)
    #######################
    # END OF YOUR CODE    #
    #######################
    return avg_accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    print("Testing model ...")
    set_seed(seed)
    test_results = {"gaussian_noise_transform": [],  # {},
                    "gaussian_blur_transform": [],  # {},
                    "contrast_transform": [],  # {},
                    "jpeg_transform": []}  # {},}
    model.to(device)
    test_dataset = get_test_set(data_dir, augmentation=None)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_acc = evaluate_model(model, test_dataloader, device)
    test_results["clean"] = [test_acc] * 5

    ''' gaussian_noise_transform'''
    print("testing gaussian_noise_transform")
    for s in range(1, 6):
        test_dataset = get_test_set(data_dir, augmentation=gaussian_noise_transform(severity=s))
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_acc = evaluate_model(model, test_dataloader, device)
        # test_results["gaussian_noise_transform"][s] = test_acc
        test_results["gaussian_noise_transform"].append(test_acc)
    ''' gaussian_blur_transform '''
    print("testing gaussian_blur_transform")
    for s in range(1, 6):
        test_dataset = get_test_set(data_dir, augmentation=gaussian_blur_transform(severity=s))
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_acc = evaluate_model(model, test_dataloader, device)
        # test_results["gaussian_blur_transform"][s] = test_acc
        test_results["gaussian_blur_transform"].append(test_acc)
    ''' contrast_transform '''
    print("testing contrast_transform")
    for s in range(1, 6):
        test_dataset = get_test_set(data_dir, augmentation=contrast_transform(severity=s))
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_acc = evaluate_model(model, test_dataloader, device)
        # test_results["contrast_transform"][s] = test_acc
        test_results["contrast_transform"].append(test_acc)
    ''' jpeg_transform '''
    print("testing jpeg_transform")
    for s in range(1, 6):
        test_dataset = get_test_set(data_dir, augmentation=jpeg_transform(severity=s))
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_acc = evaluate_model(model, test_dataloader, device)
        # test_results["jpeg_transform"][s] = test_acc
        test_results["jpeg_transform"].append(test_acc)
    #######################
    # END OF YOUR CODE    #
    #######################
    return test_results


def plot_model_acc(test_results, plot_filename):
    print("starting plotting ..")
    x = [s for s in range(1, 6)]
    sizes = (10, 5)
    plt.rcParams["figure.figsize"] = sizes
    plt.plot(x, test_results["gaussian_noise_transform"], '-b', label='gaussian_noise_transform')
    plt.plot(x, test_results["gaussian_blur_transform"], '-g', label='gaussian_blur_transform')
    plt.plot(x, test_results["contrast_transform"], '-y', label='contrast_transform')
    plt.plot(x, test_results["jpeg_transform"], '-k', label='jpeg_transform')
    plt.plot(x, test_results["clean"], '-r', label='without corruption')
    plt.grid(axis='x', color='0.95')
    plt.legend()

    plt.title('Test accuracies')
    plt.savefig('./' + plot_filename)
    print("plot saved to ./", plot_filename)
    plt.clf()


def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("DEVICE: ", device)
    set_seed(seed)
    checkpoint_name = model_name + ".ckpt"
    pretrained_filename = checkpoint_name  # os.path.join(CHECKPOINT_PATH, checkpoint_name)
    print("pretrained_filename == ", pretrained_filename)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = get_model(model_name)
        model.load_state_dict(torch.load(pretrained_filename))
    else:
        print("Starting training")
        model = train_model(model_name, lr, batch_size, epochs, data_dir, checkpoint_name, device)

    test_results_filename = model_name + '_test_results.json'
    if os.path.isfile(test_results_filename):
        with open(test_results_filename, 'r') as f:
            test_results = json.loads(f.read())
    else:
        test_results = test_model(model, batch_size, data_dir, device, seed)
        with open(test_results_filename, 'w') as fp:
            json.dump(test_results, fp, ensure_ascii=False, indent=4)
    plot_model_acc(test_results, model_name + "_test_acc_plot.png")

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--model_name', default='debug', type=str,
                        help='Name of the model to train.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
