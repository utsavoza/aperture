import torch
import torchvision
import torchsummary
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import argparse
import random
import models
import time
import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=20, type=int,
                    help="Number of training epochs")
parser.add_argument("--num-workers", default=0, type=int,
                    help="Number of dataloader workers")
parser.add_argument("--lr", default=0.1, type=float, help="Learning Rate")
parser.add_argument("--lr-cosine", default=False,
                    type=bool, help="Enable Cosine Annealing")
parser.add_argument("--optim", default="sgd", type=str, help="Optimizer")
parser.add_argument("--batch-size", default=32,
                    type=int, help="Traing batch size")
parser.add_argument("--model", default="resnet18", type=str, help="Model")
parser.add_argument("--summary", default=False,
                    type=bool, help="Print Model Summary")
parser.add_argument("--augment", default=True, type=bool,
                    help="Use standard augmentation")
parser.add_argument("--with-mixup", default=False,
                    type=bool, help="Mixup data augmentations")
args = parser.parse_args()


# Configuration
print("Configuration:")
print("-------------------------")
for key in args.__dict__:
    print("{:<18} : {}".format(key, args.__dict__[key]))


# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("{:<18} : {}".format("device", device))


# Set random seed for reproducibility
seed = 17
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# Prepare dataset
print("\nPreparing dataset:")
print("-------------------------")
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

training_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test)

train_dataloader = DataLoader(
    training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = DataLoader(
    test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


# Define the model
best_accuracy = 0
start_epoch = 0
train_loss_history = []
test_loss_history = []

model_name = args.model
exec('args.model = models.{}().to("{}")'.format(args.model, device))
net = args.model
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)


# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()


# Optimizer
lr = args.lr
if args.optim.lower() == "sgd":
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
elif args.optim.lower() == "sgdn":
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
elif args.optim.lower() == "adagrad":
    optimizer = torch.optim.Adagrad(
        net.parameters(), lr=lr, weight_decay=5e-4)
elif args.optim.lower() == "adadelta":
    optimizer = torch.optim.Adadelta(
        net.parameters(), lr=lr, weight_decay=5e-4)
elif args.optim.lower() == "adam":
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=5e-4)

if args.lr_cosine:
    scheduler = CosineAnnealingLR(optimizer, T_max=200)


# Print model summary
if args.summary:
    print("\n\nModel Summary: ")
    torchsummary.summary(net, input_size=(3, 32, 32))


# Training procedure
def train():
    train_loss = 0.0
    train_accuracy = 0.0
    net.train()

    for _, tuple in enumerate(train_dataloader):
        images, labels = tuple
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        train_accuracy += predicted.eq(labels).cpu().sum().item()

    train_accuracy = 100.0 * train_accuracy / len(training_data)
    return (train_loss, train_accuracy)


# Testing procedure
def test():
    test_loss = 0.0
    test_accuracy = 0.0
    predictions = []
    targets = []
    net.eval()

    with torch.no_grad():
        for _, tuple in enumerate(test_dataloader):
            images, labels = tuple
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            test_accuracy += predicted.eq(labels).cpu().sum().item()

            predictions.extend(predicted.cpu())
            targets.extend(labels.cpu())

    test_accuracy = 100.0 * test_accuracy / len(test_data)
    return (test_loss, test_accuracy)


# See: https://github.com/facebookresearch/mixup-cifar10/
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# See: https://github.com/facebookresearch/mixup-cifar10/
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training procedure using mixup
# See: https://github.com/facebookresearch/mixup-cifar10/
def train_with_mixup():
    train_loss = 0.0
    train_accuracy = 0.0
    net.train()
    alpha = 1

    for _, tuple in enumerate(train_dataloader):
        images, labels = tuple
        images = images.to(device)
        labels = labels.to(device)

        images, targets_a, targets_b, lam = mixup_data(images, labels, alpha)
        images, targets_a, targets_b = map(
            torch.autograd.Variable, (images, targets_a, targets_b))

        optimizer.zero_grad()
        outputs = net(images)
        loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        train_accuracy += predicted.eq(labels).cpu().sum().item()

    train_accuracy = 100.0 * train_accuracy / len(training_data)

    return (train_loss, train_accuracy)


# Train the model
print("\nTraining the model")
print("-------------------------")
for epoch in range(args.epochs):
    start_time = time.time()
    train_accuracy = 0.0
    test_accuracy = 0.0
    predictions = []
    targets = []

    # Train
    if args.with_mixup:
        train_loss, train_accuracy = train_with_mixup()
    else:
        train_loss, train_accuracy = train()

    # Test
    test_loss, test_accuracy = test()

    if args.lr_cosine:
        scheduler.step()

    train_loss = train_loss / len(train_dataloader)
    test_loss = test_loss / len(test_dataloader)
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    end_time = time.time()

    print('Epoch: %s,  Train Loss: %.8f,  Train Accuracy: %.3f,  Test Loss: %.8f,  Test Accuracy: %.3f, Time: %.3fs'
          % (start_epoch+epoch+1, train_loss, train_accuracy, test_loss, test_accuracy, end_time - start_time))

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        state = {
            'model': net,
            'accuracy': test_accuracy,
            'epoch': start_epoch + epoch,
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history,
            'optimizer': optimizer,
            'rng_state': torch.get_rng_state()
        }
        if not os.path.isdir('checkpoints'):
            print("Saving model ...")
            os.mkdir('checkpoints')
        print("Saving model ...")
        torch.save(state, './checkpoints/' + model_name +
                   '_epoch' + str(args.epochs) + '.pt')


print("\nBest Accuracy: %.3f\n" % (best_accuracy))
