from customDataset import DeepFakeSmallDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
import numpy as np


# Fake : 0
# Real : 1

# Initializing datasets
dataset = DeepFakeSmallDataset(csv_file='training_labels.csv',
                                root_dir='frames_fft/',
                                transform=transforms.ToTensor())                              

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.6 * dataset_size))
random_seed=52
if True:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]


train_loader = DataLoader(dataset,
                          batch_size=32,
                          num_workers=4, # 1 for CUDA
                          sampler=SubsetRandomSampler(train_indices)
                         # pin_memory=True # CUDA only
                         )

test_loader = DataLoader(dataset,
                          batch_size=32,
                          num_workers=4, # 1 for CUDA,
                          sampler=SubsetRandomSampler(test_indices)
                         # pin_memory=True # CUDA only
                         )

# Create basic logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 281*281
output_dim = 2
model = LogisticRegressionModel(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train model
iter = 0
n_iters = 30000
num_epochs = int(n_iters / (len(dataset) / 32))

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Load images as Variable
        images = torch.autograd.Variable(images.view(-1, input_dim))
        labels = torch.autograd.Variable(labels.long())

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        iter += 1

        # Evaluate logistic regression model 
        if iter%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = torch.autograd.Variable(images.view(-1, input_dim))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct += (predicted == labels).sum()

            # Compute confusion matrix
            confusion_vector = predicted / labels
            true_positives = torch.sum(confusion_vector == 1).item()
            false_positives = torch.sum(confusion_vector == float('inf')).item()
            true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
            false_negatives = torch.sum(confusion_vector == 0).item()
            print("TP: {} FP: {} TN: {} FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))
