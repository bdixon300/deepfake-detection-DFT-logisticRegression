from customDataset import DeepFakeSmallDataset
from LogisticRegression import LogisticRegressionModel
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
import numpy as np


# Fake : 0
# Real : 1
print(torch.cuda.is_available())
# Initializing datasets
training_dataset = DeepFakeSmallDataset(csv_file='training_labels_40_vids_youtube_15_magnitude.csv',
                                root_dir='frames_fft_40_vids_training_youtube_15_magnitude',
                                transform=transforms.ToTensor())

testing_dataset = DeepFakeSmallDataset(csv_file='testing_labels_10_vids_youtube_15_magnitude.csv',
                                root_dir='frames_fft_10_vids_testing_youtube_15_magnitude',
                                transform=transforms.ToTensor())

"""training_dataset = DeepFakeSmallDataset(csv_file='training_labels_8_vids.csv',
                                root_dir='frames_fft_8_vids_training',
                                transform=transforms.ToTensor())

testing_dataset = DeepFakeSmallDataset(csv_file='testing_labels_2_vids.csv',
                                root_dir='frames_fft_2_vids_testing',
                                transform=transforms.ToTensor()) """ 

"""training_dataset = DeepFakeSmallDataset(csv_file='training_labels_10_vids.csv',
                                root_dir='frames_fft_10_vids_training',
                                transform=transforms.ToTensor())

testing_dataset = DeepFakeSmallDataset(csv_file='testing_labels_2_vids.csv',
                                root_dir='frames_fft_2_vids_testing',
                                transform=transforms.ToTensor()) """           

# Creating data indices for training and validation splits:
"""dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.8 * dataset_size))
random_seed=52
if True:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]"""


train_loader = DataLoader(training_dataset,
                          batch_size=16,
                          num_workers=4, # 1 for CUDA,
                          shuffle=True
                         # pin_memory=True # CUDA only
                         )

test_loader = DataLoader(testing_dataset,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4 # 1 for CUDA,
                         # pin_memory=True # CUDA only
                         )

input_dim = 281*281
output_dim = 2
model = LogisticRegressionModel(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train model
iter = 0
n_iters = 70000
num_epochs = int(n_iters / (len(training_dataset) / 16))

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
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            for j, (images, labels) in enumerate(test_loader):
                images = torch.autograd.Variable(images.view(-1, input_dim))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                """print("predicted: {}".format(predicted))
                print("========")
                print("labels: {}".format(labels))"""
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct += (predicted == labels).sum()

                for tensor_value in range(0, predicted.size(0)):
                    # TP
                    if (predicted[tensor_value].item() == 1 and labels[tensor_value].item() == 1):
                        true_positives += 1
                    # FP
                    if (predicted[tensor_value].item() == 1 and labels[tensor_value].item() == 0):
                        false_positives += 1
                    #TN
                    if (predicted[tensor_value].item() == 0 and labels[tensor_value].item() == 0):
                        true_negatives += 1
                    #FN
                    if (predicted[tensor_value].item() == 0 and labels[tensor_value].item() == 1):
                        false_negatives += 1

                """confusion_vector = predicted / labels
                true_positives += torch.sum(confusion_vector == 1).item()
                false_positives += torch.sum(confusion_vector == float('inf')).item()
                true_negatives += torch.sum(torch.isnan(confusion_vector)).item()
                false_negatives += torch.sum(confusion_vector == 0).item()"""

            # Compute confusion matrix
            print("Correct: {}".format(correct))
            print("Total: {}".format(total))
            print("TP: {} FP: {} TN: {} FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

torch.save(model.state_dict(), 'model_youtube_15_magnitude.pt')
