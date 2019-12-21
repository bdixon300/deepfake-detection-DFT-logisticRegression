from customDataset import DeepFakeSmallDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torch.nn as nn
import matplotlib.pyplot as plt
import collections


# Initializing dataset
data_set = DeepFakeSmallDataset(csv_file='training_labels.csv',
                                root_dir='frames_fft/',
                                transform=transforms.ToTensor())
                                
# See dataset samples
"""plt.imshow(data_set[0]['image'], cmap='gray')
print(data_set[0]['label'])
plt.pause(0.01)
input("<Hit Enter To Close>")"""

#train_sampler, val_sampler, test_sampler = 
#train_sampler = SubsetRandomSampler(training_indices)
train_loader = DataLoader(data_set,
                          batch_size=4,
                          shuffle=False, #FIXME: SHUFFLING to true breaks it
                          num_workers=4 # 1 for CUDA
                         # pin_memory=True # CUDA only
                         )

# Create basic logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = F.softmax(self.linear(x))
        return out

input_dim = 281*281
output_dim = 2
model = LogisticRegressionModel(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train model
iter = 0
num_epochs = 100
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Load images as Variable
        images = torch.autograd.Variable(images.view(-1, 281*281))
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
        #TODO
        """if iter%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))"""
