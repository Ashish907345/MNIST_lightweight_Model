from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from collections import defaultdict
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),  # Increased channels to 12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, stride=2, padding=1),  # Stride 2 reduces spatial dimensions
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.25)
        )
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 24, 3, padding=1),  # Increased channels to 24
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3, stride=2, padding=1),  # Stride 2 reduces spatial dimensions
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.25)
        )
        
        # Global Average Pooling (GAP) + 1x1 Convolution
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP reduces spatial dimensions to (1, 1)
        self.fc = nn.Conv2d(24, 10, 1)  # 1x1 convolution for classification (24 to 10)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gap(x)  # Apply GAP
        x = self.fc(x)   # 1x1 Convolution for output
        x = x.view(x.size(0), -1)  # Flatten the output for log_softmax
        x = F.log_softmax(x, dim=1)
        return x

# Example model instantiation
model = Net()
print(model)

# Count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Trainable Parameters: {count_parameters(model)}")

model = Net().to(device)
summary(model, input_size=(1, 28, 28))
from tqdm import tqdm

training_stats = defaultdict(list)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # Store accuracy
    training_stats['accuracy'].append(accuracy)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return accuracy  # Return accuracy for tracking

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 20):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
    training_stats['epoch'].append(epoch)

# Save training stats after training
torch.save(training_stats, 'training_stats.pth')