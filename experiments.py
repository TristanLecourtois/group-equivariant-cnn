import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os 


from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Dummy forward to compute feature size

        self.conv1 = P4ConvZ2(3, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3 = P4ConvP4(10, 20, kernel_size=3)
        self.conv4 = P4ConvP4(20, 20, kernel_size=3)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            dummy = plane_group_spatial_max_pooling(dummy, 2, 2)
            dummy = F.relu(self.conv3(dummy))
            dummy = F.relu(self.conv4(dummy))
            dummy = plane_group_spatial_max_pooling(dummy, 2, 2)
            self.flatten_dim = dummy.view(1, -1).size(1)

        # Après les pooling 2×2, l'image 28×28 devient 7×7
        # et il y a un facteur 4 dû aux orientations (groupe P4)
        self.fc1 = nn.Linear(self.flatten_dim, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))          # (batch, 10*4, H, W)
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return F.log_softmax(out, dim=1)


if __name__ == "__main__":
    model = Net()
    print(model)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 10
    train_accuracy = []
    val_accuracy = []

    print("Start training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = correct / total
        print(f'Epoch {epoch+1} - Train Accuracy: {100*train_acc:.2f}% - Loss: {running_loss/len(trainloader):.4f}')
        train_accuracy.append(train_acc)
        np.save(os.path.join('./results', 'acc_train_gcnn.npy'), np.array(train_accuracy))
        
        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f'Epoch {epoch+1} - Test Accuracy: {100*val_acc:.2f}%')
        val_accuracy.append(val_acc)
        np.save(os.path.join('./results', 'acc_val_gcnn.npy'), np.array(val_accuracy))
        
        torch.save(model.state_dict(), f"model_gcnn_2.pth")

