import torch
import torch.nn as nn
import torch.nn.functional as F

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3 = P4ConvP4(10, 20, kernel_size=3)
        self.conv4 = P4ConvP4(20, 20, kernel_size=3)

        # Après les pooling 2×2, l'image 28×28 devient 7×7
        # et il y a un facteur 4 dû aux orientations (groupe P4)
        self.fc1 = nn.Linear(1280, 50)

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


model = Net()
print(model)


image = torch.randn(1, 1, 28, 28)  # 1 image, 1 canal, taille MNIST
print("\nInput shape :", image.shape)


output = model(image)
print("\nOutput logits shape :", output.shape)
print("Output logits :", output)
