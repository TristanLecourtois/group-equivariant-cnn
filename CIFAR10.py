import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from experiments_CIFAR import Net
import os

model = Net()
model.load_state_dict(torch.load("model_gcnn_2.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

images, labels = next(iter(testloader))
images = images.to(device)

with torch.no_grad():
    x = F.relu(model.conv1(images))
    x = F.relu(model.conv2(x))
    x = plane_group_spatial_max_pooling(x, 2, 2)  
    act = F.relu(model.conv3(x))
    act = act.cpu()

print(f"Shape des activations conv3: {act.shape}")

filters_to_show = [0, 5, 10, 15]

for filter_id in filters_to_show:
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    
    for i in range(4):  # images
        # Afficher l'image d'entrée
        img = images[i].cpu().permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title("Input")
        
        # Afficher les 4 orientations du filtre sélectionné
        for j in range(4):  # orientations
            fmap = act[i, filter_id, j, :, :].numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
            axes[i, j+1].imshow(fmap, cmap='viridis')
            axes[i, j+1].axis('off')
            axes[i, j+1].set_title(f"{j*90}°")
    
    plt.suptitle(f"Feature maps conv3 — Filtre {filter_id}")
    plt.tight_layout()

    plt.savefig(f'./results/conv3_filter_{filter_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
