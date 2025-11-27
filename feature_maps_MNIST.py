import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from MNIST import Net 
from PIL import Image
import numpy as np
import os

# Load model
model = Net()
model.load_state_dict(torch.load("model_gcnn_mnist.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transformation MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.MNIST(root='./mnist', train=False,
                                     download=True, transform=transform)

target_digit = 2
for img, label in testset:
    if label == target_digit:
        img_orig = img  # shape: [1,28,28]
        break

# Rotation angles à tester
angles = [0, 90, 180]

# Fonction pour extraire les feature maps à 3 niveaux
def extract_feature_maps(image, model, device):
    """Extrait les activations des 3 couches de convolution"""
    with torch.no_grad():
        # Conv1
        act1 = F.relu(model.conv1(image))  # Après conv1
        # Conv2
        act2 = F.relu(model.conv2(act1))   # Après conv2
        # Conv3
        act3 = F.relu(model.conv3(act2))   # Après conv3
    
    return act1.cpu(), act2.cpu(), act3.cpu()

# Créer les 3 versions tournées et extraire leurs features
results = {}
for angle in angles:
    pil_img = transforms.ToPILImage()(img_orig)
    pil_rot = pil_img.rotate(angle)
    tensor_rot = transform(pil_rot)  # [1,28,28]
    img_batch = tensor_rot.unsqueeze(0).to(device)  # [1,1,28,28]
    
    act1, act2, act3 = extract_feature_maps(img_batch, model, device)
    
    results[angle] = {
        'image': tensor_rot.squeeze(0).cpu().numpy(),
        'conv1': act1.squeeze(0).cpu().numpy(),
        'conv2': act2.squeeze(0).cpu().numpy(),
        'conv3': act3.squeeze(0).cpu().numpy()
    }
    print(f"Angle {angle}° - Conv1: {results[angle]['conv1'].shape}, Conv2: {results[angle]['conv2'].shape}, Conv3: {results[angle]['conv3'].shape}")

# Sélectionner les filtres à visualiser
filters_to_show = [0, 5, 10]

# Pour chaque couche de convolution
conv_levels = ['conv1', 'conv2', 'conv3']

for conv_level in conv_levels:
    for filter_id in filters_to_show:
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        
        for i, angle in enumerate(angles):
            # Colonne 1: Image d'entrée
            img = results[angle]['image']
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f"Input {angle}°")
            
            # Colonne 2: Feature map
            fmap = results[angle][conv_level]  # [num_filters, 4, H, W]
            
            # Moyenner sur les 4 orientations
            fmap_display = fmap[filter_id, :, :, :].mean(axis=0)  # [H, W]
            
            fmap_display = (fmap_display - fmap_display.min()) / (fmap_display.max() - fmap_display.min() + 1e-5)
            axes[i, 1].imshow(fmap_display, cmap='viridis')
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f"Feature map {angle}° (rotated?)")
        
        plt.suptitle(f"{conv_level.upper()} - Filter {filter_id} - Rotation Equivariance Test", fontsize=14)
        plt.tight_layout()
        
        os.makedirs('./results', exist_ok=True)
        plt.savefig(f'./results/{conv_level}_filter_{filter_id}_rotation_equivariance.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

# Visualisation comparative: même filtre, 3 angles, 3 couches
for filter_id in filters_to_show:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, angle in enumerate(angles):
        for j, conv_level in enumerate(conv_levels):
            fmap = results[angle][conv_level]  # [num_filters, 4, H, W]
            
            # Moyenner sur les 4 orientations
            fmap_display = fmap[filter_id, :, :, :].mean(axis=0)  # [H, W]
            
            fmap_display = (fmap_display - fmap_display.min()) / (fmap_display.max() - fmap_display.min() + 1e-5)
            axes[i, j].imshow(fmap_display, cmap='viridis')
            axes[i, j].axis('off')
            axes[i, j].set_title(f"{conv_level} @ {angle}°")
    
    plt.suptitle(f"Rotation Equivariance - Filter {filter_id}\n(Same filter across rotations and conv layers)", fontsize=14)
    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig(f'./results/comparison_filter_{filter_id}_all_layers.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()