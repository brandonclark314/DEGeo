import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

def plot_feature_map(model):
    img = torch.tensor(plt.imread('planet.png'))
    img = img[:, :, :3]

    # Create image
    coords = torch.meshgrid(torch.arange(-90, 90), torch.arange(-180, 180))
    coords = torch.stack(coords, dim=-1).reshape(-1, 2).float()
    colors = model.project3D(coords)

    # Get Features
    img_color = img.clone()
    for color, coord in zip(colors, coords):
        img_color[coord[0].int() + 90, coord[1].int() + 180, :] = color.detach()
        
    img = (img - img.mean()) / img.std()
    
    # Plot image
    wandb.log({"Feature Map": wandb.Image(img_color - 0.5 * img)})