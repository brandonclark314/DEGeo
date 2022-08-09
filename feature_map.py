import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import torch
import torchvision
import torchvision.transforms as T
from matplotlib import cm
from PIL import Image

def plot_feature_map(model, opt=None):
    # img = torch.tensor(plt.imread('planet.png'))
    # img = img[:, :, :3]

    # Create image
    coords = torch.meshgrid(torch.arange(-90, 90), torch.arange(-180, 180))
    coords = torch.stack(coords, dim=-1).reshape(-1, 2).float().to(opt.device)
    colors = model.project3D(coords)
    colors = colors.detach().cpu()

    # Get Features
    img_color = torch.randn((180, 360))
    for color, coord in zip(colors, coords):
        img_color[coord[0].int() + 90, coord[1].int() + 180] = color
        
    img_color = torch.clamp(img_color, 0, 255)
    
    img_final = torch.Tensor(cm.viridis(img_color))
    img_final= img_final.permute(2, 0, 1)
    
    img_final = T.ToPILImage()(img_final)
    
    new_img = img_final.resize((540, 1080),Image.BICUBIC)
    
    # Plot image
    wandb.log({"Feature Map": wandb.Image(new_img)})