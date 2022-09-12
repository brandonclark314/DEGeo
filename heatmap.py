import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm
import wandb
import torch
import torchvision
import geopy
from geopy.distance import geodesic as GD
import torchvision.transforms as T
from PIL import Image

def getRandomCoords(n):
    """Generate n random coordinates in the range [-90, 90] for latitude and [-180, 180] for longitude."""
    lat = torch.rand(n) * 180 - 90
    lon = torch.rand(n) * 360 - 180
    return torch.stack([lat, lon], dim=-1)

def plot_heatmap(ground_truth, predictions, opt=None, wandb_log=True):
    """Create two plots: one for the average error (geodesic distance between the ground truth and
    the prediction) and one for the density of the given coordinates (ground truth). 
    The plots are overlaid on two maps.

    Args:
        ground_truth (torch.Tensor of shape (N, 2)): The ground truth coordinates.
        predictions (torch.Tensor of shape (N, 2)): The predicted coordinates.
        opt (dict, optional): The options dictionary. Defaults to None.
        wandb_log (bool, optional): Whether to log the plots to wandb. Defaults to True.

    Returns:
        (PIL.Image, PIL.Image): The two heatmaps.
    """

    # Load Colormap
    cm = matplotlib.cm.get_cmap('rainbow') # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    # Load image of the Earth 
    img = torch.tensor(plt.imread('planet.png'))
    img = img[:, :, :3]

    # Create individual images for the Error Heatmap & Density Map
    img_heatmap = img.clone()
    img_density = img.clone()

    # Create tensor of coordinates [lat, long]
    coords = torch.meshgrid(torch.arange(-90, 90), torch.arange(-180, 180))
    coords = torch.stack(coords, dim=-1).reshape(-1, 2).float()#.to(opt.device)

    # Get Error & Density
    coords_error = torch.zeros((180, 360))
    coords_bucket = torch.ones((180, 360))

    for prediction, target in zip(predictions, ground_truth):
        x = target[0].int() + 90
        y = target[1].int() + 180

        coords_error[x][y] += GD(prediction, target).km # Error
        coords_bucket[x][y] += 1

    coords_average_error = coords_error / coords_bucket

    # Squash values to [0, 1]
    coords_average_error = 1 - coords_average_error / coords_average_error.max()
    coords_bucket = coords_bucket / coords_bucket.max()
    
    # Assign colors to each covered coordinate (i.e. pixel)
    for coord in ground_truth:
        x = coord[0].int() + 90
        y = coord[1].int() + 180

        img_heatmap[180 - (x + 1)][y] = torch.tensor(cm(coords_average_error[x][y].item())[:3])
        img_density[180 - (x + 1)][y] = torch.tensor(cm(coords_bucket[x][y].item())[:3])
    
    # Format the images
    img_heatmap = img_heatmap.permute(2, 0, 1) 
    img_density = img_density.permute(2, 0, 1)
    img_heatmap = T.ToPILImage()(img_heatmap)
    img_density = T.ToPILImage()(img_density)

    # Log to wandb
    if wandb_log:
        wandb.log({"Error Map": wandb.Image(img_heatmap)})
        wandb.log({"Density Map": wandb.Image(img_density)})

    return img_heatmap, img_density

def plot_accuracy_heatmap(ground_truth, predictions, thresholds, opt=None, wandb_log=True):
    """Create a plot for the accuracy of the predictions with respect to the given thresholds.
    The plot is overlaid on a map.

    Args:
        ground_truth (torch.Tensor of shape (N, 2)): The ground truth coordinates.
        predictions (torch.Tensor of shape (N, 2)): The predicted coordinates.
        thresholds (list): The thresholds to use.
        opt (dict, optional): The options dictionary. Defaults to None.
        wandb_log (bool, optional): Whether to log the plots to wandb. Defaults to True.

    Returns:
        PIL.Image: The accuracy heatmap.
    """

    # Load Colormap
    cm = matplotlib.cm.get_cmap('rainbow') # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    # Load image of the Earth 
    img = torch.tensor(plt.imread('planet.png'))
    img = img[:, :, :3]

    # Copy image for every threshold
    imgs = []
    for threshold in thresholds:
        imgs.append(img.clone())

    # Create tensor of coordinates [lat, long]
    coords = torch.meshgrid(torch.arange(-90, 90), torch.arange(-180, 180))
    coords = torch.stack(coords, dim=-1).reshape(-1, 2).float()#.to(opt.device)

    # Get Accuracy
    coords_correct = torch.zeros((len(thresholds), 180, 360))
    coords_bucket = torch.ones((len(thresholds), 180, 360))

    for prediction, target in zip(predictions, ground_truth):
        x = target[0].int() + 90
        y = target[1].int() + 180

        for i, threshold in enumerate(thresholds):
            if GD(prediction, target).km <= threshold:
                coords_correct[i][x][y] += 1

        coords_bucket[:, x, y] += 1

    coords_accuracy = coords_correct / coords_bucket

    # Squash values to [0, 1]
    for i in range(len(thresholds)):
        coords_accuracy[i] = coords_accuracy[i] / (coords_accuracy[i].max() or 1)
    
    # Assign colors to each covered coordinate (i.e. pixel)
    for i, threshold in enumerate(thresholds):
        for coord in ground_truth:
            x = coord[0].int() + 90
            y = coord[1].int() + 180

            imgs[i][180 - (x + 1)][y] = torch.tensor(cm(coords_accuracy[i][x][y].item())[:3])

    # Format the images
    for i, threshold in enumerate(thresholds):
        imgs[i] = imgs[i].permute(2, 0, 1)
        imgs[i] = T.ToPILImage()(imgs[i])

    # Log to wandb
    if wandb_log:
        for i, threshold in enumerate(thresholds):
            wandb.log({f"Accuracy Map ({threshold} km)": wandb.Image(imgs[i])})

    return imgs

if __name__ == '__main__':
    # Generate 3000 random points
    ground_truth = getRandomCoords(3000)
    predictions = getRandomCoords(3000)

    # Plot the feature map
    img_heatmap, img_density = plot_heatmap(ground_truth, predictions, wandb_log=False)

    # Plot the accuracy map
    thresholds = [1, 25, 200, 750, 2500]
    imgs = plot_accuracy_heatmap(ground_truth, predictions, thresholds, wandb_log=False)

    img_heatmap.save('heatmap.png')
    img_density.save('density.png')
    for i, img in enumerate(imgs):
        img.save(f'accuracy_{thresholds[i]}.png')