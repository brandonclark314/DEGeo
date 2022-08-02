import torch
import numpy as np

def toCartesian(L):
    L = L * np.pi / 180

    x = torch.cos(L[:, 0]) * torch.cos(L[:, 1])
    y = torch.cos(L[:, 0]) * torch.sin(L[:, 1])
    z = torch.sin(L[:, 0])
    
    R = torch.stack([x, y, z], dim=1)
    return R

def toLatLon(R):
    x = R[:, 0]
    y = R[:, 1]
    z = R[:, 2]
    
    lat = torch.arctan2(z, torch.sqrt(x**2 + y**2))
    lon = torch.arctan2(y, x)
    
    lat = lat * 180 / np.pi
    lon = lon * 180 / np.pi
    
    L = torch.stack([lat, lon], dim=1)
    return L