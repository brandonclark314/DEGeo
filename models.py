from importlib.metadata import requires
from transformers import ViTModel
from transformers import ResNetForImageClassification as ResNet

import numpy as np
import torch
import torch.nn.functional as F
from torch import logit, nn
import matplotlib.pyplot as plt
from rff.layers import GaussianEncoding
from torch.autograd import Variable
import torchvision.models as models
from config import getopt
from infonce import InfoNCE

def getLocationEncoder(km):
    Earth_Diameter = 12742
    sigma = Earth_Diameter / (3 * km)
    rff_encoding = GaussianEncoding(sigma=sigma, input_size=3, encoded_size=256)
    return nn.Sequential(rff_encoding,
                         nn.Linear(512, 1024),
                         nn.ReLU(),
                         nn.Linear(1024, 1024),
                         nn.ReLU(),
                         nn.Linear(1024, 1024),
                         nn.ReLU(),
                         nn.Linear(1024, 512))
    
def normalize(x):
    return x / x.norm(dim=1, keepdim=True)

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
    
class LocationEncoder(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt

        self.LocEnc2500k = getLocationEncoder(2500)
        self.LocEnc750k = getLocationEncoder(750)
        self.LocEnc200k = getLocationEncoder(200)
        self.LocEnc25k = getLocationEncoder(25)
        self.LocEnc1k = getLocationEncoder(1)
        
    def forward(self, location, stochastic=False):
        location = location.float()
        L2500k = self.LocEnc2500k(location)
        L750k = self.LocEnc750k(location)
        L200k = self.LocEnc200k(location)
        L25k = self.LocEnc25k(location)
        L1k = self.LocEnc1k(location)
        
        if stochastic:
            w = torch.rand(5)
            w = w / w.sum()
            location_features = (w[0] * L2500k + w[1] * L750k + w[2] * L200k + w[3] * L25k + w[4] * L1k)
        else:
            location_features = (L2500k + L750k + L200k + L25k + L1k) / 5

        return location_features
    
class ImageEncoder(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.mlp = nn.Sequential(nn.Linear(768, 512))
        
    def forward(self, image):
        image_features = self.image_encoder(image).last_hidden_state
        image_features = image_features[:,0,:]
        image_features = self.mlp(image_features)
        return image_features
        
class GeoCLIP(nn.Module):
    def __init__(self,  input_resolution=224, opt=None):
        super().__init__()
        self.opt = opt
        self.input_resolution = input_resolution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.GPS_Aug_Multiplier = 4
        
        self.location_encoder = LocationEncoder(opt)
        self.image_encoder = ImageEncoder(opt)
        
        self.gps_mlp = nn.Sequential(nn.Linear(512, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 3))
        
        if self.opt.scene:
            self.scene_predictor3 = nn.Linear(512, 3)
            self.scene_predictor16 = nn.Linear(512, 16)
            self.scene_predictor365 = nn.Linear(512, 365)
                                             
    def forward(self, image, location):
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        gps_0 = self.gps_mlp(image_features)
        
        # Normalize features
        image_features = normalize(image_features)
        location_features = normalize(location_features)
        gps_0 = normalize(gps_0)
        scene_preds = None
        
        if self.opt.scene:
            scene_preds = [self.scene_predictor3(image_features),
                           self.scene_predictor16(image_features),
                           self.scene_predictor365(image_features)]

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        
        logits_per_image = logit_scale * (image_features @ location_features.t())
        logits_per_location = logits_per_image.t()

        return logits_per_image, logits_per_location, scene_preds, gps_0

    def predict(self, image, steps = 100):    
        image_features = self.image_encoder(image)
        location = self.gps_mlp(image_features) # GPS_0
        location = normalize(location)
        location = toLatLon(location)
        location = torch.nn.Parameter(location.data, requires_grad=True)
        image_features = normalize(image_features)
        
        optimizer = torch.optim.SGD([location], lr=0.0001, momentum=0.9)
        
        with torch.no_grad():
            location = torch.nn.Parameter(location.data, requires_grad=True)
            
            for i in range(steps):
                print("Eval step: {}".format(i))
                location = location.detach()
                optimizer.zero_grad()
                
                # Forward pass
                location_features = self.location_encoder(toCartesian(location), stochastic=True)
                location_features = normalize(location_features)
                similarity = image_features @ location_features.t()
                loss = -torch.log(torch.sigmoid(similarity)).mean()
                loss.backward()
                
                # Update
                optimizer.step()
        
        return location.data

class ViT(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        
        self.coarse_classifier = nn.Linear(768, 2967)
        self.medium_classifier = nn.Linear(768, 6505)
        self.fine_classifier = nn.Linear(768, 11570)
        
    def forward(self, image):
        out = self.vit(image).last_hidden_state[:,0,:]
        
        coarse_out = self.coarse_classifier(out)
        medium_out = self.medium_classifier(out)
        fine_out = self.fine_classifier(out)

        return coarse_out, medium_out, fine_out

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = ResNet.from_pretrained('microsoft/resnet-18', output_hidden_states=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten(1,-1)

        self.coarse_classifier = nn.Linear(512, 2967)
        self.medium_classifier = nn.Linear(512, 6505)
        self.fine_classifier = nn.Linear(512, 11570)
    
    def forward(self, image):
        out = self.resnet(image).hidden_states[-1]
        out = self.flatten(self.avgpool(out))
        
        coarse_out = self.coarse_classifier(out)
        medium_out = self.medium_classifier(out)
        fine_out = self.fine_classifier(out)

        return coarse_out, medium_out, fine_out


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()


        self.resnet = ResNet.from_pretrained('microsoft/resnet-50', output_hidden_states=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten(1,-1)

        self.coarse_classifier = nn.Linear(2048, 10)
        self.medium_classifier = nn.Linear(2048, 10)
        self.fine_classifier = nn.Linear(2048, 10)
    
    def forward(self, image):
        out = self.resnet(image).hidden_states[-1]
        out = self.flatten(self.avgpool(out))
        
        coarse_out = self.coarse_classifier(out)
        medium_out = self.medium_classifier(out)
        fine_out = self.fine_classifier(out)

        return coarse_out, medium_out, fine_out

class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = ResNet.from_pretrained('microsoft/resnet-101', output_hidden_states=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten(1,-1)

        self.coarse_classifier = nn.Linear(2048, 10)
        self.medium_classifier = nn.Linear(2048, 10)
        self.fine_classifier = nn.Linear(2048, 10)
    
    def forward(self, image):
        out = self.resnet(image).hidden_states[-1]
        out = self.flatten(self.avgpool(out))
        
        coarse_out = self.coarse_classifier(out)
        medium_out = self.medium_classifier(out)
        fine_out = self.fine_classifier(out)

        return coarse_out, medium_out, fine_out

if __name__ == "__main__":
    # Test vit_model with random input
    image = torch.randn(10, 3, 224, 224)
    location = torch.randn(10, 3)
    model = GeoCLIP(opt=getopt())
    # model = ViT()
    # model = ResNet18()
    model.eval()
    with torch.no_grad():
        image_features, location_features, scenes_preds, gps_0 = model(image, location)
        
    print(image_features.dtype)
    print(location_features.dtype)

    # Plot Image features matrix as heatmap
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Get Targets (GPS Cosine Similarities)
    gps_n = location / location.norm(dim=1, keepdim=True)
    targets = (gps_n @ gps_n.t())

    torch.set_printoptions(edgeitems=30)

    # Compute the loss
    loss = 0
    img_loss = criterion(image_features, targets).float()
    gps_loss = criterion(location_features, targets).float()

    loss = (img_loss + gps_loss) / 2
    
    print(img_loss)
    print(gps_loss)
    print(loss)
    
    print(model.predict(image))
    
    print(toLatLon(gps_0))
    
    plt.figure(figsize=(10,10))
    plt.imshow(image_features, cmap='viridis')
    plt.colorbar()
    plt.show()
    
    