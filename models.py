from importlib.metadata import requires
import sched
from typing_extensions import Self

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
from coordinates import toCartesian, toLatLon

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
    
class LocationEncoderBase(nn.Module):
    def __init__(self, km=2500, opt=None):
        super().__init__()
        self.opt = opt
        Earth_Diameter = 12742
        
        self.sigma = Earth_Diameter / (3 * km)
        self.rff_encoding = GaussianEncoding(sigma=self.sigma, input_size=3, encoded_size=256)
        self.mlp = nn.Sequential(nn.Linear(512, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512))

    def forward(self, location):
        location = location.float()
        x = self.rff_encoding(location)
        x = self.mlp(x)
        return x
    
class LocationEncoderCapsule(nn.Module):
    def __init__(self, km=1, opt=None):
        super().__init__()
        self.opt = opt
        Earth_Diameter = 12742
        
        self.sigma = Earth_Diameter / (3 * km)
        self.layer_norm = nn.LayerNorm(512)
        self.rff_encoding = GaussianEncoding(sigma=self.sigma, input_size=3, encoded_size=256)
        self.mlp = nn.Sequential(nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512))

    def forward(self, location, z):
        location = location.float()
        z = nn.ReLU()(z)
        x = self.rff_encoding(location)
        x = torch.cat((x, z), dim=1)
        x = self.mlp(x)
    
        return x
    
class LocationEncoder(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt

        self.LocEnc2500k = LocationEncoderBase(km=2500, opt=opt)
        self.LocEnc750k = LocationEncoderCapsule(km=750, opt=opt)
        self.LocEnc200k = LocationEncoderCapsule(km=200, opt=opt)
        self.LocEnc25k = LocationEncoderCapsule(km=25, opt=opt)
        self.LocEnc1k = LocationEncoderCapsule(km=1, opt=opt)
        
    def forward(self, location):
        location = location.float()
        L2500k = self.LocEnc2500k(location)
        D750k = self.LocEnc750k(location, L2500k)
        D200k = self.LocEnc200k(location, L2500k + D750k)
        D25k = self.LocEnc25k(location, L2500k + D750k + D200k)
        D1k = self.LocEnc1k(location, L2500k + D750k + D200k + D25k)

        location_features = L2500k + D750k + D200k + D25k + D1k
        
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
    def __init__(self,  input_resolution=224, opt=None, dim = 512):
        super().__init__()
        self.opt = opt
        
        self.input_resolution = input_resolution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.image_encoder = ImageEncoder(opt)
        self.location_encoder = LocationEncoder(opt)
        
        if self.opt.scene:
            self.scene_predictor3 = nn.Linear(512, 3)
            self.scene_predictor16 = nn.Linear(512, 16)
            self.scene_predictor365 = nn.Linear(512, 365)
                                             
    def forward(self, image, location, train=False):
        # Compute Features
        image_features = self.image_encoder(image)
        location_features, scales_features = self.location_encoder(location)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        for scale in scales_features:
            scales_features[scale] = F.normalize(scales_features[scale], dim=1)
        
        # Cosine similarity as logits (Image Features - Location Features)
        logit_scale = self.logit_scale.exp()
        
        logits_per_image = logit_scale * (image_features @ location_features.t())
        logits_per_location = logits_per_image.t()

        logits_per_location_scales = {}
        for scale in scales_features:
            logits_per_location_scales[scale] = logit_scale * (scales_features[scale] @ image_features.t())
        
        scene_preds = None
            
        if self.opt.scene:
            scene_preds = [self.scene_predictor3(image_features),
                           self.scene_predictor16(image_features),
                           self.scene_predictor365(image_features)]

        return logits_per_image, logits_per_location, scene_preds, logits_per_location_scales

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
    opt = getopt()
    model = GeoCLIP(opt=opt)
    model.eval()
    
    # model = ViT()
    # model = ResNet18()
    for i in range(1):
        image = torch.randn(32, 3, 224, 224)
        location = torch.randn(32, 3)
        print("Image: ", i)
        with torch.no_grad():
            image_features, location_features, scenes_pred, img_loss, gps_loss = model(image, location, train=True)
        
    print(image_features.dtype)
    print(location_features.dtype)

    # Plot Image features matrix as heatmap
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Get Targets (GPS Cosine Similarities)
    targets = torch.arange(opt.batch_size, dtype=torch.long)

    torch.set_printoptions(edgeitems=30)
    
    print("Image Loss: ", img_loss)
    print("GPS Loss: ", gps_loss)

    # Compute the loss
    # loss = 0
    # img_loss = criterion(image_features_momentum, targets).float()
    # gps_loss = criterion(location_features_momentum, targets).float()

    # loss = (img_loss + gps_loss) / 2
    
    # print(img_loss)
    # print(gps_loss)
    # print(loss)

    # print(loss)
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(image_features_momentum, cmap='viridis', interpolation='none')
    # print(location_features_momentum.shape)
    # plt.colorbar()
    # plt.show()
    

    