from importlib.metadata import requires
import sched

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
from coordinates import toCartesian, toLatLon
from feature_map import plot_feature_map

def getLocationEncoder(km):
    Earth_Diameter = 12742
    sigma = Earth_Diameter / (3 * km)
    rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
    return nn.Sequential(rff_encoding,
                         nn.Linear(512, 1024),
                         nn.ReLU(),
                         nn.Linear(1024, 1024),
                         nn.ReLU(),
                         nn.Linear(1024, 1024),
                         nn.ReLU(),
                         nn.Linear(1024, 512))

def augmentGPS(coords, opt):
    # Augment the GPS coordinates
    coords = coords.detach()
    eps = torch.randn_like(coords)
    eps = (F.normalize(eps, dim=1) * 5e-4).to(opt.device)
    coords = coords + eps
    coords = F.normalize(coords, dim=1)
    return coords

class LocationEncoderCapsule(nn.Module):
    def __init__(self, km):
        super(LocationEncoderCapsule, self).__init__()
        Earth_Diameter = 12742
        sigma = Earth_Diameter / (3 * km)
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=3, encoded_size=256)
        self.km = km

        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())

        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x
    
class LocationEncoder(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt

        self.LocEnc2500k = LocationEncoderCapsule(km=2500)
        self.LocEnc750k = LocationEncoderCapsule(km=750)
        self.LocEnc200k = LocationEncoderCapsule(km=200)
        self.LocEnc25k = LocationEncoderCapsule(km=25)
        self.LocEnc1k = LocationEncoderCapsule(km=1)
        
    def forward(self, location):
        location = location.float()
        # location = location / 180
        L2500k = self.LocEnc2500k(location)
        L750k = self.LocEnc750k(location)
        L200k = self.LocEnc200k(location)
        L25k = self.LocEnc25k(location)
        L1k = self.LocEnc1k(location)
        
        location_features = L2500k + L750k + L200k + L25k + L1k
        
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
        self.K = opt.queue_size
        
        self.input_resolution = input_resolution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.image_encoder = ImageEncoder(opt)
        self.location_encoder = LocationEncoder(opt)
        
        # Create GPS queue
        self.register_buffer("gps_queue", torch.randn(3, self.K))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        if self.opt.scene:
            self.scene_predictor3 = nn.Linear(dim, 3)
            self.scene_predictor16 = nn.Linear(dim, 16)
            self.scene_predictor365 = nn.Linear(dim, 365)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, gps):
        opt = self.opt
        gps_batch_size = gps.shape[0]
        batch_size = opt.batch_size

        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + batch_size) % self.K  # move pointer
        self.gps_queue_ptr[0] = gps_ptr
                                             
    def forward(self, image, location, train=False):
        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)

        logit_scale = self.logit_scale.exp()
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        scene_preds = None
        if self.opt.scene:
            scene_preds = [self.scene_predictor3(image_features),
                           self.scene_predictor16(image_features),
                           self.scene_predictor365(image_features)]
        
        # logits_per_location_self_attention = None
        if train:
            # Get the queues
            location_queue = self.gps_queue.t().detach()

            # Get the queue features
            with torch.no_grad():
                location_queue_features = self.location_encoder(location_queue)

            # Normalize the queue features
            location_queue_features = F.normalize(location_queue_features, dim=1)

            # Concatenate Features
            location_features = torch.cat(location_features, location_queue_features, dim=0)

            # Add Encodings to Queue
            self._dequeue_and_enqueue(location)

        # Cosine similarity (Image Features - Location Feature Queue)
        logits_per_image = logit_scale * (image_features @ location_features.t())
        logits_per_location = logits_per_image.t()

        return logits_per_image, logits_per_location, scene_preds


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
            image_features, location_features, scenes_pred = model(image, location, train=True)
        
    print(image_features.dtype)
    print(location_features.dtype)

    # Plot Image features matrix as heatmap
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Get Targets (GPS Cosine Similarities)
    targets = torch.arange(opt.batch_size, dtype=torch.long)

    torch.set_printoptions(edgeitems=30)
    
    plot_feature_map(model)
    

    