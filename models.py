import encodings
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
from infonce import InfoNCE

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
    def __init__(self, km=1, opt=None):
        super().__init__()
        self.opt = opt

        Earth_Diameter = 12742
        sigma = Earth_Diameter / (3 * km)
        
        self.rff_encoding = GaussianEncoding(sigma=sigma, input_size=3, encoded_size=256)
        self.L1 = nn.Linear(512, 1024)
        self.L2 = nn.Linear(1024, 1024)
        self.L3 = nn.Linear(1024, 1024)
        self.L4 = nn.Linear(1024, 1024)
        self.L5 = nn.Linear(1536, 1024)
        self.L6 = nn.Linear(1024, 1024)
        self.L7 = nn.Linear(1024, 1024)
        self.L8 = nn.Linear(1024, 1024)
        self.L9 = nn.Linear(1024, 512)
        
    def forward(self, location):
        location = location.float()
        x_enc = self.rff_encoding(location)
        x = self.L1(x_enc)
        x = F.relu(x)
        x = self.L2(x)
        x = F.relu(x)
        x = self.L3(x)
        x = F.relu(x)
        x = self.L4(x)
        x = F.relu(x)
        x = self.L5(torch.cat([x, x_enc], dim=1))
        x = F.relu(x)
        x = self.L6(x)
        x = F.relu(x)
        x = self.L7(x)
        x = F.relu(x)
        x = self.L8(x)
        x = F.relu(x)
        x = self.L9(x)
        x = F.relu(x)
        return x
    
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
    
class GPSGaussianDecoder(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.gps_decoder = nn.Sequential(nn.Linear(512, 512),
                                          nn.ReLU(),
                                          nn.Linear(512, 512),
                                          nn.ReLU(),
                                          nn.Linear(512, 512),
                                          nn.ReLU(),
                                          nn.Linear(512, 512))
        
        self.gps_decoder_mean = nn.Sequential(nn.Linear(256, 3))
        self.gps_decoder_sigma = nn.Sequential(nn.Linear(256, 1))
        
    def forward(self, gps_features):
        gps_features = self.gps_decoder(gps_features)
        gps_mean = self.gps_decoder_mean(gps_features)
        gps_sigma = self.gps_decoder_sigma(gps_features)
        
        # Normalize Mean
        gps_mean = F.normalize(gps_mean, dim=1)
        
        # Make Sigma Positive
        gps_sigma = torch.exp(gps_sigma)
        
        return gps_mean, gps_sigma
        
class GeoCLIP(nn.Module):
    def __init__(self,  input_resolution=224, opt=None, dim = 512):
        super().__init__()
        self.opt = opt
        self.K = opt.batch_size * opt.queue_bs_multiplier # Queue Size
        self.m = 0.999 # MoCo Momentum
        self.T = 0.07 # Softmax temperature
        
        self.input_resolution = input_resolution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.image_encoder = ImageEncoder(opt=opt)
        self.location_encoder = LocationEncoder(opt=opt)
        
        self.momentum_image_encoder = ImageEncoder(opt=opt)
        self.momentum_location_encoder = LocationEncoder(opt=opt)
        
        # Copy encoders to momentum encoders
        for param, param_m in zip(self.image_encoder.parameters(), self.momentum_image_encoder.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        
        for param, param_m in zip(self.location_encoder.parameters(), self.momentum_location_encoder.parameters()):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False
        
        # create the queues
        self.register_buffer("img_queue", torch.randn(dim, self.K))
        self.img_queue = nn.functional.normalize(self.img_queue, dim=0)
        self.register_buffer("img_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.register_buffer("loc_queue", torch.randn(dim, self.K))
        self.loc_queue = nn.functional.normalize(self.loc_queue, dim=0)
        self.register_buffer("loc_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        if self.opt.scene:
            self.scene_predictor3 = nn.Linear(512, 3)
            self.scene_predictor16 = nn.Linear(512, 16)
            self.scene_predictor365 = nn.Linear(512, 365)
            
    @torch.no_grad()
    def _momentum_update(self):
        # Update Image Momentum Encoder
        for param, param_m in zip(self.image_encoder.parameters(), self.momentum_image_encoder.parameters()):
            param_m.data = param_m.data * self.m + param.data * (1. - self.m)
            
        # Update Location Momentum Encoder
        for param, param_m in zip(self.location_encoder.parameters(), self.momentum_location_encoder.parameters()):
            param_m.data = param_m.data * self.m + param.data * (1. - self.m)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_keys, loc_keys):
        opt = self.opt
        img_batch_size = img_keys.shape[0]
        loc_batch_size = loc_keys.shape[0]
        batch_size = opt.batch_size

        img_ptr = int(self.img_queue_ptr)
        loc_ptr = int(self.loc_queue_ptr)
        
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.img_queue[:, img_ptr:img_ptr + img_batch_size] = img_keys.T
        img_ptr = (img_ptr + batch_size) % self.K  # move pointer
        self.img_queue_ptr[0] = img_ptr
        
        self.loc_queue[:, loc_ptr:loc_ptr + loc_batch_size] = loc_keys.T
        loc_ptr = (loc_ptr + batch_size) % self.K  # move pointer
        self.loc_queue_ptr[0] = loc_ptr
        
                                             
    def forward(self, image, location, train=True):
        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        
        # Compute Momentum Features
        with torch.no_grad():
            if train:
                self._momentum_update() # update the momentum encoders
            
            # Compute Momentum Features
            momentum_image_features = self.momentum_image_encoder(image)
            momentum_location_features = self.momentum_location_encoder(location)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        momentum_image_features = F.normalize(momentum_image_features, dim=1)
        momentum_location_features = F.normalize(momentum_location_features, dim=1)
        scene_preds = None
        
        if self.opt.scene:
            scene_preds = [self.scene_predictor3(image_features),
                           self.scene_predictor16(image_features),
                           self.scene_predictor365(image_features)]
            
        # Get Positive + Negatives
        image_embeddings = torch.cat([momentum_image_features.t(), self.img_queue.clone().detach()], dim=1)
        location_embeddings = torch.cat([momentum_location_features.t(), self.loc_queue.clone().detach()], dim=1)

        # Cosine similarity as logits (Image Features - Location Features)
        logit_scale = self.logit_scale.exp()
        
        logits_per_image = logit_scale * (image_features @ location_features.t())
        logits_per_location = logits_per_image.t()
        
        # Cosine similarity (Image Features - Momentum Location Feature Queue)
        logits_per_image_momentum = logit_scale * (image_features @ location_embeddings)
        
        # Cosine similarity (Location Features - Momentum Image Feature Queue)
        logits_per_location_momentum = logit_scale * (location_features @ image_embeddings)
        
        if train:
            # Add Encodings to Queue
            self._dequeue_and_enqueue(momentum_image_features, momentum_location_features)

        return logits_per_image, logits_per_location, scene_preds, logits_per_image_momentum, logits_per_location_momentum

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
            image_features, location_features, scenes_pred, image_features_momentum, location_features_momentum = model(image, location)
        
    print(image_features.dtype)
    print(location_features.dtype)

    # Plot Image features matrix as heatmap
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Get Targets (GPS Cosine Similarities)
    targets = torch.arange(opt.batch_size, dtype=torch.long)

    torch.set_printoptions(edgeitems=30)
    
    loss = criterion(image_features_momentum, targets)

    # Compute the loss
    # loss = 0
    # img_loss = criterion(image_features_momentum, targets).float()
    # gps_loss = criterion(location_features_momentum, targets).float()

    # loss = (img_loss + gps_loss) / 2
    
    # print(img_loss)
    # print(gps_loss)
    # print(loss)

    print(loss)
    
    plt.figure(figsize=(10,10))
    plt.imshow(image_features_momentum, cmap='viridis', interpolation='none')
    print(location_features_momentum.shape)
    plt.colorbar()
    plt.show()
    

    