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

def getRandomGPS(n):
    coords = 2 * torch.rand(n, 3) - 1
    coords = coords / coords.norm(dim=1, keepdim=True)
    return coords
    
class LocationEncoder(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt

        self.queue = []

        self.LocEnc2500k = getLocationEncoder(2500)
        self.LocEnc750k = getLocationEncoder(750)
        self.LocEnc200k = getLocationEncoder(200)
        self.LocEnc25k = getLocationEncoder(25)
        self.LocEnc1k = getLocationEncoder(1)
        
    def forward(self, location):
        location = location.float()
        L2500k = self.LocEnc2500k(location)
        L750k = self.LocEnc750k(location)
        L200k = self.LocEnc200k(location)
        L25k = self.LocEnc25k(location)
        L1k = self.LocEnc1k(location)
        
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
    
# ================ Variational Autoencoder =============== #

class Encoder(torch.nn.Module):
    def __init__(self, D_in=512, latent_size=128):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.enc_mu = torch.nn.Linear(256, latent_size)
        self.enc_log_sigma = torch.nn.Linear(256, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        return torch.distributions.Normal(loc=mu, scale=sigma)

class Decoder(torch.nn.Module):
    def __init__(self, D_in=128, D_out=512):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 256)
        self.linear2 = torch.nn.Linear(256, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = torch.tanh(self.linear2(x))
        return torch.distributions.Normal(mu, torch.ones_like(mu))

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()
        return self.decoder(z), q_z
        
class GeoCLIP(nn.Module):
    def __init__(self,  input_resolution=224, opt=None, dim = 512):
        super().__init__()
        self.opt = opt
        self.K = opt.batch_size * opt.queue_bs_multiplier # Queue Size
        # self.K = 4096
        self.m = 0.999 # MoCo Momentum
        self.T = 0.07 # Softmax temperature
        
        self.input_resolution = input_resolution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.image_encoder = ImageEncoder(opt)
        self.location_encoder = LocationEncoder(opt)
        
        self.momentum_image_encoder = ImageEncoder(opt)
        self.momentum_location_encoder = LocationEncoder(opt)
        
        self.VAE = VAE(Encoder(512, 128), Decoder(128, 512))
        
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
        
                                             
    def forward(self, image, location, train=False):
        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        
        # Cosine similarity as logits (Image Features - Location Features)
        logit_scale = self.logit_scale.exp()
        
        logits_per_image = logit_scale * (image_features @ location_features.t())
        logits_per_location = logits_per_image.t()
        
        scene_preds = None
            
        if self.opt.scene:
            scene_preds = [self.scene_predictor3(image_features),
                           self.scene_predictor16(image_features),
                           self.scene_predictor365(image_features)]
        
        logits_per_image_momentum = logits_per_location_momentum = None
        if train:
            # Compute Momentum Features
            with torch.no_grad():
                self._momentum_update() # update the momentum encoders
            
                # Compute Momentum Features
                momentum_image_features = self.momentum_image_encoder(image)
                momentum_location_features = self.momentum_location_encoder(location)
            
            # Normalize Momentum Features
            momentum_image_features = F.normalize(momentum_image_features, dim=1)
            momentum_location_features = F.normalize(momentum_location_features, dim=1)
                
            # Get Positive + Negatives
            image_embeddings = torch.cat([momentum_image_features.t(), self.img_queue.clone().detach()], dim=1)
            location_embeddings = torch.cat([momentum_location_features.t(), self.loc_queue.clone().detach()], dim=1)
            
            # Cosine similarity (Image Features - Momentum Location Feature Queue)
            logits_per_image_momentum = logit_scale * (image_features @ location_embeddings)
            
            # Cosine similarity (Location Features - Momentum Image Feature Queue)
            logits_per_location_momentum = logit_scale * (location_features @ image_embeddings)
            
            # Add Encodings to Queue
            self._dequeue_and_enqueue(momentum_image_features, momentum_location_features)
        
        # Variational AutoEncoder Predictions
        vae_preds = self.VAE(location_features.detach())
        
        # Predict Regularization Terms
        for param in self.VAE.parameters():
            param.requires_grad = False
        
        randomGPSfeatures = self.location_encoder(getRandomGPS(128).to(self.opt.device))
        vae_reg_preds = self.VAE(randomGPSfeatures)
        
        for param in self.VAE.parameters():
            param.requires_grad = True
            
        VAEData = dict(location_features=location_features,
                         randomGPSfeatures = randomGPSfeatures,
                         vae_preds=vae_preds,
                         vae_reg_preds=vae_reg_preds,)

        return logits_per_image, logits_per_location, scene_preds, logits_per_image_momentum, logits_per_location_momentum, VAEData

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
    

    