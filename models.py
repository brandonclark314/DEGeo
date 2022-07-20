from transformers import ViTModel
from transformers import ResNetForImageClassification as ResNet

import numpy as np
import torch
import torch.nn.functional as F
from torch import logit, nn
import matplotlib.pyplot as plt
from rff.layers import GaussianEncoding
import torchvision.models as models

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

class GeoCLIP(nn.Module):
    def __init__(self,  input_resolution=224, opt=None):
        super().__init__()

        self.opt = opt
        self.GPS_Aug_Multiplier = 4
        
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.rff_encoding = GaussianEncoding(sigma=10.0, input_size=3, encoded_size=256)
        
        self.location_encoder1 = getLocationEncoder(2500)
        self.location_encoder2 = getLocationEncoder(750)
        self.location_encoder3 = getLocationEncoder(200)
        self.location_encoder4 = getLocationEncoder(25)
        self.location_encoder5 = getLocationEncoder(1)
        
        self.mlp = nn.Sequential(nn.Linear(768, 512))
        
        self.input_resolution = input_resolution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.opt.scene:
            self.scene_predictor3 = nn.Linear(512, 3)
            self.scene_predictor16 = nn.Linear(512, 16)
            self.scene_predictor365 = nn.Linear(512, 365)
        
    def encode_image(self, image):
        return self.image_encoder(image)
        
    def encode_location(self, location):
        location = location.float()
        return [self.location_encoder1(location),
                self.location_encoder2(location),
                self.location_encoder3(location),
                self.location_encoder4(location),
                self.location_encoder5(location)]
                                             
    def forward(self, image, location):
        image_features = self.encode_image(image).last_hidden_state
        location_features1, location_features2, location_features3, location_features4, \
        location_features5 = self.encode_location(location)

        image_features = image_features[:,0,:]
        image_features = self.mlp(image_features)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features1 = location_features1 / location_features1.norm(dim=1, keepdim=True)
        location_features2 = location_features2 / location_features2.norm(dim=1, keepdim=True)
        location_features3 = location_features3 / location_features3.norm(dim=1, keepdim=True)
        location_features4 = location_features4 / location_features4.norm(dim=1, keepdim=True)
        location_features5 = location_features5 / location_features5.norm(dim=1, keepdim=True)

        scene_preds = None
        if self.opt.scene:
            scene_preds = [self.scene_predictor3(image_features),
                           self.scene_predictor16(image_features),
                           self.scene_predictor365(image_features)]

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        
        s = nn.Sigmoid()
        
        # Get probabilities (similarities) from each encoder
        p1 = s(image_features @ location_features1.t())
        p2 = s(image_features @ location_features2.t())
        p3 = s(image_features @ location_features3.t())
        p4 = s(image_features @ location_features4.t())
        p5 = s(image_features @ location_features5.t())
        
        P = 1 / (1 + (1 / p1 - 1) * \
                     (1 / p2 - 1) * \
                     (1 / p3 - 1) * \
                     (1 / p4 - 1) * \
                     (1 / p5 - 1))
        
        logits_per_image = logit_scale * P
          
        logits_per_location = logits_per_image.t()
        
        # Gps Similarity
        location_features = (location_features1 + location_features2 + location_features3 + location_features4 + location_features5) / 5
        location_features = location_features / location_features.norm(dim=1, keepdim=True)
        location = location / location.norm(dim=1, keepdim=True)

        gps_features_similarity = location_features @ location_features.t()
        gps_location_similarity = location @ location.t()
        
        # Normalize
        gps_features_similarity = (gps_features_similarity / gps_features_similarity.norm(dim=1, keepdim=True)).to(torch.float64)
        gps_location_similarity = (gps_location_similarity / gps_location_similarity.norm(dim=1, keepdim=True)).to(torch.float64)
        
        gps_sim = gps_features_similarity @ gps_location_similarity.t()
        gps_sim = (gps_sim + 1) / 2
        gps_sim_loss = -torch.log(gps_sim)
        gps_sim_loss = gps_loss.mean()

        return logits_per_image, logits_per_location, scene_preds, gps_sim_loss

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
    # model = GeoCLIP()
    # model = ViT()
    # model = ResNet18()
    model.eval()
    with torch.no_grad():
        image_features, location_features, scenes_preds = model(image, location)
        
    print(image_features.dtype)
    print(location_features.dtype)

    # Plot Image features matrix as heatmap
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.MSELoss()
    
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
    
    plt.figure(figsize=(10,10))
    plt.imshow(image_features, cmap='viridis')
    plt.colorbar()
    plt.show()
    
    