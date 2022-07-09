from transformers import ViTModel, ViTFeatureExtractor

import numpy as np
import torch
import torch.nn.functional as F
from torch import logit, nn
import matplotlib.pyplot as plt
from rff.layers import GaussianEncoding
import torchvision.transforms as T

class GeoCLIP(nn.Module):
    def __init__(self,  input_resolution=224):
        super().__init__()

        self.img_augmentation = T.Compose([ T.Resize((224,224)),
                                            T.ColorJitter(hue=.05, saturation=.05),
                                            T.RandomHorizontalFlip(),
                                            T.RandomPerspective(distortion_scale=0.6, p=1.0),
                                            T.RandomSolarize(threshold=192.0),
                                            T.RandomAutocontrast()
                                            ])
        self.L2 = nn.functional.normalize
        self.Earth_Diameter = 12742 # km
        
        # Sigma Values (1km, 200km, 2500km)
        sigma1 = self.Earth_Diameter / (3 * 1)
        sigma2 = self.Earth_Diameter / (3 * 200)
        sigma3 = self.Earth_Diameter / (3 * 2500)
        
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.rff_encoding1 = GaussianEncoding(sigma=sigma1, input_size=3, encoded_size=256)
        self.rff_encoding2 = GaussianEncoding(sigma=sigma2, input_size=3, encoded_size=256)
        self.rff_encoding3 = GaussianEncoding(sigma=sigma3, input_size=3, encoded_size=256)
        
        self.location_encoder1 = nn.Sequential(self.rff_encoding1,
                                              nn.Linear(512, 1024),
                                              nn.ReLU(),
                                              nn.Linear(1024, 1024),
                                              nn.ReLU(),
                                              nn.Linear(1024, 1024),
                                              nn.ReLU(),
                                              nn.Linear(1024, 512))
        
        self.location_encoder2 = nn.Sequential(self.rff_encoding2,
                                              nn.Linear(512, 1024),
                                              nn.ReLU(),
                                              nn.Linear(1024, 1024),
                                              nn.ReLU(),
                                              nn.Linear(1024, 1024),
                                              nn.ReLU(),
                                              nn.Linear(1024, 512))
        
        self.location_encoder3 = nn.Sequential(self.rff_encoding3,
                                                nn.Linear(512, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 512))
        
        self.mlp = nn.Sequential(nn.Linear(768, 512))
        
        self.input_resolution = input_resolution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_feat = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        return self.image_encoder(image)
        
    def encode_location(self, location):
        location = location.float()
        return [self.location_encoder1(location),
                self.location_encoder2(location),
                self.location_encoder3(location)]
                                             
    def forward(self, image, location):
        image_features = self.encode_image(image).last_hidden_state
        location_features1, location_features2, \
        location_features3 = self.encode_location(location)
        
        # Augmented Image
        augmented_image = self.img_augmentation(image)
        image_aug_features = self.encode_image(augmented_image).last_hidden_state
        image_aug_features = self.mlp(image_aug_features)
        image_aug_features = image_aug_features[:,0,:]

        image_features = image_features[:,0,:]
        image_features = self.mlp(image_features)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features1 = location_features1 / location_features1.norm(dim=1, keepdim=True)
        location_features2 = location_features2 / location_features2.norm(dim=1, keepdim=True)
        location_features3 = location_features3 / location_features3.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale_feat = self.logit_scale_feat.exp()
        
        logits_per_image = logit_scale * ((image_features @ location_features1.t()) * \
                                          (image_features @ location_features2.t()) * \
                                          (image_features @ location_features3.t()))
          
        logits_per_location = logits_per_image.t()
        
        image_similarity = logit_scale_feat * (image_features @ image_aug_features.t())

        return logits_per_image, logits_per_location, image_similarity
    

if __name__ == "__main__":
    # Test vit_model with random input
    image = torch.randn(10, 3, 224, 224)
    location = torch.randn(10, 3)
    model = GeoCLIP()
    model.eval()
    with torch.no_grad():
        image_features, location_features, image_similarity = model(image, location)
        
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
    
    