from transformers import ViTModel, ViTFeatureExtractor

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

class GeoCLIP(nn.Module):
    def __init__(self,  input_resolution=224):
        super().__init__()
        
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.location_encoder = nn.Sequential(nn.Linear(3, 1000),
                                              nn.ReLU(),
                                              nn.Linear(1000, 1000),
                                              nn.ReLU(),
                                              nn.Linear(1000,512)
                                              )
        
        self.mlp = nn.Sequential(nn.Linear(768, 512))

        self.input_resolution = input_resolution

        self.L2 = nn.functional.normalize
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        return self.image_encoder(image)
        
    def encode_location(self, location):
        location = location.float()
        return self.location_encoder(location)
                                             
    def forward(self, image, location):
        image_features = self.encode_image(image).last_hidden_state
        location_features = self.encode_location(location)

        image_features = image_features[:,0,:]
        image_features = self.mlp(image_features)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features = location_features / location_features.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ location_features.t()
        logits_per_location = logits_per_image.t()

        return logits_per_image, logits_per_location
    

if __name__ == "__main__":
    # Test vit_model with random input
    image = torch.randn(8, 3, 224, 224)
    location = torch.randn(8, 3)
    model = GeoCLIP()
    model.eval()
    with torch.no_grad():
        image_features, location_features = model(image, location)
        
    print(image_features.shape)
    print(location_features.shape)
    
    # Plot Image features matrix as heatmap
    image_features = image_features.cpu().numpy()
    
    plt.figure(figsize=(10,10))
    plt.imshow(image_features, cmap='hot')
    plt.colorbar()
    plt.show()
    
    