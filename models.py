from transformers import ViTModel

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Model(nn.Module):
    def __init__(self,  input_resolution=224):
        super().__init__()
        
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.location_encoder = nn.Sequential(nn.Linear(2, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 64),
                                              nn.ReLU()
                                              # <-----
                                              )


        self.input_resolution = input_resolution
                                 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        
    def encode_image(self, image):
        return self.image_encoder(image)       
        
    def encode_location(self, location):
        return self.location_encoder(location)
                                             
    def forward(self, image, location):
        image_features = self.encode_image(image)
        location_features = self.encode_location(location)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_location = location_features / location_features.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ location_features.t()
        logits_per_location = logits_per_image.t()

        return logits_per_image, logits_per_location
    

if __name__ == "__main__":
    # Test vit_model with random input
    image = torch.randn(1, 3, 224, 224)
    location = torch.randn(1, 2)
    model = Model()
    model.eval()
    with torch.no_grad():
        image_features, location_features = model(image, location)
        
    print(image_features.shape)
    print(location_features.shape)
    
    