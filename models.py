from transformers import ViTModel, ViTFeatureExtractor

import numpy as np
import torch
import torch.nn.functional as F
from torch import logit, nn
import matplotlib.pyplot as plt
from rff.layers import GaussianEncoding

class GeoCLIP(nn.Module):
    def __init__(self,  input_resolution=224):
        super().__init__()

        self.L2 = nn.functional.normalize
        
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.rff_encoding = GaussianEncoding(sigma=10.0, input_size=3, encoded_size=600)
        self.location_encoder = nn.Sequential(self.rff_encoding,
                                              nn.Linear(1200, 1100),
                                              nn.ReLU(),
                                              nn.Linear(1100, 900),
                                              nn.ReLU(),
                                              nn.Linear(900, 500),
                                              nn.ReLU(),
                                              nn.Linear(500, 128))
        
        self.mlp = nn.Sequential(nn.Linear(768, 128))
        
        self.input_resolution = input_resolution
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
        
        image_similarity_matrix = image_features @ image_features.t()

        return logits_per_image, logits_per_location, image_similarity_matrix
    

if __name__ == "__main__":
    # Test vit_model with random input
    image = torch.randn(10, 3, 224, 224)
    location = torch.randn(10, 3)
    model = GeoCLIP()
    model.eval()
    with torch.no_grad():
        image_features, location_features, img_sim = model(image, location)
        
    print(image_features.dtype)
    print(location_features.dtype)
    print(img_sim.dtype)

    # Plot Image features matrix as heatmap
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
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_sim, cmap='viridis')
    plt.colorbar()
    plt.show()
    
    