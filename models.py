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
        self.GPS_Aug_Multiplier = 4
        self.Earth_Diameter = 12742 # km
        
        # Sigma Values (1km, 200km, 2500km)
        sigma1 = self.Earth_Diameter / (3 * 0.25)
        sigma2 = self.Earth_Diameter / (3 * 0.5)
        sigma3 = self.Earth_Diameter / (3 * 1)
        sigma4 = self.Earth_Diameter / (3 * 25)
        sigma5 = self.Earth_Diameter / (3 * 200)
        sigma6 = self.Earth_Diameter / (3 * 750)
        sigma7 = self.Earth_Diameter / (3 * 2500) 
        
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.rff_encoding1 = GaussianEncoding(sigma=sigma1, input_size=3, encoded_size=256)
        self.rff_encoding2 = GaussianEncoding(sigma=sigma2, input_size=3, encoded_size=256)
        self.rff_encoding3 = GaussianEncoding(sigma=sigma3, input_size=3, encoded_size=256)
        self.rff_encoding4 = GaussianEncoding(sigma=sigma4, input_size=3, encoded_size=256)
        self.rff_encoding5 = GaussianEncoding(sigma=sigma5, input_size=3, encoded_size=256)
        self.rff_encoding6 = GaussianEncoding(sigma=sigma6, input_size=3, encoded_size=256)
        self.rff_encoding7 = GaussianEncoding(sigma=sigma7, input_size=3, encoded_size=256)
        
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
                                                
        self.location_encoder4 = nn.Sequential(self.rff_encoding4,
                                                nn.Linear(512, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 512))
        
        self.location_encoder5 = nn.Sequential(self.rff_encoding5,
                                                nn.Linear(512, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 512))

        self.location_encoder6 = nn.Sequential(self.rff_encoding6,
                                                nn.Linear(512, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 512))
        
        self.location_encoder7 = nn.Sequential(self.rff_encoding7,
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
        
    def encode_image(self, image):
        return self.image_encoder(image)
        
    def encode_location(self, location):
        location = location.float()
        return [self.location_encoder1(location),
                self.location_encoder2(location),
                self.location_encoder3(location),
                self.location_encoder4(location),
                self.location_encoder5(location),
                self.location_encoder6(location),
                self.location_encoder7(location)]
                                             
    def forward(self, image, location):
        image_features = self.encode_image(image).last_hidden_state
        location_features1, location_features2, \
        location_features3, location_features4, \
        location_features5, location_features6, \
        location_features7 = self.encode_location(location)

        image_features = image_features[:,0,:]
        image_features = self.mlp(image_features)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features1 = location_features1 / location_features1.norm(dim=1, keepdim=True)
        location_features2 = location_features2 / location_features2.norm(dim=1, keepdim=True)
        location_features3 = location_features3 / location_features3.norm(dim=1, keepdim=True)
        location_features4 = location_features4 / location_features4.norm(dim=1, keepdim=True)
        location_features5 = location_features5 / location_features5.norm(dim=1, keepdim=True)
        location_features6 = location_features6 / location_features6.norm(dim=1, keepdim=True)
        location_features7 = location_features7 / location_features7.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        
        s = nn.Sigmoid()
        
        # Get probabilities (similarities) from each encoder
        p1 = s(image_features @ location_features1.t())
        p2 = s(image_features @ location_features2.t())
        p3 = s(image_features @ location_features3.t())
        p4 = s(image_features @ location_features4.t())
        p5 = s(image_features @ location_features5.t())
        p6 = s(image_features @ location_features6.t())
        p7 = s(image_features @ location_features7.t())
        
        P = 1 / (1 + (1 / p1 - 1) * \
                     (1 / p2 - 1) * \
                     (1 / p3 - 1) * \
                     (1 / p4 - 1) * \
                     (1 / p5 - 1) * \
                     (1 / p6 - 1) * \
                     (1 / p7 - 1))
        
        logits_per_image = logit_scale * P
          
        logits_per_location = logits_per_image.t()

        return logits_per_image, logits_per_location
    

if __name__ == "__main__":
    # Test vit_model with random input
    image = torch.randn(10, 3, 224, 224)
    location = torch.randn(10, 3)
    model = GeoCLIP()
    model.eval()
    with torch.no_grad():
        image_features, location_features = model(image, location)
        
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
    
    