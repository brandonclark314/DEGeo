from transformers import ViTModel, ViTFeatureExtractor

import numpy as np
import torch
import torch.nn.functional as F
from torch import logit, nn
import matplotlib.pyplot as plt
from rff.layers import GaussianEncoding

def getLocationEncoder(a):
    Earth_Diameter = 12742
    sigma = Earth_Diameter / (3 * Earth_Diameter / 2**a)
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
    def __init__(self,  input_resolution=224):
        super().__init__()

        self.L2 = nn.functional.normalize
        self.GPS_Aug_Multiplier = 8
        
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)

        self.location_encoder0 = getLocationEncoder(0)
        self.location_encoder1 = getLocationEncoder(1)
        self.location_encoder2 = getLocationEncoder(2)
        self.location_encoder3 = getLocationEncoder(3)
        self.location_encoder4 = getLocationEncoder(4)
        self.location_encoder5 = getLocationEncoder(5)
        self.location_encoder6 = getLocationEncoder(6)
        self.location_encoder7 = getLocationEncoder(7)
        self.location_encoder8 = getLocationEncoder(8)
        self.location_encoder9 = getLocationEncoder(9)
        self.location_encoder10 = getLocationEncoder(10)
        self.location_encoder11 = getLocationEncoder(11)
        self.location_encoder12 = getLocationEncoder(12)
        self.location_encoder13 = getLocationEncoder(13)
        self.location_encoder14 = getLocationEncoder(14)

        self.mlp = nn.Sequential(nn.Linear(768, 512))
        self.input_resolution = input_resolution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        return self.image_encoder(image)
        
    def encode_location(self, location):
        location = location.float()
        return [self.location_encoder0(location),
                self.location_encoder1(location),
                self.location_encoder2(location),
                self.location_encoder3(location),
                self.location_encoder4(location),
                self.location_encoder5(location),
                self.location_encoder6(location),
                self.location_encoder7(location),
                self.location_encoder8(location),
                self.location_encoder9(location),
                self.location_encoder10(location),
                self.location_encoder11(location),
                self.location_encoder12(location),
                self.location_encoder13(location),
                self.location_encoder14(location)]
                                             
    def forward(self, image, location):
        image_features = self.encode_image(image).last_hidden_state 
        location_features0, location_features1, location_features2, \
        location_features3, location_features4, location_features5, \
        location_features6, location_features7, location_features8, \
        location_features9, location_features10, location_features11, \
        location_features12, location_features13, location_features14 = self.encode_location(location)


        image_features = image_features[:,0,:]
        image_features = self.mlp(image_features)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features0 = location_features1 / location_features1.norm(dim=1, keepdim=True)
        location_features1 = location_features1 / location_features1.norm(dim=1, keepdim=True)
        location_features2 = location_features2 / location_features2.norm(dim=1, keepdim=True)
        location_features3 = location_features3 / location_features3.norm(dim=1, keepdim=True)
        location_features4 = location_features4 / location_features4.norm(dim=1, keepdim=True)
        location_features5 = location_features5 / location_features5.norm(dim=1, keepdim=True)
        location_features6 = location_features6 / location_features6.norm(dim=1, keepdim=True)
        location_features7 = location_features7 / location_features7.norm(dim=1, keepdim=True)
        location_features8 = location_features8 / location_features8.norm(dim=1, keepdim=True)
        location_features9 = location_features9 / location_features9.norm(dim=1, keepdim=True)
        location_features10 = location_features10 / location_features10.norm(dim=1, keepdim=True)
        location_features11 = location_features11 / location_features11.norm(dim=1, keepdim=True)
        location_features12 = location_features12 / location_features12.norm(dim=1, keepdim=True)
        location_features13 = location_features13 / location_features13.norm(dim=1, keepdim=True)
        location_features14 = location_features14 / location_features14.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        
        s = nn.Sigmoid()
        
        # Get probabilities (similarities) from each encoder
        p0 = s(image_features @ location_features0.t())
        p1 = s(image_features @ location_features1.t())
        p2 = s(image_features @ location_features2.t())
        p3 = s(image_features @ location_features3.t())
        p4 = s(image_features @ location_features4.t())
        p5 = s(image_features @ location_features5.t())
        p6 = s(image_features @ location_features6.t())
        p7 = s(image_features @ location_features7.t())
        p8 = s(image_features @ location_features8.t())
        p9 = s(image_features @ location_features9.t())
        p10 = s(image_features @ location_features10.t())
        p11 = s(image_features @ location_features11.t())
        p12 = s(image_features @ location_features12.t())
        p13 = s(image_features @ location_features13.t())
        p14 = s(image_features @ location_features14.t())
        
        P = 1 / (1 + (1 / p0 - 1) * \
                     (1 / p1 - 1) * \
                     (1 / p2 - 1) * \
                     (1 / p3 - 1) * \
                     (1 / p4 - 1) * \
                     (1 / p5 - 1) * \
                     (1 / p6 - 1) * \
                     (1 / p7 - 1) * \
                     (1 / p8 - 1) * \
                     (1 / p9 - 1) * \
                     (1 / p10 - 1) * \
                     (1 / p11 - 1) * \
                     (1 / p12 - 1) * \
                     (1 / p13 - 1) * \
                     (1 / p14 - 1))
        
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
    
    