from Transformers import ViTModel

model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
