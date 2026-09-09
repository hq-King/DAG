import clip
from PIL import Image

import torch


device = "cpu" 
model, _ = clip.load("RN101")
model = model.float()

model = model.to(device)
image = torch.randn(8, 3, 224, 224).to(device)
image_features = model.encode_image(image)
print(image_features.shape)








# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]