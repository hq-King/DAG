import datetime
import os
import warnings
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from odise.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor
from odise.modeling.backbone.feature_extractor import FeatureExtractorBackbone
from torchvision import transforms

model = FeatureExtractorBackbone(
    feature_extractor=LdmImplicitCaptionerExtractor(
        encoder_block_indices=(5, 7),
        unet_block_indices=(2, 5, 8, 11),
        decoder_block_indices=(2, 5),
        steps=(0,),
        learnable_time_embed=True,
        num_timesteps=1,
        clip_model_name="ViT-L-14-336",
    ),
    out_features=["s2", "s3", "s4", "s5"],
    use_checkpoint=False,
    slide_training=True,
).cuda()
checkpoint = torch.load("/root/ODISE-main/feature_extractor.pth")
model.load_state_dict(checkpoint, strict=False)

image_path = "/root/ovam/examples/ge_image.png"  # 替换成你的图片路径
image = Image.open(image_path).convert("RGB")


# 预处理图像，转换为张量
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 使用与训练一致的均值和标准差
])

# 将图像转换为张量
image_tensor = transform(image)

# 如果模型需要批量数据，需要为图像张量添加批量维度
image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]

# 将图像张量移动到 GPU
image_tensor = image_tensor.cuda() 
with torch.no_grad():
    features = model(image_tensor)
feats = []
for stage, feat in features.items():
    print(f"Feature map from stage {stage}: {feat.shape}")

    if stage in ["s4", "s5"]:
        feats.append(F.interpolate(feat, size=(256, 352), mode="bilinear", align_corners=False))
feats = torch.cat(feats, dim=1).cuda()
print(feats.shape)
